/*
 * 04_qk_gqa_spm.cpp — LLaMA GQA QK attention GEMM with CacheFlex SPM
 *
 * Matched comparison structure (aligned with 04_qk_gqa.cpp baseline):
 *   For each KV head g:
 *     pack_A for each Q-head (indexed by qi — no overwrite across heads)
 *     for each N-tile batch:
 *       Phase 1a: pack_K_direct (SVE gather) per tile → K_tile_buf[tiles_max,kc,NT]
 *       Phase 1b: SPMCP per tile from K_tile_buf → SPM (replaces scalar K_T transpose)
 *       for each Q-head qi:
 *         for each N-tile in batch:
 *           for each M-block:
 *             spm_kernel_mblocks(...)  → 24KB Cpanel
 *             scatter+scale → scores
 *
 * Implementation notes:
 *   - No K_T[d, T_padded] intermediate (270KB scalar transpose avoided)
 *   - Per-tile SVE gather-transpose (pack_K_direct) into K_tile_buf (identical to baseline)
 *   - A_cache indexed by qi (G × num_mc blocks, one slot per Q-head)
 *   - SPMCP from K_tile_buf[ti,kc,NT] with N_stride=NT (row-major tile)
 *   - Fully symmetric with baseline; only difference: SPMCP to SPM vs stay in L1/L2
 *
 * Compile: -DVL_2, -DVL_4, -DVL_8, or -DVL_16
 * Usage: ./04_qk_gqa_spm_vl4 [T] [H_q] [H_kv] [d] [n_iter] [KC] [MC]
 *        default: 512 32 8 64 5 64 128
 */
#include "kernels_spm.hpp"
#include <cmath>

// ----------------------------------------------------------------
// SPM capacity constants (must match kernels_spm.hpp)
// ----------------------------------------------------------------
#if defined(VL_2)
static constexpr size_t SPM_KC_FACTOR = 1;
static constexpr size_t SPM_SETS_EFF  = 2048;   // VL=2: 2 K-rows per 32B set
#elif defined(VL_8)
static constexpr size_t SPM_KC_FACTOR = 2;
static constexpr size_t SPM_SETS_EFF  = 1024;
#elif defined(VL_16)
static constexpr size_t SPM_KC_FACTOR = 3;
static constexpr size_t SPM_SETS_EFF  = 1024;
#else  // VL_4
static constexpr size_t SPM_KC_FACTOR = 1;
static constexpr size_t SPM_SETS_EFF  = 1024;
#endif

static constexpr size_t MT = 8;  // micro-kernel tile rows

// ----------------------------------------------------------------
// pack_K_direct: SVE gather-based transpose+pack K[T,d] → [d,NT] per tile.
// VL-independent (uses svcntw()/svcnth() at runtime).
// Same implementation as in llama_bench/kernels_sve.hpp.
// ----------------------------------------------------------------
static void pack_K_direct(uint16_t* out, const uint16_t* K_in, int T, int d,
                           size_t NT, int ldk = 0)
{
    if (!ldk) ldk = d;
    const size_t VLw = svcntw();
    const size_t VLh = svcnth();
    const size_t bblocks = ((size_t)T + NT - 1) / NT;

    const uint32_t bstride = (uint32_t)(ldk * sizeof(uint16_t));
    const svuint32_t voff  = svmul_n_u32_x(svptrue_b32(), svindex_u32(0u,1u), bstride);
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();

    for (size_t nb = 0; nb < bblocks; ++nb) {
        uint16_t*       dst  = out  + nb * NT * (size_t)d;
        const uint16_t* Ksub = K_in + nb * NT * (size_t)ldk;
        size_t nc = std::min(NT, (size_t)T - nb * NT);

        std::memset(dst, 0, NT * (size_t)d * sizeof(uint16_t));

        for (int k = 0; k < d; ++k) {
            uint16_t*       row_out  = dst  + (size_t)k * NT;
            const uint16_t* col_base = Ksub + (size_t)k;

            size_t n = 0;
            for (; n + VLh <= nc; n += VLh) {
                svuint32_t g0 = svld1uh_gather_u32offset_u32(pg32,
                    col_base + n       * ldk, voff);
                svuint32_t g1 = svld1uh_gather_u32offset_u32(pg32,
                    col_base + (n+VLw) * ldk, voff);
                svst1_u16(pg16, row_out + n,
                    svuzp1_u16(svreinterpret_u16_u32(g0), svreinterpret_u16_u32(g1)));
            }
            if (n + VLw <= nc) {
                svuint32_t g = svld1uh_gather_u32offset_u32(pg32,
                    col_base + n * ldk, voff);
                svst1_u16(svwhilelt_b16_u64((size_t)0, VLw), row_out + n,
                    svuzp1_u16(svreinterpret_u16_u32(g), svdup_n_u16(0)));
                n += VLw;
            }
            for (; n < nc; ++n)
                row_out[n] = col_base[n * ldk];
        }
    }
}

// ----------------------------------------------------------------
// Helper functions (used in correctness check only)
// ----------------------------------------------------------------
static void extract_Q_head(const __fp16* Q, __fp16* Q_h,
                            int T, int d, int D_q, int h) {
    for (int t = 0; t < T; ++t)
        std::memcpy(Q_h + (size_t)t * d,
                    Q   + (size_t)t * D_q + (size_t)h * d,
                    (size_t)d * sizeof(__fp16));
}

static void transpose_K_head(const __fp16* K, __fp16* K_T,
                              int T, int T_out, int d, int D_kv, int g) {
    for (int k = 0; k < d; ++k)
        for (int t = 0; t < T; ++t)
            K_T[(size_t)k * T_out + t] = K[(size_t)t * D_kv + (size_t)g * d + k];
}

static void scale_fp16(__fp16* p, size_t n, float s) {
    svbool_t pg = svptrue_b16();
    svfloat16_t vs = svdup_f16((__fp16)s);
    size_t VLh = svcnth(), i = 0;
    for (; i + VLh <= n; i += VLh)
        svst1_f16(pg, p+i, svmul_f16_x(pg, svld1_f16(pg, p+i), vs));
    for (; i < n; ++i) p[i] = (__fp16)((float)p[i] * s);
}

static void ref_gemm(const __fp16* A, const __fp16* B, __fp16* C,
                     int M, int N, int K, int ldb) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += (float)A[(size_t)m*K+k] * (float)B[(size_t)k*ldb+n];
            C[(size_t)m*N+n] = (__fp16)acc;
        }
}

// ----------------------------------------------------------------
// SPM capacity helpers
// ----------------------------------------------------------------
static size_t spm_tiles_per_batch(size_t kc) {
    return SPM_SETS_EFF / (kc * SPM_KC_FACTOR);
}

// ----------------------------------------------------------------
// Main QK attention loop — SPM version
// Identical structure to baseline run_qk_baseline except:
//   pack_K_direct → K_tile_buf → SPMCP to SPM (baseline: stays in L1/L2)
// ----------------------------------------------------------------
static void run_qk_spm(
        const __fp16* Q, const __fp16* K, __fp16* scores,
        int T, int T_padded, int H_q, int H_kv, int G, int d,
        int D_q, int D_kv, size_t MC, size_t KC, float scale,
        double& t_packB, double& t_packA, double& t_kernel, double& t_scatter)
{
    const size_t NT         = 3 * svcnth();
    const size_t n_tiles    = (size_t)T_padded / NT;
    const size_t kc         = (size_t)d;
    const size_t tiles_max  = spm_tiles_per_batch(kc);
    const size_t max_ab     = (MC + MT - 1) / MT;
    const size_t num_mc     = ((size_t)T + MC - 1) / MC;
    const svfloat16_t vs       = svdup_f16((__fp16)scale);
    const __fp16      scale_fp16 = (__fp16)scale;

    // A_cache: G × num_mc slots — qi=0..G-1 each get their own M-blocks
    AlignedBuffer<__fp16>   A_cache((size_t)G * num_mc * max_ab * MT * kc);
    // K_tile_buf: tiles_max × [kc, NT] — gather-transposed K tiles (row-major)
    // SPMCP reads from here; no 270KB K_T needed
    AlignedBuffer<uint16_t> K_tile_buf(tiles_max * kc * NT);
    // Cpanel: 24KB → stays in L1 (one M-block × one N-tile)
    AlignedBuffer<__fp16>   Cpanel(max_ab * MT * NT);

    std::memset(A_cache.data(), 0, (size_t)G * num_mc * max_ab * MT * kc * sizeof(__fp16));
    std::memset(Cpanel.data(),  0, max_ab * MT * NT * sizeof(__fp16));

    for (int g = 0; g < H_kv; ++g) {

        // --- Pack A: qi-indexed (no overwrite across G Q-heads) ---
        for (int qi = 0; qi < G; ++qi) {
            const int h = g * G + qi;
            auto ta = Clock::now();
            for (size_t m0 = 0; m0 < (size_t)T; m0 += MC) {
                pack_A_fp16_8row(Q + (size_t)h * d, (size_t)D_q,
                    A_cache.data() + ((size_t)qi * num_mc + m0/MC) * max_ab * MT * kc,
                    (size_t)T, m0, kc, 0,
                    std::min(MC, (size_t)T - m0), kc);
            }
            t_packA += us_since(ta);
        }

        // --- N-tile batching loop ---
        for (size_t ti_start = 0; ti_start < n_tiles; ti_start += tiles_max) {
            size_t tiles_this = std::min(tiles_max, n_tiles - ti_start);

            // Phase 1: SVE gather-transpose then SPMCP (identical path to baseline,
            //          only storage destination differs: SPM vs L1/L2)
            {
                auto tp = Clock::now();

                // Phase 1a: gather-transpose K tiles into K_tile_buf
                // Same pack_K_direct as baseline; K_tile_buf[ti,kc,NT] is row-major
                for (size_t ti = 0; ti < tiles_this; ++ti) {
                    size_t t_start = (ti_start + ti) * NT;
                    size_t nc      = std::min(NT, (size_t)T - t_start);
                    uint16_t* dst  = K_tile_buf.data() + ti * kc * NT;
                    const uint16_t* Kg = reinterpret_cast<const uint16_t*>(
                        K + t_start * D_kv + (size_t)g * d);
                    pack_K_direct(dst, Kg, (int)nc, d, NT, D_kv);
                }

                // Phase 1b: SPMCP all tiles from K_tile_buf → SPM
                // Fire all SPMCPs upfront → DRAM pipelines all requests
                // shadow table ensures spm.ld1qd waits on earlier-seqNum SPMCPs
                for (size_t ti = 0; ti < tiles_this; ++ti) {
                    size_t base_set = ti * kc * SPM_KC_FACTOR;
                    const __fp16* src = reinterpret_cast<const __fp16*>(
                        K_tile_buf.data() + ti * kc * NT);
                    pack_B_tile_to_spm(src, NT, kc, KC, base_set);
                }

                t_packB += us_since(tp);
            }

            // Phase 2: compute G Q-heads reading from SPM
            for (int qi = 0; qi < G; ++qi) {
                const int h = g * G + qi;

                for (size_t ti = 0; ti < tiles_this; ++ti) {
                    size_t n0       = (ti_start + ti) * NT;
                    if (n0 >= (size_t)T) break;
                    size_t nc       = std::min(NT, (size_t)T - n0);
                    size_t base_set = ti * kc * SPM_KC_FACTOR;

                    for (size_t m0 = 0; m0 < (size_t)T; m0 += MC) {
                        size_t mc      = std::min(MC, (size_t)T - m0);
                        size_t ablocks = (mc + MT - 1) / MT;
                        // qi-indexed A_cache — correct for G>1
                        const __fp16* Ap = A_cache.data() +
                            ((size_t)qi * num_mc + m0/MC) * max_ab * MT * kc;

                        if (nc == NT) {
                            // Full N-tile: fused kernel — scale+scatter directly
                            // from accumulator regs to scores, no Cpanel round-trip
                            auto tk = Clock::now();
                            __fp16* C_scat = scores + (size_t)h * T * T + m0 * T + n0;
                            spm_kernel_scatter_vl4(Ap, C_scat, (int)T, (int)kc,
                                                   (int)ablocks, base_set, scale_fp16);
                            t_kernel += us_since(tk);
                        } else {
                            // Partial N-tile: use old Cpanel path + scalar scatter
                            auto tk = Clock::now();
                            spm_kernel_mblocks(Ap, Cpanel.data(), (int)kc,
                                               (int)ablocks, kc, base_set);
                            t_kernel += us_since(tk);

                            auto ts = Clock::now();
                            __fp16* C = scores + (size_t)h * T * T;
                            for (size_t mb = 0; mb < ablocks; ++mb) {
                                const __fp16* tp2 = Cpanel.data() + mb * MT * NT;
                                for (size_t r = 0; r < MT; ++r) {
                                    size_t gr = m0 + mb * MT + r;
                                    if (gr >= (size_t)T) break;
                                    __fp16*       c_row = C + gr * T + n0;
                                    const __fp16* t_row = tp2 + r * NT;
                                    svbool_t pt = svwhilelt_b16_u64((size_t)0, nc);
                                    svst1_f16(pt, c_row,
                                        svmul_f16_x(pt, svld1_f16(pt, t_row), vs));
                                }
                            }
                            t_scatter += us_since(ts);
                        }
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------
int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int H_q    = (argc > 2) ? std::atoi(argv[2]) : 32;
    const int H_kv   = (argc > 3) ? std::atoi(argv[3]) : 8;
    const int d      = (argc > 4) ? std::atoi(argv[4]) : 64;
    const int N_ITER = (argc > 5) ? std::atoi(argv[5]) : 5;
    const size_t KC  = (argc > 6) ? (size_t)std::atoi(argv[6]) : (size_t)d;
    const size_t MC  = (argc > 7) ? (size_t)std::atoi(argv[7]) : 128;
    const size_t NT  = 3 * svcnth();
    const size_t T_padded = ((size_t)T + NT - 1) / NT * NT;
    const size_t n_tiles  = T_padded / NT;
    const int G = H_q / H_kv;

    const int D_q  = H_q  * d;
    const int D_kv = H_kv * d;
    const float scale = 1.0f / sqrtf((float)d);
    const size_t tiles_max = spm_tiles_per_batch((size_t)d);
    const size_t n_batches = (n_tiles + tiles_max - 1) / tiles_max;

    const double Q_MB     = (size_t)T * D_q  * 2.0 / 1048576;
    const double K_MB     = (size_t)T * D_kv * 2.0 / 1048576;
    const double score_MB = (size_t)H_q  * T * T * 2.0 / 1048576;

    printf("=== 04 QK GQA SPM ===  T=%d  H_q=%d  H_kv=%d  G=%d  d=%d  n_iter=%d\n",
           T, H_q, H_kv, G, d, N_ITER);
    printf("  Q      [%d, %d]  %.2f MB\n", T, D_q, Q_MB);
    printf("  K      [%d, %d]  %.2f MB\n", T, D_kv, K_MB);
    printf("  scores [%d, %d, %d]  %.2f MB\n", H_q, T, T, score_MB);
    printf("  NT=%zu  T_padded=%zu  KC=%zu  MC=%zu  n_tiles=%zu\n", NT, T_padded, KC, MC, n_tiles);
    printf("  tiles_max/batch=%zu  n_batches=%zu  (SPM_SETS_EFF=%zu)\n\n",
           tiles_max, n_batches, (size_t)SPM_SETS_EFF);
    printf("  K_tile_buf: %.1f KB  (tiles_max=%zu × d=%d × NT=%zu × 2B)\n\n",
           (double)tiles_max*d*NT*2/1024, tiles_max, d, NT);
    fflush(stdout);

    AlignedBuffer<__fp16> Q((size_t)T * D_q);
    AlignedBuffer<__fp16> K((size_t)T * D_kv);
    AlignedBuffer<__fp16> scores((size_t)H_q * T * T);

    fill_random(Q.data(), (size_t)T*D_q,  -0.1f, 0.1f, 1);
    fill_random(K.data(), (size_t)T*D_kv, -0.1f, 0.1f, 2);

    // ---- Correctness check (compare against reference scalar GEMM, head 0) ----
    {
        const int Tc = std::min(T, (int)NT);
        const size_t Tc_padded = NT;

        AlignedBuffer<__fp16> sc_spm((size_t)H_q * Tc * Tc);
        AlignedBuffer<__fp16> sc_ref((size_t)H_q * Tc * Tc);

        std::memset(sc_spm.data(), 0, (size_t)H_q*Tc*Tc*sizeof(__fp16));
        std::memset(sc_ref.data(), 0, (size_t)H_q*Tc*Tc*sizeof(__fp16));

        AlignedBuffer<__fp16> Qc_full((size_t)Tc * D_q);
        AlignedBuffer<__fp16> Kc_full((size_t)Tc * D_kv);
        for (int t = 0; t < Tc; ++t) {
            std::memcpy(Qc_full.data() + (size_t)t*D_q, Q.data() + (size_t)t*D_q, D_q*sizeof(__fp16));
            std::memcpy(Kc_full.data() + (size_t)t*D_kv, K.data() + (size_t)t*D_kv, D_kv*sizeof(__fp16));
        }

        // Reference: per-head scalar GEMM
        AlignedBuffer<__fp16> KgT_ref((size_t)d * Tc_padded);
        AlignedBuffer<__fp16> Qh_ref((size_t)Tc * d);
        for (int g = 0; g < H_kv; ++g) {
            std::memset(KgT_ref.data(), 0, (size_t)d * Tc_padded * sizeof(__fp16));
            transpose_K_head(Kc_full.data(), KgT_ref.data(), Tc, (int)Tc_padded, d, D_kv, g);
            for (int qi = 0; qi < G; ++qi) {
                int h = g*G+qi;
                extract_Q_head(Qc_full.data(), Qh_ref.data(), Tc, d, D_q, h);
                AlignedBuffer<__fp16> Ctmp((size_t)Tc * Tc);
                ref_gemm(Qh_ref.data(), KgT_ref.data(), Ctmp.data(), Tc, Tc, d, (int)Tc_padded);
                for (int r = 0; r < Tc; ++r)
                    std::memcpy(sc_ref.data() + (size_t)h*Tc*Tc + (size_t)r*Tc,
                                Ctmp.data() + (size_t)r*Tc, Tc*sizeof(__fp16));
                scale_fp16(sc_ref.data() + (size_t)h*Tc*Tc, (size_t)Tc*Tc, scale);
            }
        }

        // SPM GEMM (correctness check — timing discarded)
        double _tb=0,_ta=0,_tk=0,_ts=0;
        run_qk_spm(Qc_full.data(), Kc_full.data(), sc_spm.data(),
                   Tc, (int)Tc_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale,
                   _tb,_ta,_tk,_ts);

        float max_err = 0.f;
        for (size_t i = 0; i < (size_t)H_q*Tc*Tc; ++i)
            max_err = std::max(max_err, std::fabs((float)sc_spm.data()[i] - (float)sc_ref.data()[i]));
        fprintf(stderr, "[CORRECT] all heads, Tc=%d: max_abs_err=%.6f  %s\n",
                Tc, max_err, max_err < 0.05f ? "PASS" : "FAIL");
        fflush(stderr);
    }

    const double total_flops = 2.0 * H_q * T * T * d;

    // --- COLD ---
    flush_dcache_range(Q.data(),      (size_t)T*D_q  * sizeof(__fp16));
    flush_dcache_range(K.data(),      (size_t)T*D_kv * sizeof(__fp16));
    flush_dcache_range(scores.data(), (size_t)H_q*T*T * sizeof(__fp16));

    ROI_BEGIN();
    {
        double tb=0,ta=0,tk=0,ts=0;
        auto t0 = Clock::now();
        run_qk_spm(Q.data(), K.data(), scores.data(),
                   T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale,
                   tb, ta, tk, ts);
        double us = us_since(t0);
        printf("  COLD: %7.1f us   %.2f GFLOPS\n", us, total_flops/(us*1e3));
        printf("    packB(gather+SPMCP): %7.1f us  (%5.1f%%)\n", tb, 100.*tb/us);
        printf("    packA:               %7.1f us  (%5.1f%%)\n", ta, 100.*ta/us);
        printf("    kernel(spm.ld1qd):   %7.1f us  (%5.1f%%)\n", tk, 100.*tk/us);
        printf("    scatter(+scale):     %7.1f us  (%5.1f%%)\n\n", ts, 100.*ts/us);
        fflush(stdout);
    }
    ROI_END();

    // --- WARM (1 un-timed warmup) ---
    { double tb=0,ta=0,tk=0,ts=0;
      run_qk_spm(Q.data(), K.data(), scores.data(),
                 T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale,
                 tb,ta,tk,ts); }

    printf("--- WARM ---\n"); fflush(stdout);
    double t_total=0, tb_w=0, ta_w=0, tk_w=0, ts_w=0;
    for (int it = 0; it < N_ITER; ++it) {
        double tb=0,ta=0,tk=0,ts=0;
        auto t0 = Clock::now();
        run_qk_spm(Q.data(), K.data(), scores.data(),
                   T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale,
                   tb, ta, tk, ts);
        t_total += us_since(t0);
        tb_w+=tb; ta_w+=ta; tk_w+=tk; ts_w+=ts;
    }
    t_total/=N_ITER; tb_w/=N_ITER; ta_w/=N_ITER; tk_w/=N_ITER; ts_w/=N_ITER;
    printf("  WARM: %7.1f us/iter   %.2f GFLOPS\n", t_total, total_flops/(t_total*1e3));
    printf("    packB(gather+SPMCP): %7.1f us  (%5.1f%%)\n", tb_w, 100.*tb_w/t_total);
    printf("    packA:               %7.1f us  (%5.1f%%)\n", ta_w, 100.*ta_w/t_total);
    printf("    kernel(spm.ld1qd):   %7.1f us  (%5.1f%%)\n", tk_w, 100.*tk_w/t_total);
    printf("    scatter(+scale):     %7.1f us  (%5.1f%%)\n\n", ts_w, 100.*ts_w/t_total);

    printf("  SPM reuse: pack K_g once per N-tile batch, reuse for G=%d Q-heads\n", G);
    printf("  K_tile_buf %.1f KB (gather-transposed, row-major) → SPMCP to SPM\n",
           (double)tiles_max*d*NT*2/1024);
    return 0;
}
