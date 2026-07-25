/*
 * 04_qk_gqa_spm_mh.cpp — Method 1: Multi-head persistent SPM for QK attention
 *
 * Problem: For small K (d=64), K cache per kv head = d×T×2B = 64×512×2 = 64KB.
 *   Current 04_qk_gqa_spm: SPMCP is called once per query head invocation,
 *   meaning the SAME kv head's K matrix is SPMCP'd G=4 times (one per query head).
 *
 * Fix 1 (SPM-G): SPMCP once per kv head, compute G query heads from same SPM data.
 *   → G× fewer SPMCPs regardless of T.
 *
 * Fix 2 (SPM-MH): Pack H_BATCH kv heads simultaneously into SPM.
 *   n_heads_per_spm = SPM_SETS_EFF / (n_tiles × KC × SPM_KC_FACTOR)
 *   → T=128 (VL=4): H_BATCH=8 (all kv heads fit) → 8×G× fewer SPMCPs
 *   → T=256: H_BATCH=5 → 5×G× fewer SPMCPs
 *   → T=512: H_BATCH=2 → 2×G× fewer SPMCPs
 *   → T=1024: H_BATCH=1 (= SPM-G) → G× fewer SPMCPs
 *
 * Three methods compared across T sweep:
 *   A) SPM-CURRENT : gemm_spm() per query head (G=4 redundant SPMCPs per kv head)
 *   B) SPM-G       : SPMCP once per kv head, compute G times from slot 0
 *   C) SPM-MH      : SPMCP H_BATCH kv heads at once, compute G×H_BATCH from SPM
 *
 * Usage: ./04_qk_gqa_spm_mh_vl4 [H_q] [H_kv] [d] [MC] [KC] [n_iter]
 *        default: 32 8 64 128 64 3
 * (T sweep is built-in: 64, 128, 192, 256, 384, 512, 768, 1024)
 */

#include "kernels_spm.hpp"
#include <cmath>

// ============================================================
// SPM capacity constants (must match kernels_spm.hpp)
// ============================================================
#if defined(VL_2)
static constexpr size_t SPM_KC_FACTOR = 1;
static constexpr size_t SPM_SETS_EFF  = 2048;   // 2 K-rows per 32B set
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

static constexpr size_t MT = 8;   // micro-kernel tile height (rows)

// ============================================================
// Helpers shared with 04_qk_gqa_spm.cpp
// ============================================================
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

static inline void copy_scores(const __fp16* C_pad, __fp16* Sh,
                                int T, size_t T_padded) {
    for (int r = 0; r < T; ++r)
        std::memcpy(Sh + (size_t)r * T,
                    C_pad + (size_t)r * T_padded,
                    (size_t)T * sizeof(__fp16));
}

static void scale_fp16(__fp16* p, size_t n, float s) {
    svbool_t pg = svptrue_b16();
    svfloat16_t vs = svdup_f16((__fp16)s);
    size_t VLh = svcnth(), i = 0;
    for (; i + VLh <= n; i += VLh)
        svst1_f16(pg, p+i, svmul_f16_x(pg, svld1_f16(pg, p+i), vs));
    for (; i < n; ++i) p[i] = (__fp16)((float)p[i] * s);
}

// ============================================================
// Multi-head SPM helpers
// ============================================================

// Number of kv heads that fit simultaneously in SPM
static size_t calc_n_heads_per_spm(size_t n_tiles, size_t KC) {
    return std::max<size_t>(1, SPM_SETS_EFF / (n_tiles * KC * SPM_KC_FACTOR));
}

// SPMCP K_T[d, T_padded] into SPM slot 'slot'.
// Slot 'slot' uses sets [slot*n_tiles*KC*KCF .. (slot+1)*n_tiles*KC*KCF - 1]
static void spmcp_head_to_slot(const __fp16* K_T, size_t T_padded,
                                size_t kc, size_t KC, size_t NT,
                                size_t n_tiles, size_t slot) {
    for (size_t ti = 0; ti < n_tiles; ++ti) {
        size_t base_set = (slot * n_tiles + ti) * KC * SPM_KC_FACTOR;
        pack_B_tile_to_spm(K_T + ti * NT, T_padded, kc, KC, base_set);
    }
}

// Compute GEMM reading K from SPM slot 'slot' (no SPMCP).
// A[M,K] (row-major, lda=K), C[M,N] (row-major, ldc=N).
// K must equal KC (single K-tile).
// A_cache[max_ab * MT * KC] and Cpanel[max_ab * MT * NT] are pre-allocated scratch.
static void gemm_from_spm_slot(
        const __fp16* A, __fp16* C,
        size_t M, size_t N, size_t K,  // N is padded (stride)
        size_t MC, size_t KC, size_t NT,
        size_t n_tiles, size_t slot,
        __fp16* A_cache, __fp16* Cpanel) {

    size_t max_ab = (MC + MT - 1) / MT;

    // Phase 1: pack ALL A MC-blocks (reused across all N-tiles)
    for (size_t m0 = 0; m0 < M; m0 += MC) {
        pack_A_fp16_8row(A, K,
                         A_cache + (m0 / MC) * max_ab * MT * KC,
                         M, m0, K, 0,
                         std::min(MC, M - m0), KC);
    }

    // Phase 2: compute each N-tile from SPM
    for (size_t ti = 0; ti < n_tiles; ++ti) {
        size_t n0  = ti * NT;
        size_t nc  = std::min(NT, N - n0);
        size_t base_set = (slot * n_tiles + ti) * KC * SPM_KC_FACTOR;

        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc      = std::min(MC, M - m0);
            size_t ablocks = (mc + MT - 1) / MT;
            const __fp16* Ap = A_cache + (m0 / MC) * max_ab * MT * KC;

            spm_kernel_mblocks(Ap, Cpanel, (int)K, (int)ablocks, KC, base_set);

            // Scatter Cpanel → C[m0:m0+mc, n0:n0+nc]
            for (size_t mb = 0; mb < ablocks; ++mb) {
                const __fp16* tp = Cpanel + mb * MT * NT;
                for (size_t r = 0; r < MT; ++r) {
                    size_t gr = m0 + mb * MT + r;
                    if (gr >= M) break;
                    __fp16*       c_row = C + gr * N + n0;
                    const __fp16* t_row = tp + r * NT;
                    size_t i = 0;
                    svbool_t pg = svptrue_b16();
                    for (; i + svcnth() <= nc; i += svcnth())
                        svst1_f16(pg, c_row+i, svld1_f16(pg, t_row+i));
                    if (i < nc) {
                        svbool_t pt = svwhilelt_b16_u64(i, nc);
                        svst1_f16(pt, c_row+i, svld1_f16(pt, t_row+i));
                    }
                }
            }
        }
    }
}

// ============================================================
// Benchmark methods
// ============================================================

// Method A: current approach — gemm_spm() per query head
// (SPMCP is repeated G times for each kv head)
static double run_spm_current(
        const __fp16* Q, const __fp16* K, __fp16* scores,
        int T, int T_padded, int H_q, int H_kv, int G, int d,
        int D_q, int D_kv, size_t MC, size_t KC, float scale) {
    const size_t NT = 3 * svcnth();
    AlignedBuffer<__fp16> K_T((size_t)d * T_padded);
    AlignedBuffer<__fp16> Q_h((size_t)T * d);
    AlignedBuffer<__fp16> C_pad((size_t)T * T_padded);
    std::memset(K_T.data(),    0, (size_t)d * T_padded * sizeof(__fp16));
    std::memset(C_pad.data(),  0, (size_t)T * T_padded * sizeof(__fp16));

    auto t0 = Clock::now();
    for (int g = 0; g < H_kv; ++g) {
        transpose_K_head(K, K_T.data(), T, T_padded, d, D_kv, g);
        for (int qi = 0; qi < G; ++qi) {
            int h = g * G + qi;
            extract_Q_head(Q, Q_h.data(), T, d, D_q, h);
            gemm_spm(Q_h.data(), K_T.data(), C_pad.data(),
                     T, T_padded, d, T_padded, MC, KC);
            copy_scores(C_pad.data(), scores + (size_t)h * T * T, T, T_padded);
            scale_fp16(scores + (size_t)h * T * T, (size_t)T * T, scale);
        }
    }
    return us_since(t0);
}

// Method B: SPMCP once per kv head → G compute calls from slot 0
static double run_spm_g(
        const __fp16* Q, const __fp16* K, __fp16* scores,
        int T, int T_padded, int H_q, int H_kv, int G, int d,
        int D_q, int D_kv, size_t MC, size_t KC, float scale) {
    const size_t NT       = 3 * svcnth();
    const size_t n_tiles  = ((size_t)T_padded + NT - 1) / NT;
    const size_t kc       = (size_t)d;   // K = d = KC, single K-tile
    const size_t max_ab   = (MC + MT - 1) / MT;
    const size_t num_mc   = ((size_t)T + MC - 1) / MC;   // MC-blocks in M=T

    AlignedBuffer<__fp16> K_T((size_t)d * T_padded);
    AlignedBuffer<__fp16> Q_h((size_t)T * d);
    AlignedBuffer<__fp16> C_pad((size_t)T * T_padded);
    AlignedBuffer<__fp16> A_cache(num_mc * max_ab * MT * KC);
    AlignedBuffer<__fp16> Cpanel(max_ab * MT * NT);
    std::memset(K_T.data(),   0, (size_t)d * T_padded * sizeof(__fp16));
    std::memset(C_pad.data(), 0, (size_t)T * T_padded * sizeof(__fp16));

    auto t0 = Clock::now();
    for (int g = 0; g < H_kv; ++g) {
        transpose_K_head(K, K_T.data(), T, T_padded, d, D_kv, g);

        // SPMCP once into slot 0
        spmcp_head_to_slot(K_T.data(), T_padded, kc, KC, NT, n_tiles, 0);
        asm volatile("dsb sy\nisb\n" ::: "memory");

        // G query heads all read from slot 0
        for (int qi = 0; qi < G; ++qi) {
            int h = g * G + qi;
            extract_Q_head(Q, Q_h.data(), T, d, D_q, h);
            gemm_from_spm_slot(Q_h.data(), C_pad.data(),
                               T, T_padded, d,
                               MC, KC, NT, n_tiles, 0,
                               A_cache.data(), Cpanel.data());
            copy_scores(C_pad.data(), scores + (size_t)h * T * T, T, T_padded);
            scale_fp16(scores + (size_t)h * T * T, (size_t)T * T, scale);
        }
    }
    return us_since(t0);
}

// Method C: batch H_BATCH kv heads into SPM, compute G×H_BATCH query heads
static double run_spm_mh(
        const __fp16* Q, const __fp16* K, __fp16* scores,
        int T, int T_padded, int H_q, int H_kv, int G, int d,
        int D_q, int D_kv, size_t MC, size_t KC, float scale,
        size_t* out_H_BATCH = nullptr) {
    const size_t NT         = 3 * svcnth();
    const size_t n_tiles    = ((size_t)T_padded + NT - 1) / NT;
    const size_t kc         = (size_t)d;
    const size_t H_BATCH    = calc_n_heads_per_spm(n_tiles, KC);
    const size_t max_ab     = (MC + MT - 1) / MT;
    if (out_H_BATCH) *out_H_BATCH = H_BATCH;

    // K_batch: H_BATCH × [d × T_padded]
    const size_t num_mc   = ((size_t)T + MC - 1) / MC;   // MC-blocks in M=T
    AlignedBuffer<__fp16> K_batch((size_t)H_BATCH * d * T_padded);
    AlignedBuffer<__fp16> Q_h((size_t)T * d);
    AlignedBuffer<__fp16> C_pad((size_t)T * T_padded);
    AlignedBuffer<__fp16> A_cache(num_mc * max_ab * MT * KC);
    AlignedBuffer<__fp16> Cpanel(max_ab * MT * NT);
    std::memset(K_batch.data(), 0, (size_t)H_BATCH * d * T_padded * sizeof(__fp16));
    std::memset(C_pad.data(),   0, (size_t)T * T_padded * sizeof(__fp16));

    auto t0 = Clock::now();
    for (int g0 = 0; g0 < H_kv; g0 += (int)H_BATCH) {
        int h_batch = (int)std::min(H_BATCH, (size_t)(H_kv - g0));

        // Phase 1: transpose + SPMCP all h_batch kv heads
        for (int hb = 0; hb < h_batch; ++hb) {
            __fp16* K_T = K_batch.data() + (size_t)hb * d * T_padded;
            transpose_K_head(K, K_T, T, T_padded, d, D_kv, g0 + hb);
            spmcp_head_to_slot(K_T, T_padded, kc, KC, NT, n_tiles, (size_t)hb);
        }
        // Drain all SPMCPs before compute
        asm volatile("dsb sy\nisb\n" ::: "memory");

        // Phase 2: compute G×h_batch query heads from SPM
        for (int hb = 0; hb < h_batch; ++hb) {
            for (int qi = 0; qi < G; ++qi) {
                int h = (g0 + hb) * G + qi;
                extract_Q_head(Q, Q_h.data(), T, d, D_q, h);
                gemm_from_spm_slot(Q_h.data(), C_pad.data(),
                                   T, T_padded, d,
                                   MC, KC, NT, n_tiles, (size_t)hb,
                                   A_cache.data(), Cpanel.data());
                copy_scores(C_pad.data(), scores + (size_t)h * T * T, T, T_padded);
                scale_fp16(scores + (size_t)h * T * T, (size_t)T * T, scale);
            }
        }
    }
    return us_since(t0);
}

// ============================================================
// Correctness check
// ============================================================
static void ref_gemm(const __fp16* A, const __fp16* B, __fp16* C,
                     int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += (float)A[(size_t)m*K+k] * (float)B[(size_t)k*N+n];
            C[(size_t)m*N+n] = (__fp16)acc;
        }
}

static float max_abs_err(const __fp16* a, const __fp16* b, size_t n) {
    float e = 0.f;
    for (size_t i = 0; i < n; ++i)
        e = std::max(e, std::fabs((float)a[i] - (float)b[i]));
    return e;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    const int H_q    = (argc > 1) ? std::atoi(argv[1]) : 32;
    const int H_kv   = (argc > 2) ? std::atoi(argv[2]) : 8;
    const int d      = (argc > 3) ? std::atoi(argv[3]) : 64;
    const size_t MC  = (argc > 4) ? (size_t)std::atoi(argv[4]) : 128;
    const size_t KC  = (argc > 5) ? (size_t)std::atoi(argv[5]) : (size_t)d;
    const int N_ITER = (argc > 6) ? std::atoi(argv[6]) : 3;
    const int G      = H_q / H_kv;
    const int D_q    = H_q  * d;
    const int D_kv   = H_kv * d;
    const size_t NT  = 3 * svcnth();
    const float scale = 1.0f / sqrtf((float)d);

    printf("=== 04_qk_gqa_spm_mh ===  H_q=%d  H_kv=%d  G=%d  d=%d  MC=%zu  KC=%zu\n",
           H_q, H_kv, G, d, MC, KC);
    printf("  VL=%zu-bit  NT=%zu  SPM_KC_FACTOR=%zu  SPM_SETS_EFF=%zu\n\n",
           svcnth() * 16, NT, SPM_KC_FACTOR, SPM_SETS_EFF);

    // ---- Correctness check at small T ----
    {
        const int Tc = 96;   // one NT tile
        const size_t T_padded_c = NT;
        AlignedBuffer<__fp16> Q_c((size_t)Tc * D_q),  K_c((size_t)Tc * D_kv);
        AlignedBuffer<__fp16> sc_cur((size_t)H_q * Tc * Tc);
        AlignedBuffer<__fp16> sc_g(  (size_t)H_q * Tc * Tc);
        AlignedBuffer<__fp16> sc_mh( (size_t)H_q * Tc * Tc);
        AlignedBuffer<__fp16> sc_ref((size_t)H_q * Tc * Tc);

        fill_random(Q_c.data(), (size_t)Tc*D_q,  -0.1f, 0.1f, 1);
        fill_random(K_c.data(), (size_t)Tc*D_kv, -0.1f, 0.1f, 2);

        // Reference: per-head scalar GEMM
        AlignedBuffer<__fp16> K_T_ref((size_t)d * T_padded_c);
        AlignedBuffer<__fp16> Q_h_ref((size_t)Tc * d);
        std::memset(K_T_ref.data(), 0, (size_t)d * T_padded_c * sizeof(__fp16));
        for (int g = 0; g < H_kv; ++g) {
            transpose_K_head(K_c.data(), K_T_ref.data(), Tc, (int)T_padded_c, d, D_kv, g);
            for (int qi = 0; qi < G; ++qi) {
                int h = g*G+qi;
                extract_Q_head(Q_c.data(), Q_h_ref.data(), Tc, d, D_q, h);
                ref_gemm(Q_h_ref.data(), K_T_ref.data(),
                         sc_ref.data() + (size_t)h * Tc * Tc, Tc, Tc, d);
                scale_fp16(sc_ref.data() + (size_t)h*Tc*Tc, (size_t)Tc*Tc, scale);
            }
        }

        run_spm_current(Q_c.data(), K_c.data(), sc_cur.data(),
                        Tc, (int)T_padded_c, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
        run_spm_g(Q_c.data(), K_c.data(), sc_g.data(),
                  Tc, (int)T_padded_c, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
        run_spm_mh(Q_c.data(), K_c.data(), sc_mh.data(),
                   Tc, (int)T_padded_c, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);

        float e_cur = max_abs_err(sc_cur.data(), sc_ref.data(), (size_t)H_q*Tc*Tc);
        float e_g   = max_abs_err(sc_g.data(),   sc_ref.data(), (size_t)H_q*Tc*Tc);
        float e_mh  = max_abs_err(sc_mh.data(),  sc_ref.data(), (size_t)H_q*Tc*Tc);
        printf("[CORRECT Tc=%d]  SPM-CURRENT err=%.5f %s\n", Tc, e_cur, e_cur<0.05f?"PASS":"FAIL");
        printf("[CORRECT Tc=%d]  SPM-G       err=%.5f %s\n", Tc, e_g,   e_g  <0.05f?"PASS":"FAIL");
        printf("[CORRECT Tc=%d]  SPM-MH      err=%.5f %s\n\n", Tc, e_mh, e_mh <0.05f?"PASS":"FAIL");
        fflush(stdout);
    }

    // ---- T sweep ----
    const int T_sweep[] = {64, 128, 192, 256, 384, 512, 768, 1024};
    const int N_T = (int)(sizeof(T_sweep) / sizeof(T_sweep[0]));
    const double total_flops_per_T = 2.0 * H_q * d;  // per T²; multiply later

    printf("%-6s  %-5s  %-10s  %-12s  %-12s  %-12s  %-8s  %-8s\n",
           "T", "H_BSZ", "n_SPMCP", "SPM-CUR(us)", "SPM-G(us)", "SPM-MH(us)",
           "G/CUR", "MH/CUR");
    printf("%s\n", std::string(90, '-').c_str());

    for (int ti = 0; ti < N_T; ++ti) {
        const int T = T_sweep[ti];
        const size_t NT_loc  = 3 * svcnth();
        const size_t T_padded = ((size_t)T + NT_loc - 1) / NT_loc * NT_loc;
        const size_t n_tiles  = T_padded / NT_loc;
        const size_t H_BATCH  = calc_n_heads_per_spm(n_tiles, KC);

        // Number of SPMCP calls per method for the full H_q computation:
        // CURRENT: H_kv × G × n_tiles SPMCP calls
        // SPM-G:   H_kv × n_tiles SPMCP calls
        // SPM-MH:  ceil(H_kv/H_BATCH) × H_BATCH × n_tiles = H_kv × n_tiles SPMCP calls
        //          (same as SPM-G, but packed in fewer waves → better cache locality for SPMCP src)
        // [SPMCP count shown as number of "head-tile" units: head × n_tiles]
        size_t n_spmcp_cur = (size_t)H_kv * G * n_tiles;
        size_t n_spmcp_g   = (size_t)H_kv * n_tiles;
        size_t n_spmcp_mh  = ((size_t)H_kv + H_BATCH - 1) / H_BATCH * H_BATCH * n_tiles;

        AlignedBuffer<__fp16> Q_t((size_t)T * D_q),  K_t((size_t)T * D_kv);
        AlignedBuffer<__fp16> sc_cur((size_t)H_q * T * T);
        AlignedBuffer<__fp16> sc_g(  (size_t)H_q * T * T);
        AlignedBuffer<__fp16> sc_mh( (size_t)H_q * T * T);
        fill_random(Q_t.data(), (size_t)T*D_q,  -0.1f, 0.1f, (unsigned)T+1);
        fill_random(K_t.data(), (size_t)T*D_kv, -0.1f, 0.1f, (unsigned)T+2);

        // Warm-up (1 run to populate caches)
        run_spm_current(Q_t.data(), K_t.data(), sc_cur.data(),
                        T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
        run_spm_g(Q_t.data(), K_t.data(), sc_g.data(),
                  T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
        run_spm_mh(Q_t.data(), K_t.data(), sc_mh.data(),
                   T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);

        // Timed runs (average of N_ITER)
        double t_cur = 0, t_g = 0, t_mh = 0;
        for (int it = 0; it < N_ITER; ++it) {
            t_cur += run_spm_current(Q_t.data(), K_t.data(), sc_cur.data(),
                                     T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
            t_g   += run_spm_g(Q_t.data(), K_t.data(), sc_g.data(),
                               T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
            t_mh  += run_spm_mh(Q_t.data(), K_t.data(), sc_mh.data(),
                                T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, KC, scale);
        }
        t_cur /= N_ITER; t_g /= N_ITER; t_mh /= N_ITER;

        double total_flops = 2.0 * H_q * (double)T * T * d;
        printf("%-6d  %-5zu  %3zu/%3zu/%3zu    %10.1f    %10.1f    %10.1f  %6.2fx  %6.2fx"
               "   [%.2f/%.2f/%.2f GFLOPS]\n",
               T, H_BATCH,
               n_spmcp_cur, n_spmcp_g, n_spmcp_mh,
               t_cur, t_g, t_mh,
               t_cur / t_g, t_cur / t_mh,
               total_flops/(t_cur*1e3), total_flops/(t_g*1e3), total_flops/(t_mh*1e3));
        fflush(stdout);
    }

    printf("\nColumn 'n_SPMCP' = CURRENT/SPM-G/SPM-MH total SPMCP head-tile units\n");
    printf("  SPM-MH vs CURRENT: %.1f× fewer SPMCP waves at T=128 (H_BATCH=%zu)\n",
           (double)G * calc_n_heads_per_spm(
               (((size_t)128 + NT - 1) / NT * NT) / NT, KC),
           calc_n_heads_per_spm(
               (((size_t)128 + NT - 1) / NT * NT) / NT, KC));
    return 0;
}
