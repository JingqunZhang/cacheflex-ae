/*
 * Grouped-query PV GEMM using SPM.
 *
 * H_BATCH = SPM_SETS_EFF / (ceil(T/KC) * KC * SPM_KC_FACTOR). Each resident
 * V head is reused by its associated query heads.
 *
 * Compile: -DVL_2, -DVL_4, -DVL_8, or -DVL_16
 * Usage: ./06_pv_gqa_spm_vl4 [T] [H_q] [H_kv] [d] [n_iter] [KC] [MC]
 *        default: 512 32 8 64 5 256 128
 */
#include "kernels_spm.hpp"

// ----------------------------------------------------------------
// SPM capacity constants (must match kernels_spm.hpp)
// ----------------------------------------------------------------
#if defined(VL_2)
static constexpr size_t SPM_KC_FACTOR = 1;
static constexpr size_t SPM_SETS_EFF  = 2048;
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

static constexpr size_t MT = 8;

// ----------------------------------------------------------------
// Extract V_g[T, d] → V_g_padded[T, NT_buf], zero-pad cols d..NT_buf-1
// ----------------------------------------------------------------
static void extract_V_head_padded(const __fp16* V, __fp16* V_pad,
                                   int T, int d, int D_kv, int g, size_t NT_buf) {
    std::memset(V_pad, 0, (size_t)T * NT_buf * sizeof(__fp16));
    for (int t = 0; t < T; ++t)
        std::memcpy(V_pad + (size_t)t * NT_buf,
                    V + (size_t)t * D_kv + (size_t)g * d,
                    (size_t)d * sizeof(__fp16));
}

// ----------------------------------------------------------------
// Scatter Cpanel[ablocks * MT * NT] → C[M, NT_buf], with optional accumulate
// ----------------------------------------------------------------
static inline void scatter_cpanel(const __fp16* Cpanel, __fp16* C_base,
                                   size_t m0, size_t M,
                                   size_t mc, size_t ablocks,
                                   size_t NT_buf, size_t nc,
                                   bool accumulate) {
    const size_t NT = 3 * svcnth();
    for (size_t mb = 0; mb < ablocks; ++mb) {
        const __fp16* tp = Cpanel + mb * MT * NT;
        for (size_t r = 0; r < MT; ++r) {
            size_t gr = m0 + mb * MT + r;
            if (gr >= M) break;
            __fp16*       c_row = C_base + gr * NT_buf;
            const __fp16* t_row = tp + r * NT;
            size_t i = 0;
            svbool_t pg = svptrue_b16();
            if (!accumulate) {
                for (; i + svcnth() <= nc; i += svcnth())
                    svst1_f16(pg, c_row+i, svld1_f16(pg, t_row+i));
                if (i < nc) {
                    svbool_t pt = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pt, c_row+i, svld1_f16(pt, t_row+i));
                }
            } else {
                // RMW accumulate: C += Cpanel
                for (; i + svcnth() <= nc; i += svcnth())
                    svst1_f16(pg, c_row+i,
                              svadd_f16_x(pg, svld1_f16(pg, c_row+i),
                                              svld1_f16(pg, t_row+i)));
                if (i < nc) {
                    svbool_t pt = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pt, c_row+i,
                              svadd_f16_x(pt, svld1_f16(pt, c_row+i),
                                              svld1_f16(pt, t_row+i)));
                }
            }
        }
    }
}

// ----------------------------------------------------------------
// Number of kv heads that fit simultaneously in SPM
// K_tiles = ceil(T/KC): each head needs K_tiles * KC sets
// ----------------------------------------------------------------
static size_t calc_h_batch_pv(size_t T, size_t KC) {
    const size_t K_tiles = (T + KC - 1) / KC;
    return std::max<size_t>(1, SPM_SETS_EFF / (K_tiles * KC * SPM_KC_FACTOR));
}

// ----------------------------------------------------------------
// PV loop with adaptive head batching.
// ----------------------------------------------------------------
static void run_pv_spm(
        const __fp16* P, const __fp16* V, __fp16* ctx,
        int T, int H_q, int H_kv, int G, int d,
        int D_kv, size_t MC, size_t KC, size_t NT_buf) {

    const size_t NT      = 3 * svcnth();
    const size_t K_tiles = ((size_t)T + KC - 1) / KC;
    const size_t H_BATCH = calc_h_batch_pv((size_t)T, KC);
    const size_t max_ab  = (MC + MT - 1) / MT;
    const size_t nc      = std::min(NT, (size_t)d);   // valid output cols per NT tile

    AlignedBuffer<__fp16> V_batch(H_BATCH * (size_t)T * NT_buf);
    AlignedBuffer<__fp16> A_cache(max_ab * MT * KC);
    AlignedBuffer<__fp16> Cpanel(max_ab * MT * NT);
    std::memset(V_batch.data(), 0, H_BATCH * (size_t)T * NT_buf * sizeof(__fp16));

    for (int g0 = 0; g0 < H_kv; g0 += (int)H_BATCH) {
        const int h_batch = (int)std::min(H_BATCH, (size_t)(H_kv - g0));

        // Phase 1: extract V + SPMCP all K-tiles for h_batch heads
        for (int hb = 0; hb < h_batch; ++hb) {
            __fp16* V_g = V_batch.data() + (size_t)hb * T * NT_buf;
            extract_V_head_padded(V, V_g, T, d, D_kv, g0 + hb, NT_buf);
            for (size_t kt = 0; kt < K_tiles; ++kt) {
                size_t k0 = kt * KC;
                size_t kc = std::min(KC, (size_t)T - k0);
                size_t base_set = ((size_t)hb * K_tiles + kt) * KC * SPM_KC_FACTOR;
                // V_g[k0:k0+kc, :NT_buf]: each K-row is NT_buf elements wide
                pack_B_tile_to_spm(V_g + k0 * NT_buf, NT_buf, kc, KC, base_set);
            }
        }
        // Complete the SPMCP operations before compute.
        asm volatile("dsb sy\n" ::: "memory");

        // Phase 2: compute G * h_batch query heads from SPM
        for (int hb = 0; hb < h_batch; ++hb) {
            for (int qi = 0; qi < G; ++qi) {
                const int h = (g0 + hb) * G + qi;
                const __fp16* Ph   = P   + (size_t)h * T * T;
                __fp16*       ctxh = ctx + (size_t)h * T * NT_buf;

                // M-block loop
                for (size_t m0 = 0; m0 < (size_t)T; m0 += MC) {
                    size_t mc      = std::min(MC, (size_t)T - m0);
                    size_t ablocks = (mc + MT - 1) / MT;

                    // K-tile accumulation loop
                    for (size_t kt = 0; kt < K_tiles; ++kt) {
                        size_t k0       = kt * KC;
                        size_t kc       = std::min(KC, (size_t)T - k0);
                        size_t base_set = ((size_t)hb * K_tiles + kt) * KC * SPM_KC_FACTOR;

                        // Pack A: Ph[m0:m0+mc, k0:k0+kc], lda=T
                        pack_A_fp16_8row(Ph, (size_t)T, A_cache.data(),
                                         (size_t)T, m0, (size_t)T, k0,
                                         mc, KC);
                        // Compute Cpanel = A_tile × B_tile (from SPM)
                        spm_kernel_mblocks(A_cache.data(), Cpanel.data(),
                                           (int)kc, (int)ablocks, KC, base_set);
                        // Scatter to ctx_h (write on kt=0, accumulate on kt>0)
                        scatter_cpanel(Cpanel.data(), ctxh,
                                       m0, (size_t)T, mc, ablocks,
                                       NT_buf, nc, /*accumulate=*/(kt > 0));
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------
// Reference scalar GEMM (correctness check only)
// ----------------------------------------------------------------
static void ref_gemm(const __fp16* A, const __fp16* B, __fp16* C,
                     int M, int N, int K, int ldb) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += (float)A[(size_t)m*K+k] * (float)B[(size_t)k*ldb+n];
            C[(size_t)m*N+n] = (__fp16)acc;
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
    const size_t KC  = (argc > 6) ? (size_t)std::atoi(argv[6]) : 256;
    const size_t MC  = (argc > 7) ? (size_t)std::atoi(argv[7]) : 128;
    const size_t NT  = 3 * svcnth();
    const size_t NT_buf = std::max(NT, (size_t)d);
    const int G = H_q / H_kv;
    const int D_kv = H_kv * d;

    const size_t K_tiles = ((size_t)T + KC - 1) / KC;
    const size_t H_BATCH = calc_h_batch_pv((size_t)T, KC);

    printf("=== 06 PV GQA SPM (MH-opt) ===  T=%d  H_q=%d  H_kv=%d  G=%d  d=%d\n",
           T, H_q, H_kv, G, d);
    printf("  NT=%zu  NT_buf=%zu  KC=%zu  MC=%zu  K_tiles=%zu\n",
           NT, NT_buf, KC, MC, K_tiles);
    printf("  H_BATCH=%zu  (%zu SPMCP waves for %d kv heads)\n\n",
           H_BATCH, ((size_t)H_kv + H_BATCH - 1) / H_BATCH, H_kv);
    fflush(stdout);

    AlignedBuffer<__fp16> P((size_t)H_q * T * T);
    AlignedBuffer<__fp16> V((size_t)T * D_kv);
    AlignedBuffer<__fp16> ctx((size_t)H_q * T * NT_buf);

    fill_random(P.data(), (size_t)H_q*T*T,  0.f,   1.f,  1);
    fill_random(V.data(), (size_t)T*D_kv,  -0.1f, 0.1f,  2);

    // ---- Correctness check ----
    {
        const int Tc = std::min(T, (int)NT);   // small check size
        AlignedBuffer<__fp16> Pc((size_t)H_q * Tc * Tc);
        AlignedBuffer<__fp16> Vc((size_t)Tc * D_kv);
        AlignedBuffer<__fp16> ctx_spm((size_t)H_q * Tc * NT_buf);
        AlignedBuffer<__fp16> ctx_ref((size_t)H_q * Tc * (size_t)d);

        for (int h = 0; h < H_q; ++h)
            for (int t = 0; t < Tc; ++t)
                std::memcpy(Pc.data() + (size_t)h*Tc*Tc + (size_t)t*Tc,
                            P.data()  + (size_t)h*T*T   + (size_t)t*T,
                            Tc * sizeof(__fp16));
        for (int t = 0; t < Tc; ++t)
            std::memcpy(Vc.data() + (size_t)t*D_kv, V.data() + (size_t)t*D_kv, D_kv*sizeof(__fp16));

        std::memset(ctx_spm.data(), 0, (size_t)H_q*Tc*NT_buf*sizeof(__fp16));
        std::memset(ctx_ref.data(), 0, (size_t)H_q*Tc*(size_t)d*sizeof(__fp16));

        // Reference
        AlignedBuffer<__fp16> Vg_ref((size_t)Tc * (size_t)d);
        for (int g = 0; g < H_kv; ++g) {
            for (int t = 0; t < Tc; ++t)
                std::memcpy(Vg_ref.data() + (size_t)t*d,
                            Vc.data() + (size_t)t*D_kv + (size_t)g*d,
                            d*sizeof(__fp16));
            for (int qi = 0; qi < G; ++qi) {
                int h = g*G+qi;
                ref_gemm(Pc.data() + (size_t)h*Tc*Tc, Vg_ref.data(),
                         ctx_ref.data() + (size_t)h*Tc*(size_t)d, Tc, d, Tc, d);
            }
        }

        // SPM
        run_pv_spm(Pc.data(), Vc.data(), ctx_spm.data(),
                   Tc, H_q, H_kv, G, d, D_kv, MC, KC, NT_buf);

        // Compare (only first d cols of ctx_spm are valid)
        float max_err = 0.f;
        for (int h = 0; h < H_q; ++h)
            for (int t = 0; t < Tc; ++t)
                for (int n = 0; n < d; ++n) {
                    float s = (float)ctx_spm.data()[(size_t)h*Tc*NT_buf + (size_t)t*NT_buf + n];
                    float r = (float)ctx_ref.data()[(size_t)h*Tc*(size_t)d + (size_t)t*d + n];
                    max_err = std::max(max_err, std::fabs(s - r));
                }
        fprintf(stderr, "[CORRECT] all heads, Tc=%d: max_abs_err=%.6f  %s\n",
                Tc, max_err, max_err < 0.05f ? "PASS" : "FAIL");
        fflush(stderr);
    }

    const double total_flops = 2.0 * H_q * T * T * d;

    // --- COLD ---
    flush_dcache_range(P.data(), (size_t)H_q*T*T * sizeof(__fp16));
    flush_dcache_range(V.data(), (size_t)T*D_kv   * sizeof(__fp16));
    flush_dcache_range(ctx.data(), (size_t)H_q*T*NT_buf * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0 = Clock::now();
        run_pv_spm(P.data(), V.data(), ctx.data(),
                   T, H_q, H_kv, G, d, D_kv, MC, KC, NT_buf);
        double us = us_since(t0);
        printf("  COLD: %7.1f us   %.2f GFLOPS\n\n", us, total_flops/(us*1e3));
        fflush(stdout);
    }
    ROI_END();

    // --- WARM ---
    run_pv_spm(P.data(), V.data(), ctx.data(),
               T, H_q, H_kv, G, d, D_kv, MC, KC, NT_buf);

    printf("--- WARM ---\n"); fflush(stdout);
    double t_total = 0;
    for (int it = 0; it < N_ITER; ++it) {
        auto t0 = Clock::now();
        run_pv_spm(P.data(), V.data(), ctx.data(),
                   T, H_q, H_kv, G, d, D_kv, MC, KC, NT_buf);
        t_total += us_since(t0);
    }
    t_total /= N_ITER;
    printf("  WARM: %7.1f us/iter   %.2f GFLOPS\n\n",
           t_total, total_flops/(t_total*1e3));
    fflush(stdout);
    return 0;
}
