/*
 * 06_pv_gqa.cpp — LLaMA GQA PV GEMM (attention × values)
 *
 * GQA structure: H_kv=8 KV heads, H_q=32 query heads, G=H_q/H_kv=4
 * V[T, H_kv*d]: head-interleaved, matching QKV projection output
 * For each KV group g = 0..H_kv-1:
 *   pre-pack V_g[T, d] → V_g_pk   (prepack_B_knm: base=V+g*d, ldb=H_kv*d)
 *   for q = 0..G-1:
 *     ctx[g*G+q, T, d] = P[g*G+q, T, T] × V_g[T, d]
 *
 * V is treated as constant weight (pre-packed with prepack_B_knm).
 * Context output: ctx[H_q, T, d].
 *
 * LLaMA-3.2-1B: T=512, H_q=32, H_kv=8, G=4, d=64
 * Usage: ./06_pv_gqa [T] [H_q] [H_kv] [d] [n_iter]
 *        default: 512 32 8 64 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int H_q    = (argc > 2) ? std::atoi(argv[2]) : 32;
    const int H_kv   = (argc > 3) ? std::atoi(argv[3]) : 8;
    const int d      = (argc > 4) ? std::atoi(argv[4]) : 64;
    const int N_ITER = (argc > 5) ? std::atoi(argv[5]) : 5;
    const size_t NT_hw  = 3 * svcnth();
    const size_t KC_arg = (argc > 6) ? (size_t)std::atoi(argv[6]) : 0;
    const size_t KC     = KC_arg ? KC_arg : 24576 / NT_hw;  // default: B-panel ≈ 48KB (~¾ L1)
    const size_t MC_arg = (argc > 7) ? (size_t)std::atoi(argv[7]) : 0;
    const size_t MC     = MC_arg ? MC_arg : 16 * svcnth();

    const int G = H_q / H_kv;

    // V_g: [T, d], treated as weight B-matrix for GEMM: P[T,T] × V[T,d] → ctx[T,d]
    // prepack_B_knm(V_g, T rows as K-dim, d cols as N-dim, KC)
    const size_t Vpk_sz = prepack_B_knm_size(T, d, KC);

    // V[T, H_kv*d]: head-interleaved, matching QKV projection output
    const int    D_kv = H_kv * d;
    const double P_MB   = (size_t)H_q  * T * T * 2.0 / 1048576;
    const double V_MB   = (size_t)T * D_kv * 2.0 / 1048576;
    const double ctx_MB = (size_t)H_q  * T * d * 2.0 / 1048576;

    printf("=== 06 PV GQA GEMM ===  T=%d  H_q=%d  H_kv=%d  G=%d  d=%d  n_iter=%d\n",
           T, H_q, H_kv, G, d, N_ITER);
    printf("  P (scores) [%d, %d, %d]  %.2f MB\n", H_q, T, T, P_MB);
    printf("  V          [%d, %d]  %.2f MB  (head-interleaved, stride D_kv=%d)\n", T, D_kv, V_MB, D_kv);
    printf("  ctx output [%d, %d, %d]   %.2f MB\n\n", H_q, T, d, ctx_MB);
    printf("  GQA: pack V_g once per KV head, reuse for G=%d query heads\n\n", G);

    AlignedBuffer<__fp16> P((size_t)H_q  * T * T);
    AlignedBuffer<__fp16> V((size_t)T * D_kv);    // [T, H_kv*d] head-interleaved
    AlignedBuffer<__fp16> ctx((size_t)H_q * T * d);
    AlignedBuffer<__fp16> V_pk(Vpk_sz);   // packed buffer for one KV head's V

    fill_random(P.data(), (size_t)H_q *T*T, 0.f, 1.f, 1);   // attention weights ∈ [0,1]
    fill_random(V.data(), (size_t)T*D_kv, -0.1f, 0.1f, 2);

    // Pre-pack all V heads (in real inference, done once at start of layer)
    // For benchmarking, we pack inside the loop to model cold behavior
    // Pre-pack V[0] as baseline verification (V_g base = V + g*d, ldb = D_kv)
    prepack_B_knm(V.data(), V_pk.data(), T, d, KC, (size_t)D_kv);

    // --- COLD: flush P and V to DRAM ---
    printf("--- COLD (P and V flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(P.data(), (size_t)H_q *T*T * sizeof(__fp16));
    flush_dcache_range(V.data(), (size_t)T*D_kv * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0 = Clock::now();
        double total_flops = 0, t_packB_cold = 0;
        for (int g = 0; g < H_kv; ++g) {
            // V_g: base = V + g*d (head-interleaved), row stride = D_kv
            auto _tp = Clock::now();
            prepack_B_knm(V.data() + (size_t)g*d, V_pk.data(), T, d, KC, (size_t)D_kv);
            t_packB_cold += us_since(_tp);

            for (int qi = 0; qi < G; ++qi) {
                int h = g * G + qi;
                const __fp16* Ph   = P.data()   + (size_t)h * T * T;
                __fp16*       ctxh = ctx.data()  + (size_t)h * T * d;
                gemm_knm_prepacked(Ph, V_pk.data(), ctxh, T, d, T, MC, KC);
                total_flops += 2.0 * T * T * d;
            }
        }
        double us = us_since(t0);
        printf("  pack V : %7.1f us  (COLD, %d KV heads)\n", t_packB_cold, H_kv);
        printf("  1 run  : %7.1f us   %.2f GFLOPS\n\n", us, total_flops/(us*1e3));
    }
    ROI_END();

    // --- WARM ---
    printf("--- WARM ---\n");
    for (int it = 0; it < (N_ITER > 1 ? 1 : 0); ++it) {
        for (int g = 0; g < H_kv; ++g) {
            prepack_B_knm(V.data()+(size_t)g*d, V_pk.data(), T, d, KC, (size_t)D_kv);
            for (int qi = 0; qi < G; ++qi) {
                int h = g*G+qi;
                gemm_knm_prepacked(P.data()+(size_t)h*T*T, V_pk.data(),
                                   ctx.data()+(size_t)h*T*d, T, d, T, MC, KC);
            }
        }
    }

    double t_pack = 0, t_gemm = 0;
    GemmTiming tim_gemm;
    const double total_flops = 2.0 * H_q * T * T * d;
    for (int it = 0; it < N_ITER; ++it) {
        for (int g = 0; g < H_kv; ++g) {
            auto tp = Clock::now();
            prepack_B_knm(V.data()+(size_t)g*d, V_pk.data(), T, d, KC, (size_t)D_kv);
            t_pack += us_since(tp);
            for (int qi = 0; qi < G; ++qi) {
                int h = g*G+qi;
                tp = Clock::now();
                gemm_knm_prepacked(P.data()+(size_t)h*T*T, V_pk.data(),
                                   ctx.data()+(size_t)h*T*d, T, d, T, MC, KC, &tim_gemm);
                t_gemm += us_since(tp);
            }
        }
    }
    t_pack /= N_ITER; t_gemm /= N_ITER;
    tim_gemm.t_packA /= N_ITER; tim_gemm.t_compute /= N_ITER; tim_gemm.t_writeback /= N_ITER;
    tim_gemm.t_c_write /= N_ITER; tim_gemm.t_c_rmw /= N_ITER;
    printf("  pack V  (%d KV heads) : %7.1f us/iter\n", H_kv, t_pack);
    printf("  GEMM    (%d heads)    : %7.1f us/iter  [packA %5.1f  kernel %6.1f  C_wr %5.1f  C_rmw %5.1f]  %.2f GFLOPS\n",
           H_q, t_gemm, tim_gemm.t_packA, tim_gemm.t_compute, tim_gemm.t_c_write, tim_gemm.t_c_rmw, total_flops/(t_gemm*1e3));
    printf("  total                 : %7.1f us/iter   %.2f GFLOPS\n\n", t_pack+t_gemm, total_flops/((t_pack+t_gemm)*1e3));

    printf("  V packing reuse: pack %d times (H_kv) vs %d (H_q) in MHA — %dx savings\n",
           H_kv, H_q, G);
    printf("  P[H_q,T,T]=%.2f MB: large attention weight matrix, bandwidth critical\n", P_MB);
    printf("  V_pk[T,d] = %.2f KB per head, packed once then reused %d times\n",
           Vpk_sz*2.0/1024, G);
    return 0;
}
