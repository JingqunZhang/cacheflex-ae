/*
 * 02_qkv_gqa.cpp — LLaMA GQA QKV projection
 *
 * GQA layout:
 *   W_Q [D, H_q*d]  = [2048, 2048]  →  Q[T, D]   (square)
 *   W_K [D, H_kv*d] = [2048,  512]  →  K[T, H_kv*d]
 *   W_V [D, H_kv*d] = [2048,  512]  →  V[T, H_kv*d]
 *
 * FLOPs: 2*T*D*(H_q*d + 2*H_kv*d) = 2*T*D*(D + 2*H_kv*d)
 * LLaMA-3.2-1B: T=512, D=2048, H_q=32, H_kv=8, d=64
 * Usage: ./02_qkv_gqa [T] [D] [H_q] [H_kv] [d] [n_iter]
 *        default: 512 2048 32 8 64 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int D      = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int H_q    = (argc > 3) ? std::atoi(argv[3]) : 32;
    const int H_kv   = (argc > 4) ? std::atoi(argv[4]) : 8;
    const int d      = (argc > 5) ? std::atoi(argv[5]) : 64;
    const int N_ITER = (argc > 6) ? std::atoi(argv[6]) : 5;
    const size_t NT_hw  = 3 * svcnth();
    const size_t KC_arg = (argc > 7) ? (size_t)std::atoi(argv[7]) : 0;
    const size_t KC     = KC_arg ? KC_arg : 24576 / NT_hw;  // default: B-panel ≈ 48KB (~¾ L1)
    const size_t MC_arg = (argc > 8) ? (size_t)std::atoi(argv[8]) : 0;
    const size_t MC     = MC_arg ? MC_arg : 16 * svcnth();

    const int Nq  = H_q  * d;   // 2048
    const int Nkv = H_kv * d;   // 512

    const double WQ_MB  = D * Nq  * 2.0 / 1048576;
    const double WKV_MB = D * Nkv * 2.0 / 1048576;
    const double in_MB  = T * D   * 2.0 / 1048576;
    const double Q_MB   = T * Nq  * 2.0 / 1048576;
    const double KV_MB  = T * Nkv * 2.0 / 1048576;

    const size_t pkQ_sz  = prepack_B_knm_size(D, Nq,  KC);
    const size_t pkKV_sz = prepack_B_knm_size(D, Nkv, KC);

    printf("=== 02 QKV GQA Projection ===  T=%d  D=%d  H_q=%d  H_kv=%d  d=%d  n_iter=%d\n",
           T, D, H_q, H_kv, d, N_ITER);
    printf("  input  [%d, %d]        %.2f MB\n", T, D, in_MB);
    printf("  W_Q    [%d, %d]  %.2f MB raw\n", D, Nq, WQ_MB);
    printf("  W_K/V  [%d, %d]   %.2f MB raw each\n", D, Nkv, WKV_MB);
    printf("  Q out  [%d, %d]  %.2f MB\n", T, Nq, Q_MB);
    printf("  K/V out[%d, %d]   %.2f MB each\n\n", T, Nkv, KV_MB);

    AlignedBuffer<__fp16> x_in((size_t)T * D);
    AlignedBuffer<__fp16> W_Q((size_t)D * Nq),   W_Q_pk(pkQ_sz);
    AlignedBuffer<__fp16> W_K((size_t)D * Nkv),  W_K_pk(pkKV_sz);
    AlignedBuffer<__fp16> W_V((size_t)D * Nkv),  W_V_pk(pkKV_sz);
    AlignedBuffer<__fp16> Q_out((size_t)T * Nq);
    AlignedBuffer<__fp16> K_out((size_t)T * Nkv);
    AlignedBuffer<__fp16> V_out((size_t)T * Nkv);

    fill_random(x_in.data(), (size_t)T*D,    -0.1f, 0.1f,  1);
    fill_random(W_Q.data(),  (size_t)D*Nq,   -0.02f, 0.02f, 2);
    fill_random(W_K.data(),  (size_t)D*Nkv,  -0.02f, 0.02f, 3);
    fill_random(W_V.data(),  (size_t)D*Nkv,  -0.02f, 0.02f, 4);

    prepack_B_knm(W_Q.data(), W_Q_pk.data(), D, Nq,  KC);
    prepack_B_knm(W_K.data(), W_K_pk.data(), D, Nkv, KC);
    prepack_B_knm(W_V.data(), W_V_pk.data(), D, Nkv, KC);

    // --- COLD ---
    printf("--- COLD (all weight packs flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(W_Q_pk.data(),  pkQ_sz  * sizeof(__fp16));
    flush_dcache_range(W_K_pk.data(),  pkKV_sz * sizeof(__fp16));
    flush_dcache_range(W_V_pk.data(),  pkKV_sz * sizeof(__fp16));

    ROI_BEGIN();
    {
        double fQ = 2.0*T*D*Nq, fKV = 2.0*T*D*Nkv;
        auto t0 = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_Q_pk.data(), Q_out.data(), T, Nq, D, MC, KC);
        double uQ = us_since(t0);
        printf("  W_Q GEMM: %7.1f us   %.2f GFLOPS\n", uQ, fQ/(uQ*1e3));
        t0 = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_K_pk.data(), K_out.data(), T, Nkv, D, MC, KC);
        double uK = us_since(t0);
        printf("  W_K GEMM: %7.1f us   %.2f GFLOPS\n", uK, fKV/(uK*1e3));
        t0 = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_V_pk.data(), V_out.data(), T, Nkv, D, MC, KC);
        double uV = us_since(t0);
        printf("  W_V GEMM: %7.1f us   %.2f GFLOPS\n", uV, fKV/(uV*1e3));
    }
    ROI_END();
    printf("\n");

    // --- WARM ---
    printf("--- WARM (weights in cache) ---\n");
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i) {
        gemm_knm_prepacked(x_in.data(), W_Q_pk.data(), Q_out.data(), T, Nq,  D, MC, KC);
        gemm_knm_prepacked(x_in.data(), W_K_pk.data(), K_out.data(), T, Nkv, D, MC, KC);
        gemm_knm_prepacked(x_in.data(), W_V_pk.data(), V_out.data(), T, Nkv, D, MC, KC);
    }

    double tQ=0, tK=0, tV=0;
    GemmTiming timQ, timK, timV;
    for (int i = 0; i < N_ITER; ++i) {
        auto tp = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_Q_pk.data(), Q_out.data(), T, Nq,  D, MC, KC, &timQ);
        tQ += us_since(tp);
        tp = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_K_pk.data(), K_out.data(), T, Nkv, D, MC, KC, &timK);
        tK += us_since(tp);
        tp = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_V_pk.data(), V_out.data(), T, Nkv, D, MC, KC, &timV);
        tV += us_since(tp);
    }
    tQ/=N_ITER; tK/=N_ITER; tV/=N_ITER;
    timQ.t_packA/=N_ITER; timQ.t_compute/=N_ITER; timQ.t_writeback/=N_ITER;
    timQ.t_c_write/=N_ITER; timQ.t_c_rmw/=N_ITER;
    timK.t_packA/=N_ITER; timK.t_compute/=N_ITER; timK.t_writeback/=N_ITER;
    timK.t_c_write/=N_ITER; timK.t_c_rmw/=N_ITER;
    timV.t_packA/=N_ITER; timV.t_compute/=N_ITER; timV.t_writeback/=N_ITER;
    timV.t_c_write/=N_ITER; timV.t_c_rmw/=N_ITER;
    double fQ = 2.0*T*D*Nq, fKV = 2.0*T*D*Nkv;
    printf("  W_Q GEMM: %7.1f us  [packA %5.1f  kernel %6.1f  C_wr %5.1f  C_rmw %5.1f]  %.2f GFLOPS\n",
           tQ, timQ.t_packA, timQ.t_compute, timQ.t_c_write, timQ.t_c_rmw, fQ/(tQ*1e3));
    printf("  W_K GEMM: %7.1f us  [packA %5.1f  kernel %6.1f  C_wr %5.1f  C_rmw %5.1f]  %.2f GFLOPS\n",
           tK, timK.t_packA, timK.t_compute, timK.t_c_write, timK.t_c_rmw, fKV/(tK*1e3));
    printf("  W_V GEMM: %7.1f us  [packA %5.1f  kernel %6.1f  C_wr %5.1f  C_rmw %5.1f]  %.2f GFLOPS\n",
           tV, timV.t_packA, timV.t_compute, timV.t_c_write, timV.t_c_rmw, fKV/(tV*1e3));
    printf("  ─────────────────────────────────────\n");
    double tot = tQ+tK+tV;
    double ftot = fQ + 2*fKV;
    printf("  total   : %7.1f us   %.2f GFLOPS combined\n\n", tot, ftot/(tot*1e3));

    printf("  W_Q packed: %.2f MB, W_K+W_V packed: %.2f+%.2f = %.2f MB\n",
           pkQ_sz*2.0/1048576, pkKV_sz*2.0/1048576, pkKV_sz*2.0/1048576,
           (pkQ_sz+2*pkKV_sz)*2.0/1048576);
    printf("  GQA ratio: W_K+W_V are %dx smaller than W_Q (H_kv/H_q = %d/%d)\n",
           H_q/H_kv, H_q, H_kv);
    return 0;
}
