/*
 * 02_qkv_gqa_spm.cpp — LLaMA GQA QKV projection with CacheFlex SPM
 *
 * GQA layout:
 *   W_Q [D, H_q*d]  = [2048, 2048]  → Q[T, D]   (square)
 *   W_K [D, H_kv*d] = [2048,  512]  → K[T, H_kv*d]
 *   W_V [D, H_kv*d] = [2048,  512]  → V[T, H_kv*d]
 *
 * SPM: each weight tile pinned during the M inner loop
 *
 * LLaMA-3.2-1B: T=512, D=2048, H_q=32, H_kv=8, d=64
 *
 * Compile: -DVL_2, -DVL_4, -DVL_8, or -DVL_16
 * Usage: ./02_qkv_gqa_spm_vl4 [T] [D] [H_q] [H_kv] [d] [n_iter] [KC] [MC]
 *        default: 512 2048 32 8 64 5 256 128
 */
#include "kernels_spm.hpp"

int main(int argc, char** argv)
{
    const int    T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int    D      = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int    H_q    = (argc > 3) ? std::atoi(argv[3]) : 32;
    const int    H_kv   = (argc > 4) ? std::atoi(argv[4]) : 8;
    const int    d      = (argc > 5) ? std::atoi(argv[5]) : 64;
    const int    N_ITER = (argc > 6) ? std::atoi(argv[6]) : 5;
    const size_t KC     = (argc > 7) ? (size_t)std::atoi(argv[7]) : 256;
    const size_t MC     = (argc > 8) ? (size_t)std::atoi(argv[8]) : 128;
    const size_t NT     = 3 * svcnth();

    const int Nq  = H_q  * d;   // 2048
    const int Nkv = H_kv * d;   // 512

    const double WQ_MB  = D * Nq  * 2.0 / 1048576;
    const double WKV_MB = D * Nkv * 2.0 / 1048576;
    const double in_MB  = T * D   * 2.0 / 1048576;
    const double Q_MB   = T * Nq  * 2.0 / 1048576;
    const double KV_MB  = T * Nkv * 2.0 / 1048576;

    printf("=== 02 QKV GQA SPM ===  T=%d  D=%d  H_q=%d  H_kv=%d  d=%d  n_iter=%d\n",
           T, D, H_q, H_kv, d, N_ITER);
    printf("  NT=%zu  KC=%zu  MC=%zu\n\n", NT, KC, MC);
    printf("  input  [%d, %d]        %.2f MB\n", T, D, in_MB);
    printf("  W_Q    [%d, %d]  %.2f MB raw\n", D, Nq, WQ_MB);
    printf("  W_K/V  [%d, %d]   %.2f MB raw each\n", D, Nkv, WKV_MB);
    printf("  Q out  [%d, %d]  %.2f MB\n", T, Nq, Q_MB);
    printf("  K/V out[%d, %d]   %.2f MB each\n\n", T, Nkv, KV_MB);

    AlignedBuffer<__fp16> x_in((size_t)T * D);
    AlignedBuffer<__fp16> W_Q((size_t)D * Nq);
    AlignedBuffer<__fp16> W_K((size_t)D * Nkv);
    AlignedBuffer<__fp16> W_V((size_t)D * Nkv);
    AlignedBuffer<__fp16> Q_out((size_t)T * Nq);
    AlignedBuffer<__fp16> K_out((size_t)T * Nkv);
    AlignedBuffer<__fp16> V_out((size_t)T * Nkv);

    fill_random(x_in.data(), (size_t)T*D,    -0.1f, 0.1f,  1);
    fill_random(W_Q.data(),  (size_t)D*Nq,   -0.02f, 0.02f, 2);
    fill_random(W_K.data(),  (size_t)D*Nkv,  -0.02f, 0.02f, 3);
    fill_random(W_V.data(),  (size_t)D*Nkv,  -0.02f, 0.02f, 4);

    // --- COLD ---
    printf("--- COLD (all weights flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(W_Q.data(),  (size_t)D * Nq  * sizeof(__fp16));
    flush_dcache_range(W_K.data(),  (size_t)D * Nkv * sizeof(__fp16));
    flush_dcache_range(W_V.data(),  (size_t)D * Nkv * sizeof(__fp16));

    ROI_BEGIN();
    {
        double fQ = 2.0*T*D*Nq, fKV = 2.0*T*D*Nkv;
        auto t0 = Clock::now();
        gemm_spm(x_in.data(), W_Q.data(), Q_out.data(), T, Nq,  D, Nq,  MC, KC);
        double uQ = us_since(t0);
        printf("  W_Q GEMM: %7.1f us   %.2f GFLOPS\n", uQ, fQ/(uQ*1e3));
        t0 = Clock::now();
        gemm_spm(x_in.data(), W_K.data(), K_out.data(), T, Nkv, D, Nkv, MC, KC);
        double uK = us_since(t0);
        printf("  W_K GEMM: %7.1f us   %.2f GFLOPS\n", uK, fKV/(uK*1e3));
        t0 = Clock::now();
        gemm_spm(x_in.data(), W_V.data(), V_out.data(), T, Nkv, D, Nkv, MC, KC);
        double uV = us_since(t0);
        printf("  W_V GEMM: %7.1f us   %.2f GFLOPS\n", uV, fKV/(uV*1e3));
    }
    ROI_END();
    printf("\n");

    // --- WARM ---
    printf("--- WARM (weights in cache) ---\n");
    if (N_ITER > 1) {
        gemm_spm(x_in.data(), W_Q.data(), Q_out.data(), T, Nq,  D, Nq,  MC, KC);
        gemm_spm(x_in.data(), W_K.data(), K_out.data(), T, Nkv, D, Nkv, MC, KC);
        gemm_spm(x_in.data(), W_V.data(), V_out.data(), T, Nkv, D, Nkv, MC, KC);
    }

    double tQ=0, tK=0, tV=0;
    double fQ = 2.0*T*D*Nq, fKV = 2.0*T*D*Nkv;
    for (int i = 0; i < N_ITER; ++i) {
        auto tp = Clock::now();
        gemm_spm(x_in.data(), W_Q.data(), Q_out.data(), T, Nq,  D, Nq,  MC, KC);
        tQ += us_since(tp);
        tp = Clock::now();
        gemm_spm(x_in.data(), W_K.data(), K_out.data(), T, Nkv, D, Nkv, MC, KC);
        tK += us_since(tp);
        tp = Clock::now();
        gemm_spm(x_in.data(), W_V.data(), V_out.data(), T, Nkv, D, Nkv, MC, KC);
        tV += us_since(tp);
    }
    tQ/=N_ITER; tK/=N_ITER; tV/=N_ITER;
    printf("  W_Q GEMM: %7.1f us   %.2f GFLOPS\n", tQ, fQ/(tQ*1e3));
    printf("  W_K GEMM: %7.1f us   %.2f GFLOPS\n", tK, fKV/(tK*1e3));
    printf("  W_V GEMM: %7.1f us   %.2f GFLOPS\n", tV, fKV/(tV*1e3));
    printf("  ─────────────────────────────────────\n");
    double tot = tQ+tK+tV;
    double ftot = fQ + 2*fKV;
    printf("  total   : %7.1f us   %.2f GFLOPS combined\n\n", tot, ftot/(tot*1e3));
    printf("  GQA ratio: W_K+W_V are %dx smaller than W_Q\n", H_q/H_kv);
    return 0;
}
