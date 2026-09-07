/*
 * 08_ffn_swiglu_spm.cpp — LLaMA SwiGLU FFN with CacheFlex SPM
 *
 * SwiGLU FFN:
 *   gate = x @ W_gate    (T,D) × (D,FFN_dim)
 *   up   = x @ W_up      (T,D) × (D,FFN_dim)
 *   hidden = SiLU(gate) * up   (element-wise)
 *   out  = hidden @ W_down  (T,FFN_dim) × (FFN_dim,D)
 *
 * SPM: each weight tile pinned during the MC inner loop
 *
 * LLaMA-3.2-1B: T=512, D=2048, FFN_dim=8192
 * FLOPs: 3 × 2×T×D×FFN_dim ≈ 51.5 GFLOPs
 *
 * Compile: -DVL_2, -DVL_4, -DVL_8, or -DVL_16
 * Usage: ./08_ffn_swiglu_spm_vl4 [T] [D] [FFN_dim] [n_iter] [KC] [MC]
 *        default: 512 2048 8192 5 256 128
 */
#include "kernels_spm.hpp"

int main(int argc, char** argv)
{
    const int    T       = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int    D       = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int    FFN_dim = (argc > 3) ? std::atoi(argv[3]) : 8192;
    const int    N_ITER  = (argc > 4) ? std::atoi(argv[4]) : 5;
    const size_t KC      = (argc > 5) ? (size_t)std::atoi(argv[5]) : 256;
    const size_t MC      = (argc > 6) ? (size_t)std::atoi(argv[6]) : 128;
    const size_t NT      = 3 * svcnth();

    const double W_gate_MB = (double)D * FFN_dim * 2.0 / 1048576;
    const double W_dn_MB   = (double)FFN_dim * D * 2.0 / 1048576;
    const double in_MB     = (double)T * D * 2.0 / 1048576;
    const double hid_MB    = (double)T * FFN_dim * 2.0 / 1048576;

    printf("=== 08 FFN SwiGLU SPM ===  T=%d  D=%d  FFN_dim=%d  n_iter=%d\n",
           T, D, FFN_dim, N_ITER);
    printf("  NT=%zu  KC=%zu  MC=%zu\n\n", NT, KC, MC);
    printf("  input   [%d, %d]        %.2f MB\n", T, D, in_MB);
    printf("  W_gate  [%d, %d]   %.2f MB\n", D, FFN_dim, W_gate_MB);
    printf("  W_up    [%d, %d]   %.2f MB\n", D, FFN_dim, W_gate_MB);
    printf("  hidden  [%d, %d]  %.2f MB\n", T, FFN_dim, hid_MB);
    printf("  W_down  [%d, %d]   %.2f MB\n\n", FFN_dim, D, W_dn_MB);

    AlignedBuffer<__fp16> x_in((size_t)T * D);
    AlignedBuffer<__fp16> W_gate((size_t)D * FFN_dim);
    AlignedBuffer<__fp16> W_up((size_t)D * FFN_dim);
    AlignedBuffer<__fp16> W_down((size_t)FFN_dim * D);
    AlignedBuffer<__fp16> gate((size_t)T * FFN_dim);
    AlignedBuffer<__fp16> up((size_t)T * FFN_dim);
    AlignedBuffer<__fp16> out_buf((size_t)T * D);

    fill_random(x_in.data(),   (size_t)T*D,        -0.1f,  0.1f,  1);
    fill_random(W_gate.data(), (size_t)D*FFN_dim,  -0.02f, 0.02f, 2);
    fill_random(W_up.data(),   (size_t)D*FFN_dim,  -0.02f, 0.02f, 3);
    fill_random(W_down.data(), (size_t)FFN_dim*D,  -0.02f, 0.02f, 4);

    // --- COLD: flush all weights to DRAM ---
    printf("--- COLD (W_gate + W_up + W_down flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(W_gate.data(), (size_t)D * FFN_dim * sizeof(__fp16));
    flush_dcache_range(W_up.data(),   (size_t)D * FFN_dim * sizeof(__fp16));
    flush_dcache_range(W_down.data(), (size_t)FFN_dim * D * sizeof(__fp16));

    ROI_BEGIN();
    {
        double fg = 2.0*T*D*FFN_dim, fd = 2.0*T*FFN_dim*D;
        auto t0 = Clock::now();
        gemm_spm(x_in.data(), W_gate.data(), gate.data(), T, FFN_dim, D, FFN_dim, MC, KC);
        double ug = us_since(t0);
        printf("  W_gate GEMM: %7.1f us   %.2f GFLOPS\n", ug, fg/(ug*1e3));
        t0 = Clock::now();
        gemm_spm(x_in.data(), W_up.data(), up.data(), T, FFN_dim, D, FFN_dim, MC, KC);
        double uu = us_since(t0);
        printf("  W_up   GEMM: %7.1f us   %.2f GFLOPS\n", uu, fg/(uu*1e3));
        t0 = Clock::now();
        swiglu_fp16(gate.data(), up.data(), (size_t)T * FFN_dim);
        printf("  SwiGLU     : %7.1f us\n", us_since(t0));
        t0 = Clock::now();
        gemm_spm(gate.data(), W_down.data(), out_buf.data(), T, D, FFN_dim, D, MC, KC);
        double ud = us_since(t0);
        printf("  W_down GEMM: %7.1f us   %.2f GFLOPS\n", ud, fd/(ud*1e3));
    }
    ROI_END();
    printf("\n");

    // --- WARM ---
    printf("--- WARM (weights in cache) ---\n");
    if (N_ITER > 1) {
        gemm_spm(x_in.data(), W_gate.data(), gate.data(), T, FFN_dim, D, FFN_dim, MC, KC);
        gemm_spm(x_in.data(), W_up.data(),   up.data(),   T, FFN_dim, D, FFN_dim, MC, KC);
        swiglu_fp16(gate.data(), up.data(), (size_t)T * FFN_dim);
        gemm_spm(gate.data(), W_down.data(), out_buf.data(), T, D, FFN_dim, D, MC, KC);
    }

    double tg=0, tu=0, ts=0, td=0;
    double fg = 2.0*T*D*FFN_dim, fd = 2.0*T*FFN_dim*D;
    for (int i = 0; i < N_ITER; ++i) {
        auto tp = Clock::now();
        gemm_spm(x_in.data(), W_gate.data(), gate.data(), T, FFN_dim, D, FFN_dim, MC, KC);
        tg += us_since(tp);
        tp = Clock::now();
        gemm_spm(x_in.data(), W_up.data(), up.data(), T, FFN_dim, D, FFN_dim, MC, KC);
        tu += us_since(tp);
        tp = Clock::now();
        swiglu_fp16(gate.data(), up.data(), (size_t)T * FFN_dim);
        ts += us_since(tp);
        tp = Clock::now();
        gemm_spm(gate.data(), W_down.data(), out_buf.data(), T, D, FFN_dim, D, MC, KC);
        td += us_since(tp);
    }
    tg/=N_ITER; tu/=N_ITER; ts/=N_ITER; td/=N_ITER;
    printf("  W_gate GEMM: %7.1f us   %.2f GFLOPS\n", tg, fg/(tg*1e3));
    printf("  W_up   GEMM: %7.1f us   %.2f GFLOPS\n", tu, fg/(tu*1e3));
    printf("  SwiGLU     : %7.1f us\n", ts);
    printf("  W_down GEMM: %7.1f us   %.2f GFLOPS\n", td, fd/(td*1e3));
    printf("  ──────────────────────────────────\n");
    double tot = tg+tu+ts+td;
    printf("  total       : %7.1f us   %.2f GFLOPS combined\n\n",
           tot, (2*fg+fd)/(tot*1e3));
    return 0;
}
