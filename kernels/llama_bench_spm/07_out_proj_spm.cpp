/*
 * 07_out_proj_spm.cpp — LLaMA output projection with CacheFlex SPM
 *
 * ctx[T, D] × W_O[D, D] → out[T, D]
 * SPM: W_O tiles pinned in scratchpad (no L2 eviction during compute)
 *
 * LLaMA-3.2-1B: T=512, D=2048
 * FLOPs: 2 × T × D × D = 2 × 512 × 2048 × 2048 ≈ 4.3 GFLOPs
 *
 * Compile: -DVL_2, -DVL_4, -DVL_8, or -DVL_16
 * Usage: ./07_out_proj_spm_vl4 [T] [D] [n_iter] [KC] [MC]
 *        default: 512 2048 5 256 128
 */
#include "kernels_spm.hpp"

int main(int argc, char** argv)
{
    const int    T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int    D      = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int    N_ITER = (argc > 3) ? std::atoi(argv[3]) : 5;
    const size_t KC     = (argc > 4) ? (size_t)std::atoi(argv[4]) : 256;
    const size_t MC     = (argc > 5) ? (size_t)std::atoi(argv[5]) : 128;
    const size_t NT     = 3 * svcnth();

    const double W_MB  = (double)D * D * 2.0 / 1048576;
    const double in_MB = (double)T * D * 2.0 / 1048576;

    printf("=== 07 Out Proj SPM ===  T=%d  D=%d  n_iter=%d  KC=%zu  MC=%zu\n",
           T, D, N_ITER, KC, MC);
    printf("  NT=%zu (3×svcnth())   VL=%zu-bit\n", NT, NT/3*2*8);
    printf("  ctx/output [%d, %d]  %.2f MB\n", T, D, in_MB);
    printf("  W_O        [%d, %d]  %.2f MB raw (no prepack, direct SPM)\n\n", D, D, W_MB);

    AlignedBuffer<__fp16> ctx((size_t)T * D);
    AlignedBuffer<__fp16> W_O((size_t)D * D);
    AlignedBuffer<__fp16> out((size_t)T * D);

    fill_random(ctx.data(),  (size_t)T*D,  -0.1f, 0.1f,  1);
    fill_random(W_O.data(),  (size_t)D*D,  -0.02f, 0.02f, 2);

    // --- COLD: flush W_O to DRAM ---
    printf("--- COLD (W_O flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(W_O.data(), (size_t)D * D * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0 = Clock::now();
        gemm_spm(ctx.data(), W_O.data(), out.data(), T, D, D, D, MC, KC);
        double us = us_since(t0);
        printf("  run: %.1f us   %.2f GFLOPS\n\n", us, 2.0*T*D*D/(us*1e3));
    }
    ROI_END();

    // --- WARM ---
    printf("--- WARM ---\n");
    if (N_ITER > 1)
        gemm_spm(ctx.data(), W_O.data(), out.data(), T, D, D, D, MC, KC);

    double sum = 0;
    for (int i = 0; i < N_ITER; ++i) {
        auto t0 = Clock::now();
        gemm_spm(ctx.data(), W_O.data(), out.data(), T, D, D, D, MC, KC);
        sum += us_since(t0);
    }
    double us_w = sum / N_ITER;
    printf("  avg: %.1f us   %.2f GFLOPS\n\n", us_w, 2.0*T*D*D/(us_w*1e3));

    printf("  W_O %.2f MB → SPM tiles (%.0f tiles × KC=%zu × NT=%zu × 2B)\n",
           W_MB, (double)D/NT * D/KC, KC, NT);
    printf("  SPM: B tiles pinned, no L2 eviction during MC-loop\n");
    return 0;
}
