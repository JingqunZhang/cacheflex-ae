/*
 * 01_rmsnorm.cpp — LLaMA RMSNorm standalone benchmark
 *
 * y[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w
 *   single-pass FP32 accumulation, SVE intrinsics, no bias
 *
 * LLaMA-3.2-1B: T=512, D=2048
 * Usage: ./01_rmsnorm [T] [D] [n_iter]   default: 512 2048 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int D      = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int N_ITER = (argc > 3) ? std::atoi(argv[3]) : 5;

    printf("=== 01 RMSNorm ===  T=%d  D=%d  n_iter=%d\n", T, D, N_ITER);
    printf("  input  [%d, %d]  %.2f MB\n", T, D, T*D*2.0/1048576);
    printf("  weight [%d]      %.2f KB  (no bias)\n\n", D, D*2.0/1024);

    AlignedBuffer<__fp16> x((size_t)T*D), w(D), y((size_t)T*D);
    fill_random(x.data(), (size_t)T*D, -0.5f,  0.5f,  1);
    fill_random(w.data(), D,            0.8f,  1.2f,  2);

    // --- COLD ---
    printf("--- COLD (inputs flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(x.data(), (size_t)T*D * sizeof(__fp16));
    flush_dcache_range(w.data(), D * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0 = Clock::now();
        rmsnorm(x.data(), w.data(), y.data(), T, D);
        double us = us_since(t0);
        // read x[T*D] + w[D] + write y[T*D]
        double bytes = 2.0*(2.0*T*D + D) * sizeof(__fp16);
        printf("  1 run : %7.1f us   BW: %.2f GB/s\n\n", us, bytes/(us*1e3));
    }
    ROI_END();

    // --- WARM: w[D]=4KB fits L1, x[T,D]=2MB is the bandwidth bottleneck ---
    printf("--- WARM ---\n");
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i)
        rmsnorm(x.data(), w.data(), y.data(), T, D);

    double sum = 0;
    for (int i = 0; i < N_ITER; ++i) {
        auto t0 = Clock::now();
        rmsnorm(x.data(), w.data(), y.data(), T, D);
        sum += us_since(t0);
    }
    double us_w = sum / N_ITER;
    double bytes = 2.0*(2.0*T*D + D) * sizeof(__fp16);
    printf("  avg   : %7.1f us   BW: %.2f GB/s\n\n", us_w, bytes/(us_w*1e3));

    printf("  FLOPs per row: ~3D = %d  (square + mean + sqrt + scale)\n", 3*D);
    printf("  w[D]=%.1fKB always L1-warm; bandwidth bound on x[T,D]=%.1fMB\n",
           D*2.0/1024, T*D*2.0/1048576);
    return 0;
}
