/*
 * 07_out_proj.cpp — LLaMA output projection
 *
 * ctx[T, D] → reshape/concat heads → out[T, D] = ctx_flat[T, D] × W_O[D, D]
 * Same shape as Q projection (square GEMM).
 *
 * LLaMA-3.2-1B: T=512, D=2048
 * FLOPs: 2 × T × D × D = 2 × 512 × 2048 × 2048 ≈ 4.3 GFLOPs
 *
 * Usage: ./07_out_proj [T] [D] [n_iter]   default: 512 2048 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int    T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int    D      = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int    N_ITER = (argc > 3) ? std::atoi(argv[3]) : 5;
    const size_t NT_hw  = 3 * svcnth();
    const size_t KC_arg = (argc > 4) ? (size_t)std::atoi(argv[4]) : 0;
    const size_t KC     = KC_arg ? KC_arg : 24576 / NT_hw;  // default: B-panel ≈ 48KB (~¾ L1)
    const size_t MC_arg = (argc > 5) ? (size_t)std::atoi(argv[5]) : 0;
    const size_t MC     = MC_arg ? MC_arg : 16 * svcnth();

    const double W_MB  = (double)D * D * 2.0 / 1048576;
    const double in_MB = (double)T * D * 2.0 / 1048576;
    const size_t sz_pk = prepack_B_knm_size(D, D, KC);
    const double pk_MB = sz_pk * 2.0 / 1048576;

    printf("=== 07 Output Projection ===  T=%d  D=%d  n_iter=%d\n", T, D, N_ITER);
    printf("  ctx/output [%d, %d]  %.2f MB\n", T, D, in_MB);
    printf("  W_O        [%d, %d]  %.2f MB raw  →  %.2f MB pre-packed\n\n",
           D, D, W_MB, pk_MB);

    const size_t sz_in = (size_t)T * D;
    const size_t sz_wt = (size_t)D * D;

    AlignedBuffer<__fp16> ctx(sz_in), W_O(sz_wt), W_O_pk(sz_pk), out(sz_in);

    fill_random(ctx.data(), sz_in, -0.1f, 0.1f, 1);
    fill_random(W_O.data(), sz_wt, -0.02f, 0.02f, 2);
    prepack_B_knm(W_O.data(), W_O_pk.data(), D, D, KC);

    // --- COLD ---
    printf("--- COLD (W_O_pk flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(W_O_pk.data(), sz_pk * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0   = Clock::now();
        gemm_knm_prepacked(ctx.data(), W_O_pk.data(), out.data(), T, D, D, MC, KC);
        double us = us_since(t0);
        printf("  run: %.1f us   %.2f GFLOPS\n\n", us, 2.0*T*D*D/(us*1e3));
    }
    ROI_END();

    // --- WARM ---
    printf("--- WARM ---\n");
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i)
        gemm_knm_prepacked(ctx.data(), W_O_pk.data(), out.data(), T, D, D, MC, KC);

    double sum = 0;
    GemmTiming tim;
    for (int i = 0; i < N_ITER; ++i) {
        auto t0 = Clock::now();
        gemm_knm_prepacked(ctx.data(), W_O_pk.data(), out.data(), T, D, D, MC, KC, &tim);
        sum += us_since(t0);
    }
    double us_w = sum / N_ITER;
    tim.t_packA /= N_ITER; tim.t_compute /= N_ITER; tim.t_writeback /= N_ITER;
    printf("  avg: %.1f us  [packA %.1f  compute %.1f  wb %.1f]  %.2f GFLOPS\n\n",
           us_w, tim.t_packA, tim.t_compute, tim.t_writeback, 2.0*T*D*D/(us_w*1e3));

    printf("  Same square shape as W_Q (D×D) → identical GEMM kernel path\n");
    printf("  W_O %.2f MB packed: %s L3\n", pk_MB,
           pk_MB < 8.0 ? "fits in" : "exceeds typical");
    return 0;
}
