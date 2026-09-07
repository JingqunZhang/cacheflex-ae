/*
 * 05_softmax_causal.cpp — LLaMA causal (masked) softmax
 *
 * For each head h and query row i:
 *   mask:    scores[h, i, j] = -65504 for j > i  (FP16 most-negative-finite)
 *   softmax: scores[h, i, *] = softmax(scores[h, i, *])
 *
 * Operates in-place on scores[H_q, T, T].
 * After masking, masked positions → 0 after exp/normalize.
 *
 * LLaMA-3.2-1B: T=512, H_q=32
 * Usage: ./05_softmax_causal [T] [H_q] [n_iter]
 *        default: 512 32 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int H_q    = (argc > 2) ? std::atoi(argv[2]) : 32;
    const int N_ITER = (argc > 3) ? std::atoi(argv[3]) : 5;

    const size_t Ssz    = (size_t)H_q * T * T;
    const double S_MB   = Ssz * 2.0 / 1048576;

    printf("=== 05 Causal Softmax ===  T=%d  H_q=%d  n_iter=%d\n", T, H_q, N_ITER);
    printf("  scores [%d, %d, %d]  %.2f MB  (in-place, causal mask + softmax)\n\n",
           H_q, T, T, S_MB);

    AlignedBuffer<__fp16> scores(Ssz);
    // Fill with small values (simulating scaled QK scores)
    fill_random(scores.data(), Ssz, -2.f, 2.f, 1);

    // --- COLD ---
    printf("--- COLD (scores flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(scores.data(), Ssz * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0 = Clock::now();
        for (int h = 0; h < H_q; ++h) {
            __fp16* Sh = scores.data() + (size_t)h * T * T;
            causal_softmax_fp16(Sh, T);
        }
        double us = us_since(t0);
        // read + write scores[H_q,T,T]
        double bytes = 2.0 * Ssz * sizeof(__fp16);
        printf("  1 run : %7.1f us   BW: %.2f GB/s\n\n", us, bytes/(us*1e3));
    }
    ROI_END();

    // --- WARM ---
    printf("--- WARM ---\n");
    fill_random(scores.data(), Ssz, -2.f, 2.f, 1);
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i) {
        for (int h = 0; h < H_q; ++h)
            causal_softmax_fp16(scores.data() + (size_t)h*T*T, T);
        fill_random(scores.data(), Ssz, -2.f, 2.f, 1);  // reset for next iter
    }

    double sum = 0;
    for (int i = 0; i < N_ITER; ++i) {
        auto t0 = Clock::now();
        for (int h = 0; h < H_q; ++h)
            causal_softmax_fp16(scores.data() + (size_t)h*T*T, T);
        sum += us_since(t0);
        fill_random(scores.data(), Ssz, -2.f, 2.f, 1);
    }
    double us_w = sum / N_ITER;
    double bytes = 2.0 * Ssz * sizeof(__fp16);
    printf("  avg   : %7.1f us   BW: %.2f GB/s\n\n", us_w, bytes/(us_w*1e3));

    printf("  Causal mask: row i masks %d-i cols on average → ~50%% of T² active\n", T);
    printf("  scores[H_q,T,T]=%.2f MB: read from QK GEMM output, written back\n", S_MB);
    printf("  Triangular access: rows near start are mostly masked (cache-friendly)\n");
    return 0;
}
