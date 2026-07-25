/*
 * 09_residual_add.cpp — LLaMA residual connection
 *
 * output = x_input + attn_output   (element-wise FP16 add)
 * LLaMA has two residual connections per layer:
 *   (1) after MHA: x + out_proj(attn)
 *   (2) after FFN: x + ffn(x)
 *
 * LLaMA-3.2-1B: T=512, D=2048
 * Usage: ./09_residual_add [T] [D] [n_iter]   default: 512 2048 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int D      = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int N_ITER = (argc > 3) ? std::atoi(argv[3]) : 5;

    const size_t N  = (size_t)T * D;
    const double MB = N * 2.0 / 1048576;

    printf("=== 09 Residual Add ===  T=%d  D=%d  n_iter=%d\n", T, D, N_ITER);
    printf("  x / attn_out / output  [%d, %d]  %.2f MB each\n\n", T, D, MB);

    AlignedBuffer<__fp16> x(N), attn(N), output(N);
    fill_random(x.data(),    N, -1.f, 1.f, 1);
    fill_random(attn.data(), N, -1.f, 1.f, 2);

    // --- COLD ---
    printf("--- COLD (inputs flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(x.data(),    N * sizeof(__fp16));
    flush_dcache_range(attn.data(), N * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0  = Clock::now();
        sve_add_fp16(x.data(), attn.data(), output.data(), N);
        double us = us_since(t0);
        double bw = 3.0 * N * 2 / (us * 1e3);  // read x + read attn + write out
        printf("  1 run: %.1f us   BW: %.2f GB/s\n\n", us, bw);
    }
    ROI_END();

    // --- WARM ---
    printf("--- WARM ---\n");
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i)
        sve_add_fp16(x.data(), attn.data(), output.data(), N);

    double sum = 0;
    for (int i = 0; i < N_ITER; ++i) {
        auto t0 = Clock::now();
        sve_add_fp16(x.data(), attn.data(), output.data(), N);
        sum += us_since(t0);
    }
    double us_w = sum / N_ITER;
    double bw   = 3.0 * N * 2 / (us_w * 1e3);
    printf("  avg: %.1f us   BW: %.2f GB/s\n\n", us_w, bw);

    printf("  Both inputs fresh from prior GEMMs → cache state from out_proj/FFN\n");
    printf("  x[T,D]=%.2f MB: 4× larger than BERT residual (D=2048 vs D=768)\n", MB);
    printf("  Residual add is typically <1%%%% of total transformer layer time\n");
    return 0;
}
