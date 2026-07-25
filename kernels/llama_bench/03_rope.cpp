/*
 * 03_rope.cpp — LLaMA RoPE (Rotary Position Embedding)
 *
 * Applies RoPE in-place to Q[T, H_q*d] and K[T, H_kv*d] (head-interleaved):
 *   for each head h and token t, pair (i, i+d/2):
 *     x_out[i]     = x[i]     * cos[t,i] - x[i+d/2] * sin[t,i]
 *     x_out[i+d/2] = x[i+d/2] * cos[t,i] + x[i]     * sin[t,i]
 *
 * cos/sin table: [T, d/2], precomputed from theta_i = 1/10000^(2i/d)
 * apply_rope(x, cos, sin, H, T, d, tok_stride, head_stride): head-interleaved strides
 *
 * LLaMA-3.2-1B: T=512, H_q=32, H_kv=8, d=64
 * Usage: ./03_rope [T] [H_q] [H_kv] [d] [n_iter]
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

    // Q[T, H_q*d] and K[T, H_kv*d]: head-interleaved, matching QKV projection output
    const int    D_q  = H_q  * d;   // 2048 = full query dim
    const int    D_kv = H_kv * d;   // 512  = full KV dim
    const size_t Qsz  = (size_t)T * D_q;
    const size_t Ksz  = (size_t)T * D_kv;
    const int    half = d / 2;
    const size_t CSsz = (size_t)T * half;   // cos/sin table: T × d/2 FP32

    const double Q_MB  = Qsz  * 2.0 / 1048576;
    const double K_MB  = Ksz  * 2.0 / 1048576;
    const double CS_MB = CSsz * 4.0 / 1048576;  // FP32

    printf("=== 03 RoPE ===  T=%d  H_q=%d  H_kv=%d  d=%d  n_iter=%d\n",
           T, H_q, H_kv, d, N_ITER);
    printf("  Q   [%d, %d]  %.2f MB  (head-interleaved, in-place)\n", T, D_q, Q_MB);
    printf("  K   [%d, %d]   %.2f MB  (head-interleaved, in-place)\n", T, D_kv, K_MB);
    printf("  cos/sin [%d, %d]   %.2f MB  (FP32)\n\n", T, half, CS_MB);

    // build_cos_sin takes vectors by reference and resizes them
    std::vector<float> cos_tbl, sin_tbl;
    build_cos_sin(cos_tbl, sin_tbl, T, d);

    AlignedBuffer<__fp16> Q(Qsz), K(Ksz);
    fill_random(Q.data(), Qsz, -1.f, 1.f, 1);
    fill_random(K.data(), Ksz, -1.f, 1.f, 2);

    // --- COLD ---
    printf("--- COLD (Q and K flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(Q.data(), Qsz * sizeof(__fp16));
    flush_dcache_range(K.data(), Ksz * sizeof(__fp16));

    ROI_BEGIN();
    {
        auto t0 = Clock::now();
        // apply_rope with interleaved strides: tok_stride=H*d, head_stride=d
        apply_rope(Q.data(), cos_tbl.data(), sin_tbl.data(), H_q,  T, d, D_q,  d);
        apply_rope(K.data(), cos_tbl.data(), sin_tbl.data(), H_kv, T, d, D_kv, d);
        double us = us_since(t0);
        // read+write Q[H_q,T,d] and K[H_kv,T,d]; cos/sin read-only (likely warm)
        double bytes = 2.0 * (Qsz + Ksz) * sizeof(__fp16) + 2.0 * CSsz * sizeof(float);
        printf("  1 run : %7.1f us   BW: %.2f GB/s\n\n", us, bytes/(us*1e3));
    }
    ROI_END();

    // --- WARM: cos/sin table cached, Q/K bandwidth bound ---
    printf("--- WARM ---\n");
    // Re-fill to have valid FP16 data (RoPE is in-place and not idempotent)
    fill_random(Q.data(), Qsz, -1.f, 1.f, 1);
    fill_random(K.data(), Ksz, -1.f, 1.f, 2);
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i) {
        apply_rope(Q.data(), cos_tbl.data(), sin_tbl.data(), H_q,  T, d, D_q,  d);
        apply_rope(K.data(), cos_tbl.data(), sin_tbl.data(), H_kv, T, d, D_kv, d);
        fill_random(Q.data(), Qsz, -1.f, 1.f, 1);
        fill_random(K.data(), Ksz, -1.f, 1.f, 2);
    }

    double sum = 0;
    for (int i = 0; i < N_ITER; ++i) {
        auto t0 = Clock::now();
        apply_rope(Q.data(), cos_tbl.data(), sin_tbl.data(), H_q,  T, d, D_q,  d);
        apply_rope(K.data(), cos_tbl.data(), sin_tbl.data(), H_kv, T, d, D_kv, d);
        sum += us_since(t0);
        fill_random(Q.data(), Qsz, -1.f, 1.f, 1);
        fill_random(K.data(), Ksz, -1.f, 1.f, 2);
    }
    double us_w = sum / N_ITER;
    double bytes = 2.0 * (Qsz + Ksz) * sizeof(__fp16) + 2.0 * CSsz * sizeof(float);
    printf("  avg   : %7.1f us   BW: %.2f GB/s\n\n", us_w, bytes/(us_w*1e3));

    printf("  cos/sin [T=%d, d/2=%d] = %.2f MB FP32, likely L2-warm\n", T, half, CS_MB);
    printf("  Q[T,%d]=%.2f MB + K[T,%d]=%.2f MB bandwidth bound (head-interleaved)\n",
           D_q, Q_MB, D_kv, K_MB);
    return 0;
}
