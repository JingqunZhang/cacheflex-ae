// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * Grouped-query QK GEMM using cache-resident K tiles.
 *
 * Q[T,H_q*d] and K[T,H_kv*d] use head-interleaved storage. Each packed K
 * tile is reused across the query heads associated with one KV head.
 *
 * Usage: ./04_qk_gqa [T] [H_q] [H_kv] [d] [n_iter] [KC] [MC]
 *        default: 512 32 8 64 5 64 128
 */
#include "common.hpp"
#include "kernels_sve.hpp"
#include <cmath>

static constexpr size_t MT_BLK = 8;  // micro-kernel row tile

// ----------------------------------------------------------------
// Per-tile gather-transpose of K followed by kernel execution and scatter.
// ----------------------------------------------------------------
static void run_qk_baseline(
        const __fp16* Q, const __fp16* K, __fp16* scores,
        int T, int T_padded, int H_q, int H_kv, int G, int d,
        int D_q, int D_kv, size_t MC, float scale,
        double& t_packB, double& t_packA, double& t_kernel, double& t_scatter)
{
    const size_t NT      = 3 * svcnth();
    const size_t n_tiles = (size_t)T_padded / NT;
    const size_t kc      = (size_t)d;
    const size_t max_ab  = (MC + MT_BLK - 1) / MT_BLK;
    const size_t num_mc  = ((size_t)T + MC - 1) / MC;
    const svfloat16_t vs = svdup_f16((__fp16)scale);

    // One A-cache slot per query head and M block.
    AlignedBuffer<__fp16>   A_cache((size_t)G * num_mc * max_ab * MT_BLK * kc);
    // K_tile: one N-tile's packed K [d, NT] = 12KB, reused across G Q-heads
    AlignedBuffer<uint16_t> K_tile(kc * NT);
    // Cpanel stores one M-block by N-tile result.
    AlignedBuffer<__fp16>   Cpanel(max_ab * MT_BLK * NT);

    std::memset(A_cache.data(), 0, (size_t)G * num_mc * max_ab * MT_BLK * kc * sizeof(__fp16));
    std::memset(Cpanel.data(),  0, max_ab * MT_BLK * NT * sizeof(__fp16));

    for (int g = 0; g < H_kv; ++g) {

        // --- Pack A: each Q-head stored at its own qi-slot in A_cache ---
        for (int qi = 0; qi < G; ++qi) {
            const int h = g * G + qi;
            auto ta = Clock::now();
            for (size_t m0 = 0; m0 < (size_t)T; m0 += MC) {
                pack_A_fp16_8row(Q + (size_t)h * d, (size_t)D_q,
                    A_cache.data() + ((size_t)qi * num_mc + m0/MC) * max_ab * MT_BLK * kc,
                    (size_t)T, m0, kc, 0, std::min(MC, (size_t)T - m0), kc);
            }
            t_packA += us_since(ta);
        }

        // --- Per N-tile: SVE gather-transpose K then compute all G Q-heads ---
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            const size_t t_start = ti * NT;
            const size_t nc      = std::min(NT, (size_t)T - t_start);

            // gather-transpose: K[t_start:t_start+nc, g*d:g*d+d] → K_tile[d, NT]
            // Same SVE gather+uzp1 as pack_K_direct (stride = D_kv between tokens)
            {
                auto tp = Clock::now();
                std::memset(K_tile.data(), 0, kc * NT * sizeof(uint16_t));
                const uint16_t* Kg = reinterpret_cast<const uint16_t*>(
                    K + (size_t)t_start * D_kv + (size_t)g * d);
                pack_K_direct(K_tile.data(), Kg, (int)nc, d, NT, D_kv);
                t_packB += us_since(tp);
            }

            const __fp16* Kpk = reinterpret_cast<const __fp16*>(K_tile.data());

            // Compute G Q-heads sharing this K tile
            for (int qi = 0; qi < G; ++qi) {
                const int h   = g * G + qi;
                __fp16*   C   = scores + (size_t)h * T * T;

                for (size_t m0 = 0; m0 < (size_t)T; m0 += MC) {
                    const size_t mc      = std::min(MC, (size_t)T - m0);
                    const size_t ablocks = (mc + MT_BLK - 1) / MT_BLK;
                    const __fp16* Ap = A_cache.data() +
                        ((size_t)qi * num_mc + m0/MC) * max_ab * MT_BLK * kc;

                    // Execute one packed K tile (nB=1).
                    {
                        auto tk = Clock::now();
                        sve_interleaved_fp16_mla_8x3VL(Ap, Kpk, Cpanel.data(),
                                                        (int)ablocks, 1, (int)kc);
                        t_kernel += us_since(tk);
                    }

                    // scatter + inline scale → scores[h][m0:, t_start:]
                    // Scatter Cpanel into the head-major score matrix.
                    {
                        auto ts = Clock::now();
                        for (size_t mb = 0; mb < ablocks; ++mb) {
                            const __fp16* tp = Cpanel.data() + mb * MT_BLK * NT;
                            for (size_t r = 0; r < MT_BLK; ++r) {
                                const size_t gr = m0 + mb * MT_BLK + r;
                                if (gr >= (size_t)T) break;
                                __fp16*       c_row = C + gr * T + t_start;
                                const __fp16* t_row = tp + r * NT;
                                size_t i = 0;
                                svbool_t pg = svptrue_b16();
                                for (; i + svcnth() <= nc; i += svcnth())
                                    svst1_f16(pg, c_row+i,
                                        svmul_f16_x(pg, svld1_f16(pg, t_row+i), vs));
                                if (i < nc) {
                                    svbool_t pt = svwhilelt_b16_u64(i, nc);
                                    svst1_f16(pt, c_row+i,
                                        svmul_f16_x(pt, svld1_f16(pt, t_row+i), vs));
                                }
                            }
                        }
                        t_scatter += us_since(ts);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    const int T      = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int H_q    = (argc > 2) ? std::atoi(argv[2]) : 32;
    const int H_kv   = (argc > 3) ? std::atoi(argv[3]) : 8;
    const int d      = (argc > 4) ? std::atoi(argv[4]) : 64;
    const int N_ITER = (argc > 5) ? std::atoi(argv[5]) : 5;
    const size_t MC_arg = (argc > 6) ? (size_t)std::atoi(argv[6]) : 0;
    const size_t MC     = MC_arg ? MC_arg : 16 * svcnth();
    const int G = H_q / H_kv;

    const size_t NT       = 3 * svcnth();
    const size_t T_padded = ((size_t)T + NT - 1) / NT * NT;
    const size_t n_tiles  = T_padded / NT;
    const int D_q  = H_q  * d;
    const int D_kv = H_kv * d;
    const float scale = 1.0f / sqrtf((float)d);

    const double Q_MB     = (size_t)T * D_q  * 2.0 / 1048576;
    const double K_MB     = (size_t)T * D_kv * 2.0 / 1048576;
    const double score_MB = (size_t)H_q  * T * T * 2.0 / 1048576;

    printf("=== 04 QK GQA GEMM ===  T=%d  H_q=%d  H_kv=%d  G=%d  d=%d  n_iter=%d\n",
           T, H_q, H_kv, G, d, N_ITER);
    printf("  Q      [%d, %d]  %.2f MB  (head-interleaved, stride D_q=%d)\n", T, D_q, Q_MB, D_q);
    printf("  K      [%d, %d]  %.2f MB  (head-interleaved, stride D_kv=%d)\n", T, D_kv, K_MB, D_kv);
    printf("  scores [%d, %d, %d]  %.2f MB\n", H_q, T, T, score_MB);
    printf("  NT=%zu  T_padded=%zu  n_tiles=%zu  MC=%zu\n", NT, T_padded, n_tiles, MC);
    printf("  K_tile per tile: %.1f KB  (d=%d × NT=%zu × 2B)\n\n",
           (double)d*NT*2/1024, d, NT);
    printf("  GQA: pack K_g once per N-tile, reuse for G=%d Q-heads\n\n", G);

    AlignedBuffer<__fp16> Q((size_t)T * D_q);
    AlignedBuffer<__fp16> K((size_t)T * D_kv);
    AlignedBuffer<__fp16> scores((size_t)H_q * T * T);

    fill_random(Q.data(), (size_t)T*D_q,  -0.1f, 0.1f, 1);
    fill_random(K.data(), (size_t)T*D_kv, -0.1f, 0.1f, 2);

    const double total_flops = 2.0 * H_q * T * T * d;

    // --- COLD ---
    printf("--- COLD (Q and K flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(Q.data(), (size_t)T*D_q  * sizeof(__fp16));
    flush_dcache_range(K.data(), (size_t)T*D_kv * sizeof(__fp16));

    ROI_BEGIN();
    {
        double tb=0, ta=0, tk=0, ts=0;
        auto t0 = Clock::now();
        run_qk_baseline(Q.data(), K.data(), scores.data(),
                        T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, scale,
                        tb, ta, tk, ts);
        double us = us_since(t0);
        printf("  COLD: %7.1f us   %.2f GFLOPS\n", us, total_flops/(us*1e3));
        printf("    packB(gather+transpose): %7.1f us  (%5.1f%%)\n", tb, 100.*tb/us);
        printf("    packA:                  %7.1f us  (%5.1f%%)\n", ta, 100.*ta/us);
        printf("    kernel:                 %7.1f us  (%5.1f%%)\n", tk, 100.*tk/us);
        printf("    scatter(+scale):        %7.1f us  (%5.1f%%)\n\n", ts, 100.*ts/us);
        fflush(stdout);
    }
    ROI_END();

    // --- WARM (1 un-timed warmup) ---
    { double tb=0,ta=0,tk=0,ts=0;
      run_qk_baseline(Q.data(), K.data(), scores.data(),
                      T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, scale,
                      tb,ta,tk,ts); }

    printf("--- WARM ---\n"); fflush(stdout);
    double t_total=0, tb_w=0, ta_w=0, tk_w=0, ts_w=0;
    for (int it = 0; it < N_ITER; ++it) {
        double tb=0,ta=0,tk=0,ts=0;
        auto t0 = Clock::now();
        run_qk_baseline(Q.data(), K.data(), scores.data(),
                        T, (int)T_padded, H_q, H_kv, G, d, D_q, D_kv, MC, scale,
                        tb, ta, tk, ts);
        t_total += us_since(t0);
        tb_w+=tb; ta_w+=ta; tk_w+=tk; ts_w+=ts;
    }
    t_total/=N_ITER; tb_w/=N_ITER; ta_w/=N_ITER; tk_w/=N_ITER; ts_w/=N_ITER;
    printf("  WARM: %7.1f us/iter   %.2f GFLOPS\n", t_total, total_flops/(t_total*1e3));
    printf("    packB(gather+transpose): %7.1f us  (%5.1f%%)\n", tb_w, 100.*tb_w/t_total);
    printf("    packA:                  %7.1f us  (%5.1f%%)\n", ta_w, 100.*ta_w/t_total);
    printf("    kernel:                 %7.1f us  (%5.1f%%)\n", tk_w, 100.*tk_w/t_total);
    printf("    scatter(+scale):        %7.1f us  (%5.1f%%)\n\n", ts_w, 100.*ts_w/t_total);

    printf("  K packing: %d times (H_kv) vs %d in MHA (H_q) — %dx savings\n",
           H_kv, H_q, G);
    printf("  K_tile 12KB per tile: L1-resident during kernel, L1 hit during scatter\n");
    return 0;
}
