// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * Cache GEMM with an m0-outer loop and fused direct writes to C.
 *
 * Loop order: K → m0 → pack A → n0 → pack B → fused kernel → C.
 *
 * Usage: ./v1_fused [M] [K] [N] [n_iter] [KC] [MC]
 *        Defaults: 512 2048 2048 1 256 128
 */
#include "kernels_sve.hpp"
#include "kernels_fused.hpp"
#include "common_print.hpp"

int main(int argc, char** argv)
{
    const size_t M      = (argc > 1) ? (size_t)atoi(argv[1]) : 512;
    const size_t K      = (argc > 2) ? (size_t)atoi(argv[2]) : 2048;
    const size_t N      = (argc > 3) ? (size_t)atoi(argv[3]) : 2048;
    const int    N_ITER = (argc > 4) ? atoi(argv[4]) : 1;
    const size_t KC     = (argc > 5) ? (size_t)atoi(argv[5]) : 256;
    const size_t MC     = (argc > 6) ? (size_t)atoi(argv[6]) : 128;
    const size_t NT     = 3 * svcnth();
    const size_t MT     = 8;

    const double GFLOP    = 2.0 * M * K * N * 1e-9;
    const size_t K_tiles  = (K + KC - 1) / KC;
    const size_t M_tiles  = (M + MC - 1) / MC;
    const size_t N_tiles  = (N + NT - 1) / NT;
    const size_t max_ab   = (MC + MT - 1) / MT;

    printf("=== v1_fused ===  M=%zu K=%zu N=%zu  NT=%zu\n", M, K, N, NT);
    printf("  KC=%zu  MC=%zu  (m0-outer, fused scatter, no Cpanel)\n", KC, MC);
    printf("  A_mc=%.0fkB  B_tile=%.0fkB(JIT,L1)  C[M×N]=%.1fMB\n",
           MC*KC*2.0/1024, KC*NT*2.0/1024, M*N*2.0/1024/1024);
    printf("  GFLOP/call=%.4f\n\n", GFLOP);

    AlignedBuffer<__fp16> A(M * K);
    AlignedBuffer<__fp16> B(K * N);
    const size_t M_pad = ((M + MT - 1) / MT) * MT;
    const size_t N_pad = ((N + NT - 1) / NT) * NT;
    AlignedBuffer<__fp16> C(M_pad * N_pad);
    fill_random(A.data(), M * K, -0.1f, 0.1f, 1);
    fill_random(B.data(), K * N, -0.1f, 0.1f, 2);
    std::memset(C.data(), 0, M_pad * N_pad * sizeof(__fp16));

    AlignedBuffer<__fp16> A_mc(max_ab * MT * KC);
    AlignedBuffer<__fp16> B_tile(KC * NT);
    std::memset(A_mc.data(),    0, max_ab * MT * KC * sizeof(__fp16));
    std::memset(B_tile.data(),  0, KC * NT * sizeof(__fp16));

    printf("--- [ROI] single GEMM ---\n"); fflush(stdout);

    double t_packB=0, t_packA=0, t_kernel=0;
    size_t packB_calls=0, packA_calls=0;
    auto _now = []{ return Clock::now(); };

#ifdef GEM5
    m5_reset_stats(0, 0);
#endif
    ROI_BEGIN();

    for (size_t k0 = 0; k0 < K; k0 += KC) {
        size_t kc = std::min(KC, K - k0);

        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0);
            size_t ablocks = (mc + MT - 1) / MT;

            // Pack A[MC×KC] → L1-hot
            {
                auto _t = _now();
                pack_A_fp16_8row(A.data(), K, A_mc.data(),
                                 M, m0, K, k0, mc, kc);
                t_packA += us_since(_t);
                ++packA_calls;
            }

            // Sweep n0: JIT pack B, then fused kernel → C
            for (size_t n0 = 0; n0 < N; n0 += NT) {
                size_t nc = std::min(NT, N - n0);

                // JIT pack B tile → L1-hot
                {
                    auto _t = _now();
                    if (nc < NT || kc < KC)
                        std::memset(B_tile.data(), 0, kc * NT * sizeof(__fp16));
                    sve_pack_B_3VL((uint16_t*)B_tile.data(),
                                   (const uint16_t*)(B.data() + k0 * N + n0),
                                   nc, N * sizeof(uint16_t), kc);
                    t_packB += us_since(_t);
                    ++packB_calls;
                }

                // Fused kernel: compute + write/RMW directly to C
                {
                    auto _t = _now();
                    cache_gemm_fused_8x3VL(
                        A_mc.data(), B_tile.data(),
                        C.data() + m0 * N_pad + n0,
                        (int)ablocks, (int)kc, (int)N_pad,
                        (k0 == 0) ? 1 : 0);
                    t_kernel += us_since(_t);
                }
            }
        }
    }

    ROI_END();
#ifdef GEM5
    m5_dump_stats(0, 0);
#endif

    print_checksum_logical(C.data(), M, N, N_pad);
    print_timing("v1_fused",
                 t_packB, t_packA, t_kernel,
                 packB_calls, packA_calls, K_tiles,
                 GFLOP, KC, MC, K, N, M);
    (void)N_ITER;
    return 0;
}
