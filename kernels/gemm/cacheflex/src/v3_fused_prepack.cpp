// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * v3_fused_prepack.cpp — V3 SPM m0-outer, fused scatter, B prepacked outside ROI
 *
 * Loop order:
 *   [outside ROI] prepack_B_knm → B_pk[K_tiles × N_tiles × KC × NT] (contiguous)
 *   [inside  ROI] K → m0 → packA[MC×KC] → n0 → SPMCP(B_pk contiguous) → fused_kernel → C
 *
 * vs v3_fused.cpp: SPMCP reads from prepacked contiguous buffer (stride=NT)
 *   instead of original row-major B (stride=N). Eliminates strided DRAM access.
 *
 * MOCK_SPM mode: uses JIT cache kernel (same as v1_fused_prepack for correctness check)
 *
 * Usage: ./v3_fused_prepack [M] [K] [N] [n_iter] [KC] [MC]
 *        Defaults: 512 2048 2048 1 1024 1024
 */
#ifdef MOCK_SPM
#  include "kernels_sve.hpp"
#  include "kernels_fused.hpp"
#else
#  include "kernels_spm.hpp"
#  include "kernels_fused.hpp"
#endif
#include "common_print.hpp"

// prepack_B_knm is in kernels_sve.hpp; for SPM build we need it too
#ifdef MOCK_SPM
// already included via kernels_sve.hpp
#else
// Redefine prepack helpers (VL-independent, pure C++)
static size_t prepack_B_knm_size(size_t K, size_t N, size_t KC) {
    const size_t NT = 3 * svcnth();
    return ((K+KC-1)/KC) * ((N+NT-1)/NT) * KC * NT;
}
static void prepack_B_knm_spm(const __fp16* B, __fp16* B_packed, size_t K, size_t N, size_t KC) {
    const size_t NT = 3*svcnth(), N_tiles=(N+NT-1)/NT;
    for (size_t k0=0; k0<K; k0+=KC) {
        size_t kc=std::min(KC,K-k0), kt=k0/KC;
        for (size_t n0=0; n0<N; n0+=NT) {
            size_t nc=std::min(NT,N-n0), nt=n0/NT;
            __fp16* dst=B_packed+(kt*N_tiles+nt)*KC*NT;
            // Simple scalar pack: each row of B[k, n0..n0+nc] → dst row
            for (size_t ki=0; ki<kc; ++ki) {
                for (size_t ni=0; ni<nc; ++ni)
                    dst[ki*NT+ni] = B[(k0+ki)*N + n0+ni];
                for (size_t ni=nc; ni<NT; ++ni)
                    dst[ki*NT+ni] = (__fp16)0;
            }
            // Zero-pad remaining K rows if kc < KC
            for (size_t ki=kc; ki<KC; ++ki)
                for (size_t ni=0; ni<NT; ++ni)
                    dst[ki*NT+ni] = (__fp16)0;
        }
    }
}
#endif

int main(int argc, char** argv)
{
    const size_t M      = (argc > 1) ? (size_t)atoi(argv[1]) : 512;
    const size_t K      = (argc > 2) ? (size_t)atoi(argv[2]) : 2048;
    const size_t N      = (argc > 3) ? (size_t)atoi(argv[3]) : 2048;
    const int    N_ITER = (argc > 4) ? atoi(argv[4]) : 1;
    const size_t KC     = (argc > 5) ? (size_t)atoi(argv[5]) : 1024;
    const size_t MC     = (argc > 6) ? (size_t)atoi(argv[6]) : 1024;
    // SPM capacity guard: KC is the number of SPM sets consumed per B slice.
    //   VL2/VL4: 1 k-step/set -> KC <= 1024;  VL8: 2x KC stacking -> KC <= 512;
    //   VL16: 3x KC stacking -> KC <= 341. Exceeding silently aliases sets/ways.
#if defined(VL_16)
    const size_t KC_MAX = 341;
#elif defined(VL_8)
    const size_t KC_MAX = 512;
#else
    const size_t KC_MAX = 1024;
#endif
#ifndef MOCK_SPM
    if (KC > KC_MAX) {
        fprintf(stderr, "FATAL: KC=%zu exceeds SPM capacity cap %zu for this VL\n", KC, KC_MAX);
        return 1;
    }
#endif
    const size_t NT     = 3 * svcnth();
    const size_t MT     = 8;
    const size_t M_pad  = ((M + MT - 1) / MT) * MT;
    const size_t N_pad  = ((N + NT - 1) / NT) * NT;

    const double GFLOP    = 2.0 * M * K * N * 1e-9;
    const size_t K_tiles  = (K + KC - 1) / KC;
    const size_t N_tiles  = (N + NT - 1) / NT;
    const size_t max_ab   = (MC + MT - 1) / MT;

#ifdef MOCK_SPM
    printf("=== v3_fused_prepack [MOCK_SPM] ===  M=%zu K=%zu N=%zu  NT=%zu\n", M, K, N, NT);
#else
    printf("=== v3_fused_prepack ===  M=%zu K=%zu N=%zu  NT=%zu\n", M, K, N, NT);
#endif
    printf("  KC=%zu  MC=%zu  (m0-outer, fused scatter, B prepacked, SPM)\n", KC, MC);
    printf("  GFLOP/call=%.4f\n", GFLOP);

    AlignedBuffer<__fp16> A(M * K);
    AlignedBuffer<__fp16> B(K * N);
    AlignedBuffer<__fp16> C(M_pad * N_pad);
    fill_random(A.data(), M * K, -0.1f, 0.1f, 1);
    fill_random(B.data(), K * N, -0.02f, 0.02f, 2);
    std::memset(C.data(), 0, M_pad * N_pad * sizeof(__fp16));

    // Prepack B outside ROI
#ifdef MOCK_SPM
    size_t bpk_size = prepack_B_knm_size(K, N, KC);
    AlignedBuffer<__fp16> B_pk(bpk_size);
    prepack_B_knm(B.data(), B_pk.data(), K, N, KC);
#else
    size_t bpk_size = prepack_B_knm_size(K, N, KC);
    AlignedBuffer<__fp16> B_pk(bpk_size);
    prepack_B_knm_spm(B.data(), B_pk.data(), K, N, KC);
#endif
    printf("  B prepacked: %.1f MB (outside ROI)\n", bpk_size*2.0/1024/1024);

    AlignedBuffer<__fp16> A_mc(max_ab * MT * KC);
    std::memset(A_mc.data(), 0, max_ab * MT * KC * sizeof(__fp16));

    printf("--- [ROI] single GEMM (packB excluded) ---\n"); fflush(stdout);

    double t_spmcp=0, t_packA=0, t_kernel=0;
    size_t spmcp_calls=0, packA_calls=0;
    auto _now = []{ return Clock::now(); };

#ifdef GEM5
    m5_reset_stats(0, 0);
#endif
    ROI_BEGIN();

    for (size_t k0 = 0; k0 < K; k0 += KC) {
        size_t kc = std::min(KC, K - k0);
        size_t kt = k0 / KC;

        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0);
            size_t ablocks = (mc + MT - 1) / MT;

            {
                auto _t = _now();
                pack_A_fp16_8row(A.data(), K, A_mc.data(),
                                 M, m0, K, k0, mc, kc);
                t_packA += us_since(_t);
                ++packA_calls;
            }

            for (size_t n0 = 0; n0 < N; n0 += NT) {
                size_t nt = n0 / NT;
                const __fp16* Bp = B_pk.data() + (kt * N_tiles + nt) * KC * NT;

                // SPMCP from prepacked contiguous buffer (stride = NT, not N)
                {
                    auto _t = _now();
#ifdef MOCK_SPM
                    // Mock: just use prepacked B directly (already in correct layout)
#else
                    pack_B_tile_to_spm(Bp, NT, kc, KC, 0);
#endif
                    t_spmcp += us_since(_t);
                    ++spmcp_calls;
                }

                {
                    auto _t = _now();
#ifdef MOCK_SPM
                    cache_gemm_fused_8x3VL(
                        A_mc.data(), Bp,
                        C.data() + m0 * N_pad + n0,
                        (int)ablocks, (int)kc, (int)N_pad,
                        (k0 == 0) ? 1 : 0);
#else
#if defined(VL_8) || defined(VL_16)
                    spm_gemm_fused_8x3VL(
                        A_mc.data(),
                        C.data() + m0 * N_pad + n0,
                        (int)ablocks, (int)kc, (int)N_pad,
                        (k0 == 0) ? 1 : 0, KC, 0);
#else
                    spm_gemm_fused_8x3VL(
                        A_mc.data(),
                        C.data() + m0 * N + n0,
                        (int)ablocks, (int)kc, (int)N,
                        (k0 == 0) ? 1 : 0, 0);
#endif
#endif
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
    print_timing("v3_fused_prepack",
                 t_spmcp, t_packA, t_kernel,
                 spmcp_calls, packA_calls, K_tiles,
                 GFLOP, KC, MC, K, N, M);
    (void)N_ITER;
    return 0;
}
