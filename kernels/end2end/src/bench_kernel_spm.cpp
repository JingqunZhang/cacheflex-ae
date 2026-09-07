// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * bench_kernel_spm.cpp — SPM GEMM benchmark (V3, fused kernel, B prepacked)
 *
 * Loop order (same as v3_fused_prepack):
 *   [outside ROI] prepack_B → B_pk[K_tiles × N_tiles × KC × NT]
 *   [inside  ROI] K → m0 → packA → n0 → SPMCP(B_pk) → spm_fused_kernel → C
 *
 * Fused kernel writes directly to C (no Cpanel intermediate).
 * SPMCP reads from prepacked contiguous buffer (stride=NT, not N).
 *
 * Usage: ./bench_kernel_spm gemm <model:0/1> <T> <K> <N> <MC> <KC> [n_iter]
 */

#include "kernels_spm.hpp"
#include "kernels_fused.hpp"
#include "layer_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Scalar prepack for SPM build (can't include kernels_sve.hpp)
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
            for (size_t ki=0; ki<kc; ++ki) {
                for (size_t ni=0; ni<nc; ++ni)
                    dst[ki*NT+ni] = B[(k0+ki)*N + n0+ni];
                for (size_t ni=nc; ni<NT; ++ni)
                    dst[ki*NT+ni] = (__fp16)0;
            }
            for (size_t ki=kc; ki<KC; ++ki)
                for (size_t ni=0; ni<NT; ++ni)
                    dst[ki*NT+ni] = (__fp16)0;
        }
    }
}

// V3 SPM fused GEMM
static double gemm_v3_spm_fused(
    const __fp16* A, const __fp16* B_pk, __fp16* C,
    size_t M, size_t K, size_t N, size_t MC, size_t KC,
    size_t ldc = 0)
{
    const size_t MT = 8, NT = 3 * svcnth();
    if (ldc == 0) ldc = N;
    const size_t N_tiles = (N + NT - 1) / NT;
    const size_t max_ab  = (MC + MT - 1) / MT;
    AlignedBuffer<__fp16> A_mc(max_ab * MT * KC);

    auto t0 = Clock::now();
    for (size_t k0 = 0; k0 < K; k0 += KC) {
        size_t kc = std::min(KC, K - k0), kt = k0 / KC;

        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0), ablocks = (mc + MT - 1) / MT;

            pack_A_fp16_8row(A, K, A_mc.data(), M, m0, K, k0, mc, kc);

            for (size_t n0 = 0; n0 < N; n0 += NT) {
                size_t nt = n0 / NT;
                const __fp16* Bp = B_pk + (kt * N_tiles + nt) * KC * NT;

                pack_B_tile_to_spm(Bp, NT, kc, KC, 0);

                spm_gemm_fused_8x3VL(
                    A_mc.data(),
                    C + m0 * ldc + n0,
                    (int)ablocks, (int)kc, (int)ldc,
                    (k0 == 0) ? 1 : 0,
#if defined(VL_8) || defined(VL_16)
                    KC,
#endif
                    0);
            }
        }
    }
    return us_since(t0);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s gemm <model:0/1> <T> <K> <N> <MC> <KC> [n_iter]\n", argv[0]);
        return 1;
    }

    const char* kernel = argv[1];

    if (strcmp(kernel, "gemm") == 0) {
        if (argc < 8) {
            fprintf(stderr, "gemm needs: gemm <model> <T> <K> <N> <MC> <KC> [n_iter]\n");
            return 1;
        }
        const bool is_llama = (atoi(argv[2]) == 0);
        const size_t M = (size_t)atoi(argv[3]), K = (size_t)atoi(argv[4]), N = (size_t)atoi(argv[5]);
        const size_t MC = (size_t)atoi(argv[6]), KC = (size_t)atoi(argv[7]);
        const int n_iter = (argc > 8) ? atoi(argv[8]) : 1;
        const char* mdl = is_llama ? "llama" : "bert";

        const size_t NT = 3 * svcnth();
        const size_t N_pad = ((N + NT - 1) / NT) * NT;
        AlignedBuffer<__fp16> A(M * K), W(K * N), C(M * N_pad);
        fill_random(A.data(), M * K, -0.1f, 0.1f, 1);
        fill_random(W.data(), K * N, -0.02f, 0.02f, 2);

        // Prepack B outside ROI
        AlignedBuffer<__fp16> B_pk(prepack_B_knm_size(K, N, KC));
        prepack_B_knm_spm(W.data(), B_pk.data(), K, N, KC);

        // Warm
        gemm_v3_spm_fused(A.data(), B_pk.data(), C.data(), M, K, N, MC, KC, N_pad);

        double total_us = 0;
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int i = 0; i < n_iter; ++i)
            total_us += gemm_v3_spm_fused(A.data(), B_pk.data(), C.data(), M, K, N, MC, KC, N_pad);
        ROI_END();
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif

        double avg = total_us / n_iter;
        printf("RESULT kernel=gemm loop=V3_SPM_fused model=%s T=%zu M=%zu K=%zu N=%zu MC=%zu KC=%zu time_us=%.2f gflops=%.2f\n",
               mdl, M, M, K, N, MC, KC, avg,
               2.0 * M * (double)K * N * 1e-9 / (avg * 1e-6));
        return 0;
    }

    fprintf(stderr, "ERROR: '%s' is not a GEMM kernel. Use bench_kernel (cache) for rmsnorm/swiglu/residual/rope.\n", kernel);
    return 1;
}
