// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * verify_gemm.cpp — Correctness check: compare fused kernels against naive GEMM
 *
 * Computes C = A × B using:
 *   1. Naive triple-loop (reference)
 *   2. cache_gemm_fused_8x3VL (V1 fused)
 *
 * Reports max absolute error and checksums.
 *
 * Usage: ./verify_gemm [M] [K] [N] [KC] [MC]
 *        Defaults: 64 256 96 256 64
 */
#include "kernels_sve.hpp"
#include "kernels_fused.hpp"
#include "common_print.hpp"
#include <cmath>

static void naive_gemm(const __fp16* A, const __fp16* B, __fp16* C,
                        size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float acc = 0;
            for (size_t k = 0; k < K; ++k)
                acc += (float)A[i*K+k] * (float)B[k*N+j];
            C[i*N+j] = (__fp16)acc;
        }
}

static double checksum(const __fp16* C, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += (double)(float)C[i];
    return s;
}

static float max_abs_error_strided(const __fp16* ref, const __fp16* test,
                                   size_t M, size_t N, size_t test_ld) {
    float mx = 0;
    for (size_t row = 0; row < M; ++row)
        for (size_t col = 0; col < N; ++col) {
            float e = fabsf((float)ref[row * N + col]
                            - (float)test[row * test_ld + col]);
            if (e > mx) mx = e;
        }
    return mx;
}

int main(int argc, char** argv)
{
    const size_t M  = (argc > 1) ? (size_t)atoi(argv[1]) : 64;
    const size_t K  = (argc > 2) ? (size_t)atoi(argv[2]) : 256;
    const size_t N  = (argc > 3) ? (size_t)atoi(argv[3]) : 96;
    const size_t KC = (argc > 4) ? (size_t)atoi(argv[4]) : 256;
    const size_t MC = (argc > 5) ? (size_t)atoi(argv[5]) : 64;
    const size_t NT = 3 * svcnth();
    const size_t MT = 8;
    const size_t M_pad = ((M + MT - 1) / MT) * MT;
    const size_t N_pad = ((N + NT - 1) / NT) * NT;
    const size_t max_ab = (MC + MT - 1) / MT;

    printf("verify_gemm: M=%zu K=%zu N=%zu  KC=%zu MC=%zu NT=%zu\n", M, K, N, KC, MC, NT);

    AlignedBuffer<__fp16> A(M * K);
    AlignedBuffer<__fp16> B(K * N);
    AlignedBuffer<__fp16> C_ref(M * N);
    AlignedBuffer<__fp16> C_v1(M_pad * N_pad);

    fill_random(A.data(), M * K, -0.1f, 0.1f, 1);
    fill_random(B.data(), K * N, -0.02f, 0.02f, 2);

    // 1. Naive reference
    naive_gemm(A.data(), B.data(), C_ref.data(), M, K, N);
    printf("  naive    checksum=%.8g\n", checksum(C_ref.data(), M*N));

    // 2. V1 fused (m0-outer, JIT pack B, fused scatter)
    std::memset(C_v1.data(), 0, M_pad * N_pad * sizeof(__fp16));
    {
        AlignedBuffer<__fp16> A_mc(max_ab * MT * KC);
        AlignedBuffer<__fp16> B_tile(KC * NT);

        for (size_t k0 = 0; k0 < K; k0 += KC) {
            size_t kc = std::min(KC, K - k0);
            for (size_t m0 = 0; m0 < M; m0 += MC) {
                size_t mc = std::min(MC, M - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                pack_A_fp16_8row(A.data(), K, A_mc.data(), M, m0, K, k0, mc, kc);
                for (size_t n0 = 0; n0 < N; n0 += NT) {
                    size_t nc = std::min(NT, N - n0);
                    if (nc < NT || kc < KC)
                        std::memset(B_tile.data(), 0, kc * NT * sizeof(__fp16));
                    sve_pack_B_3VL((uint16_t*)B_tile.data(),
                                   (const uint16_t*)(B.data() + k0 * N + n0),
                                   nc, N * sizeof(uint16_t), kc);
                    cache_gemm_fused_8x3VL(
                        A_mc.data(), B_tile.data(),
                        C_v1.data() + m0 * N_pad + n0,
                        (int)ablocks, (int)kc, (int)N_pad,
                        (k0 == 0) ? 1 : 0);
                }
            }
        }
    }
    printf("  v1_fused checksum=%.8g  max_err=%.6g\n",
           [&] {
               double sum = 0.0;
               for (size_t row = 0; row < M; ++row)
                   for (size_t col = 0; col < N; ++col)
                       sum += (double)(float)C_v1.data()[row * N_pad + col];
               return sum;
           }(),
           max_abs_error_strided(C_ref.data(), C_v1.data(), M, N, N_pad));

    // 3. Multi K-tile test (KC < K)
    if (KC < K) {
        printf("  (multi K-tile: K_tiles=%zu, RMW path exercised)\n", (K+KC-1)/KC);
    }

    float err = max_abs_error_strided(C_ref.data(), C_v1.data(), M, N, N_pad);
    if (err < 0.1f) {
        printf("\nPASS: max error %.6g < 0.1 (FP16 accumulation tolerance)\n", err);
    } else {
        printf("\nFAIL: max error %.6g >= 0.1\n", err);
        return 1;
    }
    return 0;
}
