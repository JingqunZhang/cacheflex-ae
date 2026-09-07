// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * bench_kernel.cpp — Benchmark individual layer kernels (cache, fused GEMM)
 *
 * GEMM uses fused kernel (direct C write, no Cpanel intermediate):
 *   loop=0  V1 (m0-outer): K→m0→packA[MC×KC]→n0→fused_kernel→C
 * Weight B prepacked outside ROI.
 *
 * Usage:
 *   GEMM:  ./bench_kernel gemm <model:0/1> <T> <K> <N> <MC> <KC> [n_iter]
 *   Other: ./bench_kernel <rmsnorm|swiglu|residual|rope> <model:0/1> <T> [n_iter]
 */

#include "common.hpp"
#include "kernels_sve.hpp"
#include "kernels_fused.hpp"
#include "layer_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ============================================================
// V1 fused GEMM: K → m0 → packA → n0 → fused_kernel → C
// B prepacked. No Cpanel, no scatter.
// ============================================================
static double gemm_v1_fused(
    const __fp16* A, const __fp16* B_pk, __fp16* C,
    size_t M, size_t K, size_t N, size_t MC, size_t KC,
    size_t ldc = 0)
{
    const size_t MT = 8, NT = 3 * svcnth();
    if (ldc == 0) ldc = N;  // default: no padding
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

                cache_gemm_fused_8x3VL(
                    A_mc.data(), Bp,
                    C + m0 * ldc + n0,
                    (int)ablocks, (int)kc, (int)ldc,
                    (k0 == 0) ? 1 : 0);
            }
        }
    }
    return us_since(t0);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage:\n"
            "  %s gemm <model:0/1> <T> <K> <N> <MC> <KC> [n_iter]\n"
            "  %s rmsnorm  <model:0/1> <T> [n_iter]\n"
            "  %s swiglu   <model:0/1> <T> [n_iter]\n"
            "  %s residual <model:0/1> <T> [n_iter]\n"
            "  %s rope     <model:0/1> <T> [n_iter]\n",
            argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* kernel = argv[1];
    const bool is_llama = (atoi(argv[2]) == 0);
    const int T = atoi(argv[3]);
    const int D    = is_llama ? LlamaConfig::D    : BertConfig::D;
    const int H    = is_llama ? LlamaConfig::H    : BertConfig::H;
    const int H_kv = is_llama ? LlamaConfig::H_kv : BertConfig::H_kv;
    const int d    = 64;
    const int FFN  = is_llama ? LlamaConfig::FFN_dim : BertConfig::FFN_dim;
    const char* mdl = is_llama ? "llama" : "bert";

    // ================================================================
    // GEMM (fused, V1 m0-outer)
    // ================================================================
    if (strcmp(kernel, "gemm") == 0) {
        if (argc < 8) {
            fprintf(stderr, "gemm needs: gemm <model> <T> <K> <N> <MC> <KC> [n_iter]\n");
            return 1;
        }
        const size_t M = (size_t)T, K = (size_t)atoi(argv[4]), N = (size_t)atoi(argv[5]);
        const size_t MC = (size_t)atoi(argv[6]), KC = (size_t)atoi(argv[7]);
        const int n_iter = (argc > 8) ? atoi(argv[8]) : 1;

        // Pad C columns to NT so fused kernel's full-tile store doesn't overrun
        const size_t NT = 3 * svcnth();
        const size_t N_pad = ((N + NT - 1) / NT) * NT;  // round up to NT
        AlignedBuffer<__fp16> A(M * K), W(K * N), C(M * N_pad);
        fill_random(A.data(), M * K, -0.1f, 0.1f, 1);
        fill_random(W.data(), K * N, -0.02f, 0.02f, 2);

        // Prepack B outside ROI
        AlignedBuffer<__fp16> B_pk(prepack_B_knm_size(K, N, KC));
        prepack_B_knm(W.data(), B_pk.data(), K, N, KC);

        // Warm
        gemm_v1_fused(A.data(), B_pk.data(), C.data(), M, K, N, MC, KC, N_pad);

        double total_us = 0;
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int i = 0; i < n_iter; ++i)
            total_us += gemm_v1_fused(A.data(), B_pk.data(), C.data(), M, K, N, MC, KC, N_pad);
        ROI_END();
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif

        double avg = total_us / n_iter;
        printf("RESULT kernel=gemm loop=V1_fused model=%s T=%d M=%zu K=%zu N=%zu MC=%zu KC=%zu time_us=%.2f gflops=%.2f\n",
               mdl, T, M, K, N, MC, KC, avg,
               2.0 * M * (double)K * N * 1e-9 / (avg * 1e-6));
        return 0;
    }

    // ================================================================
    // Non-GEMM kernels (unchanged)
    // ================================================================
    const int n_iter = (argc > 4) ? atoi(argv[4]) : 1;
    double total_us = 0;

    if (strcmp(kernel, "rmsnorm") == 0) {
        AlignedBuffer<__fp16> x((size_t)T * D), w(D), y((size_t)T * D);
        fill_random(x.data(), (size_t)T*D, -0.1f, 0.1f, 1);
        fill_random(w.data(), D, 0.5f, 1.5f, 2);
        rmsnorm(x.data(), w.data(), y.data(), T, D);
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int i = 0; i < n_iter; ++i) {
            auto t0 = Clock::now();
            rmsnorm(x.data(), w.data(), y.data(), T, D);
            total_us += us_since(t0);
        }
        ROI_END();
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif
        printf("RESULT kernel=rmsnorm model=%s T=%d D=%d time_us=%.2f\n",
               mdl, T, D, total_us / n_iter);

    } else if (strcmp(kernel, "swiglu") == 0) {
        size_t n = (size_t)T * FFN;
        AlignedBuffer<__fp16> gate(n), up(n);
        fill_random(gate.data(), n, -1.f, 1.f, 1);
        fill_random(up.data(), n, -1.f, 1.f, 2);
        swiglu_fp16(gate.data(), up.data(), n);
        fill_random(gate.data(), n, -1.f, 1.f, 1);
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int i = 0; i < n_iter; ++i) {
            auto t0 = Clock::now();
            swiglu_fp16(gate.data(), up.data(), n);
            total_us += us_since(t0);
        }
        ROI_END();
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif
        printf("RESULT kernel=swiglu model=%s T=%d FFN=%d time_us=%.2f\n",
               mdl, T, FFN, total_us / n_iter);

    } else if (strcmp(kernel, "residual") == 0) {
        size_t n = (size_t)T * D;
        AlignedBuffer<__fp16> a(n), b(n), c(n);
        fill_random(a.data(), n, -0.1f, 0.1f, 1);
        fill_random(b.data(), n, -0.1f, 0.1f, 2);
        sve_add_fp16(a.data(), b.data(), c.data(), n);
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int i = 0; i < n_iter; ++i) {
            auto t0 = Clock::now();
            sve_add_fp16(a.data(), b.data(), c.data(), n);
            total_us += us_since(t0);
        }
        ROI_END();
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif
        printf("RESULT kernel=residual model=%s T=%d D=%d time_us=%.2f\n",
               mdl, T, D, total_us / n_iter);

    } else if (strcmp(kernel, "rope") == 0) {
        if (!is_llama) {
            printf("RESULT kernel=rope model=bert T=%d time_us=0.00\n", T);
            return 0;
        }
        AlignedBuffer<__fp16> Q((size_t)H * T * d), K((size_t)H_kv * T * d);
        fill_random(Q.data(), (size_t)H*T*d, -0.1f, 0.1f, 1);
        fill_random(K.data(), (size_t)H_kv*T*d, -0.1f, 0.1f, 2);
        std::vector<float> cos_t, sin_t;
        build_cos_sin(cos_t, sin_t, T, d);
        apply_rope(Q.data(), cos_t.data(), sin_t.data(), H, T, d);
        apply_rope(K.data(), cos_t.data(), sin_t.data(), H_kv, T, d);
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int i = 0; i < n_iter; ++i) {
            auto t0 = Clock::now();
            apply_rope(Q.data(), cos_t.data(), sin_t.data(), H, T, d);
            apply_rope(K.data(), cos_t.data(), sin_t.data(), H_kv, T, d);
            total_us += us_since(t0);
        }
        ROI_END();
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif
        printf("RESULT kernel=rope model=%s T=%d Hq=%d Hkv=%d time_us=%.2f\n",
               mdl, T, H, H_kv, total_us / n_iter);

    } else {
        fprintf(stderr, "Unknown kernel: %s\nValid: gemm, rmsnorm, swiglu, residual, rope\n", kernel);
        return 1;
    }
    return 0;
}
