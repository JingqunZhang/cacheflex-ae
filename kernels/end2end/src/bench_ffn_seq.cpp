// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * bench_ffn_seq.cpp — Sequential FFN benchmark (cache, fused GEMM)
 *
 * LLaMA SwiGLU: gate_GEMM + up_GEMM + SVE swiglu_fp16  (3 steps connected)
 * BERT GELU:    W1_GEMM + SVE gelu_fp16                 (2 steps connected)
 *
 * Uses fused kernel (direct C write, no Cpanel).
 * Weight B prepacked outside ROI.
 *
 * Usage: ./bench_ffn_seq <model:0=llama,1=bert> <T> <MC> <KC> [n_iter]
 */

#include "common.hpp"
#include "kernels_sve.hpp"
#include "kernels_fused.hpp"
#include "layer_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// V1 fused GEMM (same as bench_kernel). C rows use stride N_pad (a multiple
// of the 3*VL tile width) because the micro-kernel stores full tiles; an
// unpadded stride would let tail tiles spill into the next row / past the
// allocation (N=8192 is not a multiple of 3*VL for VL=4/16).
static void gemm_v1_fused(
    const __fp16* A, const __fp16* B_pk, __fp16* C,
    size_t M, size_t K, size_t N, size_t MC, size_t KC, size_t N_pad)
{
    const size_t MT = 8, NT = 3 * svcnth();
    const size_t N_tiles = (N + NT - 1) / NT;
    const size_t max_ab  = (MC + MT - 1) / MT;
    AlignedBuffer<__fp16> A_mc(max_ab * MT * KC);

    for (size_t k0 = 0; k0 < K; k0 += KC) {
        size_t kc = std::min(KC, K - k0), kt = k0 / KC;
        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0), ablocks = (mc + MT - 1) / MT;
            pack_A_fp16_8row(A, K, A_mc.data(), M, m0, K, k0, mc, kc);
            for (size_t n0 = 0; n0 < N; n0 += NT) {
                size_t nt = n0 / NT;
                cache_gemm_fused_8x3VL(
                    A_mc.data(), B_pk + (kt * N_tiles + nt) * KC * NT,
                    C + m0 * N_pad + n0,
                    (int)ablocks, (int)kc, (int)N_pad,
                    (k0 == 0) ? 1 : 0);
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model:0=llama,1=bert> <T> <MC> <KC> [n_iter]\n", argv[0]);
        return 1;
    }
    const bool is_llama = (atoi(argv[1]) == 0);
    const size_t M  = (size_t)atoi(argv[2]);
    const size_t K  = is_llama ? LlamaConfig::D : BertConfig::D;
    const size_t N  = is_llama ? LlamaConfig::FFN_dim : BertConfig::FFN_dim;
    const size_t MC = (size_t)atoi(argv[3]);
    const size_t KC = (size_t)atoi(argv[4]);
    const int n_iter = (argc > 5) ? atoi(argv[5]) : 1;
    const char* mdl = is_llama ? "llama" : "bert";

    AlignedBuffer<__fp16> X(M * K), output(M * N);
    fill_random(X.data(), M * K, -0.1f, 0.1f, 1);

    AlignedBuffer<__fp16> Wg_raw(K * N), Wg_pk(prepack_B_knm_size(K, N, KC));
    fill_random(Wg_raw.data(), K * N, -0.02f, 0.02f, 2);
    prepack_B_knm(Wg_raw.data(), Wg_pk.data(), K, N, KC);

    AlignedBuffer<__fp16> Wu_raw(is_llama ? K * N : 1);
    AlignedBuffer<__fp16> Wu_pk(is_llama ? prepack_B_knm_size(K, N, KC) : 1);
    if (is_llama) {
        fill_random(Wu_raw.data(), K * N, -0.02f, 0.02f, 3);
        prepack_B_knm(Wu_raw.data(), Wu_pk.data(), K, N, KC);
    }

    const size_t NT = 3 * svcnth();
    const size_t N_pad = ((N + NT - 1) / NT) * NT;  // full-tile row stride
    AlignedBuffer<__fp16> gate_buf(M * N_pad), up_buf(is_llama ? M * N_pad : 1);

    auto run = [&]() -> double {
        auto t0 = Clock::now();
        if (is_llama) {
            gemm_v1_fused(X.data(), Wg_pk.data(), gate_buf.data(), M, K, N, MC, KC, N_pad);
            gemm_v1_fused(X.data(), Wu_pk.data(), up_buf.data(), M, K, N, MC, KC, N_pad);
            for (size_t r = 0; r < M; ++r)
                swiglu_fp16(gate_buf.data() + r * N_pad, up_buf.data() + r * N_pad, N);
        } else {
            gemm_v1_fused(X.data(), Wg_pk.data(), gate_buf.data(), M, K, N, MC, KC, N_pad);
            for (size_t r = 0; r < M; ++r)
                gelu_fp16_inplace(gate_buf.data() + r * N_pad, N);
        }
        return us_since(t0);
    };

    run();  // warm
    double total_us = 0;
#ifdef GEM5
    m5_reset_stats(0, 0);
#endif
    ROI_BEGIN();
    for (int i = 0; i < n_iter; ++i) total_us += run();
    ROI_END();
#ifdef GEM5
    m5_dump_stats(0, 0);
#endif

    double avg = total_us / n_iter;
    int ng = is_llama ? 2 : 1;
    printf("RESULT kernel=ffn_seq model=%s T=%zu M=%zu K=%zu N=%zu MC=%zu KC=%zu time_us=%.2f gflops=%.2f\n",
           mdl, M, M, K, N, MC, KC, avg, 2.0 * ng * M * (double)K * N * 1e-9 / (avg * 1e-6));
    { double _ck=0; const __fp16* _p=gate_buf.data();
      for (size_t _r=0;_r<M;++_r)
        for (size_t _c=0;_c<N;++_c) _ck += (double)_p[_r*N_pad+_c];
      printf("CHECKSUM: %.9f\n", _ck); }
    return 0;
}
