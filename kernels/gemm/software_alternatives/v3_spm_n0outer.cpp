// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * v3_spm_n0outer.cpp — SPM GEMM with n0-outer loop (current SPM design)
 *
 * Loop order:  KC → packA[M×KC] → n0 → SPMCP[KC×NT→SPM] → m0 → spm_kernel → scatter
 *
 * Design rationale:
 *   - Pack all M-rows of A once per K-tile (A_cache[M×KC])
 *   - SPMCP one NT-tile of B into SPM per n0 (slot 0)
 *   - spm_kernel sweeps all M-blocks with B pinned in SPM → no B cache pressure
 *   - n0 outer → each SPMCP amortized over all M-blocks (K_tiles × N_tiles SPMCP calls)
 *
 * MOCK_SPM mode (QEMU): replaces SPMCP+spm.ld1qd with
 *   sve_pack_B_3VL → local B_tile  +  sve_interleaved_fp16_mla_8x3VL
 *   (produces bit-identical output to real SPM)
 *
 * Compile (QEMU):
 *   $CXX $CFLAGS -DMOCK_SPM $BASE_INC v3_spm_n0outer.cpp -lm \
 *     -o bin_qemu/v3_spm_n0outer_mock
 * Compile (gem5, 3-step SPM build):
 *   $CXX $SPM_FLAGS -DVL_2 $GEM5_FLAGS $SPM_INC $BASE_INC v3_spm_n0outer.cpp \
 *     -S -o v3_tmp.s && python3 spm_compiler.py v3_tmp.s v3_enc.s && \
 *     $CXX -static -o bin_gem5/v3_spm_n0outer_vl2 v3_enc.s $M5OP -lm
 *
 * Usage: ./v3_spm_n0outer [M] [K] [N] [n_iter] [KC] [MC]
 *        Defaults: 512 2048 2048 1 256 128
 */
#ifdef MOCK_SPM
#  include "kernels_sve.hpp"   // -I../../llama_bench  (no VL flag needed)
#else
#  include "kernels_spm.hpp"   // -I../../llama_bench_spm  (needs -DVL_N)
#endif
#include "loop_compare_common.hpp"

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
    const size_t num_mc   = (M + MC - 1) / MC;
    const size_t max_ab   = (MC + MT - 1) / MT;

#ifdef MOCK_SPM
    printf("=== v3_spm_n0outer [MOCK_SPM] ===  M=%zu K=%zu N=%zu  NT=%zu\n", M, K, N, NT);
#else
    printf("=== v3_spm_n0outer ===  M=%zu K=%zu N=%zu  NT=%zu\n", M, K, N, NT);
#endif
    printf("  KC=%zu  MC=%zu  (no NC, SPM B-pack per n0)\n", KC, MC);
    printf("  A_cache=%.0fkB(L2-warm)  B_tile_SPM=%.0fkB  C[M×N]=%.1fMB\n",
           M*KC*2.0/1024, KC*NT*2.0/1024, M*N*2.0/1024/1024);
    printf("  packA calls=%zu (K_tiles=%zu × M_tiles=%zu)\n",
           K_tiles*M_tiles, K_tiles, M_tiles);
    printf("  SPMCP calls=%zu (K_tiles=%zu × N_tiles=%zu)\n",
           K_tiles*N_tiles, K_tiles, N_tiles);
    printf("  GFLOP/call=%.4f\n\n", GFLOP);

    AlignedBuffer<__fp16> A(M * K);
    AlignedBuffer<__fp16> B(K * N);
    AlignedBuffer<__fp16> C(M * N);
    fill_random(A.data(), M * K, -0.1f, 0.1f, 1);
    fill_random(B.data(), K * N, -0.1f, 0.1f, 2);
    std::memset(C.data(), 0, M * N * sizeof(__fp16));

    // A_cache: all M-blocks packed once per K-tile
    AlignedBuffer<__fp16> A_cache(num_mc * max_ab * MT * KC);
    AlignedBuffer<__fp16> Cpanel(max_ab * MT * NT);
#ifdef MOCK_SPM
    // Mock SPM: local buffer mimics SPM slot 0 (KC×NT, same layout as spm.ld1qd)
    AlignedBuffer<__fp16> B_spm(KC * NT);
    std::memset(B_spm.data(), 0, KC * NT * sizeof(__fp16));
#else
    // B_pad: zero-padded partial B tile for tail n0
    AlignedBuffer<__fp16> B_pad(KC * NT);
    std::memset(B_pad.data(), 0, KC * NT * sizeof(__fp16));
#endif
    std::memset(A_cache.data(), 0, num_mc * max_ab * MT * KC * sizeof(__fp16));
    std::memset(Cpanel.data(),  0, max_ab * MT * NT           * sizeof(__fp16));

    printf("--- [ROI] single GEMM ---\n"); fflush(stdout);

    double t_packB=0, t_packA=0, t_kernel=0, t_cwrite=0, t_crmw=0;
    size_t spmcp_calls=0, packA_calls=0;
    auto _now = []{ return Clock::now(); };

#ifdef GEM5
    m5_reset_stats(0, 0);
#endif
    ROI_BEGIN();

    for (size_t k0 = 0; k0 < K; k0 += KC) {               // K-tile
        size_t kc = std::min(KC, K - k0);

        // Phase 1: pack ALL M-blocks of A for this K-tile
        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0);
            auto _t = _now();
            pack_A_fp16_8row(A.data(), K,
                             A_cache.data() + (m0/MC)*max_ab*MT*KC,
                             M, m0, K, k0, mc, kc);
            t_packA += us_since(_t);
            ++packA_calls;
        }

        // Phase 2: sweep N; SPMCP once per n0, kernel sweeps all m0
        for (size_t n0 = 0; n0 < N; n0 += NT) {           // n0: SPMCP → SPM
            size_t nc = std::min(NT, N - n0);

            // SPMCP one NT-tile into SPM slot 0
            {
                auto _t = _now();
#ifdef MOCK_SPM
                if (nc < NT || kc < KC)
                    std::memset(B_spm.data(), 0, kc * NT * sizeof(__fp16));
                sve_pack_B_3VL((uint16_t*)B_spm.data(),
                               (const uint16_t*)(B.data() + k0 * N + n0),
                               nc, N * sizeof(uint16_t), kc);
#else
                if (nc == NT) {
                    pack_B_tile_to_spm(B.data() + k0 * N + n0, N, kc, KC, 0);
                } else {
                    size_t kc_cur = kc;
                    std::memset(B_pad.data(), 0, kc_cur * NT * sizeof(__fp16));
                    for (size_t ki = 0; ki < kc_cur; ++ki)
                        std::memcpy(B_pad.data() + ki * NT,
                                    B.data() + (k0 + ki) * N + n0,
                                    nc * sizeof(__fp16));
                    pack_B_tile_to_spm(B_pad.data(), NT, kc, KC, 0);
                }
#endif
                t_packB += us_since(_t);
                ++spmcp_calls;
            }

            // Phase 3: sweep all M-blocks; B pinned in SPM for all m0
            for (size_t m0 = 0; m0 < M; m0 += MC) {       // m0: A from A_cache
                size_t mc      = std::min(MC, M - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                const __fp16* Ap = A_cache.data() + (m0/MC)*max_ab*MT*KC;

                { auto _t = _now();
#ifdef MOCK_SPM
                  sve_interleaved_fp16_mla_8x3VL(
                      Ap, B_spm.data(), Cpanel.data(),
                      (int)ablocks, 1, (int)kc);
#else
                  spm_kernel_mblocks(Ap, Cpanel.data(),
                                     (int)kc, (int)ablocks, KC, 0);
#endif
                  t_kernel += us_since(_t); }

                { auto _t = _now();
                  scatter_C_to_output(C.data(), N, M, Cpanel.data(),
                                      mc, m0, n0, nc, k0 == 0);
                  if (k0 == 0) t_cwrite += us_since(_t);
                  else         t_crmw   += us_since(_t); }
            }
        }
    }

    ROI_END();
#ifdef GEM5
    m5_dump_stats(0, 0);
#endif

    print_checksum(C.data(), M, N);
    print_timing("v3_spm_n0outer",
                 t_packB, t_packA, t_kernel, t_cwrite, t_crmw,
                 spmcp_calls, packA_calls, K_tiles,
                 GFLOP, KC, MC, N/*NC=N: no NC loop*/, K, N, M);
    (void)N_ITER;
    return 0;
}
