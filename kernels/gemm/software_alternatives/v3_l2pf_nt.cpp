// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * L2-prefetch and non-temporal-load GEMM variant.
 *
 * B is packed by N tile into a cacheable buffer and reused across M blocks.
 * The selected microkernel controls the load and prefetch instructions.
 *
 * Usage: ./v3_l2pf_nt [M] [K] [N] [n_iter] [KC] [MC]
 */
// Select microkernel variant:
//   default:       ldnt1h for B and SVE svprfh.
//   -DL2PF_ONLY:   ld1h for B and scalar prfm pldl2keep.
#ifdef COMBO
// COMBO_NT, PF_L2B_DIST, PF_L1B_DIST, and PF_L1A_DIST are build-time controls.
#  include "kernels_sve_combo.hpp"
#  define MICROKERNEL sve_interleaved_fp16_mla_8x3VL_combo
#  define VARIANT_TAG "COMBO"
#  define PF_PARAM PF_L2B_DIST
#elif defined(L1SWPF)
// ld1h/ld1rqh with optional pldl1keep prefetches for A and B.
#  include "kernels_sve_l1swpf.hpp"
#  define MICROKERNEL sve_interleaved_fp16_mla_8x3VL_l1swpf
#  define VARIANT_TAG "L1SWPF (ld1h + hand-tuned prfm pldl1keep A/B)"
#  ifndef PF_DIST
#    define PF_DIST 512
#  endif
#  define PF_PARAM PF_DIST
#elif defined(L1PIN)
// ld1h for B with scalar pldl1keep; CF_L1_PIN_KEEP enables modeled L1 pinning.
#  include "kernels_sve_l1pin.hpp"
#  define MICROKERNEL sve_interleaved_fp16_mla_8x3VL_l1pin
#  define VARIANT_TAG "L1PIN (ld1h + scalar prfm pldl1keep)"
#  ifndef PF_OFF_BYTES
#    define PF_OFF_BYTES 512
#  endif
#  define PF_PARAM PF_OFF_BYTES
#elif defined(L2PIN_NT)
// ldnt1h for B with scalar pldl2keep; NO_L1_ALLOC models the NT allocation policy.
#  include "kernels_sve_l2pin_nt.hpp"
#  define MICROKERNEL sve_interleaved_fp16_mla_8x3VL_l2pin_nt
#  define VARIANT_TAG "L2PIN+NT (ldnt1h + scalar prfm pldl2keep)"
#  ifndef PF_OFF_BYTES
#    define PF_OFF_BYTES 512
#  endif
#  define PF_PARAM PF_OFF_BYTES
#elif defined(L2PF_ONLY)
#  include "kernels_sve_l2pf.hpp"
#  define MICROKERNEL sve_interleaved_fp16_mla_8x3VL_l2pf
#  define VARIANT_TAG "L2PF-only (ld1h + scalar prfm pldl2keep)"
#  ifndef PF_OFF_BYTES
#    define PF_OFF_BYTES 512
#  endif
#  define PF_PARAM PF_OFF_BYTES
#else
#  warning "No variant macro (-DL2PIN_NT/-DL2PF_ONLY/-DL1SWPF/-DL1PIN/-DCOMBO): defaulting to ntpf, whose SVE prfh decodes as a NOP under this gem5 fork. Fine for real HW; NOT a modeled-prefetch config in simulation."
#  include "kernels_sve_ntpf.hpp"
#  define MICROKERNEL sve_interleaved_fp16_mla_8x3VL_ntpf
#  define VARIANT_TAG "NT+PF (ldnt1h + svprfh)"
#  define PF_PARAM PF_DIST_VL
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

    printf("=== v3_l2pf_nt [%s, PF=%d] ===  M=%zu K=%zu N=%zu  NT=%zu\n",
           VARIANT_TAG, PF_PARAM, M, K, N, NT);
    printf("  KC=%zu  MC=%zu  (no NC, B-pack per n0 into cacheable buffer)\n", KC, MC);
    printf("  A_cache=%.0fkB(L2-warm)  B_tile=%.0fkB  C[M×N]=%.1fMB\n",
           M*KC*2.0/1024, KC*NT*2.0/1024, M*N*2.0/1024/1024);
    printf("  packA calls=%zu (K_tiles=%zu × M_tiles=%zu)\n",
           K_tiles*M_tiles, K_tiles, M_tiles);
    printf("  packB calls=%zu (K_tiles=%zu × N_tiles=%zu)\n",
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
    // Cacheable KC×NT B-tile buffer.
    AlignedBuffer<__fp16> B_spm(KC * NT);
    std::memset(B_spm.data(), 0, KC * NT * sizeof(__fp16));
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

        // Phase 2: sweep N; pack B once per n0, kernel sweeps all m0
        for (size_t n0 = 0; n0 < N; n0 += NT) {           // n0: pack B → buffer
            size_t nc = std::min(NT, N - n0);

            // pack one NT-tile of B into the cacheable B_spm buffer (slot 0)
            {
                auto _t = _now();
                if (nc < NT || kc < KC)
                    std::memset(B_spm.data(), 0, kc * NT * sizeof(__fp16));
                sve_pack_B_3VL((uint16_t*)B_spm.data(),
                               (const uint16_t*)(B.data() + k0 * N + n0),
                               nc, N * sizeof(uint16_t), kc);
#ifdef CVAC_B
                // Clean the packed B tile before NT reads; charge this work to t_packB.
                {
                    const char *p0 = (const char *)B_spm.data();
                    const char *p1 = p0 + kc * NT * sizeof(__fp16);
                    for (const char *p = p0; p < p1; p += 64)
                        asm volatile("dc cvac, %0" :: "r"(p) : "memory");
                    asm volatile("dsb sy" ::: "memory");
                }
#endif
                t_packB += us_since(_t);
                ++spmcp_calls;
            }

            // Phase 3: sweep all M-blocks; B read via NT-load + L2 prefetch
            for (size_t m0 = 0; m0 < M; m0 += MC) {       // m0: A from A_cache
                size_t mc      = std::min(MC, M - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                const __fp16* Ap = A_cache.data() + (m0/MC)*max_ab*MT*KC;

                { auto _t = _now();
                  MICROKERNEL(
                      Ap, B_spm.data(), Cpanel.data(),
                      (int)ablocks, 1, (int)kc);
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
    print_timing("v3_l2pf_nt",
                 t_packB, t_packA, t_kernel, t_cwrite, t_crmw,
                 spmcp_calls, packA_calls, K_tiles,
                 GFLOP, KC, MC, N/*NC=N: no NC loop*/, K, N, M);
    (void)N_ITER;
    return 0;
}
