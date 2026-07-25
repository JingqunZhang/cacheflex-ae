/* GEMM checksum and timing output utilities. */
#pragma once
#include <cstdio>
#include <cstdint>
#include <arm_sve.h>

static inline double logical_weight(size_t index)
{
    const uint64_t mixed =
        (static_cast<uint64_t>(index) + 1ULL) * 11400714819323198485ULL;
    return 1.0 + static_cast<double>(mixed >> 48) / 65535.0;
}

static void print_checksum_logical(const __fp16* C, size_t M, size_t N,
                                   size_t ldc)
{
    double sum = 0.0;
    double l1 = 0.0;
    double weighted = 0.0;
    for (size_t row = 0; row < M; ++row)
        for (size_t col = 0; col < N; ++col) {
            const double value = (double)(float)C[row * ldc + col];
            const double magnitude = value < 0.0 ? -value : value;
            sum += value;
            l1 += magnitude;
            weighted += magnitude * logical_weight(row * N + col);
        }
    printf("  CHECKSUM: %.12g\n", sum);
    printf("  CHECKSUM_LOGICAL: %.12g\n", sum);
    printf("  CHECKSUM_LOGICAL_L1: %.12g\n", l1);
    printf("  CHECKSUM_LOGICAL_WEIGHTED_L1: %.12g\n", weighted);
    fflush(stdout);
}

static void print_timing(const char* label,
                         double t_packB, double t_packA, double t_kernel,
                         size_t packB_calls, size_t packA_calls, size_t K_tiles,
                         double GFLOP,
                         size_t KC, size_t MC, size_t K, size_t N, size_t M)
{
    double t_total = t_packB + t_packA + t_kernel;
    double gflops  = GFLOP / (t_total * 1e-6);
    printf("\n=== %s ===\n", label);
    printf("  M=%zu K=%zu N=%zu  KC=%zu MC=%zu  NT=%zu\n",
           M, K, N, KC, MC, 3*svcnth());
    printf("  pack_B/SPMCP :  %8.1f us  (%5.1f%%)  [%zu calls]\n",
           t_packB,  100.*t_packB /t_total, packB_calls);
    printf("  pack_A       :  %8.1f us  (%5.1f%%)  [%zu calls]\n",
           t_packA,  100.*t_packA /t_total, packA_calls);
    printf("  kernel+C_rw  :  %8.1f us  (%5.1f%%)  (fused scatter)\n",
           t_kernel, 100.*t_kernel/t_total);
    printf("  total        :  %8.1f us  →  %.2f GFLOPS\n", t_total, gflops);
    fflush(stdout);
}
