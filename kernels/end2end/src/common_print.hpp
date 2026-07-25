/*
 * common_print.hpp — checksum + timing printer for new_gemm
 */
#pragma once
#include <cstdio>
#include <arm_sve.h>

static void print_checksum(const __fp16* C, size_t M, size_t N)
{
    double sum = 0.0;
    for (size_t i = 0; i < M * N; ++i)
        sum += (double)(float)C[i];
    printf("  CHECKSUM: %.12g\n", sum);
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
