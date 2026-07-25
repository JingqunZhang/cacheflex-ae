/*
 * loop_compare_common.hpp — shared helpers for v1/v2/v3/v4 GEMM loop order comparison
 *
 * Included by all 4 versions AFTER their respective kernel header.
 * Provides: scatter_C_to_output(), print_timing()
 */
#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Scatter Cpanel[ablocks × MT × NT] → C[M × N], rows m0..m0+mc, cols n0..n0+nc
// first_k=true  → pure store  (k0==0, C_write)
// first_k=false → load+add+store (k0>0,  C_RMW)
// ---------------------------------------------------------------------------
static void scatter_C_to_output(
    __fp16* C, size_t N, size_t M,
    const __fp16* Cpanel, size_t mc,
    size_t m0, size_t n0, size_t nc,
    bool first_k)
{
    const size_t MT = 8;
    const size_t NT = 3 * svcnth();
    size_t ablocks = (mc + MT - 1) / MT;
    svbool_t pg = svptrue_b16();
    for (size_t mb = 0; mb < ablocks; ++mb) {
        const __fp16* tp = Cpanel + mb * MT * NT;
        for (size_t r = 0; r < MT; ++r) {
            size_t gr = m0 + mb * MT + r;
            if (gr >= M) break;
            __fp16*       c_row = C + gr * N + n0;
            const __fp16* t_row = tp + r * NT;
            size_t i = 0;
            if (first_k) {
                for (; i + svcnth() <= nc; i += svcnth())
                    svst1_f16(pg, c_row+i, svld1_f16(pg, t_row+i));
                if (i < nc) { svbool_t pt = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pt, c_row+i, svld1_f16(pt, t_row+i)); }
            } else {
                for (; i + svcnth() <= nc; i += svcnth())
                    svst1_f16(pg, c_row+i, svadd_f16_x(pg,
                        svld1_f16(pg, c_row+i), svld1_f16(pg, t_row+i)));
                if (i < nc) { svbool_t pt = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pt, c_row+i, svadd_f16_x(pt,
                        svld1_f16(pt, c_row+i), svld1_f16(pt, t_row+i))); }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Checksum: sum all C elements as float64 — used for cross-version verification
// ---------------------------------------------------------------------------
static void print_checksum(const __fp16* C, size_t M, size_t N)
{
    double sum = 0.0;
    double l1 = 0.0;
    double weighted = 0.0;
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            const double value = (double)(float)C[i * N + j];
            const double magnitude = value < 0.0 ? -value : value;
            sum += value;
            l1 += magnitude;
            const uint64_t mixed =
                (static_cast<uint64_t>(i * N + j) + 1ULL)
                * 11400714819323198485ULL;
            const double weight =
                1.0 + static_cast<double>(mixed >> 48) / 65535.0;
            weighted += magnitude * weight;
        }
    printf("  CHECKSUM: %.12g\n", sum);
    printf("  CHECKSUM_LOGICAL_L1: %.12g\n", l1);
    printf("  CHECKSUM_LOGICAL_WEIGHTED_L1: %.12g\n", weighted);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Timing result printer (consistent format across all 4 versions)
// ---------------------------------------------------------------------------
static void print_timing(const char* label,
                         double t_packB, double t_packA,
                         double t_kernel, double t_cwrite, double t_crmw,
                         size_t spmcp_calls, size_t packA_calls, size_t K_tiles,
                         double GFLOP,
                         size_t KC, size_t MC, size_t NC,
                         size_t K, size_t N, size_t M)
{
    double t_total = t_packB + t_packA + t_kernel + t_cwrite + t_crmw;
    double gflops  = GFLOP / (t_total * 1e-6);
    printf("\n=== %s ===\n", label);
    printf("  M=%zu K=%zu N=%zu  KC=%zu MC=%zu NC=%zu  NT=%zu\n",
           M, K, N, KC, MC, NC, 3*svcnth());
    printf("  pack_B/SPMCP :  %8.1f us  (%5.1f%%)  [%zu calls]\n",
           t_packB,  100.*t_packB /t_total, spmcp_calls);
    printf("  pack_A       :  %8.1f us  (%5.1f%%)  [%zu calls, %.0fkB/call]\n",
           t_packA,  100.*t_packA /t_total, packA_calls, MC*KC*2.0/1024);
    printf("  kernel       :  %8.1f us  (%5.1f%%)\n",
           t_kernel, 100.*t_kernel/t_total);
    printf("  C_write(k0=0):  %8.1f us  (%5.1f%%)  [1 pass]\n",
           t_cwrite, 100.*t_cwrite/t_total);
    printf("  C_RMW (k0>0) :  %8.1f us  (%5.1f%%)  [%zu passes]\n",
           t_crmw,   100.*t_crmw  /t_total, K_tiles > 0 ? K_tiles - 1 : 0);
    printf("  total        :  %8.1f us  →  %.2f GFLOPS\n", t_total, gflops);
    fflush(stdout);
}
