// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * pv_kernel_8x64_vl16.hpp — PV GEMM micro-kernel for VL=16: 8 rows × 64 cols
 *
 * At VL=16: svcnth=128, but N=d=64. Uses predicate to mask first 64 elements.
 * 8 accumulators (z8-z15), 1 VL-wide load per K step with predicate.
 *
 * Cache version: V loaded via ld1h with p64 predicate
 * SPM version:   V loaded via spm.ld1qd (4 ways, only Way0+Way1 have data)
 */
#pragma once
#include <arm_sve.h>
#include <cstddef>
#include <cstring>

// ── Cache PV kernel: 8×64 at VL=16 ──
// Apanel: packed P [8 × Kc], k-major 8-way interleaved
// Bpanel: V [Kc × 64], row-major (stride = d = 64 FP16)
// Cpanel: output PV [8 × 64], row-major
static void cache_pv_8x64_vl16(
    const __fp16* Apanel,
    const __fp16* Bpanel,
    __fp16*       Cpanel,
    int           K,
    int           ldc_elements)  // = d = 64
{
    size_t ldc_bytes = (size_t)ldc_elements * 2;
    size_t V_stride = 64 * 2;  // V row stride = d * sizeof(fp16) = 128 bytes

    __asm__ __volatile__(
        // p0 = ptrue (full VL), p2 = first 64 elements only
        "ptrue p0.h\n"
        "mov x3, #64\n"
        "whilelt p2.h, xzr, x3\n"     // p2 = true for lanes 0..63

        "mov x21, %x[Ap]\n"
        "mov x22, %x[Vp]\n"
        "mov x0,  %x[Cp]\n"
        "mov x20, %x[K]\n"
        "mov x13, %x[ldc_bytes]\n"
        "mov x14, %x[V_stride]\n"

        // Zero 8 accumulators
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"

        // K loop
        "1:\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"       // P rows 0-7 at k
        "ld1h   { z2.h }, p2/z, [x22]\n"        // V[k, 0..63] (upper 64 = zero)

        "fmla z8.h,  z2.h, z0.h[0]\n"
        "fmla z9.h,  z2.h, z0.h[1]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"
        "fmla z11.h, z2.h, z0.h[3]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"
        "fmla z13.h, z2.h, z0.h[5]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"
        "fmla z15.h, z2.h, z0.h[7]\n"

        "add x21, x21, #16\n"          // A advance 16B (8 FP16 per k)
        "add x22, x22, x14\n"          // V advance by row stride
        "subs x20, x20, #1\n"
        "bne 1b\n"

        // Store 8 rows × 64 cols (predicated)
        "mov x5, x0\n"
        "st1h { z8.h },  p2, [x5]\n"   "add x5, x5, x13\n"
        "st1h { z9.h },  p2, [x5]\n"   "add x5, x5, x13\n"
        "st1h { z10.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z11.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z12.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z13.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z14.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z15.h }, p2, [x5]\n"

        :
        : [Ap] "r" (Apanel), [Vp] "r" (Bpanel), [Cp] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc_bytes] "r" (ldc_bytes),
          [V_stride] "r" (V_stride)
        : "cc", "memory", "p0", "p2",
          "x0", "x3", "x5", "x13", "x14", "x20", "x21", "x22",
          "z0", "z2",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15"
    );
}

// ── SPM PV kernel: 8×64 at VL=16 ──
// V in SPM: 2 ways per K row (Way0: V[k,0..31], Way1: V[k,32..63])
// spm.ld1qd at VL=16 loads all 4 ways (256B = 128 FP16)
// Only Way0+Way1 have valid data (64 FP16), Way2+Way3 are garbage
// Use predicate p2 for store (only first 64 cols)
// FMLA on full register is fine — upper garbage doesn't affect stored output
#ifndef MOCK_SPM
static void spm_pv_8x64_vl16(
    const __fp16* Apanel,
    __fp16*       Cpanel,
    int           K,
    int           ldc_elements,
    size_t        spm_base_v)
{
    size_t ldc_bytes = (size_t)ldc_elements * 2;

    __asm__ __volatile__(
        "ptrue p0.h\n"
        // p1: predicate for spm.ld1qd — only 16 doubleword lanes (2 ways = 128B = 64 FP16)
        "mov x3, #16\n"
        "whilelt p1.d, xzr, x3\n"
        // p2: predicate for store — first 64 FP16 elements
        "mov x3, #64\n"
        "whilelt p2.h, xzr, x3\n"

        "mov x21, %x[Ap]\n"
        "mov x0,  %x[Cp]\n"
        "mov x20, %x[K]\n"
        "mov x13, %x[ldc_bytes]\n"
        "mov x22, %x[spm_base]\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"

        "1:\n"
        "lsl x16, x22, #6\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"      // loads 2 ways only (128B = 64 FP16)
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z2.h, z0.h[0]\n"
        "fmla z9.h,  z2.h, z0.h[1]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"
        "fmla z11.h, z2.h, z0.h[3]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"
        "fmla z13.h, z2.h, z0.h[5]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"
        "fmla z15.h, z2.h, z0.h[7]\n"

        "add x21, x21, #16\n"
        "add x22, x22, #1\n"
        "subs x20, x20, #1\n"
        "bne 1b\n"

        // Store predicated
        "mov x5, x0\n"
        "st1h { z8.h },  p2, [x5]\n"   "add x5, x5, x13\n"
        "st1h { z9.h },  p2, [x5]\n"   "add x5, x5, x13\n"
        "st1h { z10.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z11.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z12.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z13.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z14.h }, p2, [x5]\n"    "add x5, x5, x13\n"
        "st1h { z15.h }, p2, [x5]\n"

        :
        : [Ap] "r" (Apanel), [Cp] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc_bytes] "r" (ldc_bytes),
          [spm_base] "r" (spm_base_v)
        : "cc", "memory", "p0", "p1", "p2",
          "x0", "x3", "x5", "x13", "x16", "x20", "x21", "x22",
          "z0", "z2",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15"
    );
}

// ── SPMCP V to SPM for PV: 2 ways per K row (64B + 64B = 64 FP16) ──
static void spmcp_V_2way(const __fp16* V_row, size_t V_stride_bytes,
                          size_t kc, size_t spm_base) {
    __asm__ __volatile__(
        "mov x10, %x[base]\n"
        "mov x12, %x[src]\n"
        "1:\n"
        "lsl x16, x10, #6\n"
        "SPMCP_64_IMM x16, [x12, #0]\n"           // Way0: V[k, 0..31]
        "orr x14, x16, #0x10000\n"
        "SPMCP_64_IMM x14, [x12, #64]\n"          // Way1: V[k, 32..63]
        "add x10, x10, #1\n"
        "add x12, x12, %x[stride]\n"
        "subs %x[kc], %x[kc], #1\n"
        "bne 1b\n"
        : [kc] "+r" (kc)
        : [src] "r" (V_row), [stride] "r" (V_stride_bytes),
          [base] "r" (spm_base)
        : "cc", "memory", "x10", "x12", "x14", "x16"
    );
}
#endif // !MOCK_SPM
