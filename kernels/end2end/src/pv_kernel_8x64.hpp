// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * pv_kernel_8x64.hpp — Cache PV GEMM micro-kernel: 8 rows × 64 cols (VL=4, d=64)
 *
 * Computes: PV[8, 64] = P[8, Kc] × V[Kc, 64]
 * V is row-major [Kc, 64], packed as [Kc × 64 FP16] contiguous.
 * Each V row: 64 FP16 = 128 bytes = 2 SVE vectors at VL=4.
 *
 * Uses 16 accumulators (vs 24 for 8×96):
 *   z8-z15  : 8 rows × first 32 cols
 *   z16-z23 : 8 rows × last 32 cols
 *
 * Eliminates 33% compute waste of 8×96 kernel on d=64.
 * Output stored row-major with stride ldc_elements.
 */
#pragma once
#include <arm_sve.h>
#include <cstddef>
#include <cstring>

static void sve_cache_pv_8x64(
    const __fp16* Apanel,       // packed P [8 × Kc], k-major 8-way interleaved
    const __fp16* Bpanel,       // V [Kc × 64], row-major (stride=64)
    __fp16*       Cpanel,       // output PV [8 × 64], row-major
    int           K,            // Kc
    int           ldc_elements) // output row stride (= d = 64)
{
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "mov x21, %x[Apanel]\n"
        "mov x22, %x[Bpanel]\n"
        "mov x20, %x[K]\n"
        "sub x20, x20, #1\n"

        // Zero 16 accumulators
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"

        // Pre-load V[k=0] two halves and P[k=0]
        "ld1h { z2.h }, p0/z, [x22]\n"           // V[0, 0..31]
        "ld1h { z3.h }, p0/z, [x22, #1, MUL VL]\n"  // V[0, 32..63]
        "ld1rqh { z0.h }, p0/z, [x21]\n"         // P[8 rows, k=0]
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // ── Main K-loop (2-step unrolled) ──
        "3:\n"
        // Step k: FMLA first half (cols 0..31)
        "fmla z8.h,  z2.h, z0.h[0]\n"
        "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"    // P[k+1]
        "fmla z10.h, z2.h, z0.h[2]\n"
        "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #128\n"                    // advance V by 1 row = 64×2B = 128B
        "fmla z12.h, z2.h, z0.h[4]\n"
        "fmla z13.h, z2.h, z0.h[5]\n"
        "ld1h { z6.h }, p0/z, [x22]\n"           // V[k+1, 0..31]
        "fmla z14.h, z2.h, z0.h[6]\n"
        "fmla z15.h, z2.h, z0.h[7]\n"
        "ld1h { z5.h }, p0/z, [x22, #1, MUL VL]\n"  // V[k+1, 32..63]

        // Step k: FMLA second half (cols 32..63)
        "fmla z16.h, z3.h, z0.h[0]\n"
        "fmla z17.h, z3.h, z0.h[1]\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"
        "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"
        "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"                   // advance A by 2 k-steps
        "fmla z22.h, z3.h, z0.h[6]\n"
        "fmla z23.h, z3.h, z0.h[7]\n"

        // Step k+1: FMLA first half
        "fmla z8.h,  z6.h, z7.h[0]\n"
        "fmla z9.h,  z6.h, z7.h[1]\n"
        "add x22, x22, #128\n"
        "fmla z10.h, z6.h, z7.h[2]\n"
        "fmla z11.h, z6.h, z7.h[3]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"
        "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"
        "fmla z15.h, z6.h, z7.h[7]\n"

        // Step k+1: FMLA second half
        "fmla z16.h, z5.h, z7.h[0]\n"
        "fmla z17.h, z5.h, z7.h[1]\n"
        "ld1h { z2.h }, p0/z, [x22]\n"           // V[k+2, 0..31] prefetch
        "fmla z18.h, z5.h, z7.h[2]\n"
        "fmla z19.h, z5.h, z7.h[3]\n"
        "ld1h { z3.h }, p0/z, [x22, #1, MUL VL]\n"  // V[k+2, 32..63]
        "fmla z20.h, z5.h, z7.h[4]\n"
        "fmla z21.h, z5.h, z7.h[5]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"         // P[k+2]
        "fmla z22.h, z5.h, z7.h[6]\n"
        "fmla z23.h, z5.h, z7.h[7]\n"
        "bge 3b\n"

        // ── Cleanup ──
        "4:\n"
        "add x21, x21, #0x10\n"
        "add x22, x22, #128\n"

        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "cbz x20, 5f\n"
        "ld1rqh { z1.h }, p0/z, [x21]\n"
        "ld1h { z2.h }, p0/z, [x22]\n"
        "ld1h { z3.h }, p0/z, [x22, #1, MUL VL]\n"

        "fmla z8.h,  z2.h, z1.h[0]\n"  "fmla z9.h,  z2.h, z1.h[1]\n"
        "fmla z10.h, z2.h, z1.h[2]\n"  "fmla z11.h, z2.h, z1.h[3]\n"
        "fmla z12.h, z2.h, z1.h[4]\n"  "fmla z13.h, z2.h, z1.h[5]\n"
        "fmla z14.h, z2.h, z1.h[6]\n"  "fmla z15.h, z2.h, z1.h[7]\n"
        "fmla z16.h, z3.h, z1.h[0]\n"  "fmla z17.h, z3.h, z1.h[1]\n"
        "fmla z18.h, z3.h, z1.h[2]\n"  "fmla z19.h, z3.h, z1.h[3]\n"
        "fmla z20.h, z3.h, z1.h[4]\n"  "fmla z21.h, z3.h, z1.h[5]\n"
        "fmla z22.h, z3.h, z1.h[6]\n"  "fmla z23.h, z3.h, z1.h[7]\n"

        // ── Store 8×64 with ldc stride ──
        "5:\n"
        "lsl x4, %x[ldc], #1\n"       // x4 = ldc * 2 bytes
        "mov x5, %x[Cpanel]\n"
        "st1h { z8.h },  p0, [x5]\n"              "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z9.h },  p0, [x5]\n"              "st1h { z17.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z10.h }, p0, [x5]\n"              "st1h { z18.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z11.h }, p0, [x5]\n"              "st1h { z19.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z12.h }, p0, [x5]\n"              "st1h { z20.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z13.h }, p0, [x5]\n"              "st1h { z21.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z14.h }, p0, [x5]\n"              "st1h { z22.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"
        "st1h { z15.h }, p0, [x5]\n"              "st1h { z23.h }, p0, [x5, #1, MUL VL]\n"

        : [Apanel] "+&r" (Apanel)
        : [Bpanel] "r" (Bpanel), [Cpanel] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc] "r" ((size_t)ldc_elements)
        : "cc", "memory",
          "x4", "x5", "x14", "x16", "x20", "x21", "x22",
          "p0",
          "z0", "z1", "z2", "z3", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23"
    );
}

// Pack V[Kc, d=64] row-major into B panel format for 8×64 kernel
// V is already row-major [Kc, 64], just need contiguous copy
// (Unlike 8×96 kernel which needs KNM packing with NT=96 padding)
static void pack_V_64(const __fp16* V, __fp16* V_packed, size_t Kc, size_t ldb) {
    for (size_t k = 0; k < Kc; ++k)
        std::memcpy(V_packed + k * 64, V + k * ldb, 64 * sizeof(__fp16));
}
