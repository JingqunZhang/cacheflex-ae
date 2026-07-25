// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

#pragma once
static void sve_spm_pv_8x64_vl4(
    const __fp16* Apanel,      // packed P [8 × Kc], k-major 8-way interleaved
    __fp16*       Cpanel,      // output PV [8 × 64], row-major
    int           K,           // Kc = number of KV positions (inner dim)
    int           ldc_elements,// output row stride in elements (= d = 64)
    size_t        spm_base_v)  // first SPM row for V[j_start]
{
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "mov x21, %x[Apanel]\n"
        "mov x0,  %x[Cpanel]\n"
        "mov x3,  %x[ldc]\n"
        "mov x20, %x[K]\n"
        "sub x20, x20, #1\n"           // K-1 for 2-unroll
        "mov x22, %x[spm_base]\n"      // current V SPM row

        // ── Zero 16 accumulators ─────────────────────────────────────────
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"

        // ── Pre-load V[k=0] Way0/Way1 and P[k=0] ────────────────────────
        "lsl x16, x22, #6\n"
        "orr x14, x16, #0x10000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"   // V[spm_base, 0..31]  Way0
        "spm.ld1qd z3.d, p1/z, [x14]\n"   // V[spm_base, 32..63] Way1
        "ld1rqh { z0.h }, p0/z, [x21]\n"  // P[8 rows, k=0]
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // ── Main K-loop (2-step unrolled) ────────────────────────────────
        "3:\n"
        // Step k: FMLA Way0 rows
        "fmla z8.h,  z2.h, z0.h[0]\n"
        "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"  // P[k+1]
        "fmla z10.h, z2.h, z0.h[2]\n"
        "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #6\n"
        "orr x14, x16, #0x10000\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"         // V[k+1] Way0
        "fmla z12.h, z2.h, z0.h[4]\n"
        "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x14]\n"         // V[k+1] Way1
        "fmla z14.h, z2.h, z0.h[6]\n"
        "fmla z15.h, z2.h, z0.h[7]\n"

        // Step k: FMLA Way1 rows
        "fmla z16.h, z3.h, z0.h[0]\n"
        "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"
        "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"
        "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"               // P advances 2 steps × 16B
        "fmla z22.h, z3.h, z0.h[6]\n"
        "fmla z23.h, z3.h, z0.h[7]\n"

        // Step k+1: FMLA Way0 rows (z6), Way1 rows (z5)
        "fmla z8.h,  z6.h, z7.h[0]\n"
        "fmla z9.h,  z6.h, z7.h[1]\n"
        "lsl x16, x22, #6\n"
        "fmla z10.h, z6.h, z7.h[2]\n"
        "fmla z11.h, z6.h, z7.h[3]\n"
        "orr x14, x16, #0x10000\n"
        "fmla z12.h, z6.h, z7.h[4]\n"
        "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"
        "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"
        "fmla z17.h, z5.h, z7.h[1]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"         // V[k+2] Way0 (prefetch next)
        "fmla z18.h, z5.h, z7.h[2]\n"
        "fmla z19.h, z5.h, z7.h[3]\n"
        "spm.ld1qd z3.d, p1/z, [x14]\n"         // V[k+2] Way1
        "fmla z20.h, z5.h, z7.h[4]\n"
        "fmla z21.h, z5.h, z7.h[5]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"        // P[k+2]
        "fmla z22.h, z5.h, z7.h[6]\n"
        "fmla z23.h, z5.h, z7.h[7]\n"
        "bge 3b\n"

        // ── Cleanup: last 1 or 2 K-steps ─────────────────────────────────
        "4:\n"
        "add x21, x21, #0x10\n"
        "add x22, x22, #1\n"

        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        // Extra k+1 if K was odd
        "cbz x20, 5f\n"
        "ld1rqh { z1.h }, p0/z, [x21]\n"
        "lsl x16, x22, #6\n"
        "orr x14, x16, #0x10000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x14]\n"

        "fmla z8.h,  z2.h, z1.h[0]\n"  "fmla z9.h,  z2.h, z1.h[1]\n"
        "fmla z10.h, z2.h, z1.h[2]\n"  "fmla z11.h, z2.h, z1.h[3]\n"
        "fmla z12.h, z2.h, z1.h[4]\n"  "fmla z13.h, z2.h, z1.h[5]\n"
        "fmla z14.h, z2.h, z1.h[6]\n"  "fmla z15.h, z2.h, z1.h[7]\n"
        "fmla z16.h, z3.h, z1.h[0]\n"  "fmla z17.h, z3.h, z1.h[1]\n"
        "fmla z18.h, z3.h, z1.h[2]\n"  "fmla z19.h, z3.h, z1.h[3]\n"
        "fmla z20.h, z3.h, z1.h[4]\n"  "fmla z21.h, z3.h, z1.h[5]\n"
        "fmla z22.h, z3.h, z1.h[6]\n"  "fmla z23.h, z3.h, z1.h[7]\n"

        // ── Store 8 rows × 64 cols (2 st1h per row) ──────────────────────
        "5:\n"
        "lsl x4, x3, #1\n"           // x4 = ldc*2 bytes = d*2 = 128B (row stride)
        "mov x5, x0\n"               // row 0
        "st1h { z8.h },  p0, [x5]\n"              "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 1
        "st1h { z9.h },  p0, [x5]\n"              "st1h { z17.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 2
        "st1h { z10.h }, p0, [x5]\n"              "st1h { z18.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 3
        "st1h { z11.h }, p0, [x5]\n"              "st1h { z19.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 4
        "st1h { z12.h }, p0, [x5]\n"              "st1h { z20.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 5
        "st1h { z13.h }, p0, [x5]\n"              "st1h { z21.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 6
        "st1h { z14.h }, p0, [x5]\n"              "st1h { z22.h }, p0, [x5, #1, MUL VL]\n"
        "add x5, x5, x4\n"           // row 7
        "st1h { z15.h }, p0, [x5]\n"              "st1h { z23.h }, p0, [x5, #1, MUL VL]\n"

        :
        : [Apanel]   "r" (Apanel),
          [Cpanel]   "r" (Cpanel),
          [K]        "r" ((size_t)K),
          [ldc]      "r" ((size_t)ldc_elements),
          [spm_base] "r" (spm_base_v)
        : "cc", "memory",
          "x0", "x3", "x4", "x5", "x14", "x16", "x20", "x21", "x22",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23"
    );
}
