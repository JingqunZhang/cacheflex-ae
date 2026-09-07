// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * spm_qk_kernel_vl16.hpp — SPM QK kernel for VL=16: 8×1VL (NT=128)
 *
 * Non-fused: writes to Cpanel (contiguous, stride = ldc_elements).
 * Used by flash attention QK GEMM at VL=16.
 *
 * SPM B layout: 4-way per K row (same as kernels_tile_spm.hpp)
 *   Way0+1+2+3 @ set k = 256B = 128 FP16 = 1VL
 *   spm.ld1qd: 1 load per K step (loads all 4 ways)
 *   KC_max = 1024
 *
 * Also includes 4-way SPMCP for K_T packing.
 */
#pragma once

// ── SPMCP: 4-way layout for 1VL (128 FP16 = 256B per K row) ──
static void pack_B_tile_to_spm_4way_vl16(const __fp16* B_row0, size_t N_stride,
                                          size_t kc, size_t /*KC*/,
                                          size_t spm_base_set = 0) {
    if (kc == 0) return;
    size_t stride_bytes = N_stride * sizeof(__fp16);
    __asm__ __volatile__(
        "mov x10, %x[base]\n"
        "mov x12, %x[src]\n"
        "1:\n"
        "lsl x16, x10, #6\n"
        "SPMCP_64_IMM x16, [x12, #0]\n"
        "orr x14, x16, #0x10000\n"
        "SPMCP_64_IMM x14, [x12, #64]\n"
        "orr x14, x16, #0x20000\n"
        "SPMCP_64_IMM x14, [x12, #128]\n"
        "orr x14, x16, #0x30000\n"
        "SPMCP_64_IMM x14, [x12, #192]\n"
        "add x10, x10, #1\n"
        "add x12, x12, %x[stride]\n"
        "subs %x[kc], %x[kc], #1\n"
        "bne 1b\n"
        : [kc] "+r" (kc)
        : [src] "r" (B_row0), [stride] "r" (stride_bytes),
          [base] "r" (spm_base_set)
        : "cc", "memory", "x10", "x12", "x14", "x16"
    );
}

// ── Non-fused SPM kernel: 8×1VL, writes to Cpanel ──
// Apanel: packed A [8 × K], interleaved (same as spm_kernel)
// Cpanel: output [8 × NT], row-major stride = ldc_elements
// spm.ld1qd at VL=16: loads all 4 ways = 128 FP16
static void spm_kernel_1vl_vl16(const __fp16* Apanel, __fp16* Cpanel,
                                 int K, int ldc_elements,
                                 size_t /*KC*/, size_t base_set = 0) {
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "mov x21, %x[Ap]\n"
        "mov x0,  %x[Cp]\n"
        "mov x3,  %x[ldc]\n"
        "mov x20, %x[K]\n"
        "mov x22, %x[base]\n"

        // Zero 8 accumulators
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"

        // K loop
        "1:\n"
        "lsl x16, x22, #6\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"      // 128 FP16 = 1VL
        "ld1rqh { z0.h }, p0/z, [x21]\n"      // A rows 0-7

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

        // Store 8 rows × 1VL to Cpanel
        "lsl x4, x3, #1\n"   // x4 = ldc * 2 bytes
        "mov x5, x0\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"

        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z15.h }, p0, [x12]\n"

        :
        : [Ap] "r" (Apanel), [Cp] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc] "r" ((size_t)ldc_elements),
          [base] "r" (base_set)
        : "cc", "memory",
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x16", "x20", "x21", "x22",
          "p0", "p1",
          "z0", "z2",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15"
    );
}
