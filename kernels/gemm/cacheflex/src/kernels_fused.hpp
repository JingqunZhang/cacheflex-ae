// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * kernels_fused.hpp — Fused GEMM micro-kernels that write/RMW directly to C
 *
 * Eliminates the Cpanel intermediate buffer:
 *   Old: kernel → Cpanel → scatter → C   (3 mem ops/element)
 *   New: kernel → C directly              (1-2 mem ops/element)
 *
 * Two epilogue modes:
 *   first_k=true  (k0==0): pure store to C (stride ldc)
 *   first_k=false (k0>0):  load C + add accumulator + store C (stride ldc)
 *
 * Micro-kernel tile: 8 rows × 3VL cols (same as sve_interleaved_fp16_mla_8x3VL)
 * M-block loop (ablocks) is inside the kernel.
 *
 * Cache variant:  B loaded via ld1h from cache (L1/L2)
 * SPM variant:    B loaded via spm.ld1qd from SPM
 */
#pragma once

// ============================================================
// Cache fused kernel: 8×3VL, B from cache, direct C write/RMW
// Used by V1 (m0-outer)
//
// Apanel:  packed A [ablocks × 8 × K], interleaved col-major
// Bpanel:  packed B [K × 3VL], contiguous
// C_out:   pointer to C[m0, n0]
// ldc:     C row stride in FP16 elements (= N)
// ablocks: number of 8-row M-blocks
// bblocks: must be 1 (single NT tile)
// K:       number of K steps
// first_k: true → pure store, false → load+add+store
// ============================================================
static void cache_gemm_fused_8x3VL(
    const __fp16 *Apanel, const __fp16 *Bpanel, __fp16 *C_out,
    int ablocks, int K, int ldc, int first_k)
{
    size_t ldc_bytes = (size_t)ldc * 2;
    size_t A_stride = (size_t)K * 16;  // K * 8rows * 2bytes

    __asm__ __volatile__(
        "ptrue p0.b\n"
        "mov x26, %x[Apanel]\n"        // Abase
        "mov x27, %x[Cout]\n"          // Cbase
        "mov x23, %x[ablks]\n"         // ablocks counter
        "mov x28, %x[K_val]\n"         // K constant
        "mov x25, %x[A_stride]\n"      // A_stride = K*16
        "mov x13, %x[ldc_bytes]\n"     // ldc in bytes

        // ======== M-block outer loop ========
        "2:\n"
        "mov x24, %x[Bpanel]\n"        // Bp = Bpanel (reset per M-block)
        "mov x21, x26\n"               // Ap = Abase
        "mov x0,  x27\n"               // Cp = Cbase
        "mov x20, x28\n"
        "sub x20, x20, #1\n"           // x20 = K-1

        // Zero 24 accumulators
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // Preload B[k=0] and A[k=0]
        "ld1h   { z2.h }, p0/Z, [x24]\n"
        "ld1h   { z3.h }, p0/Z, [x24, #1, MUL VL]\n"
        "ld1h   { z4.h }, p0/Z, [x24, #2, MUL VL]\n"
        "ld1rqh { z0.h }, p0/Z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // ---- K 2-unroll main loop ----
        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z11.h, z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/Z, [x21, #16]\n"
        "fmla z14.h, z2.h, z0.h[2]\n"  "fmla z17.h, z2.h, z0.h[3]\n"
        "ld1h { z6.h }, p0/Z, [x24, #3, MUL VL]\n"
        "fmla z20.h, z2.h, z0.h[4]\n"  "fmla z23.h, z2.h, z0.h[5]\n"
        "ld1h { z5.h }, p0/Z, [x24, #4, MUL VL]\n"
        "fmla z26.h, z2.h, z0.h[6]\n"  "fmla z29.h, z2.h, z0.h[7]\n"
        "ld1h { z1.h }, p0/Z, [x24, #5, MUL VL]\n"
        "fmla z9.h,  z3.h, z0.h[0]\n"  "fmla z12.h, z3.h, z0.h[1]\n"
        "addvl x24, x24, #6\n"
        "fmla z15.h, z3.h, z0.h[2]\n"  "fmla z18.h, z3.h, z0.h[3]\n"
        "sub x20, x20, #0x2\n"
        "fmla z21.h, z3.h, z0.h[4]\n"  "fmla z24.h, z3.h, z0.h[5]\n"
        "cmp x20, #0x2\n"
        "fmla z27.h, z3.h, z0.h[6]\n"  "fmla z30.h, z3.h, z0.h[7]\n"
        "add x21, x21, #0x20\n"
        "fmla z10.h, z4.h, z0.h[0]\n"  "fmla z13.h, z4.h, z0.h[1]\n"
        "ld1h { z2.h }, p0/Z, [x24]\n"
        "fmla z16.h, z4.h, z0.h[2]\n"  "fmla z19.h, z4.h, z0.h[3]\n"
        "ld1h { z3.h }, p0/Z, [x24, #1, MUL VL]\n"
        "fmla z22.h, z4.h, z0.h[4]\n"  "fmla z25.h, z4.h, z0.h[5]\n"
        "fmla z28.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/Z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z11.h, z6.h, z7.h[1]\n"
        "ld1h { z4.h }, p0/Z, [x24, #2, MUL VL]\n"
        "fmla z14.h, z6.h, z7.h[2]\n"  "fmla z17.h, z6.h, z7.h[3]\n"
        "fmla z20.h, z6.h, z7.h[4]\n"  "fmla z23.h, z6.h, z7.h[5]\n"
        "fmla z26.h, z6.h, z7.h[6]\n"  "fmla z29.h, z6.h, z7.h[7]\n"
        "fmla z9.h,  z5.h, z7.h[0]\n"  "fmla z12.h, z5.h, z7.h[1]\n"
        "fmla z15.h, z5.h, z7.h[2]\n"  "fmla z18.h, z5.h, z7.h[3]\n"
        "fmla z21.h, z5.h, z7.h[4]\n"  "fmla z24.h, z5.h, z7.h[5]\n"
        "fmla z27.h, z5.h, z7.h[6]\n"  "fmla z30.h, z5.h, z7.h[7]\n"
        "fmla z10.h, z1.h, z7.h[0]\n"  "fmla z13.h, z1.h, z7.h[1]\n"
        "fmla z16.h, z1.h, z7.h[2]\n"  "fmla z19.h, z1.h, z7.h[3]\n"
        "fmla z22.h, z1.h, z7.h[4]\n"  "fmla z25.h, z1.h, z7.h[5]\n"
        "fmla z28.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

        // ---- K cleanup: one more step ----
        "4:\n"
        "add x21, x21, #0x10\n"
        "addvl x24, x24, #3\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z11.h, z2.h, z0.h[1]\n"
        "fmla z14.h, z2.h, z0.h[2]\n"  "fmla z17.h, z2.h, z0.h[3]\n"
        "fmla z20.h, z2.h, z0.h[4]\n"  "fmla z23.h, z2.h, z0.h[5]\n"
        "fmla z26.h, z2.h, z0.h[6]\n"  "fmla z29.h, z2.h, z0.h[7]\n"
        "fmla z9.h,  z3.h, z0.h[0]\n"  "fmla z12.h, z3.h, z0.h[1]\n"
        "fmla z15.h, z3.h, z0.h[2]\n"  "fmla z18.h, z3.h, z0.h[3]\n"
        "fmla z21.h, z3.h, z0.h[4]\n"  "fmla z24.h, z3.h, z0.h[5]\n"
        "fmla z27.h, z3.h, z0.h[6]\n"  "fmla z30.h, z3.h, z0.h[7]\n"
        "fmla z10.h, z4.h, z0.h[0]\n"  "fmla z13.h, z4.h, z0.h[1]\n"
        "fmla z16.h, z4.h, z0.h[2]\n"  "fmla z19.h, z4.h, z0.h[3]\n"
        "fmla z22.h, z4.h, z0.h[4]\n"  "fmla z25.h, z4.h, z0.h[5]\n"
        "fmla z28.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "cbz x20, 5f\n"

        // Final K step (odd K)
        "ld1rqh { z3.h }, p0/Z, [x21]\n"
        "ld1h   { z2.h }, p0/Z, [x24]\n"
        "add x21, x21, #0x10\n"
        "ld1h   { z1.h }, p0/Z, [x24, #1, MUL VL]\n"
        "ld1h   { z0.h }, p0/Z, [x24, #2, MUL VL]\n"
        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z11.h, z2.h, z3.h[1]\n"
        "fmla z14.h, z2.h, z3.h[2]\n"  "fmla z17.h, z2.h, z3.h[3]\n"
        "fmla z20.h, z2.h, z3.h[4]\n"  "fmla z23.h, z2.h, z3.h[5]\n"
        "fmla z26.h, z2.h, z3.h[6]\n"  "fmla z29.h, z2.h, z3.h[7]\n"
        "fmla z9.h,  z1.h, z3.h[0]\n"  "fmla z12.h, z1.h, z3.h[1]\n"
        "fmla z15.h, z1.h, z3.h[2]\n"  "fmla z18.h, z1.h, z3.h[3]\n"
        "fmla z21.h, z1.h, z3.h[4]\n"  "fmla z24.h, z1.h, z3.h[5]\n"
        "fmla z27.h, z1.h, z3.h[6]\n"  "fmla z30.h, z1.h, z3.h[7]\n"
        "fmla z10.h, z0.h, z3.h[0]\n"  "fmla z13.h, z0.h, z3.h[1]\n"
        "fmla z16.h, z0.h, z3.h[2]\n"  "fmla z19.h, z0.h, z3.h[3]\n"
        "fmla z22.h, z0.h, z3.h[4]\n"  "fmla z25.h, z0.h, z3.h[5]\n"
        "fmla z28.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        // ---- Epilogue: fused store/RMW to C with stride ldc ----
        "5:\n"
        "mov x5, x0\n"
        "add x6,  x5,  x13\n"  "add x7,  x6,  x13\n"  "add x8,  x7,  x13\n"
        "add x9,  x8,  x13\n"  "add x10, x9,  x13\n"  "add x11, x10, x13\n"
        "add x12, x11, x13\n"

        "cbz %x[first_k], 7f\n"

        // ---- first_k: pure store ----
        "st1h { z8.h },  p0, [x5]\n"   "st1h { z9.h },  p0, [x5, #1, MUL VL]\n"  "st1h { z10.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x6]\n"   "st1h { z12.h }, p0, [x6, #1, MUL VL]\n"  "st1h { z13.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x7]\n"   "st1h { z15.h }, p0, [x7, #1, MUL VL]\n"  "st1h { z16.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z17.h }, p0, [x8]\n"   "st1h { z18.h }, p0, [x8, #1, MUL VL]\n"  "st1h { z19.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z20.h }, p0, [x9]\n"   "st1h { z21.h }, p0, [x9, #1, MUL VL]\n"  "st1h { z22.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z23.h }, p0, [x10]\n"  "st1h { z24.h }, p0, [x10, #1, MUL VL]\n" "st1h { z25.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z26.h }, p0, [x11]\n"  "st1h { z27.h }, p0, [x11, #1, MUL VL]\n" "st1h { z28.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z29.h }, p0, [x12]\n"  "st1h { z30.h }, p0, [x12, #1, MUL VL]\n" "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"
        "b 8f\n"

        // ---- RMW: load C + add + store ----
        "7:\n"
        // row 0
        "ld1h { z0.h }, p0/Z, [x5]\n"             "fadd z0.h, z8.h,  z0.h\n"  "st1h { z0.h }, p0, [x5]\n"
        "ld1h { z1.h }, p0/Z, [x5, #1, MUL VL]\n" "fadd z1.h, z9.h,  z1.h\n"  "st1h { z1.h }, p0, [x5, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x5, #2, MUL VL]\n" "fadd z2.h, z10.h, z2.h\n"  "st1h { z2.h }, p0, [x5, #2, MUL VL]\n"
        // row 1
        "ld1h { z0.h }, p0/Z, [x6]\n"             "fadd z0.h, z11.h, z0.h\n"  "st1h { z0.h }, p0, [x6]\n"
        "ld1h { z1.h }, p0/Z, [x6, #1, MUL VL]\n" "fadd z1.h, z12.h, z1.h\n"  "st1h { z1.h }, p0, [x6, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x6, #2, MUL VL]\n" "fadd z2.h, z13.h, z2.h\n"  "st1h { z2.h }, p0, [x6, #2, MUL VL]\n"
        // row 2
        "ld1h { z0.h }, p0/Z, [x7]\n"             "fadd z0.h, z14.h, z0.h\n"  "st1h { z0.h }, p0, [x7]\n"
        "ld1h { z1.h }, p0/Z, [x7, #1, MUL VL]\n" "fadd z1.h, z15.h, z1.h\n"  "st1h { z1.h }, p0, [x7, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x7, #2, MUL VL]\n" "fadd z2.h, z16.h, z2.h\n"  "st1h { z2.h }, p0, [x7, #2, MUL VL]\n"
        // row 3
        "ld1h { z0.h }, p0/Z, [x8]\n"             "fadd z0.h, z17.h, z0.h\n"  "st1h { z0.h }, p0, [x8]\n"
        "ld1h { z1.h }, p0/Z, [x8, #1, MUL VL]\n" "fadd z1.h, z18.h, z1.h\n"  "st1h { z1.h }, p0, [x8, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x8, #2, MUL VL]\n" "fadd z2.h, z19.h, z2.h\n"  "st1h { z2.h }, p0, [x8, #2, MUL VL]\n"
        // row 4
        "ld1h { z0.h }, p0/Z, [x9]\n"             "fadd z0.h, z20.h, z0.h\n"  "st1h { z0.h }, p0, [x9]\n"
        "ld1h { z1.h }, p0/Z, [x9, #1, MUL VL]\n" "fadd z1.h, z21.h, z1.h\n"  "st1h { z1.h }, p0, [x9, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x9, #2, MUL VL]\n" "fadd z2.h, z22.h, z2.h\n"  "st1h { z2.h }, p0, [x9, #2, MUL VL]\n"
        // row 5
        "ld1h { z0.h }, p0/Z, [x10]\n"            "fadd z0.h, z23.h, z0.h\n"  "st1h { z0.h }, p0, [x10]\n"
        "ld1h { z1.h }, p0/Z, [x10, #1, MUL VL]\n""fadd z1.h, z24.h, z1.h\n"  "st1h { z1.h }, p0, [x10, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x10, #2, MUL VL]\n""fadd z2.h, z25.h, z2.h\n"  "st1h { z2.h }, p0, [x10, #2, MUL VL]\n"
        // row 6
        "ld1h { z0.h }, p0/Z, [x11]\n"            "fadd z0.h, z26.h, z0.h\n"  "st1h { z0.h }, p0, [x11]\n"
        "ld1h { z1.h }, p0/Z, [x11, #1, MUL VL]\n""fadd z1.h, z27.h, z1.h\n"  "st1h { z1.h }, p0, [x11, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x11, #2, MUL VL]\n""fadd z2.h, z28.h, z2.h\n"  "st1h { z2.h }, p0, [x11, #2, MUL VL]\n"
        // row 7
        "ld1h { z0.h }, p0/Z, [x12]\n"            "fadd z0.h, z29.h, z0.h\n"  "st1h { z0.h }, p0, [x12]\n"
        "ld1h { z1.h }, p0/Z, [x12, #1, MUL VL]\n""fadd z1.h, z30.h, z1.h\n"  "st1h { z1.h }, p0, [x12, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x12, #2, MUL VL]\n""fadd z2.h, z31.h, z2.h\n"  "st1h { z2.h }, p0, [x12, #2, MUL VL]\n"

        // ---- Advance to next M-block ----
        "8:\n"
        "add x26, x26, x25\n"          // Abase += A_stride
        "add x27, x12, x13\n"          // Cbase = row7_ptr + ldc_bytes
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        :
        : [Apanel] "r" (Apanel), [Bpanel] "r" (Bpanel),
          [Cout] "r" (C_out), [ablks] "r" ((size_t)ablocks),
          [K_val] "r" ((size_t)K), [ldc_bytes] "r" (ldc_bytes),
          [A_stride] "r" (A_stride), [first_k] "r" ((size_t)first_k)
        : "cc", "memory", "p0",
          "x0", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
          "x20", "x21", "x23", "x24", "x25", "x26", "x27", "x28",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

// ============================================================
// SPM fused kernel: 8×3VL, B from SPM, direct C write/RMW
// Used by V3 (m0-outer with SPM)
//
// Same structure as cache variant but B loaded via spm.ld1qd.
// Multi-VL: compile with exactly one of -DVL_2, -DVL_4, -DVL_8, -DVL_16.
//
// SPM addressing per VL (matches kernels_spm.hpp layout):
//   VL=2:  3 ways, lsl#5. z2=Way0[k*32], z3=Way1[k*32|0x10000], z4=Way2[k*32|0x20000]
//   VL=4:  3 ways, lsl#6. z2=Way0[k*64], z3=Way1[k*64|0x10000], z4=Way2[k*64|0x20000]
//   VL=8:  4 ways, KC-offset. z2=Way01[k*64], z3=Way23[k*64|0x20000], z4=Way01[k*64+KC*64]
//   VL=16: 4 ways, 3×KC-stacking. z2=[k*64], z3=[k*64+KC*64], z4=[k*64+2*KC*64]
// ============================================================
#ifndef MOCK_SPM

// VL=8/16 need KC parameter for KC-offset addressing; VL=2/4 do not.
#if defined(VL_8) || defined(VL_16)
static void spm_gemm_fused_8x3VL(
    const __fp16 *Apanel, __fp16 *C_out,
    int ablocks, int K, int ldc, int first_k,
    size_t KC, size_t base_set)
{
    size_t KC64 = KC * 64;
#else
static void spm_gemm_fused_8x3VL(
    const __fp16 *Apanel, __fp16 *C_out,
    int ablocks, int K, int ldc, int first_k,
    size_t base_set)
{
#endif
    struct SpmFusedArgs {
        const __fp16* Apanel;
        __fp16* C_out;
        size_t ablocks;
        size_t K_val;
        size_t A_stride;
        size_t ldc_bytes;
        size_t first_k;
        size_t base_set;
#if defined(VL_8) || defined(VL_16)
        size_t KC64;
#endif
    } args;
    args.Apanel = Apanel;
    args.C_out = C_out;
    args.ablocks = (size_t)ablocks;
    args.K_val = (size_t)K;
    args.A_stride = (size_t)K * 16;
    args.ldc_bytes = (size_t)ldc * 2;
    args.first_k = (size_t)first_k;
    args.base_set = base_set;
#if defined(VL_8) || defined(VL_16)
    args.KC64 = KC64;
#endif

    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "ldr x26, [%x[args], #0]\n"    // Apanel
        "ldr x27, [%x[args], #8]\n"    // C_out
        "ldr x23, [%x[args], #16]\n"   // ablocks
        "ldr x28, [%x[args], #24]\n"   // K_val
        "ldr x25, [%x[args], #32]\n"   // A_stride
        "ldr x13, [%x[args], #40]\n"   // ldc_bytes
        "ldr x17, [%x[args], #48]\n"   // first_k
        "ldr x19, [%x[args], #56]\n"   // base_set
#if defined(VL_8) || defined(VL_16)
        "ldr x14, [%x[args], #64]\n"   // KC64
#endif

        // ======== M-block outer loop ========
        "2:\n"
        "mov x21, x26\n"               // Ap
        "mov x0,  x27\n"               // Cp
        "mov x22, x19\n"               // SPM row counter = base_set
        "mov x20, x28\n"
        "sub x20, x20, #1\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // ---- Preload B[k=0] from SPM (VL-dependent addressing) ----
#if defined(VL_2)
        "lsl x16, x22, #5\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_4)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_8)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x20000\n"
        "add x18, x16, x14\n"
#elif defined(VL_16)
        "lsl x16, x22, #6\n"
        "add x15, x16, x14\n"
        "add x18, x15, x14\n"
#endif
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // ---- K 2-unroll main loop ----
        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z11.h, z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z14.h, z2.h, z0.h[2]\n"  "fmla z17.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
#if defined(VL_2)
        "lsl x16, x22, #5\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_4)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_8)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x20000\n"
        "add x18, x16, x14\n"
#elif defined(VL_16)
        "lsl x16, x22, #6\n"
        "add x15, x16, x14\n"
        "add x18, x15, x14\n"
#endif
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z20.h, z2.h, z0.h[4]\n"  "fmla z23.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x15]\n"
        "fmla z26.h, z2.h, z0.h[6]\n"  "fmla z29.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x18]\n"

        "fmla z9.h,  z3.h, z0.h[0]\n"  "fmla z12.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z15.h, z3.h, z0.h[2]\n"  "fmla z18.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z21.h, z3.h, z0.h[4]\n"  "fmla z24.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z27.h, z3.h, z0.h[6]\n"  "fmla z30.h, z3.h, z0.h[7]\n"

        "fmla z10.h, z4.h, z0.h[0]\n"  "fmla z13.h, z4.h, z0.h[1]\n"
#if defined(VL_2)
        "lsl x16, x22, #5\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_4)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_8)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x20000\n"
        "add x18, x16, x14\n"
#elif defined(VL_16)
        "lsl x16, x22, #6\n"
        "add x15, x16, x14\n"
        "add x18, x15, x14\n"
#endif
        "fmla z16.h, z4.h, z0.h[2]\n"  "fmla z19.h, z4.h, z0.h[3]\n"
        "fmla z22.h, z4.h, z0.h[4]\n"  "fmla z25.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z28.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z11.h, z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "fmla z14.h, z6.h, z7.h[2]\n"  "fmla z17.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "fmla z20.h, z6.h, z7.h[4]\n"  "fmla z23.h, z6.h, z7.h[5]\n"
        "fmla z26.h, z6.h, z7.h[6]\n"  "fmla z29.h, z6.h, z7.h[7]\n"
        "fmla z9.h,  z5.h, z7.h[0]\n"  "fmla z12.h, z5.h, z7.h[1]\n"
        "fmla z15.h, z5.h, z7.h[2]\n"  "fmla z18.h, z5.h, z7.h[3]\n"
        "fmla z21.h, z5.h, z7.h[4]\n"  "fmla z24.h, z5.h, z7.h[5]\n"
        "fmla z27.h, z5.h, z7.h[6]\n"  "fmla z30.h, z5.h, z7.h[7]\n"
        "fmla z10.h, z1.h, z7.h[0]\n"  "fmla z13.h, z1.h, z7.h[1]\n"
        "fmla z16.h, z1.h, z7.h[2]\n"  "fmla z19.h, z1.h, z7.h[3]\n"
        "fmla z22.h, z1.h, z7.h[4]\n"  "fmla z25.h, z1.h, z7.h[5]\n"
        "fmla z28.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

        // ---- K cleanup ----
        "4:\n"
        "add x21, x21, #0x10\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z11.h, z2.h, z0.h[1]\n"
        "fmla z14.h, z2.h, z0.h[2]\n"  "fmla z17.h, z2.h, z0.h[3]\n"
        "fmla z20.h, z2.h, z0.h[4]\n"  "fmla z23.h, z2.h, z0.h[5]\n"
        "fmla z26.h, z2.h, z0.h[6]\n"  "fmla z29.h, z2.h, z0.h[7]\n"
        "fmla z9.h,  z3.h, z0.h[0]\n"  "fmla z12.h, z3.h, z0.h[1]\n"
        "fmla z15.h, z3.h, z0.h[2]\n"  "fmla z18.h, z3.h, z0.h[3]\n"
        "fmla z21.h, z3.h, z0.h[4]\n"  "fmla z24.h, z3.h, z0.h[5]\n"
        "fmla z27.h, z3.h, z0.h[6]\n"  "fmla z30.h, z3.h, z0.h[7]\n"
        "fmla z10.h, z4.h, z0.h[0]\n"  "fmla z13.h, z4.h, z0.h[1]\n"
        "fmla z16.h, z4.h, z0.h[2]\n"  "fmla z19.h, z4.h, z0.h[3]\n"
        "fmla z22.h, z4.h, z0.h[4]\n"  "fmla z25.h, z4.h, z0.h[5]\n"
        "fmla z28.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "cbz x20, 5f\n"

        // Final K step (odd K)
        "add x22, x22, #1\n"
#if defined(VL_2)
        "lsl x16, x22, #5\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_4)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x10000\n"
        "orr x18, x16, #0x20000\n"
#elif defined(VL_8)
        "lsl x16, x22, #6\n"
        "orr x15, x16, #0x20000\n"
        "add x18, x16, x14\n"
#elif defined(VL_16)
        "lsl x16, x22, #6\n"
        "add x15, x16, x14\n"
        "add x18, x15, x14\n"
#endif
        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x15]\n"
        "spm.ld1qd z0.d, p1/z, [x18]\n"
        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z11.h, z2.h, z3.h[1]\n"
        "fmla z14.h, z2.h, z3.h[2]\n"  "fmla z17.h, z2.h, z3.h[3]\n"
        "fmla z20.h, z2.h, z3.h[4]\n"  "fmla z23.h, z2.h, z3.h[5]\n"
        "fmla z26.h, z2.h, z3.h[6]\n"  "fmla z29.h, z2.h, z3.h[7]\n"
        "fmla z9.h,  z1.h, z3.h[0]\n"  "fmla z12.h, z1.h, z3.h[1]\n"
        "fmla z15.h, z1.h, z3.h[2]\n"  "fmla z18.h, z1.h, z3.h[3]\n"
        "fmla z21.h, z1.h, z3.h[4]\n"  "fmla z24.h, z1.h, z3.h[5]\n"
        "fmla z27.h, z1.h, z3.h[6]\n"  "fmla z30.h, z1.h, z3.h[7]\n"
        "fmla z10.h, z0.h, z3.h[0]\n"  "fmla z13.h, z0.h, z3.h[1]\n"
        "fmla z16.h, z0.h, z3.h[2]\n"  "fmla z19.h, z0.h, z3.h[3]\n"
        "fmla z22.h, z0.h, z3.h[4]\n"  "fmla z25.h, z0.h, z3.h[5]\n"
        "fmla z28.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        // ---- Epilogue: fused store/RMW to C ----
        "5:\n"
        "mov x5, x0\n"
        "add x6,  x5,  x13\n"  "add x7,  x6,  x13\n"  "add x8,  x7,  x13\n"
        "add x9,  x8,  x13\n"  "add x10, x9,  x13\n"  "add x11, x10, x13\n"
        "add x12, x11, x13\n"

        "cbz x17, 7f\n"

        // ---- first_k: pure store ----
        "st1h { z8.h },  p0, [x5]\n"   "st1h { z9.h },  p0, [x5, #1, MUL VL]\n"  "st1h { z10.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x6]\n"   "st1h { z12.h }, p0, [x6, #1, MUL VL]\n"  "st1h { z13.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x7]\n"   "st1h { z15.h }, p0, [x7, #1, MUL VL]\n"  "st1h { z16.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z17.h }, p0, [x8]\n"   "st1h { z18.h }, p0, [x8, #1, MUL VL]\n"  "st1h { z19.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z20.h }, p0, [x9]\n"   "st1h { z21.h }, p0, [x9, #1, MUL VL]\n"  "st1h { z22.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z23.h }, p0, [x10]\n"  "st1h { z24.h }, p0, [x10, #1, MUL VL]\n" "st1h { z25.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z26.h }, p0, [x11]\n"  "st1h { z27.h }, p0, [x11, #1, MUL VL]\n" "st1h { z28.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z29.h }, p0, [x12]\n"  "st1h { z30.h }, p0, [x12, #1, MUL VL]\n" "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"
        "b 8f\n"

        // ---- RMW: load C + add + store ----
        "7:\n"
        "ld1h { z0.h }, p0/Z, [x5]\n"             "fadd z0.h, z8.h,  z0.h\n"  "st1h { z0.h }, p0, [x5]\n"
        "ld1h { z1.h }, p0/Z, [x5, #1, MUL VL]\n" "fadd z1.h, z9.h,  z1.h\n"  "st1h { z1.h }, p0, [x5, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x5, #2, MUL VL]\n" "fadd z2.h, z10.h, z2.h\n"  "st1h { z2.h }, p0, [x5, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x6]\n"             "fadd z0.h, z11.h, z0.h\n"  "st1h { z0.h }, p0, [x6]\n"
        "ld1h { z1.h }, p0/Z, [x6, #1, MUL VL]\n" "fadd z1.h, z12.h, z1.h\n"  "st1h { z1.h }, p0, [x6, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x6, #2, MUL VL]\n" "fadd z2.h, z13.h, z2.h\n"  "st1h { z2.h }, p0, [x6, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x7]\n"             "fadd z0.h, z14.h, z0.h\n"  "st1h { z0.h }, p0, [x7]\n"
        "ld1h { z1.h }, p0/Z, [x7, #1, MUL VL]\n" "fadd z1.h, z15.h, z1.h\n"  "st1h { z1.h }, p0, [x7, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x7, #2, MUL VL]\n" "fadd z2.h, z16.h, z2.h\n"  "st1h { z2.h }, p0, [x7, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x8]\n"             "fadd z0.h, z17.h, z0.h\n"  "st1h { z0.h }, p0, [x8]\n"
        "ld1h { z1.h }, p0/Z, [x8, #1, MUL VL]\n" "fadd z1.h, z18.h, z1.h\n"  "st1h { z1.h }, p0, [x8, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x8, #2, MUL VL]\n" "fadd z2.h, z19.h, z2.h\n"  "st1h { z2.h }, p0, [x8, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x9]\n"             "fadd z0.h, z20.h, z0.h\n"  "st1h { z0.h }, p0, [x9]\n"
        "ld1h { z1.h }, p0/Z, [x9, #1, MUL VL]\n" "fadd z1.h, z21.h, z1.h\n"  "st1h { z1.h }, p0, [x9, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x9, #2, MUL VL]\n" "fadd z2.h, z22.h, z2.h\n"  "st1h { z2.h }, p0, [x9, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x10]\n"            "fadd z0.h, z23.h, z0.h\n"  "st1h { z0.h }, p0, [x10]\n"
        "ld1h { z1.h }, p0/Z, [x10, #1, MUL VL]\n""fadd z1.h, z24.h, z1.h\n"  "st1h { z1.h }, p0, [x10, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x10, #2, MUL VL]\n""fadd z2.h, z25.h, z2.h\n"  "st1h { z2.h }, p0, [x10, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x11]\n"            "fadd z0.h, z26.h, z0.h\n"  "st1h { z0.h }, p0, [x11]\n"
        "ld1h { z1.h }, p0/Z, [x11, #1, MUL VL]\n""fadd z1.h, z27.h, z1.h\n"  "st1h { z1.h }, p0, [x11, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x11, #2, MUL VL]\n""fadd z2.h, z28.h, z2.h\n"  "st1h { z2.h }, p0, [x11, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/Z, [x12]\n"            "fadd z0.h, z29.h, z0.h\n"  "st1h { z0.h }, p0, [x12]\n"
        "ld1h { z1.h }, p0/Z, [x12, #1, MUL VL]\n""fadd z1.h, z30.h, z1.h\n"  "st1h { z1.h }, p0, [x12, #1, MUL VL]\n"
        "ld1h { z2.h }, p0/Z, [x12, #2, MUL VL]\n""fadd z2.h, z31.h, z2.h\n"  "st1h { z2.h }, p0, [x12, #2, MUL VL]\n"

        "8:\n"
        "add x26, x26, x25\n"
        "add x27, x12, x13\n"
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        :
        : [args] "r" (&args)
        : "cc", "memory", "p0", "p1",
          "x0", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
          "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22",
          "x23", "x24", "x25", "x26", "x27", "x28",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // !MOCK_SPM
