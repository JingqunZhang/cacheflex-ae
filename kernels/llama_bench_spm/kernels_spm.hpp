// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * kernels_spm.hpp — CacheFlex SPM GEMM kernels for llama_bench_spm
 *
 * Compile with exactly ONE of: -DVL_2, -DVL_4, -DVL_8, -DVL_16
 *
 * SPM layout per VL:
 *   VL=2  (256b):  NT=48  = 3×16.  3 Ways, Way0/1/2 at set k (SPMCP_32_IMM)
 *   VL=4  (512b):  NT=96  = 3×32.  3 Ways, Way0/1/2 at set k (SPMCP_64_IMM)
 *   VL=8  (1024b): NT=192 = 3×64.  4 Ways, KC-offset reuse:
 *                    slice0 = Way0+1@k, slice1 = Way2+3@k, slice2 = Way0+1@k+KC
 *   VL=16 (2048b): NT=384 = 3×128. 4 Ways, 3 KC-level stacking:
 *                    slice0 = Way0+1+2+3@k, slice1 = @k+KC, slice2 = @k+2KC
 *
 * SPM address format: (set_index << 6) | (way_id << 16)
 * SPMCP_64_IMM imm: spm_compiler.py encodes imm6 = imm_bytes / 64
 * SPMCP_32_IMM imm: spm_compiler.py encodes imm6 = imm_bytes / 32
 *
 * SPM capacity per VL (KC=256 default):
 *   VL=2:  3 ways × KC sets × 64B/set = 48KB  (8KB/way of 32B-used/set)
 *   VL=4:  3 ways × KC × 64B = 48KB  (16KB/way)
 *   VL=8:  Way0/1: 2×KC sets × 64B = 32KB. Way2/3: KC × 64B = 16KB.
 *   VL=16: 4 ways × 3×KC sets × 64B = 192KB  → use KC=64 (12KB/way) or 256 (48KB/way)
 *
 * Micro-kernel: 8 rows × NT cols (z8-z31 = 24 accumulators)
 *   z8..z15  = rows 0-7 × col-group 0  (z2 = B slice0)
 *   z16..z23 = rows 0-7 × col-group 1  (z3 = B slice1)
 *   z24..z31 = rows 0-7 × col-group 2  (z4 = B slice2)
 *
 * IMPLEMENTATION NOTES (vs reference gemm_vl_*.cpp):
 *   - SPMCP_64_PRE has writeback address drift → use SPMCP_64_IMM
 *   - Unit notation (#1,#2) → byte notation (#64,#128) for spm_compiler.py
 *   - VL=8 with 6 ways → use 4 ways with KC-offset reuse
 *   - gem5 cache.cc snoop patch required (isSPMcmd() guard)
 */
#pragma once
#include "common_spm.hpp"

#if !defined(VL_2) && !defined(VL_4) && !defined(VL_8) && !defined(VL_16)
#error "kernels_spm.hpp: define exactly one of VL_2, VL_4, VL_8, VL_16"
#endif

// ============================================================
// Pack A: 8 rows × kc cols → interleaved [r0k0,r1k0,...,r7k0, r0k1,...]
// (VL-independent, same for all configurations)
// ============================================================
static void pack_A_fp16_8row(const __fp16* A, size_t lda, __fp16* Apanel,
                              size_t M, size_t m0, size_t K, size_t k0,
                              size_t MC, size_t KC)
{
    const size_t MT  = 8;
    const size_t VLh = svcnth();
    size_t rows    = std::min(MC, M - m0);
    size_t kc      = std::min(KC, K - k0);
    size_t ablocks = (rows + MT - 1) / MT;

    __fp16* pad_row = static_cast<__fp16*>(__builtin_alloca(kc * sizeof(__fp16)));
    std::memset(pad_row, 0, kc * sizeof(__fp16));

    for (size_t mb = 0; mb < ablocks; ++mb) {
        size_t row_base = m0 + mb * MT;
        __fp16* dst = Apanel + mb * (kc * MT);

        const __fp16* rp[MT];
        for (size_t r = 0; r < MT; ++r) {
            size_t gr = row_base + r;
            rp[r] = (gr < M) ? (A + gr * lda + k0) : pad_row;
        }

        size_t ki = 0;
        for (; ki + VLh <= kc; ki += VLh) {
            svbool_t pg = svptrue_b16();
            svfloat16_t r0=svld1_f16(pg,rp[0]+ki), r1=svld1_f16(pg,rp[1]+ki);
            svfloat16_t r2=svld1_f16(pg,rp[2]+ki), r3=svld1_f16(pg,rp[3]+ki);
            svfloat16_t r4=svld1_f16(pg,rp[4]+ki), r5=svld1_f16(pg,rp[5]+ki);
            svfloat16_t r6=svld1_f16(pg,rp[6]+ki), r7=svld1_f16(pg,rp[7]+ki);
            svfloat16_t t01lo=svzip1_f16(r0,r1), t01hi=svzip2_f16(r0,r1);
            svfloat16_t t23lo=svzip1_f16(r2,r3), t23hi=svzip2_f16(r2,r3);
            svfloat16_t t45lo=svzip1_f16(r4,r5), t45hi=svzip2_f16(r4,r5);
            svfloat16_t t67lo=svzip1_f16(r6,r7), t67hi=svzip2_f16(r6,r7);
            svfloat32_t s01lo=svreinterpret_f32_f16(t01lo), s01hi=svreinterpret_f32_f16(t01hi);
            svfloat32_t s23lo=svreinterpret_f32_f16(t23lo), s23hi=svreinterpret_f32_f16(t23hi);
            svfloat32_t s45lo=svreinterpret_f32_f16(t45lo), s45hi=svreinterpret_f32_f16(t45hi);
            svfloat32_t s67lo=svreinterpret_f32_f16(t67lo), s67hi=svreinterpret_f32_f16(t67hi);
            svfloat32_t u0=svzip1_f32(s01lo,s23lo), u1=svzip2_f32(s01lo,s23lo);
            svfloat32_t u2=svzip1_f32(s01hi,s23hi), u3=svzip2_f32(s01hi,s23hi);
            svfloat32_t u4=svzip1_f32(s45lo,s67lo), u5=svzip2_f32(s45lo,s67lo);
            svfloat32_t u6=svzip1_f32(s45hi,s67hi), u7=svzip2_f32(s45hi,s67hi);
            svfloat64_t d0=svzip1_f64(svreinterpret_f64_f32(u0),svreinterpret_f64_f32(u4));
            svfloat64_t d1=svzip2_f64(svreinterpret_f64_f32(u0),svreinterpret_f64_f32(u4));
            svfloat64_t d2=svzip1_f64(svreinterpret_f64_f32(u1),svreinterpret_f64_f32(u5));
            svfloat64_t d3=svzip2_f64(svreinterpret_f64_f32(u1),svreinterpret_f64_f32(u5));
            svfloat64_t d4=svzip1_f64(svreinterpret_f64_f32(u2),svreinterpret_f64_f32(u6));
            svfloat64_t d5=svzip2_f64(svreinterpret_f64_f32(u2),svreinterpret_f64_f32(u6));
            svfloat64_t d6=svzip1_f64(svreinterpret_f64_f32(u3),svreinterpret_f64_f32(u7));
            svfloat64_t d7=svzip2_f64(svreinterpret_f64_f32(u3),svreinterpret_f64_f32(u7));
            svst1_f16(pg, dst+ki*MT+0*VLh, svreinterpret_f16_f64(d0));
            svst1_f16(pg, dst+ki*MT+1*VLh, svreinterpret_f16_f64(d1));
            svst1_f16(pg, dst+ki*MT+2*VLh, svreinterpret_f16_f64(d2));
            svst1_f16(pg, dst+ki*MT+3*VLh, svreinterpret_f16_f64(d3));
            svst1_f16(pg, dst+ki*MT+4*VLh, svreinterpret_f16_f64(d4));
            svst1_f16(pg, dst+ki*MT+5*VLh, svreinterpret_f16_f64(d5));
            svst1_f16(pg, dst+ki*MT+6*VLh, svreinterpret_f16_f64(d6));
            svst1_f16(pg, dst+ki*MT+7*VLh, svreinterpret_f16_f64(d7));
        }
        for (; ki < kc; ++ki)
            for (size_t r = 0; r < MT; ++r)
                dst[ki*MT + r] = rp[r][ki];
    }
}

// ============================================================
// SwiGLU: gate[i] = SiLU(gate[i]) * up[i]  (in-place on gate)
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Scalar FP32 implementation — SwiGLU is tiny vs GEMMs.
// (VL-independent)
// ============================================================
static void swiglu_fp16(__fp16* gate, const __fp16* up, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = (float)gate[i];
        float u = (float)up[i];
        gate[i] = (__fp16)(g / (1.0f + expf(-g)) * u);
    }
}

// ============================================================
//  VL-specific section
// ============================================================

#if defined(VL_2) || defined(VL_4)
// -------------------------------------------------------------------
// VL=2 (256-bit) and VL=4 (512-bit): identical kernel
//   VL=2: NT=48  = 3×svcnth(), spm.ld1qd = 32B per call, SPMCP_32_IMM
//   VL=4: NT=96  = 3×svcnth(), spm.ld1qd = 64B per call, SPMCP_64_IMM
//
// SPM layout: 3 Ways, Way0/1/2 at set k (one KC-chunk at a time)
//   SPMCP Way0@k: B[k, n+0      .. n+NT/3-1]
//   SPMCP Way1@k: B[k, n+NT/3   .. n+2NT/3-1]
//   SPMCP Way2@k: B[k, n+2NT/3  .. n+NT-1]
// -------------------------------------------------------------------
// SPM address shift: VL=4 uses lsl#6 (64B/set, 1 row/set, KC_max=1024)
// VL=2 uses kc/2 split: Way0 (lsl#6) ← near k=0..kc/2-1 (B[k,0..31]);
//   Way1 (lsl#5) ← all k (near: B[k,32..47], far: B[k,0..15]); Way2 (lsl#6) ← far k (B[k,16..47])
//   KC_max=2048 (Way0/2 hold kc/2 rows each at lsl#6; Way1 holds kc rows at lsl#5=2048 slots)
// Kernel: Phase1 k=0..kc/2-1 → Way0[k*64]+Way0[k*64+32]+Way1[k*32|0x10000]
//         Phase2 k=kc/2..kc-1 → Way1[k*32|0x10000]+Way2[j*64|0x20000]+Way2[j*64+32|0x20000]
//   FMLA order identical in both phases: z2→z8-z15, z3→z16-z23, z4→z24-z31
#define SPM_ADDR_SHIFT "6"

// Pack B tile [kc rows × NT cols] → SPM Ways 0/1/2
//   B_row0  : pointer to B[k0, n0]
//   N_stride: row stride of B in FP16 elements (full N)
//   kc      : number of rows to pack (KC chunk)
//   KC      : unused for VL=2/4 (parameter kept for API uniformity)
static void pack_B_tile_to_spm(const __fp16* B_row0, size_t N_stride,
                                size_t kc, size_t /*KC*/,
                                size_t spm_base_set = 0) {
    if (kc == 0) return;
    size_t stride_bytes = N_stride * sizeof(__fp16);
    __asm__ __volatile__(
        "mov x17, #0x10000\n"          // Way1 offset
        "mov x18, #0x20000\n"          // Way2 offset
        "mov x10, %x[base]\n"          // SPM row counter starts at spm_base_set
        "mov x12, %x[src]\n"           // source row pointer
        "1:\n"
#if defined(VL_2)
        // VL=2 lsl#5: Way0[k*32]=B[k,0..15], Way1[k*32|0x10000]=B[k,16..31],
        //             Way2[k*32|0x20000]=B[k,32..47]. KC_max=2048.
        "lsl x16, x10, #5\n"             // x16 = k*32 (lsl#5, 2 rows per 64B set)
        "SPMCP_32_IMM x16, [x12, #0]\n"  // Way0 ← B[k, 0..15]  (32B)
        "orr x15, x16, x17\n"            // Way1 SPM addr = k*32 | 0x10000
        "SPMCP_32_IMM x15, [x12, #32]\n" // Way1 ← B[k, 16..31] (32B)
        "orr x14, x16, x18\n"            // Way2 SPM addr = k*32 | 0x20000
        "SPMCP_32_IMM x14, [x12, #64]\n" // Way2 ← B[k, 32..47] (32B)
#else  // VL=4
        "lsl x16, x10, #" SPM_ADDR_SHIFT "\n"  // x16 = row * 64
        "orr x14, x16, x17\n"            // Way1 SPM addr
        // SPMCP_64_IMM: 64B per copy; imm6 = imm_bytes/64
        "SPMCP_64_IMM x16, [x12, #0]\n"    // Way0 ← B[k, n+0..31]  (64B)
        "SPMCP_64_IMM x14, [x12, #64]\n"   // Way1 ← B[k, n+32..63] (64B)
        "orr x14, x16, x18\n"
        "SPMCP_64_IMM x14, [x12, #128]\n"  // Way2 ← B[k, n+64..95] (64B)
#endif
        "add x10, x10, #1\n"
        "add x12, x12, %x[stride]\n"
        "subs %x[kc], %x[kc], #1\n"
        "bne 1b\n"
        : [kc] "+r" (kc)
        : [src] "r" (B_row0), [stride] "r" (stride_bytes),
          [base] "r" (spm_base_set)
        : "cc", "memory", "x10", "x12", "x14", "x15", "x16", "x17", "x18"
    );
}

// Pack B tile (prepacked, VL=2): 3×SPMCP_32 per k-row, lsl#5 layout.
// SPM layout (lsl#5): Way0[k*32]=B[k,0..15], Way1[k*32|0x10000]=B[k,16..31],
//                     Way2[k*32|0x20000]=B[k,32..47].
// Each SPM set holds 2 k-rows (k*32: byte_off=0 for even k, byte_off=32 for odd k).
// KC_max = 2048 (2 rows/set × 1024 sets). Supports KC sweep 128..2048 step 128.
#if defined(VL_2)
static void pack_B_tile_to_spm_vl2(const __fp16* Bp,
                                    size_t kc, size_t /*KC*/,
                                    size_t spm_base_set = 0) {
    if (kc == 0) return;
    __asm__ __volatile__(
        "mov x10, %x[base]\n"            // SPM row counter (starts at spm_base_set)
        "mov x12, %x[src]\n"             // source pointer
        "cbz %x[kc], 2f\n"
        "1:\n"
        "lsl x16, x10, #5\n"             // k*32 (lsl#5, 2 rows per 64B set)
        "SPMCP_32_IMM x16, [x12, #0]\n"  // Way0[k*32]          ← B[k, 0..15]
        "orr x15, x16, #0x10000\n"       // Way1[k*32 | 0x10000]
        "SPMCP_32_IMM x15, [x12, #32]\n" // Way1[k*32]          ← B[k, 16..31]
        "orr x14, x16, #0x20000\n"       // Way2[k*32 | 0x20000]
        "SPMCP_32_IMM x14, [x12, #64]\n" // Way2[k*32]          ← B[k, 32..47]
        "add x10, x10, #1\n"
        "add x12, x12, #96\n"            // advance source by 1 row (96B)
        "subs %x[kc], %x[kc], #1\n"
        "bne 1b\n"
        "2:\n"
        : [kc] "+r" (kc)
        : [src] "r" (Bp), [base] "r" (spm_base_set)
        : "cc", "memory", "x10", "x12", "x14", "x15", "x16"
    );
}
#endif  // VL_2

// SPM micro-kernel: 8 rows × NT cols (VL=2 or VL=4)
//   Apanel      : packed A [8 × kc], interleaved column-major within 8-row group
//   Cpanel      : output  [8 × NT], row-major, stride = NT elements
//   K           : number of K steps (= kc)
//   ldc_elements: NT
//   KC          : unused for VL=2/4
static void spm_kernel(const __fp16* Apanel, __fp16* Cpanel,
                       int K, int ldc_elements, size_t /*KC*/,
                       size_t base_set = 0) {
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "mov x21, %x[Apanel]\n"
        "mov x0,  %x[Cpanel]\n"
        "mov x3,  %x[ldc]\n"
        "mov x20, %x[K]\n"
        "sub x20, x20, #1\n"
        "mov x22, %x[base]\n"       // SPM row counter starts at base_set

        // Zero 24 accumulators: z8..z15 = row0-7 × way0,
        //                       z16..z23 = row0-7 × way1,
        //                       z24..z31 = row0-7 × way2
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // Preload first B row from SPM (row 0)
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"   // Way0[0]
        "spm.ld1qd z3.d, p1/z, [x15]\n"   // Way1[0]
        "spm.ld1qd z4.d, p1/z, [x14]\n"   // Way2[0]
        "ld1rqh { z0.h }, p0/z, [x21]\n"  // A rows 0-7, k=0
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // Main 2-unroll loop
        "3:\n"
        // --- K step (current, using z2/z3/z4 and z0) ---
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"    // A rows 0-7, k+1
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"          // prefetch B[k+1] way0
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x15]\n"          // prefetch B[k+1] way1
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x14]\n"          // prefetch B[k+1] way2

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"                   // advance A by 2 K-steps (2×8×2B=32B)
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"          // prefetch B[k+2] way0
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"         // A rows 0-7, k+2

        // --- K step+1 (using z6/z5/z1 and z7) ---
        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x14]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

        // Cleanup: one more K step
        "4:\n"
        "add x21, x21, #0x10\n"        // advance A by 1 K-step (1×8×2B=16B)
        "add x22, x22, #1\n"

        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"

        "cbz x20, 5f\n"

        // Final K step (odd K)
        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x15]\n"
        "spm.ld1qd z0.d, p1/z, [x14]\n"

        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"

        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"

        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        // Store: 8 rows × 3 VL cols, row-major stride = ldc
        "5:\n"
        "lsl x4, x3, #1\n"            // x4 = ldc * 2 bytes/row
        "mov x5, x0\n"
        "add x6,  x5, x4\n"
        "add x7,  x6, x4\n"
        "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"
        "add x10, x9, x4\n"
        "add x11, x10, x4\n"
        "add x12, x11, x4\n"

        // Row 0: z8 (way0), z16 (way1), z24 (way2)
        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        // Row 1
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        // Row 2
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        // Row 3
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        // Row 4
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        // Row 5
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        // Row 6
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        // Row 7
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        :
        : [Apanel] "r" (Apanel), [Cpanel] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc] "r" ((size_t)ldc_elements),
          [base] "r" (base_set)
        : "cc", "memory",
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x15", "x16", "x20", "x21", "x22",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

// SPM micro-kernel: ablocks × 8 rows × NT cols (VL=2 or VL=4)
// Mirrors baseline sve_interleaved_fp16_mla_8x3VL: the ablocks (M-block) loop is
// INSIDE the kernel so B stays hot in SPM across all M-blocks of one N-tile.
//   Apanel  : packed A [ablocks × 8 × kc], interleaved col-major within each 8-row group
//   Cpanel  : output  [ablocks × 8 × NT], row-major stride=NT elements per row
//   K       : kc (actual K steps ≥ 1)
//   ablocks : number of 8-row M-blocks
//   KC      : unused for VL=2/4
//   base_set: SPM set offset for this N-tile
static void spm_kernel_mblocks(const __fp16* Apanel, __fp16* Cpanel,
                                int K, int ablocks, size_t /*KC*/,
                                size_t base_set = 0) {
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"

        // Outer-loop setup (x23-x28 are callee-saved; GCC saves/restores them)
        // ldc computed via cnth to reduce "r" input count (avoids register pressure)
        "cnth x3, all, mul #3\n"    // x3 = ldc = NT = 3*svcnth()
        "mov x26, %x[Apanel]\n"     // Abase: advances by A_stride per M-block
        "mov x27, %x[Cpanel]\n"     // Cbase: advances by 8*ldc_bytes per M-block
        "mov x23, %x[ablks]\n"      // ablocks counter
        "mov x24, %x[base]\n"       // base_set (constant)
        "mov x28, %x[K_val]\n"      // K constant for per-block x20 reset
        "lsl x25, x28, #4\n"        // A_stride = K * 8 rows * 2 bytes = K<<4

        // ======== M-block outer loop ========
        "2:\n"
        // Per-block reset: restore A/C pointers, SPM counter, K countdown
        "mov x21, x26\n"            // Ap = Abase (current A block)
        "mov x0,  x27\n"            // Cp = Cbase (current C block)
        "mov x22, x24\n"            // SPM row counter = base_set
        "mov x20, x28\n"
        "sub x20, x20, #1\n"        // x20 = K-1

        // Zero 24 accumulators: z8..z15 = rows 0-7 x way0,
        //                       z16..z23 = rows 0-7 x way1,
        //                       z24..z31 = rows 0-7 x way2
        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // Preload B[k=0] from SPM
        // VL=2: Way0 x16=k*32, Way1 x15=x16|0x10000, Way2 x14=x16|0x20000  (lsl#5)
        // VL=4: Way0 x16=k*64, Way1 x15=x16|0x10000, Way2 x14=x16|0x20000  (lsl#6)
#if defined(VL_2)
        "lsl x16, x22, #5\n"
#else
        "lsl x16, x22, #6\n"
#endif
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "spm.ld1qd z4.d, p1/z, [x14]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // ---- K 2-unroll main loop (identical to spm_kernel) ----
        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
#if defined(VL_2)
        "lsl x16, x22, #5\n"
#else
        "lsl x16, x22, #6\n"
#endif
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x15]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x14]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
#if defined(VL_2)
        "lsl x16, x22, #5\n"
#else
        "lsl x16, x22, #6\n"
#endif
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x14]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

        // ---- K cleanup (identical to spm_kernel) ----
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

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"


        "cbz x20, 5f\n"

        // Final K step (odd K)
        "ld1rqh { z3.h }, p0/z, [x21]\n"
#if defined(VL_2)
        "lsl x16, x22, #5\n"
#else
        "lsl x16, x22, #6\n"
#endif
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x15]\n"
        "spm.ld1qd z0.d, p1/z, [x14]\n"

        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"

        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"

        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        // ---- Store 8 rows, row-major stride=NT (identical to spm_kernel) ----
        "5:\n"
        "lsl x4, x3, #1\n"             // x4 = ldc_bytes = NT * 2
        "mov x5, x0\n"
        "add x6,  x5, x4\n"
        "add x7,  x6, x4\n"
        "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"
        "add x10, x9, x4\n"
        "add x11, x10, x4\n"
        "add x12, x11, x4\n"            // row 7 pointer

        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        // Advance to next M-block
        "add x26, x26, x25\n"           // Abase += A_stride (= K*16 bytes)
        "add x27, x12, x4\n"            // Cbase = row7_ptr + ldc_bytes (= Cbase + 8*ldc_bytes)
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        :
        : [Apanel] "r" (Apanel), [Cpanel] "r" (Cpanel),
          [K_val]  "r" ((size_t)K),
          [ablks]  "r" ((size_t)ablocks), [base] "r" (base_set)
        : "cc", "memory",
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x15", "x16", "x17", "x18", "x20", "x21", "x22",
          "x23", "x24", "x25", "x26", "x27", "x28",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

// VL=4: fused compute + scale + scatter to scores
// Eliminates the Cpanel intermediate buffer round-trip.
// ONLY for FULL N-tiles (nc == NT=96). Partial tiles use spm_kernel_mblocks.
//
// C_scores: scores[h, m0, n0]  — already offset by caller
// ldc_s:    T  (row stride of scores in FP16 elements)
// scale_val: __fp16 scale = 1/sqrt(d)
//
// Register changes vs spm_kernel_mblocks:
//   x13 = ldc_bytes = ldc_s*2  (was x4 = NT*2)
//   z0  = scale broadcast       (loaded after K-loop, free register)
//   store stride: x13 (T*2) instead of x4 (NT*2)
static void spm_kernel_scatter_vl4(
    const __fp16* Apanel,
    __fp16*       C_scores,   // scores[h, m0, n0]
    int           ldc_s,      // T (row stride in FP16 elements)
    int           K,
    int           ablocks,
    size_t        base_set,
    __fp16        scale_val)
{
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"

        "cnth x3, all, mul #3\n"       // x3 = NT = 3*svcnth() (kept for compatibility)
        "lsl x13, %x[ldc_s], #1\n"     // x13 = ldc_bytes = ldc_s * 2  (scores row stride)
        "mov x26, %x[Apanel]\n"
        "mov x27, %x[Cscores]\n"       // Cscores_base: advances by 8*ldc_bytes per M-block
        "mov x23, %x[ablks]\n"
        "mov x24, %x[base]\n"
        "mov x28, %x[K_val]\n"
        "lsl x25, x28, #4\n"           // A_stride = K * 8 rows * 2B

        // ======== M-block outer loop ========
        "2:\n"
        "mov x21, x26\n"
        "mov x0,  x27\n"
        "mov x22, x24\n"
        "mov x20, x28\n"
        "sub x20, x20, #1\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // Preload B[k=0] from SPM
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "spm.ld1qd z4.d, p1/z, [x14]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        // ---- K 2-unroll main loop ----
        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x15]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x14]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x15]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x14]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

        // ---- K cleanup ----
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

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"

        "cbz x20, 5f\n"

        // Final K step (odd K)
        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #" SPM_ADDR_SHIFT "\n"
        "orr x15, x16, #0x10000\n"
        "orr x14, x16, #0x20000\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x15]\n"
        "spm.ld1qd z0.d, p1/z, [x14]\n"

        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"

        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"

        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        // ---- Scale + direct scatter to scores (no Cpanel round-trip) ----
        "5:\n"
        // Broadcast scale into z0 (free after K loop)
        "ld1rh { z0.h }, p0/z, [%x[scl]]\n"

        // Row pointers: stride = ldc_bytes = ldc_s*2 = T*2
        "mov x5, x0\n"
        "add x6,  x5, x13\n"
        "add x7,  x6, x13\n"
        "add x8,  x7, x13\n"
        "add x9,  x8, x13\n"
        "add x10, x9, x13\n"
        "add x11, x10, x13\n"
        "add x12, x11, x13\n"

        // Row 0: scale z8/z16/z24, store directly to scores
        "fmul z8.h,  p0/m, z8.h,  z0.h\n"
        "st1h { z8.h },  p0, [x5]\n"
        "fmul z16.h, p0/m, z16.h, z0.h\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "fmul z24.h, p0/m, z24.h, z0.h\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"

        // Row 1
        "fmul z9.h,  p0/m, z9.h,  z0.h\n"
        "st1h { z9.h },  p0, [x6]\n"
        "fmul z17.h, p0/m, z17.h, z0.h\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "fmul z25.h, p0/m, z25.h, z0.h\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"

        // Row 2
        "fmul z10.h, p0/m, z10.h, z0.h\n"
        "st1h { z10.h }, p0, [x7]\n"
        "fmul z18.h, p0/m, z18.h, z0.h\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "fmul z26.h, p0/m, z26.h, z0.h\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"

        // Row 3
        "fmul z11.h, p0/m, z11.h, z0.h\n"
        "st1h { z11.h }, p0, [x8]\n"
        "fmul z19.h, p0/m, z19.h, z0.h\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "fmul z27.h, p0/m, z27.h, z0.h\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"

        // Row 4
        "fmul z12.h, p0/m, z12.h, z0.h\n"
        "st1h { z12.h }, p0, [x9]\n"
        "fmul z20.h, p0/m, z20.h, z0.h\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "fmul z28.h, p0/m, z28.h, z0.h\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"

        // Row 5
        "fmul z13.h, p0/m, z13.h, z0.h\n"
        "st1h { z13.h }, p0, [x10]\n"
        "fmul z21.h, p0/m, z21.h, z0.h\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "fmul z29.h, p0/m, z29.h, z0.h\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"

        // Row 6
        "fmul z14.h, p0/m, z14.h, z0.h\n"
        "st1h { z14.h }, p0, [x11]\n"
        "fmul z22.h, p0/m, z22.h, z0.h\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "fmul z30.h, p0/m, z30.h, z0.h\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"

        // Row 7
        "fmul z15.h, p0/m, z15.h, z0.h\n"
        "st1h { z15.h }, p0, [x12]\n"
        "fmul z23.h, p0/m, z23.h, z0.h\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "fmul z31.h, p0/m, z31.h, z0.h\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        // Advance to next M-block
        "add x26, x26, x25\n"          // Abase += A_stride
        "add x27, x12, x13\n"          // Cscores_base = row7_ptr + ldc_bytes
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        :
        : [Apanel]  "r" (Apanel),
          [Cscores] "r" (C_scores),
          [ldc_s]   "r" ((size_t)ldc_s),
          [K_val]   "r" ((size_t)K),
          [ablks]   "r" ((size_t)ablocks),
          [base]    "r" (base_set),
          [scl]     "r" (&scale_val)
        : "cc", "memory",
          "x0", "x3", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
          "x14", "x15", "x16", "x20", "x21", "x22",
          "x23", "x24", "x25", "x26", "x27", "x28",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#elif defined(VL_8)
// -------------------------------------------------------------------
// VL=8 (1024-bit): NT=192 = 3×64 FP16
//   spm.ld1qd loads 128B = 2 consecutive ways per call
//   4-way KC-offset layout:
//     slice0 = Way0+1 @ set k   (B[k, n+0..63])
//     slice1 = Way2+3 @ set k   (B[k, n+64..127])
//     slice2 = Way0+1 @ set k+KC (B[k, n+128..191])
//   Way capacity: Way0/1 need 2×KC sets = 32KB; Way2/3 need KC sets = 16KB
// -------------------------------------------------------------------

static void pack_B_tile_to_spm(const __fp16* B_row0, size_t N_stride,
                                size_t kc, size_t KC,
                                size_t spm_base_set = 0) {
    if (kc == 0) return;
    size_t stride_bytes = N_stride * sizeof(__fp16);
    size_t KC64 = KC * 64;  // SPM set offset for KC-level
    __asm__ __volatile__(
        "mov x10, %x[base]\n"          // row counter starts at spm_base_set
        "mov x12, %x[src]\n"           // source row pointer
        "mov x19, %x[KC64]\n"          // KC * 64
        "1:\n"
        "lsl x16, x10, #6\n"           // set_k = k*64
        "add x18, x16, x19\n"          // set_{k+KC}

        // Slice 0: Way0@k + Way1@k = B[k, n+0..63]
        "SPMCP_64_IMM x16, [x12, #0]\n"           // Way0@k ← B[k,n+0..31]
        "orr x14, x16, #0x10000\n"
        "SPMCP_64_IMM x14, [x12, #64]\n"          // Way1@k ← B[k,n+32..63]

        // Slice 1: Way2@k + Way3@k = B[k, n+64..127]
        "orr x14, x16, #0x20000\n"
        "SPMCP_64_IMM x14, [x12, #128]\n"         // Way2@k ← B[k,n+64..95]
        "orr x14, x16, #0x30000\n"
        "SPMCP_64_IMM x14, [x12, #192]\n"         // Way3@k ← B[k,n+96..127]

        // Slice 2: Way0@k+KC + Way1@k+KC = B[k, n+128..191]
        "SPMCP_64_IMM x18, [x12, #256]\n"         // Way0@k+KC ← B[k,n+128..159]
        "orr x14, x18, #0x10000\n"
        "SPMCP_64_IMM x14, [x12, #320]\n"         // Way1@k+KC ← B[k,n+160..191]

        "add x10, x10, #1\n"
        "add x12, x12, %x[stride]\n"
        "subs %x[kc], %x[kc], #1\n"
        "bne 1b\n"
        : [kc] "+r" (kc)
        : [src] "r" (B_row0), [stride] "r" (stride_bytes), [KC64] "r" (KC64),
          [base] "r" (spm_base_set)
        : "cc", "memory", "x10", "x12", "x14", "x16", "x18", "x19"
    );
}

// VL=8 kernel: SPM loads from Way0+1@k, Way2+3@k, Way0+1@k+KC
static void spm_kernel(const __fp16* Apanel, __fp16* Cpanel,
                       int K, int ldc_elements, size_t KC,
                       size_t base_set = 0) {
    size_t KC64 = KC * 64;
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "mov x21, %x[Apanel]\n"
        "mov x0,  %x[Cpanel]\n"
        "mov x3,  %x[ldc]\n"
        "mov x20, %x[K]\n"
        "sub x20, x20, #1\n"
        "mov x22, %x[base]\n"          // SPM row counter starts at base_set
        "mov x19, %x[KC64]\n"          // KC * 64

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // Preload first B row (k=0)
        // slice0 = Way0+1@k=0: addr = 0
        // slice1 = Way2+3@k=0: addr = 0 | 0x20000
        // slice2 = Way0+1@k+KC: addr = KC*64
        "lsl x16, x22, #6\n"           // 0
        "orr x17, x16, #0x20000\n"     // Way2@k=0
        "add x18, x16, x19\n"          // Way0@k+KC
        "spm.ld1qd z2.d, p1/z, [x16]\n"  // slice0
        "spm.ld1qd z3.d, p1/z, [x17]\n"  // slice1
        "spm.ld1qd z4.d, p1/z, [x18]\n"  // slice2
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #6\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x17]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x18]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #6\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

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
        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"

        "cbz x20, 5f\n"

        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #6\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x17]\n"
        "spm.ld1qd z0.d, p1/z, [x18]\n"

        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"
        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"
        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        "5:\n"
        "lsl x4, x3, #1\n"
        "mov x5, x0\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"

        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        :
        : [Apanel] "r" (Apanel), [Cpanel] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc] "r" ((size_t)ldc_elements),
          [KC64] "r" (KC64), [base] "r" (base_set)
        : "cc", "memory",
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x16", "x17", "x18", "x19", "x20", "x21", "x22",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

// VL=8 spm_kernel_mblocks: ablocks loop inside kernel, B stays hot in SPM.
// Register pressure fix: force Abase(x26)/Cbase(x27)/KC64(x19) into specific named
// registers via `register ... asm("xN")`, removing them from the clobber list and
// freeing GCC to allocate the remaining 3 pure "r" inputs (K_val, ablks, base)
// from the ~4 unclobbered GP regs (x1,x2,x13,x15). x19 is read-only in K-loop.
static void spm_kernel_mblocks(const __fp16* Apanel, __fp16* Cpanel,
                                int K, int ablocks, size_t KC,
                                size_t base_set = 0) {
    size_t KC64 = KC * 64;
    register const __fp16* abase_r asm("x26") = Apanel;  // "+r": Abase, advances per block
    register __fp16*       cbase_r asm("x27") = Cpanel;  // "+r": Cbase, advances per block
    register size_t        kc64_r  asm("x19") = KC64;    // "r":  KC*64 read-only in K-loop
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "cnth x3, all, mul #3\n"      // ldc = NT = 3*svcnth()
        // x26=Abase, x27=Cbase, x19=KC64 already in place via register constraints
        "mov x23, %x[ablks]\n"
        "mov x24, %x[base]\n"
        "mov x28, %x[K_val]\n"
        "lsl x25, x28, #4\n"          // A_stride = K * 16 bytes

        "2:\n"
        "mov x21, x26\n"
        "mov x0,  x27\n"
        "mov x22, x24\n"
        "mov x20, x28\n"
        "sub x20, x20, #1\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        "lsl x16, x22, #6\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #6\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x17]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x18]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #6\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

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
        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "cbz x20, 5f\n"

        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #6\n"
        "orr x17, x16, #0x20000\n"
        "add x18, x16, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x17]\n"
        "spm.ld1qd z0.d, p1/z, [x18]\n"
        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"
        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"
        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        "5:\n"
        "lsl x4, x3, #1\n"
        "mov x5, x0\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"
        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        "add x26, x26, x25\n"           // Abase += A_stride (updates abase_r)
        "add x27, x12, x4\n"            // Cbase = row7 + ldc  (updates cbase_r)
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        : [abase] "+r" (abase_r), [cbase] "+r" (cbase_r)   // x26/x27: modified each block
        : [K_val] "r" ((size_t)K), [ablks] "r" ((size_t)ablocks),
          [base]  "r" (base_set),
          [kc64]  "r" (kc64_r)   // x19 = KC64 read-only; NOT in clobber list
        : "cc", "memory",
          // x19(kc64), x26(abase), x27(cbase) NOT clobbered — they are operands
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x16", "x17", "x18", "x20", "x21", "x22",
          "x23", "x24", "x25", "x28",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#elif defined(VL_16)
// -------------------------------------------------------------------
// VL=16 (2048-bit): NT=384 = 3×128 FP16
//   spm.ld1qd loads 256B = 4 consecutive ways per call
//   4-way, 3 KC-level stacking:
//     slice0 = Way0+1+2+3 @ set k        (B[k, n+0..127])
//     slice1 = Way0+1+2+3 @ set k+KC     (B[k, n+128..255])
//     slice2 = Way0+1+2+3 @ set k+2KC    (B[k, n+256..383])
//   Way capacity: 3×KC sets × 64B/set — use KC ≤ 341 (for 64KB/way)
// -------------------------------------------------------------------

static void pack_B_tile_to_spm(const __fp16* B_row0, size_t N_stride,
                                size_t kc, size_t KC,
                                size_t spm_base_set = 0) {
    if (kc == 0) return;
    size_t stride_bytes = N_stride * sizeof(__fp16);
    size_t KC64 = KC * 64;
    __asm__ __volatile__(
        "mov x10, %x[base]\n"          // row counter starts at spm_base_set
        "mov x12, %x[src]\n"
        "mov x19, %x[KC64]\n"          // KC * 64
        "1:\n"
        "lsl x16, x10, #6\n"           // set_k = k*64
        "add x17, x16, x19\n"          // set_{k+KC}
        "add x18, x17, x19\n"          // set_{k+2KC}

        // Slice 0: Way0+1+2+3 @ k = B[k, n+0..127] = 4×64B
        "SPMCP_64_IMM x16, [x12, #0]\n"
        "orr x14, x16, #0x10000\n"   "SPMCP_64_IMM x14, [x12, #64]\n"
        "orr x14, x16, #0x20000\n"   "SPMCP_64_IMM x14, [x12, #128]\n"
        "orr x14, x16, #0x30000\n"   "SPMCP_64_IMM x14, [x12, #192]\n"

        // Slice 1: Way0+1+2+3 @ k+KC = B[k, n+128..255] = 4×64B
        "SPMCP_64_IMM x17, [x12, #256]\n"
        "orr x14, x17, #0x10000\n"   "SPMCP_64_IMM x14, [x12, #320]\n"
        "orr x14, x17, #0x20000\n"   "SPMCP_64_IMM x14, [x12, #384]\n"
        "orr x14, x17, #0x30000\n"   "SPMCP_64_IMM x14, [x12, #448]\n"

        // Slice 2: Way0+1+2+3 @ k+2KC = B[k, n+256..383] = 4×64B
        "SPMCP_64_IMM x18, [x12, #512]\n"
        "orr x14, x18, #0x10000\n"   "SPMCP_64_IMM x14, [x12, #576]\n"
        "orr x14, x18, #0x20000\n"   "SPMCP_64_IMM x14, [x12, #640]\n"
        "orr x14, x18, #0x30000\n"   "SPMCP_64_IMM x14, [x12, #704]\n"

        "add x10, x10, #1\n"
        "add x12, x12, %x[stride]\n"
        "subs %x[kc], %x[kc], #1\n"
        "bne 1b\n"
        // No DSB SY needed: Bug 9 seqNum-based shadow table ensures all SPMCPs
        // (including slice1/slice2 at k+KC/k+2KC) defer to spm.ld1qd ordering.
        : [kc] "+r" (kc)
        : [src] "r" (B_row0), [stride] "r" (stride_bytes), [KC64] "r" (KC64),
          [base] "r" (spm_base_set)
        : "cc", "memory", "x10", "x12", "x14", "x16", "x17", "x18", "x19"
    );
}

// VL=16 kernel: SPM loads from Way0+1+2+3@k, @k+KC, @k+2KC
static void spm_kernel(const __fp16* Apanel, __fp16* Cpanel,
                       int K, int ldc_elements, size_t KC,
                       size_t base_set = 0) {
    size_t KC64 = KC * 64;
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"
        "mov x21, %x[Apanel]\n"
        "mov x0,  %x[Cpanel]\n"
        "mov x3,  %x[ldc]\n"
        "mov x20, %x[K]\n"
        "sub x20, x20, #1\n"
        "mov x22, %x[base]\n"          // SPM row counter starts at base_set
        "mov x19, %x[KC64]\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        // Preload first B row: k=0, 3 slices
        "lsl x16, x22, #6\n"           // k=0 → 0
        "add x17, x16, x19\n"          // Way0@KC
        "add x18, x17, x19\n"          // Way0@2KC
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x17]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x18]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #6\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

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
        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"

        "cbz x20, 5f\n"

        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x17]\n"
        "spm.ld1qd z0.d, p1/z, [x18]\n"

        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"
        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"
        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        "5:\n"
        "lsl x4, x3, #1\n"
        "mov x5, x0\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"

        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        :
        : [Apanel] "r" (Apanel), [Cpanel] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc] "r" ((size_t)ldc_elements),
          [KC64] "r" (KC64), [base] "r" (base_set)
        : "cc", "memory",
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x16", "x17", "x18", "x19", "x20", "x21", "x22",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

// VL=16 spm_kernel_mblocks: ablocks loop inside kernel, B stays hot in SPM.
// Register pressure fix: force Abase(x26)/Cbase(x27)/KC64(x19) into specific named
// registers via `register ... asm("xN")`, removing them from the clobber list and
// freeing GCC to allocate the remaining 3 pure "r" inputs (K_val, ablks, base)
// from the ~4 unclobbered GP regs (x1,x2,x13,x15). x19 is read-only in K-loop.
// x19 = KC*64 constant (K-loop only reads it). 3 slices stacked at KC intervals:
//   x16 = slice0 addr (k*64), x17 = slice1 addr (k*64 + KC*64), x18 = slice2 (+2KC*64)
// ldc computed via cnth (same reason as VL=8 — avoid 7-input register pressure).
static void spm_kernel_mblocks(const __fp16* Apanel, __fp16* Cpanel,
                                int K, int ablocks, size_t KC,
                                size_t base_set = 0) {
    size_t KC64 = KC * 64;
    register const __fp16* abase_r asm("x26") = Apanel;  // "+r": Abase, advances per block
    register __fp16*       cbase_r asm("x27") = Cpanel;  // "+r": Cbase, advances per block
    register size_t        kc64_r  asm("x19") = KC64;    // "r":  KC*64 read-only in K-loop
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"

        "cnth x3, all, mul #3\n"
        // x26=Abase, x27=Cbase, x19=KC64 already in place via register constraints
        "mov x23, %x[ablks]\n"
        "mov x24, %x[base]\n"
        "mov x28, %x[K_val]\n"
        "lsl x25, x28, #4\n"

        "2:\n"
        "mov x21, x26\n"
        "mov x0,  x27\n"
        "mov x22, x24\n"
        "mov x20, x28\n"
        "sub x20, x20, #1\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x17]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x18]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #6\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

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
        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "cbz x20, 5f\n"

        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x17]\n"
        "spm.ld1qd z0.d, p1/z, [x18]\n"
        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"
        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"
        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        "5:\n"
        "lsl x4, x3, #1\n"
        "mov x5, x0\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"
        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        "add x26, x26, x25\n"
        "add x27, x12, x4\n"
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        : [abase] "+r" (abase_r), [cbase] "+r" (cbase_r)   // x26/x27 advance per block
        : [K_val] "r" ((size_t)K),
          [ablks] "r" ((size_t)ablocks), [base] "r" (base_set),
          [kc64]  "r" (kc64_r)   // x19 = KC64 read-only; NOT in clobber list
        : "cc", "memory",
          // x19(kc64), x26(abase), x27(cbase) NOT clobbered — they are operands
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x16", "x17", "x18", "x20", "x21", "x22",
          "x23", "x24", "x25", "x28",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

// VL=16 spm_kernel_mblocks_acc: identical to spm_kernel_mblocks (B read from SPM
// via spm.ld1qd) but the C-store epilogue gains an `accumulate` mode that mirrors
// sve_interleaved_fp16_mla_8x3VL_cres's fp16 RMW epilogue:
//   accumulate==0 : st1h Cpanel  = acc          (k0==0, initialise the L2 tile)
//   accumulate!=0 : ld1h+fadd+st1h Cpanel += acc (k0>0, fp16 add — bit-identical
//                   addition order to the cres kernel: fadd zAcc, zOld, zAcc)
// Cpanel lives in regular L2 (NOT SPM); only the B operand is read from SPM. This
// lets CacheFlex put ONLY B in SPM while C stays L2-resident across all k0.
// acc pinned to x13 (free reg, read-only) to keep the input register count low.
static void spm_kernel_mblocks_acc(const __fp16* Apanel, __fp16* Cpanel,
                                   int K, int ablocks, size_t KC,
                                   size_t base_set, int accumulate) {
    size_t KC64 = KC * 64;
    register const __fp16* abase_r asm("x26") = Apanel;  // "+r": Abase, advances per block
    register __fp16*       cbase_r asm("x27") = Cpanel;  // "+r": Cbase, advances per block
    register size_t        kc64_r  asm("x19") = KC64;    // "r":  KC*64 read-only in K-loop
    register size_t        acc_r   asm("x13") = (size_t)(unsigned)accumulate; // read-only
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "ptrue p1.d\n"

        "cnth x3, all, mul #3\n"
        "mov x23, %x[ablks]\n"
        "mov x24, %x[base]\n"
        "mov x28, %x[K_val]\n"
        "lsl x25, x28, #4\n"

        "2:\n"
        "mov x21, x26\n"
        "mov x0,  x27\n"
        "mov x22, x24\n"
        "mov x20, x28\n"
        "sub x20, x20, #1\n"

        "mov z8.b,  #0\n"  "mov z9.b,  #0\n"  "mov z10.b, #0\n"  "mov z11.b, #0\n"
        "mov z12.b, #0\n"  "mov z13.b, #0\n"  "mov z14.b, #0\n"  "mov z15.b, #0\n"
        "mov z16.b, #0\n"  "mov z17.b, #0\n"  "mov z18.b, #0\n"  "mov z19.b, #0\n"
        "mov z20.b, #0\n"  "mov z21.b, #0\n"  "mov z22.b, #0\n"  "mov z23.b, #0\n"
        "mov z24.b, #0\n"  "mov z25.b, #0\n"  "mov z26.b, #0\n"  "mov z27.b, #0\n"
        "mov z28.b, #0\n"  "mov z29.b, #0\n"  "mov z30.b, #0\n"  "mov z31.b, #0\n"

        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"
        "cmp x20, #0x2\n"
        "blt 4f\n"

        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n"  "fmla z9.h,  z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/z, [x21, #16]\n"
        "fmla z10.h, z2.h, z0.h[2]\n"  "fmla z11.h, z2.h, z0.h[3]\n"
        "add x22, x22, #1\n"
        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z6.d, p1/z, [x16]\n"
        "fmla z12.h, z2.h, z0.h[4]\n"  "fmla z13.h, z2.h, z0.h[5]\n"
        "spm.ld1qd z5.d, p1/z, [x17]\n"
        "fmla z14.h, z2.h, z0.h[6]\n"  "fmla z15.h, z2.h, z0.h[7]\n"
        "spm.ld1qd z1.d, p1/z, [x18]\n"

        "fmla z16.h, z3.h, z0.h[0]\n"  "fmla z17.h, z3.h, z0.h[1]\n"
        "add x22, x22, #1\n"
        "sub x20, x20, #0x2\n"
        "fmla z18.h, z3.h, z0.h[2]\n"  "fmla z19.h, z3.h, z0.h[3]\n"
        "cmp x20, #0x2\n"
        "fmla z20.h, z3.h, z0.h[4]\n"  "fmla z21.h, z3.h, z0.h[5]\n"
        "add x21, x21, #0x20\n"
        "fmla z22.h, z3.h, z0.h[6]\n"  "fmla z23.h, z3.h, z0.h[7]\n"

        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "lsl x16, x22, #6\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/z, [x21]\n"

        "fmla z8.h,  z6.h, z7.h[0]\n"  "fmla z9.h,  z6.h, z7.h[1]\n"
        "spm.ld1qd z3.d, p1/z, [x17]\n"
        "fmla z10.h, z6.h, z7.h[2]\n"  "fmla z11.h, z6.h, z7.h[3]\n"
        "spm.ld1qd z4.d, p1/z, [x18]\n"
        "fmla z12.h, z6.h, z7.h[4]\n"  "fmla z13.h, z6.h, z7.h[5]\n"
        "fmla z14.h, z6.h, z7.h[6]\n"  "fmla z15.h, z6.h, z7.h[7]\n"

        "fmla z16.h, z5.h, z7.h[0]\n"  "fmla z17.h, z5.h, z7.h[1]\n"
        "fmla z18.h, z5.h, z7.h[2]\n"  "fmla z19.h, z5.h, z7.h[3]\n"
        "fmla z20.h, z5.h, z7.h[4]\n"  "fmla z21.h, z5.h, z7.h[5]\n"
        "fmla z22.h, z5.h, z7.h[6]\n"  "fmla z23.h, z5.h, z7.h[7]\n"

        "fmla z24.h, z1.h, z7.h[0]\n"  "fmla z25.h, z1.h, z7.h[1]\n"
        "fmla z26.h, z1.h, z7.h[2]\n"  "fmla z27.h, z1.h, z7.h[3]\n"
        "fmla z28.h, z1.h, z7.h[4]\n"  "fmla z29.h, z1.h, z7.h[5]\n"
        "fmla z30.h, z1.h, z7.h[6]\n"  "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"

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
        "fmla z24.h, z4.h, z0.h[0]\n"  "fmla z25.h, z4.h, z0.h[1]\n"
        "fmla z26.h, z4.h, z0.h[2]\n"  "fmla z27.h, z4.h, z0.h[3]\n"
        "fmla z28.h, z4.h, z0.h[4]\n"  "fmla z29.h, z4.h, z0.h[5]\n"
        "fmla z30.h, z4.h, z0.h[6]\n"  "fmla z31.h, z4.h, z0.h[7]\n"
        "cbz x20, 5f\n"

        "ld1rqh { z3.h }, p0/z, [x21]\n"
        "lsl x16, x22, #6\n"
        "add x17, x16, x19\n"
        "add x18, x17, x19\n"
        "spm.ld1qd z2.d, p1/z, [x16]\n"
        "spm.ld1qd z1.d, p1/z, [x17]\n"
        "spm.ld1qd z0.d, p1/z, [x18]\n"
        "fmla z8.h,  z2.h, z3.h[0]\n"  "fmla z9.h,  z2.h, z3.h[1]\n"
        "fmla z10.h, z2.h, z3.h[2]\n"  "fmla z11.h, z2.h, z3.h[3]\n"
        "fmla z12.h, z2.h, z3.h[4]\n"  "fmla z13.h, z2.h, z3.h[5]\n"
        "fmla z14.h, z2.h, z3.h[6]\n"  "fmla z15.h, z2.h, z3.h[7]\n"
        "fmla z16.h, z1.h, z3.h[0]\n"  "fmla z17.h, z1.h, z3.h[1]\n"
        "fmla z18.h, z1.h, z3.h[2]\n"  "fmla z19.h, z1.h, z3.h[3]\n"
        "fmla z20.h, z1.h, z3.h[4]\n"  "fmla z21.h, z1.h, z3.h[5]\n"
        "fmla z22.h, z1.h, z3.h[6]\n"  "fmla z23.h, z1.h, z3.h[7]\n"
        "fmla z24.h, z0.h, z3.h[0]\n"  "fmla z25.h, z0.h, z3.h[1]\n"
        "fmla z26.h, z0.h, z3.h[2]\n"  "fmla z27.h, z0.h, z3.h[3]\n"
        "fmla z28.h, z0.h, z3.h[4]\n"  "fmla z29.h, z0.h, z3.h[5]\n"
        "fmla z30.h, z0.h, z3.h[6]\n"  "fmla z31.h, z0.h, z3.h[7]\n"

        "5:\n"
        "lsl x4, x3, #1\n"
        "mov x5, x0\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"

        "cbz %x[acc], 6f\n"
        // ---- accumulate epilogue: Cpanel += acc, fp16 fadd (old first) ----
        "ld1h { z0.h }, p0/z, [x5]\n"  "ld1h { z1.h }, p0/z, [x5, #1, MUL VL]\n"  "ld1h { z2.h }, p0/z, [x5, #2, MUL VL]\n"
        "fadd z8.h,  z0.h, z8.h\n"   "fadd z16.h, z1.h, z16.h\n"  "fadd z24.h, z2.h, z24.h\n"
        "st1h { z8.h },  p0, [x5]\n"  "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"  "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x6]\n"  "ld1h { z1.h }, p0/z, [x6, #1, MUL VL]\n"  "ld1h { z2.h }, p0/z, [x6, #2, MUL VL]\n"
        "fadd z9.h,  z0.h, z9.h\n"   "fadd z17.h, z1.h, z17.h\n"  "fadd z25.h, z2.h, z25.h\n"
        "st1h { z9.h },  p0, [x6]\n"  "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"  "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x7]\n"  "ld1h { z1.h }, p0/z, [x7, #1, MUL VL]\n"  "ld1h { z2.h }, p0/z, [x7, #2, MUL VL]\n"
        "fadd z10.h, z0.h, z10.h\n"  "fadd z18.h, z1.h, z18.h\n"  "fadd z26.h, z2.h, z26.h\n"
        "st1h { z10.h }, p0, [x7]\n"  "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"  "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x8]\n"  "ld1h { z1.h }, p0/z, [x8, #1, MUL VL]\n"  "ld1h { z2.h }, p0/z, [x8, #2, MUL VL]\n"
        "fadd z11.h, z0.h, z11.h\n"  "fadd z19.h, z1.h, z19.h\n"  "fadd z27.h, z2.h, z27.h\n"
        "st1h { z11.h }, p0, [x8]\n"  "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"  "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x9]\n"  "ld1h { z1.h }, p0/z, [x9, #1, MUL VL]\n"  "ld1h { z2.h }, p0/z, [x9, #2, MUL VL]\n"
        "fadd z12.h, z0.h, z12.h\n"  "fadd z20.h, z1.h, z20.h\n"  "fadd z28.h, z2.h, z28.h\n"
        "st1h { z12.h }, p0, [x9]\n"  "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"  "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x10]\n" "ld1h { z1.h }, p0/z, [x10, #1, MUL VL]\n" "ld1h { z2.h }, p0/z, [x10, #2, MUL VL]\n"
        "fadd z13.h, z0.h, z13.h\n"  "fadd z21.h, z1.h, z21.h\n"  "fadd z29.h, z2.h, z29.h\n"
        "st1h { z13.h }, p0, [x10]\n" "st1h { z21.h }, p0, [x10, #1, MUL VL]\n" "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x11]\n" "ld1h { z1.h }, p0/z, [x11, #1, MUL VL]\n" "ld1h { z2.h }, p0/z, [x11, #2, MUL VL]\n"
        "fadd z14.h, z0.h, z14.h\n"  "fadd z22.h, z1.h, z22.h\n"  "fadd z30.h, z2.h, z30.h\n"
        "st1h { z14.h }, p0, [x11]\n" "st1h { z22.h }, p0, [x11, #1, MUL VL]\n" "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "ld1h { z0.h }, p0/z, [x12]\n" "ld1h { z1.h }, p0/z, [x12, #1, MUL VL]\n" "ld1h { z2.h }, p0/z, [x12, #2, MUL VL]\n"
        "fadd z15.h, z0.h, z15.h\n"  "fadd z23.h, z1.h, z23.h\n"  "fadd z31.h, z2.h, z31.h\n"
        "st1h { z15.h }, p0, [x12]\n" "st1h { z23.h }, p0, [x12, #1, MUL VL]\n" "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"
        "b 7f\n"

        "6:\n"
        // ---- plain store epilogue (k0==0): Cpanel = acc (verbatim original) ----
        "st1h { z8.h },  p0, [x5]\n"
        "st1h { z16.h }, p0, [x5, #1, MUL VL]\n"
        "st1h { z24.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z9.h },  p0, [x6]\n"
        "st1h { z17.h }, p0, [x6, #1, MUL VL]\n"
        "st1h { z25.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z10.h }, p0, [x7]\n"
        "st1h { z18.h }, p0, [x7, #1, MUL VL]\n"
        "st1h { z26.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x8]\n"
        "st1h { z19.h }, p0, [x8, #1, MUL VL]\n"
        "st1h { z27.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z12.h }, p0, [x9]\n"
        "st1h { z20.h }, p0, [x9, #1, MUL VL]\n"
        "st1h { z28.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z13.h }, p0, [x10]\n"
        "st1h { z21.h }, p0, [x10, #1, MUL VL]\n"
        "st1h { z29.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x11]\n"
        "st1h { z22.h }, p0, [x11, #1, MUL VL]\n"
        "st1h { z30.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z15.h }, p0, [x12]\n"
        "st1h { z23.h }, p0, [x12, #1, MUL VL]\n"
        "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"

        "7:\n"
        "add x26, x26, x25\n"
        "add x27, x12, x4\n"
        "subs x23, x23, #1\n"
        "bgt 2b\n"

        : [abase] "+r" (abase_r), [cbase] "+r" (cbase_r)   // x26/x27 advance per block
        : [K_val] "r" ((size_t)K),
          [ablks] "r" ((size_t)ablocks), [base] "r" (base_set),
          [kc64]  "r" (kc64_r),  // x19 = KC64 read-only; NOT in clobber list
          [acc]   "r" (acc_r)    // x13 = accumulate flag, read-only
        : "cc", "memory",
          // x13(acc), x19(kc64), x26(abase), x27(cbase) NOT clobbered — operands
          "x0", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x14", "x16", "x17", "x18", "x20", "x21", "x22",
          "x23", "x24", "x25", "x28",
          "p0", "p1",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8",  "z9",  "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}
#endif  // VL_* section end

// ============================================================
// GEMM with SPM B-panel — NSP-batched loop order (VL-independent driver)
//
//   A[M, K] × B[K, N] → C[M, N]
//   B: row-major with ldb_B elements per row (may be > N for strided B)
//
//   Loop: K-outer → N-batch (NSP tiles packed to SPM) → M-inner (A_panel in L1)
//     NSP = SPM_SETS / (KC × SPM_KC_FACTOR) tiles packed simultaneously
//     A_panel_mc (one MC-block, ~16KB) stays in L1 for NSP kernel calls
//
// SPM_KC_FACTOR per VL:
//   VL=2/4: 1  (3 ways, 1 KC-level)    → NSP=16 at KC=64
//   VL=8:   2  (4 ways, 2 KC-levels)   → NSP=8  at KC=64
//   VL=16:  3  (4 ways, 3 KC-levels)   → NSP=5  at KC=64
// ============================================================
// Optional timing output for gemm_spm (mirrors GemmTiming in kernels_sve.hpp)
struct SpmGemmTiming {
    double t_packA   = 0;   // pack_A_fp16_8row total
    double t_packB   = 0;   // pack_B_tile_to_spm (SPMCP only, excl. memset)
    double t_kernel  = 0;   // spm_kernel_mblocks total
    double t_c_write = 0;   // C scatter k0==0 (pure store, 1 pass)
    double t_c_rmw   = 0;   // C scatter k0>0  (load+add+store)
    void reset() { t_packA=t_packB=t_kernel=t_c_write=t_c_rmw=0; }
};

static void gemm_spm(const __fp16* A, const __fp16* B, __fp16* C,
                     size_t M, size_t N, size_t K, size_t ldb_B,
                     size_t MC = 128, size_t KC = 256,
                     SpmGemmTiming* tim = nullptr)
{
    const size_t NT  = 3 * svcnth();   // 48/96/192/384 depending on compiled VL
    const size_t MT  = 8;
    const size_t num_mc = (M + MC - 1) / MC;
    const size_t max_ab = (MC + MT - 1) / MT;


    // A_cache: all MC-blocks packed once per K-tile, reused across all N-batches
    // (mirrors gemm_knm_prepacked loop order: K → M(pack_A) → N)
    AlignedBuffer<__fp16> A_cache(num_mc * max_ab * MT * KC);
    // Cpanel for one spm_kernel call (8 rows × NT cols per ablocks block)
    AlignedBuffer<__fp16> Cpanel_tmp(max_ab * MT * NT);
    // Scratch for zero-padding partial B tiles
    AlignedBuffer<__fp16> B_pad(KC * NT);
    // Pre-touch all internal buffers to demand-fault their pages before
    // any SPM or O3 speculative access. With large heap (T=512), these
    // buffers are at high addresses and their pages are unmapped until first touch.
    // Uninitialized page faults during O3 speculative spm_kernel_mblocks stores
    // (st1h to Cpanel_tmp) corrupt accumulators when re-execution misorders.
    std::memset(A_cache.data(),   0, num_mc * max_ab * MT * KC * sizeof(__fp16));
    std::memset(Cpanel_tmp.data(),0, max_ab * MT * NT          * sizeof(__fp16));
    std::memset(B_pad.data(),     0, KC * NT                   * sizeof(__fp16));

    // Per-call timing (microseconds)
    static int _gemm_call_id = 0;
    int _this_call = _gemm_call_id++;
    double _t_packA = 0, _t_packB = 0, _t_kernel = 0, _t_c_write = 0, _t_c_rmw = 0;
    auto _now = []() { return Clock::now(); };

    for (size_t k0 = 0; k0 < K; k0 += KC) {
        size_t kc = std::min(KC, K - k0);

        // Phase 1: pack ALL M-tiles of A for this K-tile (amortized over all N-batches)
        // pack_A calls: K_tiles × M_tiles  (same count as gemm_knm_prepacked)
        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0);
            auto _t0 = _now();
            pack_A_fp16_8row(A, K,
                             A_cache.data() + (m0 / MC) * max_ab * MT * KC,
                             M, m0, K, k0, mc, kc);
            _t_packA += us_since(_t0);
        }

        // Phase 2: process N one NT-tile at a time
        for (size_t n0 = 0; n0 < N; n0 += NT) {
            size_t nc = std::min(NT, N - n0);

            // SPMCP one tile to SPM
            {
                // Pre-prepare partial B tile (tail padding) outside the packB timer
                if (nc != NT) {
                    size_t kc_cur = std::min(KC, K - k0);
                    std::memset(B_pad.data(), 0, kc_cur * NT * sizeof(__fp16));
                    for (size_t ki = 0; ki < kc_cur; ++ki)
                        std::memcpy(B_pad.data() + ki * NT,
                                    B + (k0 + ki) * ldb_B + n0,
                                    nc * sizeof(__fp16));
                }
                auto _t0 = _now();
                if (nc == NT) {
                    pack_B_tile_to_spm(B + k0 * ldb_B + n0, ldb_B, kc, KC, 0);
                } else {
                    pack_B_tile_to_spm(B_pad.data(), NT, kc, KC, 0);
                }
                _t_packB += us_since(_t0);
            }

            // Compute: sweep M with pre-packed A_cache
            for (size_t m0 = 0; m0 < M; m0 += MC) {
                size_t mc      = std::min(MC, M - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                const __fp16* Ap = A_cache.data() + (m0 / MC) * max_ab * MT * KC;

                {
                    auto _t0 = _now();
                    spm_kernel_mblocks(Ap, Cpanel_tmp.data(),
                                       (int)kc, (int)ablocks, KC, 0);
                    _t_kernel += us_since(_t0);
                }

                // Scatter Cpanel_tmp → C[m0:m0+mc, n0:n0+nc]
                // k0==0: pure store (C write); k0>0: load+add+store (C RMW)
                {
                    auto _t0 = _now();
                    for (size_t mb = 0; mb < ablocks; ++mb) {
                        const __fp16* tp = Cpanel_tmp.data() + mb * MT * NT;
                        for (size_t r = 0; r < MT; ++r) {
                            size_t gr = m0 + mb * MT + r;
                            if (gr >= M) break;
                            __fp16*       c_row = C + gr * N + n0;
                            const __fp16* t_row = tp + r * NT;
                            size_t i = 0;
                            svbool_t pg = svptrue_b16();
                            if (k0 == 0) {
                                for (; i + svcnth() <= nc; i += svcnth())
                                    svst1_f16(pg, c_row+i, svld1_f16(pg, t_row+i));
                                if (i < nc) {
                                    svbool_t pt = svwhilelt_b16_u64(i, nc);
                                    svst1_f16(pt, c_row+i, svld1_f16(pt, t_row+i));
                                }
                            } else {
                                for (; i + svcnth() <= nc; i += svcnth())
                                    svst1_f16(pg, c_row+i, svadd_f16_x(pg,
                                        svld1_f16(pg, c_row+i), svld1_f16(pg, t_row+i)));
                                if (i < nc) {
                                    svbool_t pt = svwhilelt_b16_u64(i, nc);
                                    svst1_f16(pt, c_row+i, svadd_f16_x(pt,
                                        svld1_f16(pt, c_row+i), svld1_f16(pt, t_row+i)));
                                }
                            }
                        }
                    }
                    if (k0 == 0) _t_c_write += us_since(_t0);
                    else         _t_c_rmw   += us_since(_t0);
                }
            }
        }
    }

    // Export to caller struct if provided
    if (tim) {
        tim->t_packA   += _t_packA;
        tim->t_packB   += _t_packB;
        tim->t_kernel  += _t_kernel;
        tim->t_c_write += _t_c_write;
        tim->t_c_rmw   += _t_c_rmw;
    }

    // Print per-call breakdown
    {
        size_t K_tiles = (K + KC - 1) / KC;
        size_t N_tiles = (N + NT - 1) / NT;
        double t_total = _t_packA + _t_packB + _t_kernel + _t_c_write + _t_c_rmw;
        printf("[gemm_spm #%d] M=%zu K=%zu N=%zu  KC=%zu MC=%zu NT=%zu"
               "  K_tiles=%zu N_tiles=%zu\n",
               _this_call, M, K, N, KC, MC, NT, K_tiles, N_tiles);
        printf("  packA:       %8.1f us  (%5.1f%%)\n", _t_packA,   100.*_t_packA/t_total);
        printf("  packB(SPMCP):%8.1f us  (%5.1f%%)\n", _t_packB,   100.*_t_packB/t_total);
        printf("  kernel:      %8.1f us  (%5.1f%%)\n", _t_kernel,  100.*_t_kernel/t_total);
        printf("  C write(k0=0):%7.1f us  (%5.1f%%)  [pure store, 1 pass]\n",
               _t_c_write, 100.*_t_c_write/t_total);
        printf("  C RMW(k0>0): %8.1f us  (%5.1f%%)  [load+add+store, %zu passes]\n",
               _t_c_rmw,   100.*_t_c_rmw/t_total, K_tiles > 0 ? K_tiles - 1 : 0);
        printf("  total:       %8.1f us\n", t_total);
        fflush(stdout);
    }
}

// =============================================================================
// prepack_B_for_spm: Convert B[K×N] row-major → B_packed[K_tiles×N_tiles×KC×NT]
//
// Layout: B_packed[(k_idx * N_tiles + n_idx) * KC * NT + ki * NT + ni]
//           = B[(k_idx*KC + ki) * N + (n_idx*NT + ni)]
//
// With this layout, pack_B_tile_to_spm reads with stride=NT*2 bytes (192B at VL=4)
// instead of N*2 bytes (4096B), enabling AMPM to prefetch effectively.
//
// Call once before inference loop; B_packed must be allocated with size:
//   K_tiles * N_tiles * KC * NT * sizeof(__fp16)  (zero-padded for partial tiles)
// =============================================================================
static void prepack_B_for_spm(const __fp16* B, size_t K, size_t N,
                               size_t KC, __fp16* B_packed)
{
    const size_t NT      = 3 * svcnth();
    const size_t K_tiles = (K + KC - 1) / KC;
    const size_t N_tiles = (N + NT - 1) / NT;

    for (size_t k_idx = 0; k_idx < K_tiles; ++k_idx) {
        size_t k0 = k_idx * KC;
        size_t kc = std::min(KC, K - k0);
        for (size_t n_idx = 0; n_idx < N_tiles; ++n_idx) {
            size_t n0 = n_idx * NT;
            size_t nc = std::min(NT, N - n0);
            __fp16* Bp = B_packed + (k_idx * N_tiles + n_idx) * KC * NT;
            // Row-major layout [kc × NT]: B[k, n0..n0+NT-1] in order k=0..kc-1
            for (size_t ki = 0; ki < kc; ++ki) {
                std::memcpy(Bp + ki * NT, B + (k0 + ki) * N + n0,
                            nc * sizeof(__fp16));
                if (nc < NT)
                    std::memset(Bp + ki * NT + nc, 0,
                                (NT - nc) * sizeof(__fp16));
            }
            // Zero-pad rows beyond kc (partial last K-tile)
            if (kc < KC)
                std::memset(Bp + kc * NT, 0,
                            (KC - kc) * NT * sizeof(__fp16));
        }
    }
}

// =============================================================================
// gemm_spm_prepacked: SPM GEMM with B pre-packed in KNM layout.
//
// B_packed must be prepared by prepack_B_for_spm() before the inference loop.
// SPMCP source addresses are now sequential (stride=NT*2=192B at VL=4),
// enabling AMPM to prefetch B into L2 ahead of SPMCP demand.
// =============================================================================
static void gemm_spm_prepacked(const __fp16* A, const __fp16* B_packed, __fp16* C,
                                size_t M, size_t N, size_t K,
                                size_t MC = 128, size_t KC = 256,
                                SpmGemmTiming* tim = nullptr)
{
    const size_t NT  = 3 * svcnth();
    const size_t MT  = 8;
    const size_t num_mc  = (M + MC - 1) / MC;
    const size_t max_ab  = (MC + MT - 1) / MT;
    const size_t N_tiles = (N + NT - 1) / NT;

#if defined(VL_2) || defined(VL_4)
    const size_t SPM_KC_FACTOR = 1;
#elif defined(VL_8)
    const size_t SPM_KC_FACTOR = 2;
#elif defined(VL_16)
    const size_t SPM_KC_FACTOR = 3;
#endif
    const size_t SPM_SETS = 1024;
#if defined(VL_2)
    const size_t NSP = std::max<size_t>(1, SPM_SETS * 2 / KC);
#else
    const size_t NSP = std::max<size_t>(1, SPM_SETS / (KC * SPM_KC_FACTOR));
#endif

    AlignedBuffer<__fp16> A_cache(num_mc * max_ab * MT * KC);
    AlignedBuffer<__fp16> Cpanel_tmp(max_ab * MT * NT);
    std::memset(A_cache.data(),    0, num_mc * max_ab * MT * KC * sizeof(__fp16));
    std::memset(Cpanel_tmp.data(), 0, max_ab * MT * NT          * sizeof(__fp16));

    static int _call_id = 0;
    int _this_call = _call_id++;
    double _t_packA = 0, _t_packB = 0, _t_kernel = 0, _t_c_write = 0, _t_c_rmw = 0;
    auto _now = []() { return Clock::now(); };

    for (size_t k0 = 0, k_idx = 0; k0 < K; k0 += KC, ++k_idx) {
        size_t kc = std::min(KC, K - k0);

        // Pack all M-tiles of A for this K-tile
        for (size_t m0 = 0; m0 < M; m0 += MC) {
            size_t mc = std::min(MC, M - m0);
            auto _t0 = _now();
            pack_A_fp16_8row(A, K,
                             A_cache.data() + (m0 / MC) * max_ab * MT * KC,
                             M, m0, K, k0, mc, kc);
            _t_packA += us_since(_t0);
        }

        // Process N in batches of NSP tiles
        for (size_t n_batch = 0; n_batch < N; n_batch += NSP * NT) {
            size_t n_tiles = std::min(NSP, (N - n_batch + NT - 1) / NT);

            // SPMCP: read from pre-packed B (stride = NT*2 bytes, sequential)
            {
                auto _t0 = _now();
                for (size_t ti = 0; ti < n_tiles; ++ti) {
                    size_t n_idx    = (n_batch / NT) + ti;
                    size_t base_set = ti * KC * SPM_KC_FACTOR;
                    // B_packed for this (k_idx, n_idx) tile: contiguous KC×NT block
                    const __fp16* Bp = B_packed + (k_idx * N_tiles + n_idx) * KC * NT;
                    // stride = NT (elements) → NT*2 bytes between consecutive K rows
#if defined(VL_2)
                    // VL=2: row-major prepack, SPMCP_64+SPMCP_32 per row, kc/2 pairing
                    pack_B_tile_to_spm_vl2(Bp, kc, KC, base_set);
#else
                    pack_B_tile_to_spm(Bp, NT, kc, KC, base_set);
#endif
                }
                _t_packB += us_since(_t0);
            }

            // Kernel: sweep all M-tiles
            for (size_t m0 = 0; m0 < M; m0 += MC) {
                size_t mc      = std::min(MC, M - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                const __fp16* Ap = A_cache.data() + (m0 / MC) * max_ab * MT * KC;

                for (size_t ti = 0; ti < n_tiles; ++ti) {
                    size_t n0       = n_batch + ti * NT;
                    size_t nc       = std::min(NT, N - n0);
                    size_t base_set = ti * KC * SPM_KC_FACTOR;

                    {
                        auto _t0 = _now();
                        spm_kernel_mblocks(Ap, Cpanel_tmp.data(),
                                           (int)kc, (int)ablocks, KC, base_set);
                        _t_kernel += us_since(_t0);
                    }

                    // Scatter Cpanel → C
                    {
                        auto _t0 = _now();
                        for (size_t mb = 0; mb < ablocks; ++mb) {
                            const __fp16* tp = Cpanel_tmp.data() + mb * MT * NT;
                            for (size_t r = 0; r < MT; ++r) {
                                size_t gr = m0 + mb * MT + r;
                                if (gr >= M) break;
                                __fp16*       c_row = C + gr * N + n0;
                                const __fp16* t_row = tp + r * NT;
                                size_t i = 0;
                                svbool_t pg = svptrue_b16();
                                if (k0 == 0) {
                                    for (; i + svcnth() <= nc; i += svcnth())
                                        svst1_f16(pg, c_row+i, svld1_f16(pg, t_row+i));
                                    if (i < nc) {
                                        svbool_t pt = svwhilelt_b16_u64(i, nc);
                                        svst1_f16(pt, c_row+i, svld1_f16(pt, t_row+i));
                                    }
                                } else {
                                    for (; i + svcnth() <= nc; i += svcnth())
                                        svst1_f16(pg, c_row+i, svadd_f16_x(pg,
                                            svld1_f16(pg, c_row+i), svld1_f16(pg, t_row+i)));
                                    if (i < nc) {
                                        svbool_t pt = svwhilelt_b16_u64(i, nc);
                                        svst1_f16(pt, c_row+i, svadd_f16_x(pt,
                                            svld1_f16(pt, c_row+i), svld1_f16(pt, t_row+i)));
                                    }
                                }
                            }
                        }
                        if (k0 == 0) _t_c_write += us_since(_t0);
                        else         _t_c_rmw   += us_since(_t0);
                    }
                }
            }
        }
    }

    if (tim) {
        tim->t_packA   += _t_packA;
        tim->t_packB   += _t_packB;
        tim->t_kernel  += _t_kernel;
        tim->t_c_write += _t_c_write;
        tim->t_c_rmw   += _t_c_rmw;
    }

    {
        size_t K_tiles = (K + KC - 1) / KC;
        double t_total = _t_packA + _t_packB + _t_kernel + _t_c_write + _t_c_rmw;
        printf("[gemm_spm_prepacked #%d] M=%zu K=%zu N=%zu  KC=%zu MC=%zu NT=%zu"
               "  K_tiles=%zu N_tiles=%zu NSP=%zu\n",
               _this_call, M, K, N, KC, MC, NT, K_tiles, N_tiles, NSP);
        printf("  packA:          %8.1f us  (%5.1f%%)\n", _t_packA,   100.*_t_packA/t_total);
        printf("  packB(SPMCP):   %8.1f us  (%5.1f%%)  [stride=NT*2=%zuB, sequential]\n",
               _t_packB, 100.*_t_packB/t_total, NT * sizeof(__fp16));
        printf("  kernel:         %8.1f us  (%5.1f%%)\n", _t_kernel,  100.*_t_kernel/t_total);
        printf("  C write(k0=0):  %8.1f us  (%5.1f%%)\n", _t_c_write, 100.*_t_c_write/t_total);
        printf("  C RMW(k0>0):    %8.1f us  (%5.1f%%)  [%zu passes]\n",
               _t_c_rmw, 100.*_t_c_rmw/t_total, K_tiles > 0 ? K_tiles - 1 : 0);
        printf("  total:          %8.1f us\n", t_total);
        fflush(stdout);
    }
}
