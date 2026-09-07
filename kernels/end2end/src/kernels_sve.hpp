// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * kernels_sve.hpp — SVE FP16 kernels used by bench_kernel.cpp
 *
 * Only includes kernels actually needed for end-to-end evaluation:
 *   GEMM:    sve_interleaved_fp16_mla_8x3VL, sve_pack_B_3VL, pack_A_fp16_8row
 *            prepack_B_knm, prepack_B_knm_size
 *   Norm:    rmsnorm_row, rmsnorm
 *   RoPE:    build_cos_sin, apply_rope
 *   Act:     swiglu_fp16 (uses sve_exp_f32 internally)
 *   Add:     sve_add_fp16
 *
 * Attention kernels are in separate binaries (cache_flash_attn.cpp etc.)
 * and do NOT use this header.
 */
#pragma once
#include "common.hpp"

// ============================================================
// SVE FP16 GEMM micro-kernel: 8 rows × 3VL cols
// ============================================================
extern "C" {

void sve_interleaved_fp16_mla_8x3VL(
    const __fp16 *Apanel, const __fp16 *Bpanel, __fp16 *Cpanel,
    int ablocks, int bblocks, int K)
{
    struct KernelArgs {
        size_t Kmain; size_t Kfull;
        const __fp16 *Bpanel_base; const __fp16 *Apanel_base;
        size_t Mblocks;
    } ka;
    ka.Kmain = K - 1; ka.Kfull = K; ka.Bpanel_base = Bpanel;
    ka.Apanel_base = Apanel; ka.Mblocks = ablocks;
    int N_outer = bblocks;

    __asm__ __volatile__(
        "ptrue p0.b\n"
        "ldr x24, [%x[args], %[off_Bbase]]\n"
        "ldr x27, [%x[args], %[off_Abase]]\n"
        "1:\n"
        "ldr x23, [%x[args], %[off_Mblocks]]\n"
        "mov x21, x27\n"
        "2:\n"
        "mov x22, x24\n"
        "ldr x20, [%x[args], %[off_Kmain]]\n"
        "mov %x[Apanel], x21\n"
        "mov z8.b, #0\n"  "mov z9.b, #0\n"  "mov z10.b,#0\n" "mov z11.b,#0\n"
        "mov z12.b,#0\n"  "mov z13.b,#0\n"  "mov z14.b,#0\n" "mov z15.b,#0\n"
        "mov z16.b,#0\n"  "mov z17.b,#0\n"  "mov z18.b,#0\n" "mov z19.b,#0\n"
        "mov z20.b,#0\n"  "mov z21.b,#0\n"  "mov z22.b,#0\n" "mov z23.b,#0\n"
        "mov z24.b,#0\n"  "mov z25.b,#0\n"  "mov z26.b,#0\n" "mov z27.b,#0\n"
        "mov z28.b,#0\n"  "mov z29.b,#0\n"  "mov z30.b,#0\n" "mov z31.b,#0\n"
        "ld1h   { z2.h }, p0/Z, [x22]\n"
        "ld1h   { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
        "ld1h   { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
        "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
        "cmp x20, #0x2\n" "blt 4f\n"
        "3:\n"
        "fmla z8.h,  z2.h, z0.h[0]\n" "fmla z11.h, z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/Z, [%x[Apanel], #16]\n"
        "fmla z14.h, z2.h, z0.h[2]\n" "fmla z17.h, z2.h, z0.h[3]\n"
        "ld1h { z6.h }, p0/Z, [x22, #3, MUL VL]\n"
        "fmla z20.h, z2.h, z0.h[4]\n" "fmla z23.h, z2.h, z0.h[5]\n"
        "ld1h { z5.h }, p0/Z, [x22, #4, MUL VL]\n"
        "fmla z26.h, z2.h, z0.h[6]\n" "fmla z29.h, z2.h, z0.h[7]\n"
        "ld1h { z1.h }, p0/Z, [x22, #5, MUL VL]\n"
        "fmla z9.h,  z3.h, z0.h[0]\n" "fmla z12.h, z3.h, z0.h[1]\n"
        "addvl x22, x22, #6\n"
        "fmla z15.h, z3.h, z0.h[2]\n" "fmla z18.h, z3.h, z0.h[3]\n"
        "sub x20, x20, #0x2\n"
        "fmla z21.h, z3.h, z0.h[4]\n" "fmla z24.h, z3.h, z0.h[5]\n"
        "cmp x20, #0x2\n"
        "fmla z27.h, z3.h, z0.h[6]\n" "fmla z30.h, z3.h, z0.h[7]\n"
        "add %x[Apanel], %x[Apanel], #0x20\n"
        "fmla z10.h, z4.h, z0.h[0]\n" "fmla z13.h, z4.h, z0.h[1]\n"
        "ld1h { z2.h }, p0/Z, [x22]\n"
        "fmla z16.h, z4.h, z0.h[2]\n" "fmla z19.h, z4.h, z0.h[3]\n"
        "ld1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
        "fmla z22.h, z4.h, z0.h[4]\n" "fmla z25.h, z4.h, z0.h[5]\n"
        "fmla z28.h, z4.h, z0.h[6]\n" "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
        "fmla z8.h,  z6.h, z7.h[0]\n" "fmla z11.h, z6.h, z7.h[1]\n"
        "ld1h { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
        "fmla z14.h, z6.h, z7.h[2]\n" "fmla z17.h, z6.h, z7.h[3]\n"
        "fmla z20.h, z6.h, z7.h[4]\n" "fmla z23.h, z6.h, z7.h[5]\n"
        "fmla z26.h, z6.h, z7.h[6]\n" "fmla z29.h, z6.h, z7.h[7]\n"
        "fmla z9.h,  z5.h, z7.h[0]\n" "fmla z12.h, z5.h, z7.h[1]\n"
        "fmla z15.h, z5.h, z7.h[2]\n" "fmla z18.h, z5.h, z7.h[3]\n"
        "fmla z21.h, z5.h, z7.h[4]\n" "fmla z24.h, z5.h, z7.h[5]\n"
        "fmla z27.h, z5.h, z7.h[6]\n" "fmla z30.h, z5.h, z7.h[7]\n"
        "fmla z10.h, z1.h, z7.h[0]\n" "fmla z13.h, z1.h, z7.h[1]\n"
        "fmla z16.h, z1.h, z7.h[2]\n" "fmla z19.h, z1.h, z7.h[3]\n"
        "fmla z22.h, z1.h, z7.h[4]\n" "fmla z25.h, z1.h, z7.h[5]\n"
        "fmla z28.h, z1.h, z7.h[6]\n" "fmla z31.h, z1.h, z7.h[7]\n"
        "bge 3b\n"
        "4:\n"
        "add %x[Apanel], %x[Apanel], #0x10\n"
        "addvl x22, x22, #3\n"
        "fmla z8.h,  z2.h, z0.h[0]\n" "fmla z11.h, z2.h, z0.h[1]\n"
        "fmla z14.h, z2.h, z0.h[2]\n" "fmla z17.h, z2.h, z0.h[3]\n"
        "fmla z20.h, z2.h, z0.h[4]\n" "fmla z23.h, z2.h, z0.h[5]\n"
        "fmla z26.h, z2.h, z0.h[6]\n" "fmla z29.h, z2.h, z0.h[7]\n"
        "fmla z9.h,  z3.h, z0.h[0]\n" "fmla z12.h, z3.h, z0.h[1]\n"
        "fmla z15.h, z3.h, z0.h[2]\n" "fmla z18.h, z3.h, z0.h[3]\n"
        "fmla z21.h, z3.h, z0.h[4]\n" "fmla z24.h, z3.h, z0.h[5]\n"
        "fmla z27.h, z3.h, z0.h[6]\n" "fmla z30.h, z3.h, z0.h[7]\n"
        "fmla z10.h, z4.h, z0.h[0]\n" "fmla z13.h, z4.h, z0.h[1]\n"
        "fmla z16.h, z4.h, z0.h[2]\n" "fmla z19.h, z4.h, z0.h[3]\n"
        "fmla z22.h, z4.h, z0.h[4]\n" "fmla z25.h, z4.h, z0.h[5]\n"
        "fmla z28.h, z4.h, z0.h[6]\n" "fmla z31.h, z4.h, z0.h[7]\n"
        "cbz x20, 5f\n"
        "ld1rqh { z3.h }, p0/Z, [%x[Apanel]]\n"
        "ld1h   { z2.h }, p0/Z, [x22]\n"
        "add %x[Apanel], %x[Apanel], #0x10\n"
        "ld1h   { z1.h }, p0/Z, [x22, #1, MUL VL]\n"
        "ld1h   { z0.h }, p0/Z, [x22, #2, MUL VL]\n"
        "addvl x22, x22, #3\n"
        "fmla z8.h,  z2.h, z3.h[0]\n" "fmla z11.h, z2.h, z3.h[1]\n"
        "fmla z14.h, z2.h, z3.h[2]\n" "fmla z17.h, z2.h, z3.h[3]\n"
        "fmla z20.h, z2.h, z3.h[4]\n" "fmla z23.h, z2.h, z3.h[5]\n"
        "fmla z26.h, z2.h, z3.h[6]\n" "fmla z29.h, z2.h, z3.h[7]\n"
        "fmla z9.h,  z1.h, z3.h[0]\n" "fmla z12.h, z1.h, z3.h[1]\n"
        "fmla z15.h, z1.h, z3.h[2]\n" "fmla z18.h, z1.h, z3.h[3]\n"
        "fmla z21.h, z1.h, z3.h[4]\n" "fmla z24.h, z1.h, z3.h[5]\n"
        "fmla z27.h, z1.h, z3.h[6]\n" "fmla z30.h, z1.h, z3.h[7]\n"
        "fmla z10.h, z0.h, z3.h[0]\n" "fmla z13.h, z0.h, z3.h[1]\n"
        "fmla z16.h, z0.h, z3.h[2]\n" "fmla z19.h, z0.h, z3.h[3]\n"
        "fmla z22.h, z0.h, z3.h[4]\n" "fmla z25.h, z0.h, z3.h[5]\n"
        "fmla z28.h, z0.h, z3.h[6]\n" "fmla z31.h, z0.h, z3.h[7]\n"
        "5:\n"
        "st1h { z8.h },  p0, [%x[Cpanel]]\n" "st1h { z9.h },  p0, [%x[Cpanel], #1, MUL VL]\n"
        "st1h { z10.h }, p0, [%x[Cpanel], #2, MUL VL]\n" "st1h { z11.h }, p0, [%x[Cpanel], #3, MUL VL]\n"
        "st1h { z12.h }, p0, [%x[Cpanel], #4, MUL VL]\n" "st1h { z13.h }, p0, [%x[Cpanel], #5, MUL VL]\n"
        "st1h { z14.h }, p0, [%x[Cpanel], #6, MUL VL]\n" "st1h { z15.h }, p0, [%x[Cpanel], #7, MUL VL]\n"
        "addvl %x[Cpanel], %x[Cpanel], #16\n"
        "st1h { z16.h }, p0, [%x[Cpanel], #-8, MUL VL]\n" "st1h { z17.h }, p0, [%x[Cpanel], #-7, MUL VL]\n"
        "st1h { z18.h }, p0, [%x[Cpanel], #-6, MUL VL]\n" "st1h { z19.h }, p0, [%x[Cpanel], #-5, MUL VL]\n"
        "st1h { z20.h }, p0, [%x[Cpanel], #-4, MUL VL]\n" "st1h { z21.h }, p0, [%x[Cpanel], #-3, MUL VL]\n"
        "st1h { z22.h }, p0, [%x[Cpanel], #-2, MUL VL]\n" "st1h { z23.h }, p0, [%x[Cpanel], #-1, MUL VL]\n"
        "st1h { z24.h }, p0, [%x[Cpanel]]\n" "st1h { z25.h }, p0, [%x[Cpanel], #1, MUL VL]\n"
        "st1h { z26.h }, p0, [%x[Cpanel], #2, MUL VL]\n" "st1h { z27.h }, p0, [%x[Cpanel], #3, MUL VL]\n"
        "st1h { z28.h }, p0, [%x[Cpanel], #4, MUL VL]\n" "st1h { z29.h }, p0, [%x[Cpanel], #5, MUL VL]\n"
        "st1h { z30.h }, p0, [%x[Cpanel], #6, MUL VL]\n" "st1h { z31.h }, p0, [%x[Cpanel], #7, MUL VL]\n"
        "addvl %x[Cpanel], %x[Cpanel], #8\n"
        "mov x21, %x[Apanel]\n"
        "subs x23, x23, #1\n"
        "bgt 2b\n"
        "ldr x26, [%x[args], %[off_Kfull]]\n"
        "cntb x25\n" "mul x25, x25, x26\n" "add x25, x25, x25, lsl #1\n"
        "add x24, x24, x25\n"
        "subs %x[Nouter], %x[Nouter], #1\n"
        "bne 1b\n"
        : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [Nouter] "+&r" (N_outer)
        : [args] "r" (&ka),
          [off_Kmain]  "I" (offsetof(KernelArgs, Kmain)),
          [off_Kfull]  "I" (offsetof(KernelArgs, Kfull)),
          [off_Bbase]  "I" (offsetof(KernelArgs, Bpanel_base)),
          [off_Abase]  "I" (offsetof(KernelArgs, Apanel_base)),
          [off_Mblocks]"I" (offsetof(KernelArgs, Mblocks))
        : "cc", "memory", "p0",
          "x20","x21","x22","x23","x24","x25","x26","x27",
          "z0","z1","z2","z3","z4","z5","z6","z7",
          "z8","z9","z10","z11","z12","z13","z14","z15",
          "z16","z17","z18","z19","z20","z21","z22","z23",
          "z24","z25","z26","z27","z28","z29","z30","z31"
    );
}

void sve_pack_B_3VL(uint16_t *out, const uint16_t *in,
                    size_t width, size_t in_stride_bytes, size_t height)
{
    size_t out_stride = 3 * height * svcntb();
    __asm__ __volatile__(
        "cmp %x[height], #0x4\n" "ptrue p2.b\n" "blt 4f\n"
        "1:\n" "mov x27, %x[in]\n" "mov x26, %x[out]\n" "sub %x[height], %x[height], #0x4\n"
        "mov x25, %x[width]\n" "add x24, x27, %x[in_stride]\n" "add x23, x24, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n" "add %x[in], x22, %x[in_stride]\n"
        "2:\n" "mov x21, x25\n" "mov x20, x26\n" "dech x25, ALL, MUL #3\n"
        "add x26, x26, %x[out_stride]\n" "whilelt p0.h, XZR, x21\n" "dech x21\n"
        "whilelt p1.h, XZR, x21\n" "dech x21\n"
        "ld1h { z27.h }, p0/Z, [x27]\n" "ld1h { z26.h }, p0/Z, [x24]\n"
        "ld1h { z25.h }, p0/Z, [x23]\n" "ld1h { z24.h }, p0/Z, [x22]\n"
        "whilelt p0.h, XZR, x21\n" "cmp x25, #0x0\n"
        "ld1h { z23.h }, p1/Z, [x27, #1, MUL VL]\n" "ld1h { z22.h }, p1/Z, [x24, #1, MUL VL]\n"
        "ld1h { z21.h }, p1/Z, [x23, #1, MUL VL]\n" "ld1h { z20.h }, p1/Z, [x22, #1, MUL VL]\n"
        "ld1h { z19.h }, p0/Z, [x27, #2, MUL VL]\n" "ld1h { z18.h }, p0/Z, [x24, #2, MUL VL]\n"
        "addvl x27, x27, #3\n" "addvl x24, x24, #3\n"
        "ld1h { z17.h }, p0/Z, [x23, #2, MUL VL]\n" "ld1h { z16.h }, p0/Z, [x22, #2, MUL VL]\n"
        "st1h { z27.h }, p2, [x20]\n" "addvl x23, x23, #3\n"
        "st1h { z23.h }, p2, [x20, #1, MUL VL]\n" "addvl x22, x22, #3\n"
        "st1h { z19.h }, p2, [x20, #2, MUL VL]\n" "st1h { z26.h }, p2, [x20, #3, MUL VL]\n"
        "st1h { z22.h }, p2, [x20, #4, MUL VL]\n" "st1h { z18.h }, p2, [x20, #5, MUL VL]\n"
        "st1h { z25.h }, p2, [x20, #6, MUL VL]\n" "st1h { z21.h }, p2, [x20, #7, MUL VL]\n"
        "addvl x20, x20, #12\n"
        "st1h { z17.h }, p2, [x20, #-4, MUL VL]\n" "st1h { z24.h }, p2, [x20, #-3, MUL VL]\n"
        "st1h { z20.h }, p2, [x20, #-2, MUL VL]\n" "st1h { z16.h }, p2, [x20, #-1, MUL VL]\n"
        "bgt 2b\n" "3:\n" "cmp %x[height], #0x4\n" "addvl %x[out], %x[out], #12\n" "bge 1b\n"
        "cbz %x[height], 8f\n" "4:\n" "5:\n"
        "mov x27, %x[in]\n" "mov x26, %x[out]\n" "sub %x[height], %x[height], #0x1\n"
        "mov x21, %x[width]\n" "add %x[in], x27, %x[in_stride]\n" "6:\n"
        "mov x20, x21\n" "dech x21, ALL, MUL #3\n"
        "whilelt p0.h, XZR, x20\n" "dech x20\n" "whilelt p1.h, XZR, x20\n" "dech x20\n"
        "ld1h { z18.h }, p0/Z, [x27]\n" "whilelt p0.h, XZR, x20\n" "cmp x21, #0x0\n"
        "ld1h { z17.h }, p1/Z, [x27, #1, MUL VL]\n" "ld1h { z16.h }, p0/Z, [x27, #2, MUL VL]\n"
        "addvl x27, x27, #3\n"
        "st1h { z18.h }, p2, [x26]\n" "st1h { z17.h }, p2, [x26, #1, MUL VL]\n"
        "st1h { z16.h }, p2, [x26, #2, MUL VL]\n" "add x26, x26, %x[out_stride]\n"
        "bgt 6b\n" "7:\n" "cmp %x[height], #0x1\n" "addvl %x[out], %x[out], #3\n" "bge 5b\n" "8:\n"
        : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
        : [in_stride] "r" (in_stride_bytes), [out_stride] "r" (out_stride), [width] "r" (width)
        : "cc", "memory", "p0", "p1", "p2",
          "x20","x21","x22","x23","x24","x25","x26","x27",
          "z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27"
    );
}

} // extern "C"

// ============================================================
// Pack A: 3-level zip butterfly interleave → 8-row k-major
// ============================================================
static void pack_A_fp16_8row(const __fp16* A, size_t lda, __fp16* Apanel,
                              size_t M, size_t m0, size_t K, size_t k0,
                              size_t MC, size_t KC)
{
    const size_t MT  = 8, VLh = svcnth();
    size_t rows = std::min(MC, M - m0), kc = std::min(KC, K - k0);
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
            svst1_f16(pg,dst+ki*MT+0*VLh,svreinterpret_f16_f64(d0));
            svst1_f16(pg,dst+ki*MT+1*VLh,svreinterpret_f16_f64(d1));
            svst1_f16(pg,dst+ki*MT+2*VLh,svreinterpret_f16_f64(d2));
            svst1_f16(pg,dst+ki*MT+3*VLh,svreinterpret_f16_f64(d3));
            svst1_f16(pg,dst+ki*MT+4*VLh,svreinterpret_f16_f64(d4));
            svst1_f16(pg,dst+ki*MT+5*VLh,svreinterpret_f16_f64(d5));
            svst1_f16(pg,dst+ki*MT+6*VLh,svreinterpret_f16_f64(d6));
            svst1_f16(pg,dst+ki*MT+7*VLh,svreinterpret_f16_f64(d7));
        }
        for (; ki < kc; ++ki)
            for (size_t r = 0; r < MT; ++r)
                dst[ki*MT + r] = rp[r][ki];
    }
}

// ============================================================
// Prepack B[K,N] → KNM interleaved layout [K_tiles × N_tiles × KC × NT]
// ============================================================
static size_t prepack_B_knm_size(size_t K, size_t N, size_t KC) {
    const size_t NT = 3 * svcnth();
    return ((K+KC-1)/KC) * ((N+NT-1)/NT) * KC * NT;
}

static void prepack_B_knm(const __fp16* B, __fp16* B_packed, size_t K, size_t N, size_t KC,
                           size_t ldb = 0) {
    const size_t row_stride = ldb ? ldb : N;
    const size_t NT = 3*svcnth(), N_tiles=(N+NT-1)/NT;
    for (size_t k0=0; k0<K; k0+=KC) {
        size_t kc=std::min(KC,K-k0), kt=k0/KC;
        for (size_t n0=0; n0<N; n0+=NT) {
            size_t nc=std::min(NT,N-n0), nt=n0/NT;
            __fp16* dst=B_packed+(kt*N_tiles+nt)*KC*NT;
            sve_pack_B_3VL((uint16_t*)dst,(const uint16_t*)(B+k0*row_stride+n0),
                           nc, row_stride*sizeof(uint16_t), kc);
        }
    }
}

// ============================================================
// SVE exp (5th-order polynomial, used by swiglu)
// ============================================================
static inline svfloat32_t sve_exp_f32(svbool_t pg, svfloat32_t x)
{
    x = svmax_n_f32_x(pg, x, -88.0f);
    svfloat32_t nf = svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.44269504088896f));
    svint32_t   ni = svcvt_s32_f32_x(pg, nf);
    svfloat32_t r  = svmls_n_f32_x(pg, x, nf,  6.93147182e-1f);
    r              = svmls_n_f32_x(pg, r, nf, -1.90465430e-9f);
    svfloat32_t p  = svdup_n_f32(1.0f/120.0f);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f/24.0f),r,p);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f/6.0f), r,p);
    p = svmla_f32_x(pg,svdup_n_f32(0.5f),       r,p);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f),       r,p);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f),       r,p);
    return svscale_f32_x(pg,p,ni);
}

// ============================================================
// Element-wise add: c = a + b
// ============================================================
static void sve_add_fp16(const __fp16* a,const __fp16* b,__fp16* c,size_t n) {
    const size_t VL=svcnth(); const svbool_t pg=svptrue_b16();
    size_t i=0;
    for(;i+4*VL<=n;i+=4*VL){
        svst1_f16(pg,c+i,      svadd_f16_x(pg,svld1_f16(pg,a+i),      svld1_f16(pg,b+i)));
        svst1_f16(pg,c+i+VL,   svadd_f16_x(pg,svld1_f16(pg,a+i+VL),   svld1_f16(pg,b+i+VL)));
        svst1_f16(pg,c+i+2*VL, svadd_f16_x(pg,svld1_f16(pg,a+i+2*VL), svld1_f16(pg,b+i+2*VL)));
        svst1_f16(pg,c+i+3*VL, svadd_f16_x(pg,svld1_f16(pg,a+i+3*VL), svld1_f16(pg,b+i+3*VL)));}
    while(i<n){svbool_t pt=svwhilelt_b16_u64(i,n);svst1_f16(pt,c+i,svadd_f16_x(pt,svld1_f16(pt,a+i),svld1_f16(pt,b+i)));i+=VL;}
}

// ============================================================
// RMSNorm: y = x / rms(x) * w   (LLaMA-style, no bias)
// ============================================================
static void rmsnorm_row(const __fp16* __restrict__ x, const __fp16* __restrict__ w,
                        __fp16* __restrict__ y, int D, float eps)
{
    const size_t VL32=svcntw();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    size_t d=0;
    for(;d+4*VL32<=(size_t)D;d+=4*VL32){
        svbool_t pg=svptrue_b32();
        svfloat32_t f0=svcvt_f32_f16_z(pg,svld1_f16(svwhilelt_b16_u64(d,      d+  VL32),x+d));
        svfloat32_t f1=svcvt_f32_f16_z(pg,svld1_f16(svwhilelt_b16_u64(d+VL32, d+2*VL32),x+d+VL32));
        svfloat32_t f2=svcvt_f32_f16_z(pg,svld1_f16(svwhilelt_b16_u64(d+2*VL32,d+3*VL32),x+d+2*VL32));
        svfloat32_t f3=svcvt_f32_f16_z(pg,svld1_f16(svwhilelt_b16_u64(d+3*VL32,d+4*VL32),x+d+3*VL32));
        a0=svmla_f32_x(pg,a0,f0,f0); a1=svmla_f32_x(pg,a1,f1,f1);
        a2=svmla_f32_x(pg,a2,f2,f2); a3=svmla_f32_x(pg,a3,f3,f3);}
    for(;d<(size_t)D;d+=VL32){
        svbool_t pg32=svwhilelt_b32_u64(d,(size_t)D);
        svfloat32_t f=svcvt_f32_f16_z(pg32,svld1_f16(svwhilelt_b16_u64(d,(size_t)D),x+d));
        a0=svmla_f32_m(pg32,a0,f,f);}
    float ss=svaddv_f32(svptrue_b32(),svadd_f32_x(svptrue_b32(),svadd_f32_x(svptrue_b32(),a0,a1),svadd_f32_x(svptrue_b32(),a2,a3)));
    float scale=1.0f/sqrtf(ss/D+eps);
    svfloat32_t vscale=svdup_f32(scale);
    for(d=0;d+4*VL32<=(size_t)D;d+=4*VL32){
        svbool_t pg32=svptrue_b32();
        svbool_t p0=svwhilelt_b16_u64(d,      d+  VL32);
        svbool_t p1=svwhilelt_b16_u64(d+VL32, d+2*VL32);
        svbool_t p2=svwhilelt_b16_u64(d+2*VL32,d+3*VL32);
        svbool_t p3=svwhilelt_b16_u64(d+3*VL32,d+4*VL32);
        svfloat32_t xf0=svcvt_f32_f16_z(pg32,svld1_f16(p0,x+d));
        svfloat32_t xf1=svcvt_f32_f16_z(pg32,svld1_f16(p1,x+d+VL32));
        svfloat32_t xf2=svcvt_f32_f16_z(pg32,svld1_f16(p2,x+d+2*VL32));
        svfloat32_t xf3=svcvt_f32_f16_z(pg32,svld1_f16(p3,x+d+3*VL32));
        svfloat32_t wf0=svcvt_f32_f16_z(pg32,svld1_f16(p0,w+d));
        svfloat32_t wf1=svcvt_f32_f16_z(pg32,svld1_f16(p1,w+d+VL32));
        svfloat32_t wf2=svcvt_f32_f16_z(pg32,svld1_f16(p2,w+d+2*VL32));
        svfloat32_t wf3=svcvt_f32_f16_z(pg32,svld1_f16(p3,w+d+3*VL32));
        svst1_f16(p0,y+d,      svcvt_f16_f32_z(pg32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf0,wf0),vscale)));
        svst1_f16(p1,y+d+VL32, svcvt_f16_f32_z(pg32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf1,wf1),vscale)));
        svst1_f16(p2,y+d+2*VL32,svcvt_f16_f32_z(pg32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf2,wf2),vscale)));
        svst1_f16(p3,y+d+3*VL32,svcvt_f16_f32_z(pg32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf3,wf3),vscale)));}
    for(;d+VL32<=(size_t)D;d+=VL32){
        svbool_t pg32=svptrue_b32(); svbool_t pg16=svwhilelt_b16_u64(d,d+VL32);
        svfloat32_t xf=svcvt_f32_f16_z(pg32,svld1_f16(pg16,x+d));
        svfloat32_t wf=svcvt_f32_f16_z(pg32,svld1_f16(pg16,w+d));
        svst1_f16(pg16,y+d,svcvt_f16_f32_z(pg32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf,wf),vscale)));}
    for(;d<(size_t)D;++d) y[d]=(__fp16)((float)x[d]*scale*(float)w[d]);
}

static void rmsnorm(const __fp16* x, const __fp16* w, __fp16* y, int T, int D) {
    for(int t=0;t<T;++t) rmsnorm_row(x+t*D,w,y+t*D,D,1e-5f);
}

// ============================================================
// RoPE: LLaMA-style rotary position embedding
// ============================================================
static void build_cos_sin(std::vector<float>& cos_t, std::vector<float>& sin_t, int T, int d) {
    int half=d/2; cos_t.resize(T*half); sin_t.resize(T*half);
    for(int t=0;t<T;++t)
        for(int i=0;i<half;++i){
            float theta=t*powf(10000.f,-2.f*i/d);
            cos_t[t*half+i]=cosf(theta); sin_t[t*half+i]=sinf(theta);}
}

static void apply_rope(__fp16* __restrict__ x, const float* __restrict__ cos_t,
                       const float* __restrict__ sin_t, int H, int T, int d,
                       int tok_stride = 0, int head_stride = 0)
{
    if (!tok_stride)  tok_stride  = d;
    if (!head_stride) head_stride = T*d;
    const int half=d/2; const size_t VL32=svcntw();
    for(int h=0;h<H;++h)
        for(int t=0;t<T;++t){
            __fp16* row=x + (size_t)h*head_stride + (size_t)t*tok_stride;
            const float* c=cos_t+t*half; const float* s=sin_t+t*half;
            size_t i=0;
            for(;i+2*VL32<=(size_t)half;i+=2*VL32){
                svbool_t pg32=svptrue_b32();
                svbool_t pg16a=svwhilelt_b16_u64(i,      i+  VL32);
                svbool_t pg16b=svwhilelt_b16_u64(i+VL32, i+2*VL32);
                svfloat32_t xla=svcvt_f32_f16_z(pg32,svld1_f16(pg16a,row+i));
                svfloat32_t xha=svcvt_f32_f16_z(pg32,svld1_f16(pg16a,row+i+half));
                svfloat32_t ca=svld1_f32(pg32,c+i), sa=svld1_f32(pg32,s+i);
                svfloat32_t xlb=svcvt_f32_f16_z(pg32,svld1_f16(pg16b,row+i+VL32));
                svfloat32_t xhb=svcvt_f32_f16_z(pg32,svld1_f16(pg16b,row+i+VL32+half));
                svfloat32_t cb=svld1_f32(pg32,c+i+VL32), sb=svld1_f32(pg32,s+i+VL32);
                svst1_f16(pg16a,row+i,            svcvt_f16_f32_z(pg32,svmls_f32_x(pg32,svmul_f32_x(pg32,xla,ca),xha,sa)));
                svst1_f16(pg16a,row+i+half,        svcvt_f16_f32_z(pg32,svmla_f32_x(pg32,svmul_f32_x(pg32,xha,ca),xla,sa)));
                svst1_f16(pg16b,row+i+VL32,        svcvt_f16_f32_z(pg32,svmls_f32_x(pg32,svmul_f32_x(pg32,xlb,cb),xhb,sb)));
                svst1_f16(pg16b,row+i+VL32+half,   svcvt_f16_f32_z(pg32,svmla_f32_x(pg32,svmul_f32_x(pg32,xhb,cb),xlb,sb)));}
            for(;i+VL32<=(size_t)half;i+=VL32){
                svbool_t pg32=svptrue_b32(); svbool_t pg16=svwhilelt_b16_u64(i,i+VL32);
                svfloat32_t xl=svcvt_f32_f16_z(pg32,svld1_f16(pg16,row+i));
                svfloat32_t xh=svcvt_f32_f16_z(pg32,svld1_f16(pg16,row+i+half));
                svfloat32_t cv=svld1_f32(pg32,c+i), sv_=svld1_f32(pg32,s+i);
                svst1_f16(pg16,row+i,      svcvt_f16_f32_z(pg32,svmls_f32_x(pg32,svmul_f32_x(pg32,xl,cv),xh,sv_)));
                svst1_f16(pg16,row+i+half, svcvt_f16_f32_z(pg32,svmla_f32_x(pg32,svmul_f32_x(pg32,xh,cv),xl,sv_)));}
            for(;i<(size_t)half;++i){
                float xi=(float)row[i],xi2=(float)row[i+half];
                row[i]=(__fp16)(xi*c[i]-xi2*s[i]); row[i+half]=(__fp16)(xi2*c[i]+xi*s[i]);}
        }
}

// ============================================================
// GELU: buf[i] = x * sigmoid(1.702 * x)  (sigmoid approximation)
// Same structure as SiLU but with scaled input. In-place.
// ============================================================
static void gelu_fp16_inplace(__fp16* buf, size_t n)
{
    const float GELU_SCALE = 1.702f;
    const size_t VL32 = svcntw();
    size_t i = 0;
    for (; i + 2*VL32 <= n; i += 2*VL32) {
        svbool_t pg32  = svptrue_b32();
        svbool_t pg16a = svwhilelt_b16_u64(i,      i+VL32);
        svbool_t pg16b = svwhilelt_b16_u64(i+VL32, i+2*VL32);
        svfloat32_t x0 = svcvt_f32_f16_z(pg32, svld1_f16(pg16a, buf+i));
        svfloat32_t x1 = svcvt_f32_f16_z(pg32, svld1_f16(pg16b, buf+i+VL32));
        // exp(-1.702*x)
        svfloat32_t nx0 = svmin_n_f32_x(pg32, svneg_f32_x(pg32, svmul_n_f32_x(pg32, x0, GELU_SCALE)), 88.0f);
        svfloat32_t nx1 = svmin_n_f32_x(pg32, svneg_f32_x(pg32, svmul_n_f32_x(pg32, x1, GELU_SCALE)), 88.0f);
        svfloat32_t e0 = sve_exp_f32(pg32, nx0);
        svfloat32_t e1 = sve_exp_f32(pg32, nx1);
        // 1/(1+e)
        svfloat32_t d0 = svadd_n_f32_x(pg32, e0, 1.0f);
        svfloat32_t d1 = svadd_n_f32_x(pg32, e1, 1.0f);
        svfloat32_t r0 = svrecpe_f32(d0);
        svfloat32_t r1 = svrecpe_f32(d1);
        r0 = svmul_f32_x(pg32, svrecps_f32(d0, r0), r0);
        r1 = svmul_f32_x(pg32, svrecps_f32(d1, r1), r1);
        // x * sigmoid(1.702*x)
        svst1_f16(pg16a, buf+i,      svcvt_f16_f32_z(pg32, svmul_f32_x(pg32, x0, r0)));
        svst1_f16(pg16b, buf+i+VL32, svcvt_f16_f32_z(pg32, svmul_f32_x(pg32, x1, r1)));
    }
    for (; i < n; i += VL32) {
        svbool_t pg32 = svwhilelt_b32_u64(i, n);
        svbool_t pg16 = svwhilelt_b16_u64(i, std::min(i+VL32, n));
        svfloat32_t x = svcvt_f32_f16_z(pg32, svld1_f16(pg16, buf+i));
        svfloat32_t nx = svmin_n_f32_x(pg32, svneg_f32_x(pg32, svmul_n_f32_x(pg32, x, GELU_SCALE)), 88.0f);
        svfloat32_t e = sve_exp_f32(pg32, nx);
        svfloat32_t d = svadd_n_f32_x(pg32, e, 1.0f);
        svfloat32_t r = svrecpe_f32(d);
        r = svmul_f32_x(pg32, svrecps_f32(d, r), r);
        svst1_f16(pg16, buf+i, svcvt_f16_f32_z(pg32, svmul_f32_x(pg32, x, r)));
    }
}

// ============================================================
// SwiGLU: gate[i] = SiLU(gate[i]) * up[i]
// ============================================================
static void swiglu_fp16(__fp16* __restrict__ gate, const __fp16* __restrict__ up, size_t n)
{
    const size_t VL32 = svcntw();
    size_t i = 0;
    for (; i + 2*VL32 <= n; i += 2*VL32) {
        svbool_t pg32  = svptrue_b32();
        svbool_t pg16a = svwhilelt_b16_u64(i,      i+VL32);
        svbool_t pg16b = svwhilelt_b16_u64(i+VL32, i+2*VL32);
        svfloat32_t g0 = svcvt_f32_f16_z(pg32, svld1_f16(pg16a, gate+i));
        svfloat32_t g1 = svcvt_f32_f16_z(pg32, svld1_f16(pg16b, gate+i+VL32));
        svfloat32_t u0 = svcvt_f32_f16_z(pg32, svld1_f16(pg16a, up+i));
        svfloat32_t u1 = svcvt_f32_f16_z(pg32, svld1_f16(pg16b, up+i+VL32));
        svfloat32_t ng0 = svmin_n_f32_x(pg32, svneg_f32_x(pg32, g0), 88.0f);
        svfloat32_t ng1 = svmin_n_f32_x(pg32, svneg_f32_x(pg32, g1), 88.0f);
        svfloat32_t e0  = sve_exp_f32(pg32, ng0);
        svfloat32_t e1  = sve_exp_f32(pg32, ng1);
        svfloat32_t d0  = svadd_n_f32_x(pg32, e0, 1.0f);
        svfloat32_t d1  = svadd_n_f32_x(pg32, e1, 1.0f);
        svfloat32_t r0  = svrecpe_f32(d0);
        svfloat32_t r1  = svrecpe_f32(d1);
        r0 = svmul_f32_x(pg32, svrecps_f32(d0, r0), r0);
        r1 = svmul_f32_x(pg32, svrecps_f32(d1, r1), r1);
        svst1_f16(pg16a, gate+i,      svcvt_f16_f32_z(pg32, svmul_f32_x(pg32, svmul_f32_x(pg32, g0, r0), u0)));
        svst1_f16(pg16b, gate+i+VL32, svcvt_f16_f32_z(pg32, svmul_f32_x(pg32, svmul_f32_x(pg32, g1, r1), u1)));
    }
    for (; i < n; i += VL32) {
        svbool_t pg32 = svwhilelt_b32_u64(i, n);
        svbool_t pg16 = svwhilelt_b16_u64(i, std::min(i+VL32, n));
        svfloat32_t g  = svcvt_f32_f16_z(pg32, svld1_f16(pg16, gate+i));
        svfloat32_t u  = svcvt_f32_f16_z(pg32, svld1_f16(pg16, up+i));
        svfloat32_t ng = svmin_n_f32_x(pg32, svneg_f32_x(pg32, g), 88.0f);
        svfloat32_t e  = sve_exp_f32(pg32, ng);
        svfloat32_t d  = svadd_n_f32_x(pg32, e, 1.0f);
        svfloat32_t r  = svrecpe_f32(d);
        r = svmul_f32_x(pg32, svrecps_f32(d, r), r);
        svst1_f16(pg16, gate+i, svcvt_f16_f32_z(pg32, svmul_f32_x(pg32, svmul_f32_x(pg32, g, r), u)));
    }
}

