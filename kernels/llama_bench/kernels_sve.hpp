// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/* SVE FP16 GEMM and transformer kernels. */
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
// Pack A (3-level zip butterfly), pack_K_direct, gemm helpers
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

// Optional accumulator for a per-call GEMM timing breakdown.
struct GemmTiming {
    double t_packA     = 0;   // pack_A_fp16_8row total
    double t_compute   = 0;   // sve_interleaved_fp16_mla_8x3VL total
    double t_writeback = 0;   // Cpanel_tmp → C scatter total (c_write + c_rmw)
    double t_c_write   = 0;   // C scatter k0==0 (pure store, 1 pass)
    double t_c_rmw     = 0;   // C scatter k0>0  (load+add+store, K_tiles-1 passes)
    void reset() { t_packA = t_compute = t_writeback = t_c_write = t_c_rmw = 0; }
};

static size_t prepack_B_knm_size(size_t K, size_t N, size_t KC) {
    const size_t NT = 3 * svcnth();
    return ((K+KC-1)/KC) * ((N+NT-1)/NT) * KC * NT;
}

static void prepack_B_knm(const __fp16* B, __fp16* B_packed, size_t K, size_t N, size_t KC,
                           size_t ldb = 0) {
    const size_t row_stride = ldb ? ldb : N;   // stride (elements) between B rows
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

static void gemm_knm_prepacked(const __fp16* A, const __fp16* B_packed, __fp16* C,
                                size_t M, size_t N, size_t K,
                                size_t MC=128, size_t KC=256,
                                GemmTiming* tim=nullptr)
{
    const size_t MT=8, NT=3*svcnth(), N_tiles=(N+NT-1)/NT;
    const size_t num_mc=(M+MC-1)/MC, max_ab=(MC+MT-1)/MT;
    AlignedBuffer<__fp16> A_cache(num_mc*max_ab*MT*KC);
    AlignedBuffer<__fp16> Cpanel_tmp(max_ab*MT*NT);

    // Collect the per-call timing printed below.
    static int _gemm_call_id = 0;
    int _this_call = _gemm_call_id++;
    double _t_packA=0, _t_kernel=0, _t_c_write=0, _t_c_rmw=0;
    auto _now = []() { return Clock::now(); };

    for (size_t k0=0; k0<K; k0+=KC) {
        size_t kc=std::min(KC,K-k0), kt=k0/KC;
        for (size_t m0=0; m0<M; m0+=MC) {
            size_t mc=std::min(MC,M-m0);
            auto _ta=_now();
            pack_A_fp16_8row(A,K,A_cache.data()+(m0/MC)*max_ab*MT*KC,M,m0,K,k0,mc,kc);
            double dt=us_since(_ta); _t_packA+=dt;
            if(tim) tim->t_packA+=dt;
        }
        for (size_t n0=0; n0<N; n0+=NT) {
            size_t nc=std::min(NT,N-n0), nt=n0/NT;
            const __fp16* Bp=B_packed+(kt*N_tiles+nt)*KC*NT;
            for (size_t m0=0; m0<M; m0+=MC) {
                size_t mc=std::min(MC,M-m0), ablocks=(mc+MT-1)/MT;
                {
                    auto _tc=_now();
                    sve_interleaved_fp16_mla_8x3VL(A_cache.data()+(m0/MC)*max_ab*MT*KC,
                        Bp,Cpanel_tmp.data(),(int)ablocks,1,(int)kc);
                    double dt=us_since(_tc); _t_kernel+=dt;
                    if(tim) tim->t_compute+=dt;
                }
                {
                    auto _tw=_now();
                    for (size_t mb=0; mb<ablocks; ++mb) {
                        const __fp16* tp=Cpanel_tmp.data()+mb*MT*NT;
                        for (size_t r=0; r<MT; ++r) {
                            size_t gr=m0+mb*MT+r; if (gr>=M) break;
                            __fp16* c_row=C+gr*N+n0; const __fp16* t_row=tp+r*NT;
                            size_t i=0; svbool_t pg=svptrue_b16();
                            if (k0==0) {
                                for(;i+svcnth()<=nc;i+=svcnth()) svst1_f16(pg,c_row+i,svld1_f16(pg,t_row+i));
                                if(i<nc){svbool_t pt=svwhilelt_b16_u64(i,nc);svst1_f16(pt,c_row+i,svld1_f16(pt,t_row+i));}
                            } else {
                                for(;i+svcnth()<=nc;i+=svcnth()) svst1_f16(pg,c_row+i,svadd_f16_x(pg,svld1_f16(pg,c_row+i),svld1_f16(pg,t_row+i)));
                                if(i<nc){svbool_t pt=svwhilelt_b16_u64(i,nc);svst1_f16(pt,c_row+i,svadd_f16_x(pt,svld1_f16(pt,c_row+i),svld1_f16(pt,t_row+i)));}
                            }
                        }
                    }
                    double dt=us_since(_tw);
                    if (k0==0) { _t_c_write+=dt; if(tim) tim->t_c_write+=dt; }
                    else       { _t_c_rmw  +=dt; if(tim) tim->t_c_rmw  +=dt; }
                    if(tim) tim->t_writeback+=dt;
                }
            }
        }
    }

    // Emit the per-call timing breakdown.
    {
        size_t K_tiles=(K+KC-1)/KC, N_tiles_n=(N+NT-1)/NT;
        double t_total=_t_packA+_t_kernel+_t_c_write+_t_c_rmw;
        printf("[gemm_nonspm #%d] M=%zu K=%zu N=%zu  KC=%zu MC=%zu NT=%zu"
               "  K_tiles=%zu N_tiles=%zu\n",
               _this_call, M, K, N, KC, MC, NT, K_tiles, N_tiles_n);
        printf("  packA:        %8.1f us  (%5.1f%%)\n", _t_packA,   100.*_t_packA/t_total);
        printf("  kernel:       %8.1f us  (%5.1f%%)  [B loaded from cache during exec]\n",
               _t_kernel,  100.*_t_kernel/t_total);
        printf("  C write(k0=0):%8.1f us  (%5.1f%%)  [pure store, 1 pass]\n",
               _t_c_write, 100.*_t_c_write/t_total);
        printf("  C RMW(k0>0): %8.1f us  (%5.1f%%)  [load+add+store, %zu passes]\n",
               _t_c_rmw,   100.*_t_c_rmw/t_total, K_tiles > 0 ? K_tiles-1 : 0);
        printf("  total:        %8.1f us\n", t_total);
        fflush(stdout);
    }
}

// pack_K_direct: SVE gather-based transpose+pack K[T,d] → packed[d, NT] per NT-block.
// Gather byte-offset vector [0, ldk*2, ..., (VLw-1)*ldk*2] addresses consecutive
// token rows for a fixed column k. Two gathers + uzp1 (extracts low u16 of each
// zero-extended u32 lane) → 1 VLh sequential store. Mirrors sve_pack_B_3VL's
// column-load strategy applied to the transposed access pattern.
static void pack_K_direct(uint16_t* out, const uint16_t* K_in, int T, int d, size_t NT, int ldk=0)
{
    if (!ldk) ldk=d;
    const size_t VLw     = svcntw();  // 8 at VL=256-bit
    const size_t VLh     = svcnth();  // 16 = 2×VLw
    const size_t bblocks = ((size_t)T+NT-1)/NT;

    const uint32_t bstride = (uint32_t)(ldk * sizeof(uint16_t));
    const svuint32_t voff  = svmul_n_u32_x(svptrue_b32(), svindex_u32(0u,1u), bstride);
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();

    for (size_t nb=0; nb<bblocks; ++nb) {
        uint16_t*       dst  = out   + nb*NT*(size_t)d;
        const uint16_t* Ksub = K_in  + nb*NT*(size_t)ldk;
        size_t nc = std::min(NT,(size_t)T-nb*NT);

        std::memset(dst, 0, NT*(size_t)d*sizeof(uint16_t));

        for (int k=0; k<d; ++k) {
            uint16_t*       row_out  = dst  + (size_t)k*NT;
            const uint16_t* col_base = Ksub + (size_t)k;

            size_t n=0;
            // 2×VLw per iter: 2 gathers → uzp1 → VLh store
            for (; n+VLh<=nc; n+=VLh) {
                svuint32_t g0=svld1uh_gather_u32offset_u32(pg32,
                    col_base+ n     *ldk, voff);
                svuint32_t g1=svld1uh_gather_u32offset_u32(pg32,
                    col_base+(n+VLw)*ldk, voff);
                svst1_u16(pg16, row_out+n,
                    svuzp1_u16(svreinterpret_u16_u32(g0), svreinterpret_u16_u32(g1)));
            }
            // One remaining full VLw group
            if (n+VLw<=nc) {
                svuint32_t g=svld1uh_gather_u32offset_u32(pg32,
                    col_base+n*ldk, voff);
                svst1_u16(svwhilelt_b16_u64((size_t)0,VLw), row_out+n,
                    svuzp1_u16(svreinterpret_u16_u32(g), svdup_n_u16(0)));
                n+=VLw;
            }
            for (; n<nc; ++n) row_out[n]=Ksub[n*ldk+k];
        }
    }
}

static void gemm_prepacked_B(const __fp16* A, const __fp16* B_packed, __fp16* C,
                              size_t M, size_t N, size_t K,
                              size_t MC=128, size_t KC=256, size_t lda=0,
                              GemmTiming* tim=nullptr, float scale=1.0f)
{
    if (!lda) lda=K;
    const size_t MT=8, NT=3*svcntb()/2, kc=std::min(KC,K);
    size_t bblocks_total=(N+NT-1)/NT, max_ablocks=(MC+MT-1)/MT;
    AlignedBuffer<__fp16> Apanel(max_ablocks*MT*kc);
    AlignedBuffer<__fp16> Cpanel_tmp(max_ablocks*MT*bblocks_total*NT);

    // Collect timing for this single-K-block call.
    static int _gemm_call_id = 0;
    int _this_call = _gemm_call_id++;
    double _t_packA=0, _t_kernel=0, _t_c_write=0;
    auto _now = []() { return Clock::now(); };

    for (size_t m0=0; m0<M; m0+=MC) {
        size_t mc=std::min(MC,M-m0), ablocks=(mc+MT-1)/MT;
        {
            auto _ta=_now();
            pack_A_fp16_8row(A,lda,Apanel.data(),M,m0,K,0,mc,kc);
            double dt=us_since(_ta); _t_packA+=dt;
            if(tim) tim->t_packA+=dt;
        }
        {
            auto _tc=_now();
            sve_interleaved_fp16_mla_8x3VL(Apanel.data(),B_packed,Cpanel_tmp.data(),(int)ablocks,(int)bblocks_total,(int)kc);
            double dt=us_since(_tc); _t_kernel+=dt;
            if(tim) tim->t_compute+=dt;
        }
        // SVE scatter Cpanel_tmp → C row-major, inline scale (always k0==0: pure write, no RMW)
        {
            auto _tw=_now();
            const svfloat16_t vs = svdup_f16((__fp16)scale);
            for (size_t nb=0; nb<bblocks_total; ++nb) {
                size_t n0=nb*NT, nc=std::min(NT,N-n0);
                for (size_t mb=0; mb<ablocks; ++mb) {
                    const __fp16* tp=Cpanel_tmp.data()+(nb*ablocks+mb)*MT*NT;
                    for (size_t r=0; r<MT; ++r) {
                        size_t gr=m0+mb*MT+r; if(gr>=M) break;
                        __fp16* c_row=C+gr*N+n0; const __fp16* t_row=tp+r*NT;
                        size_t i=0; svbool_t pg=svptrue_b16();
                        for(;i+svcnth()<=nc;i+=svcnth()) svst1_f16(pg,c_row+i,svmul_f16_x(pg,svld1_f16(pg,t_row+i),vs));
                        if(i<nc){svbool_t pt=svwhilelt_b16_u64(i,nc);svst1_f16(pt,c_row+i,svmul_f16_x(pt,svld1_f16(pt,t_row+i),vs));}
                    }
                }
            }
            double dt=us_since(_tw); _t_c_write+=dt;
            if(tim) { tim->t_c_write+=dt; tim->t_writeback+=dt; }
        }
    }

    // Emit the per-call timing breakdown; a single K block requires no C RMW.
    {
        double t_total=_t_packA+_t_kernel+_t_c_write;
        printf("[gemm_nonspm_qk #%d] M=%zu K=%zu N=%zu  KC=%zu MC=%zu NT=%zu  (single K-block)\n",
               _this_call, M, K, N, KC, MC, NT);
        printf("  packA:        %8.1f us  (%5.1f%%)\n", _t_packA,   100.*_t_packA/t_total);
        printf("  kernel:       %8.1f us  (%5.1f%%)  [B from cache, no C RMW]\n",
               _t_kernel,  100.*_t_kernel/t_total);
        printf("  C write:      %8.1f us  (%5.1f%%)\n", _t_c_write, 100.*_t_c_write/t_total);
        printf("  total:        %8.1f us\n", t_total);
        fflush(stdout);
    }
}

static void scale_fp16(__fp16* buf, size_t n, float sv) {
    svfloat16_t vs=svdup_f16((__fp16)sv);
    for(size_t i=0;i<n;i+=svcnth()){svbool_t pg=svwhilelt_b16_u64(i,n);svst1_f16(pg,buf+i,svmul_f16_x(pg,svld1_f16(pg,buf+i),vs));}
}

// ============================================================
// SVE exp and softmax
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

// Widen/narrow one FP16 value per active FP32 lane.  Loading a packed
// svfloat16_t and converting it directly would consume only alternating
// halfwords because the source and destination lane widths differ.
static inline svfloat32_t load_f16_as_f32(
    svbool_t pg32, const __fp16* source)
{
    return svcvt_f32_f16_z(
        pg32,
        svreinterpret_f16_u32(svld1uh_u32(
            pg32, reinterpret_cast<const uint16_t*>(source))));
}

static inline void store_f32_as_f16(
    svbool_t pg32, __fp16* destination, svfloat32_t values)
{
    svst1h_u32(
        pg32,
        reinterpret_cast<uint16_t*>(destination),
        svreinterpret_u32_f16(svcvt_f16_f32_z(pg32, values)));
}

// softmax_fp16_rowwise: 2×VL32 unroll for max and exp+sum passes; 4×VL16 for scale.
// Two independent vmax/vsum accumulators hide the 4-cycle svmax/svadd latency.
static void softmax_fp16_rowwise(__fp16* buf, int rows, int cols)
{
    const size_t vl16 = svcnth();
    const size_t vl32 = svcntw();
    const svbool_t pg32f = svptrue_b32();
    const svbool_t pg16f = svptrue_b16();

    for (int r=0; r<rows; ++r) {
        __fp16* row = buf + (size_t)r*cols;
        size_t i;

        // --- Pass 1: max (2× VL32) ---
        svfloat32_t vmax0=svdup_f32(-1e30f), vmax1=svdup_f32(-1e30f);
        for (i=0; i+2*vl32<=(size_t)cols; i+=2*vl32) {
            svfloat32_t f0=load_f16_as_f32(pg32f,row+i);
            svfloat32_t f1=load_f16_as_f32(pg32f,row+i+vl32);
            vmax0=svmax_f32_x(pg32f,vmax0,f0);
            vmax1=svmax_f32_x(pg32f,vmax1,f1);
        }
        svfloat32_t vmax32=svmax_f32_x(pg32f,vmax0,vmax1);
        for (; i<(size_t)cols; i+=vl32) {
            svbool_t pg32=svwhilelt_b32_u64(i,(size_t)cols);
            svfloat32_t f=load_f16_as_f32(pg32,row+i);
            vmax32=svmax_f32_m(pg32,vmax32,f);
        }
        float row_max=svmaxv_f32(svptrue_b32(),vmax32);

        // --- Pass 2: exp + sum (2× VL32) ---
        svfloat32_t vsum0=svdup_f32(0.f), vsum1=svdup_f32(0.f);
        for (i=0; i+2*vl32<=(size_t)cols; i+=2*vl32) {
            svfloat32_t v0=load_f16_as_f32(pg32f,row+i);
            svfloat32_t v1=load_f16_as_f32(pg32f,row+i+vl32);
            svfloat32_t e0=sve_exp_f32(pg32f,svsub_n_f32_x(pg32f,v0,row_max));
            svfloat32_t e1=sve_exp_f32(pg32f,svsub_n_f32_x(pg32f,v1,row_max));
            vsum0=svadd_f32_x(pg32f,vsum0,e0);
            vsum1=svadd_f32_x(pg32f,vsum1,e1);
            store_f32_as_f16(pg32f,row+i,e0);
            store_f32_as_f16(pg32f,row+i+vl32,e1);
        }
        svfloat32_t vsum=svadd_f32_x(pg32f,vsum0,vsum1);
        for (; i<(size_t)cols; i+=vl32) {
            svbool_t pg32=svwhilelt_b32_u64(i,(size_t)cols);
            svfloat32_t vf=svsub_n_f32_x(pg32,
                load_f16_as_f32(pg32,row+i), row_max);
            svfloat32_t e=sve_exp_f32(pg32,vf);
            vsum=svadd_f32_m(pg32,vsum,e);
            store_f32_as_f16(pg32,row+i,e);
        }
        float inv_sum=1.f/svaddv_f32(svptrue_b32(),vsum);

        // --- Pass 3: scale (4× VL16) ---
        svfloat16_t vs=svdup_f16((__fp16)inv_sum);
        for (i=0; i+4*vl16<=(size_t)cols; i+=4*vl16) {
            svst1_f16(pg16f,row+i,        svmul_f16_x(pg16f,svld1_f16(pg16f,row+i),        vs));
            svst1_f16(pg16f,row+i+vl16,   svmul_f16_x(pg16f,svld1_f16(pg16f,row+i+vl16),   vs));
            svst1_f16(pg16f,row+i+2*vl16, svmul_f16_x(pg16f,svld1_f16(pg16f,row+i+2*vl16), vs));
            svst1_f16(pg16f,row+i+3*vl16, svmul_f16_x(pg16f,svld1_f16(pg16f,row+i+3*vl16), vs));
        }
        for (; i<(size_t)cols; i+=vl16) {
            svbool_t pg=svwhilelt_b16_u64(i,(size_t)cols);
            svst1_f16(pg,row+i,svmul_f16_x(pg,svld1_f16(pg,row+i),vs));
        }
    }
}

// ============================================================
// SVE element-wise add
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
// RMSNorm: y = x / rms(x) * w   (no bias, LLaMA-style)
// ============================================================
static void rmsnorm_row(const __fp16* __restrict__ x, const __fp16* __restrict__ w,
                        __fp16* __restrict__ y, int D, float eps)
{
    const size_t VL32=svcntw();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    size_t d=0;
    for(;d+4*VL32<=(size_t)D;d+=4*VL32){
        svbool_t pg=svptrue_b32();
        svfloat32_t f0=load_f16_as_f32(pg,x+d);
        svfloat32_t f1=load_f16_as_f32(pg,x+d+VL32);
        svfloat32_t f2=load_f16_as_f32(pg,x+d+2*VL32);
        svfloat32_t f3=load_f16_as_f32(pg,x+d+3*VL32);
        a0=svmla_f32_x(pg,a0,f0,f0); a1=svmla_f32_x(pg,a1,f1,f1);
        a2=svmla_f32_x(pg,a2,f2,f2); a3=svmla_f32_x(pg,a3,f3,f3);}
    for(;d<(size_t)D;d+=VL32){
        svbool_t pg32=svwhilelt_b32_u64(d,(size_t)D);
        svfloat32_t f=load_f16_as_f32(pg32,x+d);
        a0=svmla_f32_m(pg32,a0,f,f);}
    float ss=svaddv_f32(svptrue_b32(),svadd_f32_x(svptrue_b32(),svadd_f32_x(svptrue_b32(),a0,a1),svadd_f32_x(svptrue_b32(),a2,a3)));
    float scale=1.0f/sqrtf(ss/D+eps);
    svfloat32_t vscale=svdup_f32(scale);
    for(d=0;d+4*VL32<=(size_t)D;d+=4*VL32){
        svbool_t pg32=svptrue_b32();
        svfloat32_t xf0=load_f16_as_f32(pg32,x+d);
        svfloat32_t xf1=load_f16_as_f32(pg32,x+d+VL32);
        svfloat32_t xf2=load_f16_as_f32(pg32,x+d+2*VL32);
        svfloat32_t xf3=load_f16_as_f32(pg32,x+d+3*VL32);
        svfloat32_t wf0=load_f16_as_f32(pg32,w+d);
        svfloat32_t wf1=load_f16_as_f32(pg32,w+d+VL32);
        svfloat32_t wf2=load_f16_as_f32(pg32,w+d+2*VL32);
        svfloat32_t wf3=load_f16_as_f32(pg32,w+d+3*VL32);
        store_f32_as_f16(pg32,y+d,svmul_f32_x(pg32,svmul_f32_x(pg32,xf0,wf0),vscale));
        store_f32_as_f16(pg32,y+d+VL32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf1,wf1),vscale));
        store_f32_as_f16(pg32,y+d+2*VL32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf2,wf2),vscale));
        store_f32_as_f16(pg32,y+d+3*VL32,svmul_f32_x(pg32,svmul_f32_x(pg32,xf3,wf3),vscale));}
    for(;d<(size_t)D;d+=VL32){
        svbool_t pg32=svwhilelt_b32_u64(d,(size_t)D);
        svfloat32_t xf=load_f16_as_f32(pg32,x+d);
        svfloat32_t wf=load_f16_as_f32(pg32,w+d);
        store_f32_as_f16(pg32,y+d,svmul_f32_x(pg32,svmul_f32_x(pg32,xf,wf),vscale));}
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
    if (!tok_stride)  tok_stride  = d;    // default: head-major [H,T,d]
    if (!head_stride) head_stride = T*d;  // default: head-major [H,T,d]
    const int half=d/2; const size_t VL32=svcntw();
    for(int h=0;h<H;++h)
        for(int t=0;t<T;++t){
            __fp16* row=x + (size_t)h*head_stride + (size_t)t*tok_stride;
            const float* c=cos_t+t*half; const float* s=sin_t+t*half;
            size_t i=0;
            for(;i+2*VL32<=(size_t)half;i+=2*VL32){
                svbool_t pg32=svptrue_b32();
                svfloat32_t xla=load_f16_as_f32(pg32,row+i);
                svfloat32_t xha=load_f16_as_f32(pg32,row+i+half);
                svfloat32_t ca=svld1_f32(pg32,c+i), sa=svld1_f32(pg32,s+i);
                svfloat32_t xlb=load_f16_as_f32(pg32,row+i+VL32);
                svfloat32_t xhb=load_f16_as_f32(pg32,row+i+VL32+half);
                svfloat32_t cb=svld1_f32(pg32,c+i+VL32), sb=svld1_f32(pg32,s+i+VL32);
                store_f32_as_f16(pg32,row+i,
                    svmls_f32_x(pg32,svmul_f32_x(pg32,xla,ca),xha,sa));
                store_f32_as_f16(pg32,row+i+half,
                    svmla_f32_x(pg32,svmul_f32_x(pg32,xha,ca),xla,sa));
                store_f32_as_f16(pg32,row+i+VL32,
                    svmls_f32_x(pg32,svmul_f32_x(pg32,xlb,cb),xhb,sb));
                store_f32_as_f16(pg32,row+i+VL32+half,
                    svmla_f32_x(pg32,svmul_f32_x(pg32,xhb,cb),xlb,sb));}
            for(;i+VL32<=(size_t)half;i+=VL32){
                svbool_t pg32=svptrue_b32();
                svfloat32_t xl=load_f16_as_f32(pg32,row+i);
                svfloat32_t xh=load_f16_as_f32(pg32,row+i+half);
                svfloat32_t cv=svld1_f32(pg32,c+i), sv_=svld1_f32(pg32,s+i);
                store_f32_as_f16(pg32,row+i,
                    svmls_f32_x(pg32,svmul_f32_x(pg32,xl,cv),xh,sv_));
                store_f32_as_f16(pg32,row+i+half,
                    svmla_f32_x(pg32,svmul_f32_x(pg32,xh,cv),xl,sv_));}
            for(;i<(size_t)half;++i){
                float xi=(float)row[i],xi2=(float)row[i+half];
                row[i]=(__fp16)(xi*c[i]-xi2*s[i]); row[i+half]=(__fp16)(xi2*c[i]+xi*s[i]);}
        }
}

// ============================================================
// Causal softmax: mask scores[i, j>i] = -65504 then softmax
// ============================================================
static void causal_softmax_fp16(__fp16* buf, int T)
{
    // buf layout: [T, T] row-major. For row i, mask columns j > i.
    const __fp16 neg_inf = (__fp16)(-65504.0f);
    const size_t vl16 = svcnth();
    for (int i = 0; i < T; ++i) {
        __fp16*     row = buf + (size_t)i * T;
        svfloat16_t vni = svdup_f16(neg_inf);
        // SVE predicated fill from j=i+1; svwhilelt handles partial head/tail naturally
        for (size_t j = (size_t)(i + 1); j < (size_t)T; j += vl16) {
            svbool_t pg = svwhilelt_b16_u64(j, (size_t)T);
            svst1_f16(pg, row + j, vni);
        }
    }
    softmax_fp16_rowwise(buf, T, T);
}

// ============================================================
// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
//   2× VL32 unrolled; svrecpe+svrecps replaces svdiv
// ============================================================
static void silu_fp16_inplace(__fp16* buf, size_t n)
{
    const size_t VL32 = svcntw();
    size_t i = 0;
    for (; i + 2*VL32 <= n; i += 2*VL32) {
        svbool_t pg32  = svptrue_b32();
        svfloat32_t x0 = load_f16_as_f32(pg32, buf+i);
        svfloat32_t x1 = load_f16_as_f32(pg32, buf+i+VL32);
        // exp(-x), clamped to [-88, 88] to avoid overflow
        svfloat32_t nx0 = svmin_n_f32_x(pg32, svneg_f32_x(pg32, x0), 88.0f);
        svfloat32_t nx1 = svmin_n_f32_x(pg32, svneg_f32_x(pg32, x1), 88.0f);
        svfloat32_t e0  = sve_exp_f32(pg32, nx0);
        svfloat32_t e1  = sve_exp_f32(pg32, nx1);
        // 1/(1+e): Newton-Raphson reciprocal
        svfloat32_t d0  = svadd_n_f32_x(pg32, e0, 1.0f);
        svfloat32_t d1  = svadd_n_f32_x(pg32, e1, 1.0f);
        svfloat32_t r0  = svrecpe_f32(d0);
        svfloat32_t r1  = svrecpe_f32(d1);
        r0 = svmul_f32_x(pg32, svrecps_f32(d0, r0), r0);
        r1 = svmul_f32_x(pg32, svrecps_f32(d1, r1), r1);
        store_f32_as_f16(pg32, buf+i, svmul_f32_x(pg32, x0, r0));
        store_f32_as_f16(pg32, buf+i+VL32, svmul_f32_x(pg32, x1, r1));
    }
    for (; i < n; i += VL32) {
        svbool_t pg32 = svwhilelt_b32_u64(i, n);
        svfloat32_t x  = load_f16_as_f32(pg32, buf+i);
        svfloat32_t nx = svmin_n_f32_x(pg32, svneg_f32_x(pg32, x), 88.0f);
        svfloat32_t e  = sve_exp_f32(pg32, nx);
        svfloat32_t d  = svadd_n_f32_x(pg32, e, 1.0f);
        svfloat32_t r  = svrecpe_f32(d);
        r = svmul_f32_x(pg32, svrecps_f32(d, r), r);
        store_f32_as_f16(pg32, buf+i, svmul_f32_x(pg32, x, r));
    }
}

// ============================================================
// SwiGLU: gate[i] = SiLU(gate[i]) * up[i]  (in-place on gate)
//   2× VL32 unrolled; svrecpe+svrecps replaces svdiv
// ============================================================
static void swiglu_fp16(__fp16* __restrict__ gate, const __fp16* __restrict__ up, size_t n)
{
    const size_t VL32 = svcntw();
    size_t i = 0;
    for (; i + 2*VL32 <= n; i += 2*VL32) {
        svbool_t pg32  = svptrue_b32();
        svfloat32_t g0 = load_f16_as_f32(pg32, gate+i);
        svfloat32_t g1 = load_f16_as_f32(pg32, gate+i+VL32);
        svfloat32_t u0 = load_f16_as_f32(pg32, up+i);
        svfloat32_t u1 = load_f16_as_f32(pg32, up+i+VL32);
        // SiLU(gate) = gate * sigmoid(gate); sigmoid via Newton-Raphson reciprocal
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
        // SwiGLU = SiLU(gate) * up
        store_f32_as_f16(pg32, gate+i,
            svmul_f32_x(pg32, svmul_f32_x(pg32, g0, r0), u0));
        store_f32_as_f16(pg32, gate+i+VL32,
            svmul_f32_x(pg32, svmul_f32_x(pg32, g1, r1), u1));
    }
    for (; i < n; i += VL32) {
        svbool_t pg32 = svwhilelt_b32_u64(i, n);
        svfloat32_t g  = load_f16_as_f32(pg32, gate+i);
        svfloat32_t u  = load_f16_as_f32(pg32, up+i);
        svfloat32_t ng = svmin_n_f32_x(pg32, svneg_f32_x(pg32, g), 88.0f);
        svfloat32_t e  = sve_exp_f32(pg32, ng);
        svfloat32_t d  = svadd_n_f32_x(pg32, e, 1.0f);
        svfloat32_t r  = svrecpe_f32(d);
        r = svmul_f32_x(pg32, svrecps_f32(d, r), r);
        store_f32_as_f16(pg32, gate+i,
            svmul_f32_x(pg32, svmul_f32_x(pg32, g, r), u));
    }
}
