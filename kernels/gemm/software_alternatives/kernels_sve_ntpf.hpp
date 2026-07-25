// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * kernels_sve_ntpf.hpp — L2-prefetch + non-temporal-load SUBSTITUTE microkernel
 *
 * This is a clone of sve_interleaved_fp16_mla_8x3VL from kernels_sve.hpp, but:
 *   - All B-panel loads (from x22) are NON-TEMPORAL: ld1h -> ldnt1h.
 *     This models CacheFlex's "B bypasses L1" using the STOCK SVE NT-load.
 *   - A software L2 prefetch (prfh PLDL2KEEP) of the B panel is issued ahead
 *     of use by PF_DIST_VL vector-loads (i.e. #(PF_DIST_VL), MUL VL bytes from
 *     the current B pointer). This models the "L2PF" half of the substitute.
 *
 * PF_DIST_VL is a compile-time macro (default 8). Sweep it at build time:
 *   -DPF_DIST_VL=4 / 8 / 16 / 32 ...   (in MUL-VL units; 0 disables prefetch)
 *
 * A-panel loads (ld1rqh) stay NORMAL (temporal) — only B becomes NT+prefetched.
 *
 * Output is bit-identical to sve_interleaved_fp16_mla_8x3VL (same FMA order);
 * NT vs temporal load and prefetch only affect cache placement / timing.
 */
#pragma once
#include "kernels_sve.hpp"   // pulls in sve_pack_B_3VL, pack_A_fp16_8row, etc.

#ifndef PF_DIST_VL
#  define PF_DIST_VL 8
#endif

// Stringize helper so PF_DIST_VL macro expands inside the inline-asm string.
#define NTPF_STR2(x) #x
#define NTPF_STR(x)  NTPF_STR2(x)

#if PF_DIST_VL > 0
#  define NTPF_PRFH_B \
        "prfh PLDL2KEEP, p0, [x22, #" NTPF_STR(PF_DIST_VL) ", MUL VL]\n"
#else
#  define NTPF_PRFH_B ""
#endif

extern "C" {

// Same signature & semantics as sve_interleaved_fp16_mla_8x3VL.
void sve_interleaved_fp16_mla_8x3VL_ntpf(
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
        NTPF_PRFH_B
        "ldnt1h { z2.h }, p0/Z, [x22]\n"
        "ldnt1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
        "ldnt1h { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
        "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
        "cmp x20, #0x2\n" "blt 4f\n"
        "3:\n"
        NTPF_PRFH_B
        "fmla z8.h,  z2.h, z0.h[0]\n" "fmla z11.h, z2.h, z0.h[1]\n"
        "ld1rqh { z7.h }, p0/Z, [%x[Apanel], #16]\n"
        "fmla z14.h, z2.h, z0.h[2]\n" "fmla z17.h, z2.h, z0.h[3]\n"
        "ldnt1h { z6.h }, p0/Z, [x22, #3, MUL VL]\n"
        "fmla z20.h, z2.h, z0.h[4]\n" "fmla z23.h, z2.h, z0.h[5]\n"
        "ldnt1h { z5.h }, p0/Z, [x22, #4, MUL VL]\n"
        "fmla z26.h, z2.h, z0.h[6]\n" "fmla z29.h, z2.h, z0.h[7]\n"
        "ldnt1h { z1.h }, p0/Z, [x22, #5, MUL VL]\n"
        "fmla z9.h,  z3.h, z0.h[0]\n" "fmla z12.h, z3.h, z0.h[1]\n"
        "addvl x22, x22, #6\n"
        "fmla z15.h, z3.h, z0.h[2]\n" "fmla z18.h, z3.h, z0.h[3]\n"
        "sub x20, x20, #0x2\n"
        "fmla z21.h, z3.h, z0.h[4]\n" "fmla z24.h, z3.h, z0.h[5]\n"
        "cmp x20, #0x2\n"
        "fmla z27.h, z3.h, z0.h[6]\n" "fmla z30.h, z3.h, z0.h[7]\n"
        "add %x[Apanel], %x[Apanel], #0x20\n"
        "fmla z10.h, z4.h, z0.h[0]\n" "fmla z13.h, z4.h, z0.h[1]\n"
        "ldnt1h { z2.h }, p0/Z, [x22]\n"
        "fmla z16.h, z4.h, z0.h[2]\n" "fmla z19.h, z4.h, z0.h[3]\n"
        "ldnt1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
        "fmla z22.h, z4.h, z0.h[4]\n" "fmla z25.h, z4.h, z0.h[5]\n"
        "fmla z28.h, z4.h, z0.h[6]\n" "fmla z31.h, z4.h, z0.h[7]\n"
        "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
        "fmla z8.h,  z6.h, z7.h[0]\n" "fmla z11.h, z6.h, z7.h[1]\n"
        "ldnt1h { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
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
        "ldnt1h  { z2.h }, p0/Z, [x22]\n"
        "add %x[Apanel], %x[Apanel], #0x10\n"
        "ldnt1h  { z1.h }, p0/Z, [x22, #1, MUL VL]\n"
        "ldnt1h  { z0.h }, p0/Z, [x22, #2, MUL VL]\n"
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

} // extern "C"
