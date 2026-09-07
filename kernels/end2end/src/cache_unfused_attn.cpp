// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * Cache-resident unfused attention baseline. QK and PV results are scattered
 * immediately after each microkernel call, and the PV path uses KC=256 so its
 * 64-KB packed-A panel fits in L1.
 *
 * Pipeline per head (no FlashAttention and no SPM):
 *   Step 1  QK GEMM:  scores_h[T,T] = Q_h[T,d] x K_h[T,d]^T x (1/sqrt(d))
 *   Step 2  Softmax:  causal softmax on scores_h[T,T]
 *   Step 3  PV GEMM:  out_h[T,d] = P_h[T,T] x V_h[T,d]
 *
 * Usage: ./bin/cache_unfused_attn [T] [H_q] [H_kv] [d] [n_iter]
 *        [MC_qk] [KC_pv] [MC_pv] [phase] <causal|noncausal>
 *        default: 512 1 1 64 1 128 256 128 0  — trailing attention mode is MANDATORY
 *
 * GFLOP accounting (FP16 FMA):
 *   QK:  2 x T x d x T x H   (A[T,d] x B[d,T])
 *   PV:  2 x T x T x d x H   (A[T,T] x B[T,d])
 *   total = 4 x T x T x d x H FLOP
 */
#include <arm_neon.h>
#include <arm_sve.h>
#include "pv_kernel_8x64.hpp"
#include "pv_kernel_8x64_vl16.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>

#ifdef GEM5
#  include <gem5/m5ops.h>
#  define ROI_BEGIN()   m5_work_begin(0, 0)
#  define ROI_END()     m5_work_end(0, 0)
#else
#  define ROI_BEGIN()   do{}while(0)
#  define ROI_END()     do{}while(0)
#endif

using Clock = std::chrono::high_resolution_clock;
static double us_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

static inline size_t my_svcnth() { size_t n; __asm__("cnth %0":"=r"(n)); return n; }

// VL=16 (svcnth=128): use 1VL=128 to avoid NT=384 > T issues
// VL<=8: use 3VL as before
static inline size_t get_NT() {
    size_t s = my_svcnth();
    return (s >= 128) ? s : 3 * s;
}

// ── Cache QK kernel: 8×1VL (for VL=16 where 3VL=384 is too wide) ──
static void cache_qk_8x1VL_ldc(
    const __fp16 *Apanel, const __fp16 *Bpanel, __fp16 *Cpanel,
    int K, int ldc_elements)
{
    size_t ldc_bytes = (size_t)ldc_elements * 2;
    __asm__ __volatile__(
        "ptrue p0.h\n"
        "mov x21, %x[Ap]\n" "mov x22, %x[Bp]\n"
        "mov x20, %x[K]\n"  "mov x13, %x[ldc_bytes]\n"

        "mov z8.b,#0\n" "mov z9.b,#0\n" "mov z10.b,#0\n" "mov z11.b,#0\n"
        "mov z12.b,#0\n" "mov z13.b,#0\n" "mov z14.b,#0\n" "mov z15.b,#0\n"

        "1:\n"
        "ld1rqh {z0.h},p0/z,[x21]\n"
        "ld1h {z2.h},p0/z,[x22]\n"
        "fmla z8.h,z2.h,z0.h[0]\n"  "fmla z9.h,z2.h,z0.h[1]\n"
        "fmla z10.h,z2.h,z0.h[2]\n" "fmla z11.h,z2.h,z0.h[3]\n"
        "fmla z12.h,z2.h,z0.h[4]\n" "fmla z13.h,z2.h,z0.h[5]\n"
        "fmla z14.h,z2.h,z0.h[6]\n" "fmla z15.h,z2.h,z0.h[7]\n"
        "add x21,x21,#16\n" "addvl x22,x22,#1\n"
        "subs x20,x20,#1\n" "bne 1b\n"

        "mov x5,%x[Cp]\n"
        "st1h {z8.h},p0,[x5]\n"  "add x5,x5,x13\n"
        "st1h {z9.h},p0,[x5]\n"  "add x5,x5,x13\n"
        "st1h {z10.h},p0,[x5]\n" "add x5,x5,x13\n"
        "st1h {z11.h},p0,[x5]\n" "add x5,x5,x13\n"
        "st1h {z12.h},p0,[x5]\n" "add x5,x5,x13\n"
        "st1h {z13.h},p0,[x5]\n" "add x5,x5,x13\n"
        "st1h {z14.h},p0,[x5]\n" "add x5,x5,x13\n"
        "st1h {z15.h},p0,[x5]\n"
        :
        : [Ap] "r" (Apanel), [Bp] "r" (Bpanel), [Cp] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc_bytes] "r" (ldc_bytes)
        : "cc","memory","p0",
          "x5","x13","x20","x21","x22",
          "z0","z2","z8","z9","z10","z11","z12","z13","z14","z15"
    );
}

// --- Aligned buffer ---------------------------------------------------------
template <typename T>
struct AlignedBuffer {
    T* ptr; size_t sz;
    AlignedBuffer() : ptr(nullptr), sz(0) {}
    explicit AlignedBuffer(size_t n) : sz(n) {
        if (!n) { ptr = nullptr; return; }
        size_t bytes = n * sizeof(T);
        if (bytes % 64) bytes += 64 - bytes % 64;
        ptr = static_cast<T*>(std::aligned_alloc(64, bytes));
        if (!ptr) throw std::bad_alloc();
    }
    ~AlignedBuffer() { std::free(ptr); }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

// --- Cache flush (no-op under GEM5_SE) --------------------------------------
static void flush_dcache_range(const void* ptr, size_t bytes)
{
#ifndef GEM5_SE
    uintptr_t addr = (uintptr_t)ptr & ~63UL;
    uintptr_t end  = (uintptr_t)ptr + bytes;
    for (; addr < end; addr += 64)
        __asm__ volatile("dc civac, %0" : : "r"(addr) : "memory");
    __asm__ volatile("dsb sy" : : : "memory");
    __asm__ volatile("isb"    : : : "memory");
#else
    (void)ptr; (void)bytes;
    __asm__ volatile("" : : : "memory");
#endif
}

static inline void fill_random(__fp16* p, size_t n, float lo, float hi, unsigned seed=42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) p[i] = (__fp16)dist(rng);
}

// =============================================================================
// SVE FP16 GEMM micro-kernel: 8 rows x 3VL cols
// Inputs: Apanel [ablocks x 8 x K], Bpanel [bblocks x K x NT] (KNM packed)
// Output: Cpanel [ablocks x bblocks x 8 x NT] (overwrites, then caller scatters)
//
// This is the SAME kernel as v12 (12_attn_baseline_seq_vl4.cpp). It is the
// SOTA 8x3VL micro-kernel used in ARM's Compute Library. Do NOT modify.
// =============================================================================
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
        // --- N-block loop ---
        "1:\n"
        "ldr x23, [%x[args], %[off_Mblocks]]\n"
        "mov x21, x27\n"
        // --- M-block loop ---
        "2:\n"
        "mov x22, x24\n"
        "ldr x20, [%x[args], %[off_Kmain]]\n"
        "mov %x[Apanel], x21\n"
        // Zero accumulators (24 registers: 8 rows x 3 VL-width cols)
        "mov z8.b, #0\n"  "mov z9.b, #0\n"  "mov z10.b,#0\n" "mov z11.b,#0\n"
        "mov z12.b,#0\n"  "mov z13.b,#0\n"  "mov z14.b,#0\n" "mov z15.b,#0\n"
        "mov z16.b,#0\n"  "mov z17.b,#0\n"  "mov z18.b,#0\n" "mov z19.b,#0\n"
        "mov z20.b,#0\n"  "mov z21.b,#0\n"  "mov z22.b,#0\n" "mov z23.b,#0\n"
        "mov z24.b,#0\n"  "mov z25.b,#0\n"  "mov z26.b,#0\n" "mov z27.b,#0\n"
        "mov z28.b,#0\n"  "mov z29.b,#0\n"  "mov z30.b,#0\n" "mov z31.b,#0\n"
        // Preload first K iteration
        "ld1h   { z2.h }, p0/Z, [x22]\n"
        "ld1h   { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
        "ld1h   { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
        "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
        "cmp x20, #0x2\n" "blt 4f\n"
        // --- K main loop (unrolled x2) ---
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
        // --- K tail (1 remaining) ---
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
        // Check for odd K tail
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
        // --- Store Cpanel (8 rows x 3VL cols = 24 vectors) ---
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
        // Advance Apanel for next M-block
        "mov x21, %x[Apanel]\n"
        "subs x23, x23, #1\n"
        "bgt 2b\n"
        // Advance Bpanel to next N-block: B_stride = 3*VL_bytes*K
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

// =============================================================================
// pack_A_fp16_8row: pack A[M, K] row-major -> Apanel [num_mc x max_ab x 8 x KC]
//   k-interleaved 8-row format for sve_interleaved_fp16_mla_8x3VL
// =============================================================================
static void pack_A_fp16_8row(const __fp16* A, size_t lda, __fp16* Apanel,
                              size_t M, size_t m0, size_t K, size_t k0,
                              size_t mc, size_t kc)
{
    const size_t MT  = 8, VLh = my_svcnth();
    size_t rows = std::min(mc, M - m0);
    kc = std::min(kc, K - k0);
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

// =============================================================================
// pack_K_tile: gather-transpose K_h[T, d] -> K_tile[d, NT] (one N-block)
//   K_h is stored row-major: K_h[t, k] = K[t * d + k]
//   K_tile[k, n] = K_h[n_start + n, k]  (zero-padded for n >= nc)
// =============================================================================
static void pack_K_tile(uint16_t* out, const uint16_t* K_h, int T, int d, size_t NT,
                        size_t n_start)
{
    const size_t VLw = svcntw();
    const size_t VLh = svcnth();
    size_t nc = std::min(NT, (size_t)T - n_start);
    uint16_t* dst = out;
    std::memset(dst, 0, NT * (size_t)d * sizeof(uint16_t));

    const uint32_t bstride = (uint32_t)((size_t)d * sizeof(uint16_t));
    const svuint32_t voff = svmul_n_u32_x(svptrue_b32(), svindex_u32(0u,1u), bstride);

    for (int k = 0; k < d; ++k) {
        uint16_t* row_out = dst + (size_t)k * NT;
        const uint16_t* col_base = K_h + (n_start * (size_t)d) + (size_t)k;
        size_t n = 0;
        for (; n + VLh <= nc; n += VLh) {
            svuint32_t g0 = svld1uh_gather_u32offset_u32(svptrue_b32(),
                col_base + n * (size_t)d, voff);
            svuint32_t g1 = svld1uh_gather_u32offset_u32(svptrue_b32(),
                col_base + (n + VLw) * (size_t)d, voff);
            svst1_u16(svptrue_b16(), row_out + n,
                svuzp1_u16(svreinterpret_u16_u32(g0), svreinterpret_u16_u32(g1)));
        }
        if (n + VLw <= nc) {
            svuint32_t g = svld1uh_gather_u32offset_u32(svptrue_b32(),
                col_base + n * (size_t)d, voff);
            svst1_u16(svwhilelt_b16_u64(0, VLw), row_out + n,
                svuzp1_u16(svreinterpret_u16_u32(g), svdup_n_u16(0)));
            n += VLw;
        }
        for (; n < nc; ++n)
            row_out[n] = K_h[(n_start + n) * (size_t)d + (size_t)k];
    }
}

// =============================================================================
// prepack_B_knm: pack V_h[T, d] -> V_pk [K_tiles x N_tiles x KC x NT]
//   KNM layout for use with sve_interleaved_fp16_mla_8x3VL.
// =============================================================================
static size_t prepack_B_knm_size(size_t K, size_t N, size_t KC)
{
    const size_t NT = get_NT();
    size_t K_tiles = (K + KC - 1) / KC;
    size_t N_tiles = (N + NT - 1) / NT;
    return K_tiles * N_tiles * KC * NT;
}

static void prepack_B_knm(const __fp16* B, __fp16* B_packed,
                           size_t K, size_t N, size_t KC, size_t ldb = 0)
{
    if (!ldb) ldb = N;
    const size_t NT = get_NT();
    const size_t N_tiles = (N + NT - 1) / NT;
    const size_t K_tiles = (K + KC - 1) / KC;
    for (size_t kt = 0; kt < K_tiles; ++kt) {
        size_t k0 = kt * KC;
        size_t kc = std::min(KC, K - k0);
        for (size_t nt = 0; nt < N_tiles; ++nt) {
            size_t n0 = nt * NT;
            size_t nc = std::min(NT, N - n0);
            __fp16* dst = B_packed + (kt * N_tiles + nt) * KC * NT;
            std::memset(dst, 0, KC * NT * sizeof(__fp16));
            for (size_t k = 0; k < kc; ++k) {
                const __fp16* src_row = B + (k0 + k) * ldb + n0;
                __fp16* dst_row = dst + k * NT;
                size_t i = 0;
                svbool_t pg_full = svptrue_b16();
                for (; i + my_svcnth() <= nc; i += my_svcnth())
                    svst1_f16(pg_full, dst_row + i, svld1_f16(pg_full, src_row + i));
                if (i < nc) {
                    svbool_t pg = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pg, dst_row + i, svld1_f16(pg, src_row + i));
                }
            }
        }
    }
}

// =============================================================================
// SVE exp and causal softmax (SAME as v12 -- already SOTA with 2-unrolled exp)
// =============================================================================
static inline svfloat32_t sve_exp_f32(svbool_t pg, svfloat32_t x)
{
    x = svmax_n_f32_x(pg, x, -88.0f);
    svfloat32_t nf = svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.44269504088896f));
    svint32_t   ni = svcvt_s32_f32_x(pg, nf);
    svfloat32_t r  = svmls_n_f32_x(pg, x, nf,  6.93147182e-1f);
    r              = svmls_n_f32_x(pg, r, nf, -1.90465430e-9f);
    svfloat32_t p  = svdup_n_f32(1.0f/120.0f);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f/24.0f), r,p);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f/6.0f),  r,p);
    p = svmla_f32_x(pg,svdup_n_f32(0.5f),        r,p);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f),        r,p);
    p = svmla_f32_x(pg,svdup_n_f32(1.0f),        r,p);
    return svscale_f32_x(pg,p,ni);
}

static void softmax_fp16_rowwise(__fp16* buf, int rows, int cols)
{
    const size_t vl16 = my_svcnth();
    const size_t vl32 = vl16 / 2;
    const svbool_t pg32f = svptrue_b32();
    const svbool_t pg16f = svptrue_b16();
    for (int r = 0; r < rows; ++r) {
        __fp16* row = buf + (size_t)r * cols;
        size_t i;
        // Pass 1: find row max (2-unrolled for ILP)
        // LANE-FIX: fp16->fp32 widening uses ld1uh (consecutive fp16 -> low half
        // of each 32-bit lane), matching the flash benches' lane-fixed idiom.
        svfloat32_t vmax0 = svdup_f32(-1e30f), vmax1 = svdup_f32(-1e30f);
        for (i = 0; i + 2*vl32 <= (size_t)cols; i += 2*vl32) {
            svfloat32_t f0 = svcvt_f32_f16_x(pg32f, svreinterpret_f16_u32(svld1uh_u32(pg32f, (const uint16_t*)(row+i))));
            svfloat32_t f1 = svcvt_f32_f16_x(pg32f, svreinterpret_f16_u32(svld1uh_u32(pg32f, (const uint16_t*)(row+i+vl32))));
            vmax0 = svmax_f32_x(pg32f, vmax0, f0);
            vmax1 = svmax_f32_x(pg32f, vmax1, f1);
        }
        svfloat32_t vmax32 = svmax_f32_x(pg32f, vmax0, vmax1);
        for (; i < (size_t)cols; i += vl32) {
            svbool_t pg32 = svwhilelt_b32_u64(i,(size_t)cols);
            svfloat32_t f = svcvt_f32_f16_z(pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(row+i))));
            vmax32 = svmax_f32_m(pg32, vmax32, f);
        }
        float row_max = svmaxv_f32(svptrue_b32(), vmax32);
        // Pass 2: exp(x - max) and sum (2-unrolled)
        // LANE-FIX: fp32->fp16 narrowing uses st1h (low half of each 32-bit
        // lane -> consecutive fp16), matching the flash benches' idiom.
        svfloat32_t vsum0 = svdup_f32(0.f), vsum1 = svdup_f32(0.f);
        for (i = 0; i + 2*vl32 <= (size_t)cols; i += 2*vl32) {
            svfloat32_t v0 = svcvt_f32_f16_x(pg32f, svreinterpret_f16_u32(svld1uh_u32(pg32f, (const uint16_t*)(row+i))));
            svfloat32_t v1 = svcvt_f32_f16_x(pg32f, svreinterpret_f16_u32(svld1uh_u32(pg32f, (const uint16_t*)(row+i+vl32))));
            svfloat32_t e0 = sve_exp_f32(pg32f, svsub_n_f32_x(pg32f, v0, row_max));
            svfloat32_t e1 = sve_exp_f32(pg32f, svsub_n_f32_x(pg32f, v1, row_max));
            vsum0 = svadd_f32_x(pg32f, vsum0, e0);
            vsum1 = svadd_f32_x(pg32f, vsum1, e1);
            svst1h_u32(pg32f, (uint16_t*)(row+i),      svreinterpret_u32_f16(svcvt_f16_f32_x(pg32f, e0)));
            svst1h_u32(pg32f, (uint16_t*)(row+i+vl32), svreinterpret_u32_f16(svcvt_f16_f32_x(pg32f, e1)));
        }
        svfloat32_t vsum = svadd_f32_x(pg32f, vsum0, vsum1);
        for (; i < (size_t)cols; i += vl32) {
            svbool_t pg32 = svwhilelt_b32_u64(i,(size_t)cols);
            svfloat32_t e = sve_exp_f32(pg32, svsub_n_f32_x(pg32,
                svcvt_f32_f16_z(pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(row+i)))), row_max));
            vsum = svadd_f32_m(pg32, vsum, e);
            svst1h_u32(pg32, (uint16_t*)(row+i), svreinterpret_u32_f16(svcvt_f16_f32_x(pg32, e)));
        }
        float inv_sum = 1.f / svaddv_f32(svptrue_b32(), vsum);
        // Pass 3: scale by 1/sum (4-unrolled, data is L1-hot)
        svfloat16_t vs = svdup_f16((__fp16)inv_sum);
        for (i = 0; i + 4*vl16 <= (size_t)cols; i += 4*vl16) {
            svst1_f16(pg16f, row+i,           svmul_f16_x(pg16f, svld1_f16(pg16f, row+i),           vs));
            svst1_f16(pg16f, row+i+vl16,      svmul_f16_x(pg16f, svld1_f16(pg16f, row+i+vl16),      vs));
            svst1_f16(pg16f, row+i+2*vl16,    svmul_f16_x(pg16f, svld1_f16(pg16f, row+i+2*vl16),    vs));
            svst1_f16(pg16f, row+i+3*vl16,    svmul_f16_x(pg16f, svld1_f16(pg16f, row+i+3*vl16),    vs));
        }
        for (; i < (size_t)cols; i += vl16) {
            svbool_t pg = svwhilelt_b16_u64(i,(size_t)cols);
            svst1_f16(pg, row+i, svmul_f16_x(pg, svld1_f16(pg, row+i), vs));
        }
    }
}

static void causal_softmax_fp16(__fp16* buf, int T, size_t T_pad)
{
    const __fp16 neg_inf = (__fp16)(-65504.0f);
    const size_t vl16 = my_svcnth();
    for (int row_i = 0; row_i < T; ++row_i) {
        __fp16* row = buf + (size_t)row_i * T_pad;
        // Fill padding columns (T..T_pad-1) with neg_inf to avoid softmax bias
        for (size_t j = (size_t)T; j < T_pad; j += vl16) {
            svbool_t pg = svwhilelt_b16_u64(j, T_pad);
            svst1_f16(pg, row + j, svdup_f16(neg_inf));
        }
        // Causal mask: fill positions row_i+1..T-1 with neg_inf
        for (size_t j = (size_t)(row_i + 1); j < (size_t)T; j += vl16) {
            svbool_t pg = svwhilelt_b16_u64(j, (size_t)T);
            svst1_f16(pg, row + j, svdup_f16(neg_inf));
        }
    }
    softmax_fp16_rowwise(buf, T, (int)T_pad);
}

// Non-causal variant: same tail/padding handling (pad cols T..T_pad-1 -> -inf),
// but NO causal mask — plain softmax over all T columns of each row.
static void noncausal_softmax_fp16(__fp16* buf, int T, size_t T_pad)
{
    const __fp16 neg_inf = (__fp16)(-65504.0f);
    const size_t vl16 = my_svcnth();
    for (int row_i = 0; row_i < T; ++row_i) {
        __fp16* row = buf + (size_t)row_i * T_pad;
        // Fill padding columns (T..T_pad-1) with neg_inf to avoid softmax bias
        for (size_t j = (size_t)T; j < T_pad; j += vl16) {
            svbool_t pg = svwhilelt_b16_u64(j, T_pad);
            svst1_f16(pg, row + j, svdup_f16(neg_inf));
        }
    }
    softmax_fp16_rowwise(buf, T, (int)T_pad);
}

// =============================================================================
// SELF_CHECK (env CF_SELF_CHECK=1): plain FP64 reference attention, O(T^2*d)
// simple loops, same Q/K/V and same attention mode as the timed run.
// =============================================================================
static void self_check_attention(const __fp16* Q, const __fp16* K, const __fp16* V,
                                 const __fp16* O, int T, int H_q, int H_kv, int d,
                                 bool causal)
{
    const int group_ratio = H_q / H_kv;
    const double scale = 1.0 / std::sqrt((double)d);
    double* srow = (double*)std::malloc((size_t)T * sizeof(double));
    double max_err = 0.0, sum_err = 0.0;
    size_t n_cnt = 0;
    for (int hq = 0; hq < H_q; ++hq) {
        const __fp16* Qh = Q + (size_t)hq * T * d;
        const __fp16* Kh = K + (size_t)(hq / group_ratio) * T * d;
        const __fp16* Vh = V + (size_t)(hq / group_ratio) * T * d;
        const __fp16* Oh = O + (size_t)hq * T * d;
        for (int r = 0; r < T; ++r) {
            const int cols = causal ? (r + 1) : T;
            double m = -1e300;
            for (int c = 0; c < cols; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k)
                    s += (double)(float)Qh[(size_t)r * d + k] * (double)(float)Kh[(size_t)c * d + k];
                s *= scale;
                srow[c] = s;
                if (s > m) m = s;
            }
            double l = 0.0;
            for (int c = 0; c < cols; ++c) { srow[c] = std::exp(srow[c] - m); l += srow[c]; }
            for (int k = 0; k < d; ++k) {
                double acc = 0.0;
                for (int c = 0; c < cols; ++c)
                    acc += srow[c] * (double)(float)Vh[(size_t)c * d + k];
                acc /= l;
                double err = std::fabs(acc - (double)(float)Oh[(size_t)r * d + k]);
                if (err > max_err) max_err = err;
                sum_err += err; ++n_cnt;
            }
        }
    }
    std::free(srow);
    double mean_err = sum_err / (double)n_cnt;
    printf("SELF_CHECK max_err=%.3e mean_err=%.3e\n", max_err, mean_err);
    printf("SELF_CHECK %s\n", (max_err <= 2e-2 && mean_err <= 2e-3) ? "PASS" : "FAIL");
}

// =============================================================================
// Timing struct — fine-grained breakdown for each stage
// =============================================================================
struct AttnTiming {
    double t_qk_packA   = 0;   // pack Q -> A_cache
    double t_qk_packK   = 0;   // gather-transpose K -> K_tile
    double t_qk_gemm    = 0;   // FMLA kernel + fused scatter (OPT 1)
    double t_softmax    = 0;   // causal softmax on scores[T,T]
    double t_pv_packA   = 0;   // pack scores -> A_cache for PV
    double t_pv_gemm    = 0;   // 8x64 PV kernel + scatter (OPT 2)
    // Plan B: exclude K prepack (assume pre-packed KV cache)
    double total_planB() const {
        return t_qk_packA + t_qk_gemm     // QK: pack_A(Q) + kernel
             + t_softmax                    // softmax
             + t_pv_packA + t_pv_gemm;     // PV: pack_A(scores) + kernel
        // EXCLUDED: t_qk_packK (K transpose+pack)
    }
    double total_full() const {
        return t_qk_packA + t_qk_packK + t_qk_gemm
             + t_softmax
             + t_pv_packA + t_pv_gemm;
    }
};

// =============================================================================
// FUSED SCATTER HELPERS
//
// These are called IMMEDIATELY after each micro-kernel (8 rows x NT cols).
// Cpanel is still L1-hot, so the scatter is essentially free compared to
// a separate pass that would reload Cpanel from L2.
// =============================================================================

// QK fused scatter: Cpanel[8, NT] -> scores[T, T_pad] with scale multiplication
//   Processes one M-block (up to 8 rows) for one N-block.
//   Called right after sve_interleaved_fp16_mla_8x3VL for this M-block.
static inline void scatter_qk_fused(
    const __fp16* Cpanel, __fp16* scores,
    size_t m0, size_t ablocks, size_t n_start, size_t nc,
    size_t T, size_t T_pad, svfloat16_t vs_scale)
{
    const size_t MT = 8;
    const size_t NT = get_NT();
    const size_t VLh = my_svcnth();
    svbool_t pg_full = svptrue_b16();
    for (size_t mb = 0; mb < ablocks; ++mb) {
        const __fp16* tp = Cpanel + mb * MT * NT;
        for (size_t r = 0; r < MT; ++r) {
            size_t gr = m0 + mb * MT + r;
            if (gr >= T) break;
            __fp16* s_row = scores + gr * T_pad + n_start;
            const __fp16* t_row = tp + r * NT;
            size_t i = 0;
            for (; i + VLh <= nc; i += VLh)
                svst1_f16(pg_full, s_row + i,
                    svmul_f16_x(pg_full, svld1_f16(pg_full, t_row + i), vs_scale));
            if (i < nc) {
                svbool_t pg = svwhilelt_b16_u64(i, nc);
                svst1_f16(pg, s_row + i,
                    svmul_f16_x(pg, svld1_f16(pg, t_row + i), vs_scale));
            }
        }
    }
}

// PV fused scatter: Cpanel[8, NT] -> out[T, d] with accumulation
//   kt == 0: overwrite (first K-tile); kt > 0: accumulate (+=).
//   Called right after sve_interleaved_fp16_mla_8x3VL for this M-block.
static inline void scatter_pv_fused(
    const __fp16* Cpanel, __fp16* out,
    size_t m0, size_t ablocks, size_t n0, size_t nc,
    size_t T, size_t d, size_t kt)
{
    const size_t MT = 8;
    const size_t NT = get_NT();
    const size_t VLh = my_svcnth();
    svbool_t pg_full = svptrue_b16();
    for (size_t mb = 0; mb < ablocks; ++mb) {
        const __fp16* tp = Cpanel + mb * MT * NT;
        for (size_t r = 0; r < MT; ++r) {
            size_t gr = m0 + mb * MT + r;
            if (gr >= T) break;
            __fp16* o_row = out + gr * d + n0;
            const __fp16* t_row = tp + r * NT;
            size_t i = 0;
            if (kt == 0) {
                // First K-tile: overwrite
                for (; i + VLh <= nc; i += VLh)
                    svst1_f16(pg_full, o_row + i, svld1_f16(pg_full, t_row + i));
                if (i < nc) {
                    svbool_t pg = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pg, o_row + i, svld1_f16(pg, t_row + i));
                }
            } else {
                // Subsequent K-tiles: accumulate
                for (; i + VLh <= nc; i += VLh)
                    svst1_f16(pg_full, o_row + i,
                        svadd_f16_x(pg_full,
                            svld1_f16(pg_full, o_row + i),
                            svld1_f16(pg_full, t_row + i)));
                if (i < nc) {
                    svbool_t pg = svwhilelt_b16_u64(i, nc);
                    svst1_f16(pg, o_row + i,
                        svadd_f16_x(pg,
                            svld1_f16(pg, o_row + i),
                            svld1_f16(pg, t_row + i)));
                }
            }
        }
    }
}

// =============================================================================
// run_attn_head_opt: one head's full attention pipeline (OPTIMIZED)
//
//   Q_h [T, d], K_h [T, d], V_h [T, d]  (head slices, row-major)
//   scores [T, T_padded]  (output buffer for QK; T_padded = ceil(T/NT)*NT)
//   out    [T, d]         (output buffer for PV)
//   MC_qk, KC_pv, MC_pv: tile parameters
//
// Key differences from v12:
//   - QK scatter is fused into the GEMM M-block loop (OPT 1)
//   - PV scatter is fused into the GEMM M-block loop (OPT 2)
//   - KC_pv default is 256 instead of 64 (OPT 3)
//   - PV pack_A is timed separately (not lumped into kernel time)
// =============================================================================
// phase: 0=all, 1=QK only, 2=softmax only, 3=PV only
// causal: true = causal softmax (mask col > row), false = plain softmax
static void run_attn_head_opt(
    const __fp16* Q_h, const __fp16* K_h, const __fp16* V_h,
    __fp16* scores, __fp16* out,
    int T, int d,
    size_t MC_qk, size_t KC_pv, size_t MC_pv,
    AttnTiming& tim, int phase = 0, bool causal = true)
{
    const size_t NT      = get_NT();           // 96 at VL=4
    const size_t MT      = 8;
    const size_t T_pad   = ((size_t)T + NT - 1) / NT * NT;
    const size_t n_tiles = T_pad / NT;
    const size_t kc_qk   = (size_t)d;                 // K dimension for QK = d
    const size_t num_mc_qk = ((size_t)T + MC_qk - 1) / MC_qk;
    const size_t max_ab_qk = (MC_qk + MT - 1) / MT;
    const float  scale   = 1.0f / std::sqrt((float)d);

    // Scratch buffers for QK GEMM
    AlignedBuffer<__fp16>   A_cache_qk(num_mc_qk * max_ab_qk * MT * kc_qk);
    AlignedBuffer<uint16_t> K_tile(kc_qk * NT);       // one N-block's transposed K
    // Cpanel for ONE M-block (not all M-blocks) — smaller footprint, stays L1-hot
    AlignedBuffer<__fp16>   Cpanel_qk(max_ab_qk * MT * NT);

    // ===== Step 1: QK GEMM with fused scatter (OPT 1) =======================
    if (phase == 0 || phase == 1) {
    //
    // Pack all Q M-blocks once (Q is reused across all N-tiles).
    {
        auto t0 = Clock::now();
        for (size_t m0 = 0; m0 < (size_t)T; m0 += MC_qk) {
            size_t mc = std::min(MC_qk, (size_t)T - m0);
            pack_A_fp16_8row(Q_h, (size_t)d,
                A_cache_qk.data() + (m0/MC_qk) * max_ab_qk * MT * kc_qk,
                (size_t)T, m0, (size_t)d, 0, mc, kc_qk);
        }
        tim.t_qk_packA += us_since(t0);
    }

    // For each N-tile: transpose K, then for each M-block: GEMM + scatter
    const svfloat16_t vs_scale = svdup_f16((__fp16)scale);
    for (size_t ti = 0; ti < n_tiles; ++ti) {
        size_t n_start = ti * NT;
        size_t nc = std::min(NT, (size_t)T - n_start);

        // Transpose one K tile
        {
            auto t0 = Clock::now();
            pack_K_tile(K_tile.data(), (const uint16_t*)K_h, T, d, NT, n_start);
            tim.t_qk_packK += us_since(t0);
        }

        // OPT 1: For each M-block, compute GEMM then IMMEDIATELY scatter with scale.
        // Cpanel (8*96*2 = 1536B per M-block) is still in L1 when we scatter.
        // This avoids the separate scatter pass that would reload Cpanel from L2.
        for (size_t m0 = 0; m0 < (size_t)T; m0 += MC_qk) {
            size_t mc = std::min(MC_qk, (size_t)T - m0);
            size_t ablocks = (mc + MT - 1) / MT;

            auto t0 = Clock::now();

            // Reset Cpanel pointer for the kernel (it advances Cpanel internally)
            __fp16* cp_ptr = Cpanel_qk.data();
            const __fp16* Ap = A_cache_qk.data() + (m0/MC_qk) * max_ab_qk * MT * kc_qk;
            if (my_svcnth() >= 128) {
                // VL=16: use 1VL kernel per M-block, writes to Cpanel with ldc=NT
                for (size_t mb = 0; mb < ablocks; ++mb)
                    cache_qk_8x1VL_ldc(
                        Ap + mb * 8 * kc_qk, (const __fp16*)K_tile.data(),
                        cp_ptr + mb * 8 * NT, (int)kc_qk, (int)NT);
            } else {
                sve_interleaved_fp16_mla_8x3VL(Ap, (const __fp16*)K_tile.data(),
                                               cp_ptr, (int)ablocks, 1, (int)kc_qk);
            }

            // Fused scatter: Cpanel -> scores with scale (Cpanel is L1-hot)
            scatter_qk_fused(Cpanel_qk.data(), scores,
                             m0, ablocks, n_start, nc, (size_t)T, T_pad, vs_scale);

            tim.t_qk_gemm += us_since(t0);
        }
    }

    } // end phase 0/1 (QK)

    // ===== Step 2: Softmax on scores[T, T_pad] (causal or plain) =============
    if (phase == 0 || phase == 2) {
        auto t0 = Clock::now();
        if (causal) causal_softmax_fp16(scores, T, T_pad);
        else        noncausal_softmax_fp16(scores, T, T_pad);
        tim.t_softmax += us_since(t0);
    }

    // ===== Step 3: PV GEMM with dedicated 8×64 kernel ============================
    if (phase == 0 || phase == 3) {
    // scores[T, T_pad] x V_h[T, d=64] -> out[T, d]
    //
    // Uses sve_cache_pv_8x64: reads V row-major directly (no prepack needed).
    // K-tile outer loop: pack A (scores rows) per M-block, call 8x64 kernel per
    // 8-row sub-block, scatter/accumulate to output.
    //
    // OPT 3: KC_pv=256. A_panel = MC*KC*2 = 128*256*2 = 64KB fits L1.
    {
        const size_t K_tiles_pv = ((size_t)T + KC_pv - 1) / KC_pv;
        const size_t max_ab_pv  = (MC_pv + MT - 1) / MT;

        AlignedBuffer<__fp16> A_cache_pv(max_ab_pv * MT * KC_pv);
        AlignedBuffer<__fp16> PV_panel(8 * (size_t)d);

        const size_t VLh = my_svcnth();
        svbool_t pg_full = svptrue_b16();

        for (size_t kt = 0; kt < K_tiles_pv; ++kt) {
            size_t k0 = kt * KC_pv;
            size_t kc = std::min(KC_pv, (size_t)T - k0);

            // V read row-major directly by 8x64 kernel — no prepack needed

            for (size_t m0 = 0; m0 < (size_t)T; m0 += MC_pv) {
                size_t mc = std::min(MC_pv, (size_t)T - m0);
                size_t ablocks = (mc + MT - 1) / MT;

                // Pack A (scores rows)
                {
                    auto t0 = Clock::now();
                    pack_A_fp16_8row((const __fp16*)scores, T_pad,
                        A_cache_pv.data(),
                        (size_t)T, m0, (size_t)T, k0, mc, kc);
                    tim.t_pv_packA += us_since(t0);
                }

                // PV kernel per 8-row block
                {
                    auto t0 = Clock::now();
                    for (size_t mb = 0; mb < ablocks; ++mb) {
                        size_t row_base = mb * MT;
                        size_t rows_this = std::min((size_t)MT, mc - row_base);

                        if (my_svcnth() >= 128) // VL=16: predicated 1VL PV kernel
                            cache_pv_8x64_vl16(
                                A_cache_pv.data() + mb * MT * kc,
                                V_h + k0 * (size_t)d,
                                PV_panel.data(),
                                (int)kc, (int)d);
                        else
                            sve_cache_pv_8x64(
                                A_cache_pv.data() + mb * MT * kc,
                                V_h + k0 * (size_t)d,
                                PV_panel.data(),
                                (int)kc, (int)d);

                        // Scatter PV_panel -> out (overwrite if kt==0, accumulate if kt>0)
                        for (size_t r = 0; r < rows_this; ++r) {
                            size_t gr = m0 + row_base + r;
                            if (gr >= (size_t)T) break;
                            __fp16* o_row = out + gr * (size_t)d;
                            const __fp16* pv = PV_panel.data() + r * (size_t)d;
                            size_t i = 0;
                            if (kt == 0) {
                                for (; i + VLh <= (size_t)d; i += VLh)
                                    svst1_f16(pg_full, o_row+i, svld1_f16(pg_full, pv+i));
                                if (i < (size_t)d) {
                                    svbool_t pg = svwhilelt_b16_u64(i, (size_t)d);
                                    svst1_f16(pg, o_row+i, svld1_f16(pg, pv+i));
                                }
                            } else {
                                for (; i + VLh <= (size_t)d; i += VLh)
                                    svst1_f16(pg_full, o_row+i,
                                        svadd_f16_x(pg_full,
                                            svld1_f16(pg_full, o_row+i),
                                            svld1_f16(pg_full, pv+i)));
                                if (i < (size_t)d) {
                                    svbool_t pg = svwhilelt_b16_u64(i, (size_t)d);
                                    svst1_f16(pg, o_row+i,
                                        svadd_f16_x(pg,
                                            svld1_f16(pg, o_row+i),
                                            svld1_f16(pg, pv+i)));
                                }
                            }
                        }
                    }
                    tim.t_pv_gemm += us_since(t0);
                }
            }
        }
    }
    } // end phase 0/3 (PV)
}

// =============================================================================
// Main
// =============================================================================
static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [T] [H_q] [H_kv] [d] [n_iter] [MC_qk] [KC_pv] [MC_pv] [phase] <causal|noncausal>\n"
        "       defaults: 512 1 1 64 1 128 256 128 0\n"
        "       trailing attention-mode argument is MANDATORY (causal or noncausal)\n",
        prog);
}

int main(int argc, char** argv)
{
    setbuf(stdout, NULL);  // unbuffered stdout for gem5 SE mode
    // MANDATORY trailing attention-mode argument (appended after all positional args)
    bool causal;
    {
        const char* mode = (argc >= 2) ? argv[argc - 1] : nullptr;
        if      (mode && std::strcmp(mode, "causal")    == 0) causal = true;
        else if (mode && std::strcmp(mode, "noncausal") == 0) causal = false;
        else { print_usage(argv[0]); exit(2); }
    }
    const int pargc = argc - 1;  // positional args exclude the trailing mode
    const int    T      = (pargc > 1) ? std::atoi(argv[1]) : 512;
    const int    H_q    = (pargc > 2) ? std::atoi(argv[2]) : 1;
    const int    H_kv   = (pargc > 3) ? std::atoi(argv[3]) : 1;
    const int    d      = (pargc > 4) ? std::atoi(argv[4]) : 64;
    const int    N_ITER = (pargc > 5) ? std::atoi(argv[5]) : 1;
    const size_t MC_qk  = (pargc > 6) ? (size_t)std::atoi(argv[6]) : 128;
    const size_t KC_pv  = (pargc > 7) ? (size_t)std::atoi(argv[7]) : 256;
    const size_t MC_pv  = (pargc > 8) ? (size_t)std::atoi(argv[8]) : 128;
    // phase: 0=all(default), 1=QK only, 2=softmax only, 3=PV only
    const int    phase  = (pargc > 9) ? std::atoi(argv[9]) : 0;

    if (H_q % H_kv != 0) { fprintf(stderr, "ERROR: H_q(%d) must be multiple of H_kv(%d)\n", H_q, H_kv); return 1; }
    const int group_ratio = H_q / H_kv;

    const size_t NT     = get_NT();
    const size_t T_pad  = ((size_t)T + NT - 1) / NT * NT;

    // GFLOP: QK (2*T*d*T) + PV (2*T*T*d) per Q head
    const double GFLOP_total = 4.0 * H_q * T * (double)T * d * 1e-9;

    const char* phase_names[] = {"all", "QK", "softmax", "PV"};
    printf("=== cache_unfused_attn (8x64 PV, GQA, phase=%s, %s) ===\n",
           phase_names[phase], causal ? "causal" : "noncausal");
    printf("  T=%d  H_q=%d  H_kv=%d  group_ratio=%d  d=%d  NT=%zu  n_iter=%d  phase=%d\n",
           T, H_q, H_kv, group_ratio, d, NT, N_ITER, phase);
    printf("  MC_qk=%zu  KC_pv=%zu  MC_pv=%zu\n", MC_qk, KC_pv, MC_pv);

    // Allocate: Q[H_q*T*d], K[H_kv*T*d], V[H_kv*T*d], scores[T*T_pad] (workspace), out[H_q*T*d]
    AlignedBuffer<__fp16> Q((size_t)H_q * T * d);
    AlignedBuffer<__fp16> K((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> V((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> scores((size_t)T * T_pad);  // single-head workspace, reused
    AlignedBuffer<__fp16> out((size_t)H_q * T * d);

    fill_random(Q.data(), (size_t)H_q*T*d,  -0.1f, 0.1f, 1);
    fill_random(K.data(), (size_t)H_kv*T*d, -0.1f, 0.1f, 2);
    fill_random(V.data(), (size_t)H_kv*T*d, -0.1f, 0.1f, 3);
    std::memset(scores.data(), 0, (size_t)T * T_pad * sizeof(__fp16));
    std::memset(out.data(),    0, (size_t)H_q * T * d * sizeof(__fp16));

    // === Phase-aware execution ===
    // phase=0: time all 3 steps together
    // phase=1: time QK only
    // phase=2: run QK untimed to produce scores, then time softmax only
    // phase=3: run QK+softmax untimed, then time PV only

    // For phase 2/3: pre-run earlier steps without timing to produce intermediate data
    if (phase >= 2) {
        printf("  [prep] Running QK (untimed) to produce scores...\n");
        AttnTiming dummy;
        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_ratio;
            run_attn_head_opt(Q.data()+(size_t)hq*T*d, K.data()+(size_t)hkv*T*d,
                     V.data()+(size_t)hkv*T*d, scores.data(),
                     out.data()+(size_t)hq*T*d, T, d, MC_qk, KC_pv, MC_pv, dummy, 1, causal);
        }
    }
    if (phase >= 3) {
        printf("  [prep] Running softmax (untimed) on scores...\n");
        if (causal) causal_softmax_fp16(scores.data(), T, T_pad);
        else        noncausal_softmax_fp16(scores.data(), T, T_pad);
    }

    // COLD + WARM: flush cache, then time only the selected phase
    printf("--- [COLD] ---\n");
    flush_dcache_range(Q.data(),      (size_t)H_q*T*d   * sizeof(__fp16));
    flush_dcache_range(K.data(),      (size_t)H_kv*T*d  * sizeof(__fp16));
    flush_dcache_range(V.data(),      (size_t)H_kv*T*d  * sizeof(__fp16));
    flush_dcache_range(scores.data(), (size_t)T*T_pad    * sizeof(__fp16));
    flush_dcache_range(out.data(),    (size_t)H_q*T*d   * sizeof(__fp16));

    {
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        auto t0 = Clock::now();
        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_ratio;
            AttnTiming dummy;
            run_attn_head_opt(Q.data()+(size_t)hq*T*d, K.data()+(size_t)hkv*T*d,
                     V.data()+(size_t)hkv*T*d, scores.data(),
                     out.data()+(size_t)hq*T*d, T, d, MC_qk, KC_pv, MC_pv, dummy, phase, causal);
        }
        double cold_us = us_since(t0);
        ROI_END();
        printf("  COLD: %.2f ms\n", cold_us/1000.0);
#ifdef GEM5
        m5_dump_stats(0, 0);
        m5_reset_stats(0, 0);   // reset so WARM runs don't accumulate into ROI
#endif
    }

    if (N_ITER > 0) {
        printf("--- [WARM] %d iterations ---\n", N_ITER);
        double sum_us = 0;
        for (int iter = 0; iter < N_ITER; ++iter) {
            auto t0 = Clock::now();
            for (int hq = 0; hq < H_q; ++hq) {
                int hkv = hq / group_ratio;
                AttnTiming dummy;
                run_attn_head_opt(Q.data()+(size_t)hq*T*d, K.data()+(size_t)hkv*T*d,
                         V.data()+(size_t)hkv*T*d, scores.data(),
                         out.data()+(size_t)hq*T*d, T, d, MC_qk, KC_pv, MC_pv, dummy, phase, causal);
            }
            sum_us += us_since(t0);
        }
        double avg = sum_us / N_ITER;
        printf("  WARM avg: %.2f ms  ->  %.2f GFLOPS\n", avg/1000.0, GFLOP_total/(avg*1e-6));
    }

    // ── OUTSIDE ROI: attention mode + logical-output checksum ────────────────
    // out is [H_q*T, d] row-major with no padding: logical region == full buffer.
    printf("ATTENTION_MODE=%s\n", causal ? "causal" : "noncausal");
    { double s=0; for(size_t i=0;i<(size_t)H_q*T*d;i++) s+=(double)(float)out.data()[i]; printf("CHECKSUM: %.9f\n",s); }

    // ── SELF_CHECK (env CF_SELF_CHECK=1): FP64 reference attention ───────────
    if (const char* sc = std::getenv("CF_SELF_CHECK"); sc && sc[0] == '1') {
        if (phase != 0)
            printf("SELF_CHECK SKIP (phase=%d != 0: output is not full attention)\n", phase);
        else
            self_check_attention(Q.data(), K.data(), V.data(), out.data(),
                                 T, H_q, H_kv, d, causal);
    }
    return 0;
}
