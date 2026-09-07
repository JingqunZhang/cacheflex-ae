// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * Cache-resident FlashAttention (d=64) with an 8x64 PV kernel, adaptive Bc,
 * and a strided QK output panel. K and V use the ordinary cache hierarchy.
 *
 * Algorithm (Flash Attention v2, online softmax):
 *   Prepack K → K_T[d,T] → K_T_knm[N_tiles × d × NT]  (once per head, in ROI)
 *   Prepack V[T,d] → V_knm[N_tiles × NT × NT]          (once per head, in ROI)
 *   For each i-block (Br rows of Q):
 *     Init O_acc[Br,d]=0, m[Br]=-inf, l[Br]=0
 *     Pack Q[i-block] → qk_Ap (k-major 8-way, reused across j-loop)
 *     For each j-tile (jt, NT cols of K/V):
 *       QK: sve_interleaved_fp16_mla_8x3VL(qk_Ap, K_T_knm[jt], S_panel, ablocks, 1, d)
 *       Mask S_panel cols nc..NT-1 to -inf if last tile (nc < NT)
 *       Online softmax: P = exp(S*scale - m_new), update m, l
 *       Pack P → pv_Ap
 *       PV: sve_interleaved_fp16_mla_8x3VL(pv_Ap, V_knm[jt], PV_panel, ablocks, 1, NT)
 *       Accumulate PV_panel[:,0..d-1] → O_acc (FP32)
 *     O[i-block] = O_acc / l  (FP32→FP16)
 *
 * Memory layout (T=512, d=64, NT=96):
 *   K_T_knm: 6 × 64 × 96 × 2 =  73.7 KB (in L2)
 *   V_knm:   6 × 96 × 96 × 2 = 110.6 KB (in L2)
 *   Total:  ~184 KB < L2=512 KB — K/V stay in L2 across all i-blocks
 *
 * Usage: ./bin/cache_flash_attn [T] [H_q] [H_kv] [d] [n_iter] [Br] [Bc] <causal|noncausal>
 *        default: 512 1 1 64 1 0 0  — trailing attention mode is MANDATORY
 *        causal mode: block-skipping causal flash attention (skip future KV
 *        blocks, element-wise mask on diagonal blocks)
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
#  define ROI_BEGIN()  m5_work_begin(0, 0)
#  define ROI_END()    m5_work_end(0, 0)
#else
#  define ROI_BEGIN()  do{}while(0)
#  define ROI_END()    do{}while(0)
#endif

using Clock = std::chrono::high_resolution_clock;
static double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

#ifdef NO_PHASE_TIMERS
#  define PHASE_T0() (Clock::time_point{})
#  define PHASE_US(x) ((void)(x), 0.0)
#else
#  define PHASE_T0() Clock::now()
#  define PHASE_US(x) us_since(x)
#endif


static inline size_t my_svcnth() { size_t n; __asm__("cnth %0":"=r"(n)); return n; }

// VL=16 (svcnth=128): use 1VL=128 to avoid NT=384 > T issues
// VL≤8: use 3VL as before
static inline size_t get_NT() {
    size_t s = my_svcnth();
    return (s >= 128) ? s : 3 * s;
}
static constexpr size_t MT     = 8;

// ── Aligned buffer ──────────────────────────────────────────────────────────
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

static void flush_dcache_range(const void* ptr, size_t bytes)
{
#ifndef GEM5_SE
    uintptr_t addr = (uintptr_t)ptr & ~63UL;
    uintptr_t end  = (uintptr_t)ptr + bytes;
    for (; addr < end; addr += 64)
        __asm__ volatile("dc civac, %0" :: "r"(addr) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
    __asm__ volatile("isb"    ::: "memory");
#else
    (void)ptr; (void)bytes;
    __asm__ volatile("" : : : "memory");
#endif
}

// ── SVE FP16 GEMM micro-kernel: 8 rows × 3VL cols, configurable output stride
// ldc_elements: output row stride in FP16 elements (= Bc for QK, = NT for PV)
// Single-block version (ablocks=1, bblocks=1) for use in multi-Bc flash attention
static void sve_mla_8x3VL_ldc(
    const __fp16 *Apanel, const __fp16 *Bpanel, __fp16 *Cpanel,
    int K, int ldc_elements)
{
    __asm__ __volatile__(
        "ptrue p0.b\n"
        "mov x22, %x[Bpanel]\n"
        "mov x20, %x[K]\n"
        "sub x20, x20, #1\n"
        "lsl x4, %x[ldc], #1\n"       // x4 = ldc * 2 bytes (row stride)
        // Precompute row base addresses
        "mov x5, %x[Cpanel]\n"
        "add x6,  x5, x4\n"  "add x7,  x6, x4\n"  "add x8,  x7, x4\n"
        "add x9,  x8, x4\n"  "add x10, x9, x4\n"  "add x11, x10, x4\n"
        "add x12, x11, x4\n"
        // Zero accumulators
        "mov z8.b, #0\n"  "mov z9.b, #0\n"  "mov z10.b,#0\n" "mov z11.b,#0\n"
        "mov z12.b,#0\n"  "mov z13.b,#0\n"  "mov z14.b,#0\n" "mov z15.b,#0\n"
        "mov z16.b,#0\n"  "mov z17.b,#0\n"  "mov z18.b,#0\n" "mov z19.b,#0\n"
        "mov z20.b,#0\n"  "mov z21.b,#0\n"  "mov z22.b,#0\n" "mov z23.b,#0\n"
        "mov z24.b,#0\n"  "mov z25.b,#0\n"  "mov z26.b,#0\n" "mov z27.b,#0\n"
        "mov z28.b,#0\n"  "mov z29.b,#0\n"  "mov z30.b,#0\n" "mov z31.b,#0\n"
        // Load first B
        "ld1h   { z2.h }, p0/Z, [x22]\n"
        "ld1h   { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
        "ld1h   { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
        "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
        "cmp x20, #0x2\n" "blt 4f\n"
        // ── K-loop (2-unrolled) ──
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
        // ── K tail ──
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
        // ── Store with ldc stride (same as 11 SPM kernel pattern) ──
        "5:\n"
        "st1h { z8.h },  p0, [x5]\n"    "st1h { z9.h },  p0, [x5, #1, MUL VL]\n"    "st1h { z10.h }, p0, [x5, #2, MUL VL]\n"
        "st1h { z11.h }, p0, [x6]\n"    "st1h { z12.h }, p0, [x6, #1, MUL VL]\n"    "st1h { z13.h }, p0, [x6, #2, MUL VL]\n"
        "st1h { z14.h }, p0, [x7]\n"    "st1h { z15.h }, p0, [x7, #1, MUL VL]\n"    "st1h { z16.h }, p0, [x7, #2, MUL VL]\n"
        "st1h { z17.h }, p0, [x8]\n"    "st1h { z18.h }, p0, [x8, #1, MUL VL]\n"    "st1h { z19.h }, p0, [x8, #2, MUL VL]\n"
        "st1h { z20.h }, p0, [x9]\n"    "st1h { z21.h }, p0, [x9, #1, MUL VL]\n"    "st1h { z22.h }, p0, [x9, #2, MUL VL]\n"
        "st1h { z23.h }, p0, [x10]\n"   "st1h { z24.h }, p0, [x10, #1, MUL VL]\n"   "st1h { z25.h }, p0, [x10, #2, MUL VL]\n"
        "st1h { z26.h }, p0, [x11]\n"   "st1h { z27.h }, p0, [x11, #1, MUL VL]\n"   "st1h { z28.h }, p0, [x11, #2, MUL VL]\n"
        "st1h { z29.h }, p0, [x12]\n"   "st1h { z30.h }, p0, [x12, #1, MUL VL]\n"   "st1h { z31.h }, p0, [x12, #2, MUL VL]\n"
        : [Apanel] "+&r" (Apanel)
        : [Bpanel] "r" (Bpanel), [Cpanel] "r" (Cpanel),
          [K] "r" ((size_t)K), [ldc] "r" ((size_t)ldc_elements)
        : "cc", "memory", "p0",
          "x4","x5","x6","x7","x8","x9","x10","x11","x12",
          "x20","x22",
          "z0","z1","z2","z3","z4","z5","z6","z7",
          "z8","z9","z10","z11","z12","z13","z14","z15",
          "z16","z17","z18","z19","z20","z21","z22","z23",
          "z24","z25","z26","z27","z28","z29","z30","z31"
    );
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

// ── Original kernel (for PV GEMM where stride=NT is fine) ────────────────────
extern "C" {
void sve_interleaved_fp16_mla_8x3VL_13(
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
        "st1h { z8.h },  p0, [%x[Cpanel]]\n"
        "st1h { z9.h },  p0, [%x[Cpanel], #1, MUL VL]\n"
        "st1h { z10.h }, p0, [%x[Cpanel], #2, MUL VL]\n"
        "st1h { z11.h }, p0, [%x[Cpanel], #3, MUL VL]\n"
        "st1h { z12.h }, p0, [%x[Cpanel], #4, MUL VL]\n"
        "st1h { z13.h }, p0, [%x[Cpanel], #5, MUL VL]\n"
        "st1h { z14.h }, p0, [%x[Cpanel], #6, MUL VL]\n"
        "st1h { z15.h }, p0, [%x[Cpanel], #7, MUL VL]\n"
        "addvl %x[Cpanel], %x[Cpanel], #16\n"
        "st1h { z16.h }, p0, [%x[Cpanel], #-8, MUL VL]\n"
        "st1h { z17.h }, p0, [%x[Cpanel], #-7, MUL VL]\n"
        "st1h { z18.h }, p0, [%x[Cpanel], #-6, MUL VL]\n"
        "st1h { z19.h }, p0, [%x[Cpanel], #-5, MUL VL]\n"
        "st1h { z20.h }, p0, [%x[Cpanel], #-4, MUL VL]\n"
        "st1h { z21.h }, p0, [%x[Cpanel], #-3, MUL VL]\n"
        "st1h { z22.h }, p0, [%x[Cpanel], #-2, MUL VL]\n"
        "st1h { z23.h }, p0, [%x[Cpanel], #-1, MUL VL]\n"
        "st1h { z24.h }, p0, [%x[Cpanel]]\n"
        "st1h { z25.h }, p0, [%x[Cpanel], #1, MUL VL]\n"
        "st1h { z26.h }, p0, [%x[Cpanel], #2, MUL VL]\n"
        "st1h { z27.h }, p0, [%x[Cpanel], #3, MUL VL]\n"
        "st1h { z28.h }, p0, [%x[Cpanel], #4, MUL VL]\n"
        "st1h { z29.h }, p0, [%x[Cpanel], #5, MUL VL]\n"
        "st1h { z30.h }, p0, [%x[Cpanel], #6, MUL VL]\n"
        "st1h { z31.h }, p0, [%x[Cpanel], #7, MUL VL]\n"
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

// ── pack_A_fp16_8row (SVE vectorized, same as kernels_sve.hpp) ───────────────
static void pack_A_fp16_8row(
    const __fp16* A, size_t lda, __fp16* Apanel,
    size_t M, size_t m0, size_t K, size_t k0, size_t mc, size_t kc)
{
    const size_t VLh = my_svcnth();
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

// ── prepack_B_knm ────────────────────────────────────────────────────────────
// Pack B[K,N] row-major (stride ldb=N) → KNM layout [K_tiles × N_tiles × KC × NT]
static void prepack_B_knm(const __fp16* B, __fp16* B_packed,
                           size_t K, size_t N, size_t KC, size_t ldb = 0)
{
    if (!ldb) ldb = N;
    const size_t NT   = get_NT();
    const size_t N_tiles = (N + NT - 1) / NT;
    const size_t K_tiles = (K + KC - 1) / KC;
    for (size_t kt = 0; kt < K_tiles; ++kt) {
        size_t k0 = kt * KC, kc = std::min(KC, K - k0);
        for (size_t nt = 0; nt < N_tiles; ++nt) {
            size_t n0 = nt * NT, nc = std::min(NT, N - n0);
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

// ── K transpose ─────────────────────────────────────────────────────────────
static void transpose_K(const __fp16* K, __fp16* K_T, int T, int d)
{
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < d; ++k)
            K_T[(size_t)k * T + t] = K[(size_t)t * d + k];
}

// ── SVE exp + helpers ────────────────────────────────────────────────────────
static inline svfloat32_t sve_exp_f32(svbool_t pg, svfloat32_t x)
{
    x  = svmax_n_f32_x(pg, x, -88.0f);
    svfloat32_t nf = svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.44269504088896f));
    svint32_t   ni = svcvt_s32_f32_x(pg, nf);
    svfloat32_t r  = svmls_n_f32_x(pg, x, nf,  6.93147182e-1f);
    r              = svmls_n_f32_x(pg, r, nf, -1.90465430e-9f);
    svfloat32_t p  = svdup_n_f32(1.0f/120.0f);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f/24.0f), r, p);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f/ 6.0f), r, p);
    p = svmla_f32_x(pg, svdup_n_f32(0.5f),        r, p);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f),        r, p);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f),        r, p);
    return svscale_f32_x(pg, p, ni);
}

static inline void scale_f32_sve(float* p, float alpha, size_t n)
{
    size_t VL = svcntw(), i = 0;
    for (; i + VL <= n; i += VL) {
        svbool_t pg = svptrue_b32();
        svst1_f32(pg, p+i, svmul_n_f32_x(pg, svld1_f32(pg, p+i), alpha));
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32_u64(i, n);
        svst1_f32(pg, p+i, svmul_n_f32_x(pg, svld1_f32(pg, p+i), alpha));
    }
}

// Accumulate n contiguous FP16 values into FP32 dst
// Full-width fp16 loads; zip1/zip2 keep the two f32 half-blocks contiguous,
// so per-element semantics match the half-width formulation exactly.
static inline void acc_f16_f32_n(const __fp16* src, float* dst, size_t n)
{
    const size_t VLh = svcnth(), VL32 = svcntw();
    const svbool_t pth = svptrue_b16(), pt32 = svptrue_b32();
    size_t i = 0;
    for (; i + VLh <= n; i += VLh) {
        svfloat16_t v = svld1_f16(pth, src + i);
        svst1_f32(pt32, dst+i, svadd_f32_x(pt32, svld1_f32(pt32, dst+i),
                                             svcvt_f32_f16_x(pt32, svzip1_f16(v, v))));
        svst1_f32(pt32, dst+i+VL32, svadd_f32_x(pt32, svld1_f32(pt32, dst+i+VL32),
                                             svcvt_f32_f16_x(pt32, svzip2_f16(v, v))));
    }
    for (; i < n; i += VL32) {
        svbool_t pg32 = svwhilelt_b32_u64(i, n);
        svfloat16_t v16 = svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(src + i)));
        svst1_f32(pg32, dst+i, svadd_f32_x(pg32, svld1_f32(pg32, dst+i),
                                             svcvt_f32_f16_z(pg32, v16)));
    }
}

// ── Flash Attention head (no SPM) ────────────────────────────────────────────
//
// K_T_knm layout: [1 K-tile × N_tiles × d × NT]  (from prepack of K_T[d,T])
// V_knm layout:   [N_tiles × 1 N-tile × NT × NT] (from prepack of V[T,d])
//
// For each jt: K_T_knm_j = K_T_knm + jt*d*NT,  V_knm_j = V_knm + jt*NT*NT
// QK  GEMM K-dim = d   (inner dim = head dimension)
// PV  GEMM K-dim = NT  (inner dim = kv block width; last block may have nc<NT)
//
struct FlashTiming {
    double t_packA_q=0, t_qk=0, t_sm=0, t_pv_packA=0, t_pv_kern=0, t_fin=0;
};

static void flash_attn_head_nospm(
    const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O,
    int T, int d, float scale, int Br, int Bc,
    __fp16* K_T, __fp16* K_T_knm,
    __fp16* Q_tile, __fp16* S_panel, __fp16* P_tile, __fp16* PV_panel,
    __fp16* qk_Ap, __fp16* pv_Ap,
    float*  O_acc, float*  m_vec, float*  l_vec,
    bool causal = false, FlashTiming* ft = nullptr)
{
    const size_t NT    = get_NT();
    auto _now = []{ return Clock::now(); };

    // Plan B: K_T_knm already pre-packed (simulating pre-packed KV cache)
    // transpose_K and prepack_B_knm called BEFORE ROI by caller
    // V read row-major directly by 8×64 PV kernel (no prepack needed)

    for (int i = 0; i < T; i += Br) {
        int br = std::min(Br, T - i);
        size_t ablocks = ((size_t)br + MT - 1) / MT;

        std::memcpy(Q_tile, Q + (size_t)i * d, (size_t)br * d * sizeof(__fp16));
        std::fill_n(O_acc,  (size_t)br * d, 0.f);
        std::fill_n(m_vec,  br, -1e30f);
        std::fill_n(l_vec,  br, 0.f);

        { auto _t = PHASE_T0();
        pack_A_fp16_8row(Q_tile, (size_t)d, qk_Ap, (size_t)br, 0, (size_t)d, 0, (size_t)br, (size_t)d);
        if (ft) ft->t_packA_q += PHASE_US(_t); }

        // ── j-block loop: step by Bc (multiple of NT) ────────────────────────
        for (int j = 0; j < T; j += Bc) {
            int bc = std::min(Bc, T - j);
            if (causal) {
                // Block-skip: KV block starts beyond the last query row of this
                // i-block (block_col_start > row_block_end) -> entirely future,
                // and so are all later j-blocks: stop the j-loop.
                if (j >= i + br) break;
                // Diagonal block: clamp KV width to the causal frontier so the
                // QK GEMM / softmax / PV skip fully-masked NT sub-blocks.
                bc = std::min(bc, i + br - j);
            }
            size_t bblocks = ((size_t)bc + NT - 1) / NT;
            size_t bc_padded = bblocks * NT;  // stride for S_panel/P_tile (NT-aligned)

            // ── QK GEMM: Q[br,d] × K_T[d, bc] → S_panel[br, bc] ────────────
            { auto _t = PHASE_T0();
            for (size_t nb = 0; nb < bblocks; ++nb) {
                size_t jt = (size_t)j / NT + nb;
                size_t nc_this = std::min(NT, (size_t)T - jt * NT);
                const __fp16* K_T_knm_j = K_T_knm + jt * (size_t)d * NT;

                for (size_t mb = 0; mb < ablocks; ++mb) {
                    // ldc kernel: writes directly to S_panel with stride=bc (no scatter!)
                    if (my_svcnth() >= 128) // VL=16: use 1VL kernel
                        cache_qk_8x1VL_ldc(
                            qk_Ap + mb * 8 * (size_t)d, K_T_knm_j,
                            S_panel + mb * 8 * bc_padded + nb * NT,
                            (int)d, (int)bc_padded);
                    else
                        sve_mla_8x3VL_ldc(
                            qk_Ap + mb * 8 * (size_t)d, K_T_knm_j,
                            S_panel + mb * 8 * bc_padded + nb * NT,
                            (int)d, (int)bc_padded);
                }

                // Mask invalid cols in last NT-block
                if (nc_this < NT) {
                    const __fp16 neg_inf = (__fp16)(-65504.f);
                    for (size_t r = 0; r < (size_t)br; ++r) {
                        __fp16* s_row = S_panel + r * bc_padded + nb * NT;
                        for (size_t c = nc_this; c < NT; ++c)
                            s_row[c] = neg_inf;
                    }
                }
            }

            // CAUSAL: element-wise mask on the diagonal block (col > row -> -inf)
            // before the online-softmax update (ported from
            // flash_attention/src/cache_flash_attn_causal.cpp, LLaMA prefill semantics)
            if (causal) {
                const __fp16 neg_inf = (__fp16)(-65504.f);
                for (int r = 0; r < br; ++r) {
                    int qi = i + r;                // global query row
                    int start = qi - j + 1;        // first masked column in this j-block
                    if (start < 0) start = 0;
                    for (int c = start; c < bc; ++c)
                        S_panel[(size_t)r * bc_padded + c] = neg_inf;
                }
            }
            if (ft) ft->t_qk += PHASE_US(_t); }

            // ── Online softmax on S_panel[br, bc] ────────────────────────────
            // ── Online softmax (predicated loop, same as spm_flash) ──
            { auto _t = PHASE_T0();
            const size_t VLh_sm = svcnth();
            const size_t VL32_sm = svcntw();
            const svbool_t pth_sm = svptrue_b16();
            const svbool_t pt32_sm = svptrue_b32();
            for (int r = 0; r < br; ++r) {
                const __fp16* s_row = S_panel + (size_t)r * bc_padded;
                __fp16*       p_row = P_tile  + (size_t)r * bc_padded;

                // Row max computed in fp16 (exact: fp16 compare is exact and
                // x -> scale*x is monotone for scale > 0, so scaling the max
                // afterwards yields bit-identical m_ij).
                svfloat16_t vmax16 = svdup_n_f16((__fp16)-65504.f);
                size_t c = 0;
                for (; c + VLh_sm <= (size_t)bc; c += VLh_sm)
                    vmax16 = svmax_f16_m(pth_sm, vmax16, svld1_f16(pth_sm, s_row + c));
                if (c < (size_t)bc) {
                    svbool_t pg = svwhilelt_b16_u64(c, (size_t)bc);
                    vmax16 = svmax_f16_m(pg, vmax16, svld1_f16(pg, s_row + c));
                }
                float m_ij = scale * (float)svmaxv_f16(pth_sm, vmax16);
                float m_old = m_vec[r];
                float m_new = std::max(m_old, m_ij);
                float alpha = expf(m_old - m_new);

                // exp + sum + P store, full-width fp16 loads/stores.
                // zip1/zip2 expand to the SAME contiguous 16-element f32
                // blocks the half-width loop processed, in the same order,
                // so vsum accumulation is bit-identical; uzp1 re-packs the
                // two converted halves for a single full-width store.
                const svfloat32_t vneg_m = svdup_n_f32(-m_new);
                svfloat32_t vsum = svdup_n_f32(0.f);
                c = 0;
                for (; c + VLh_sm <= (size_t)bc; c += VLh_sm) {
                    svfloat16_t v = svld1_f16(pth_sm, s_row + c);
                    svfloat32_t s_lo = svcvt_f32_f16_x(pt32_sm, svzip1_f16(v, v));
                    svfloat32_t s_hi = svcvt_f32_f16_x(pt32_sm, svzip2_f16(v, v));
                    svfloat32_t e_lo = sve_exp_f32(pt32_sm,
                        svmla_n_f32_x(pt32_sm, vneg_m, s_lo, scale));
                    svfloat32_t e_hi = sve_exp_f32(pt32_sm,
                        svmla_n_f32_x(pt32_sm, vneg_m, s_hi, scale));
                    vsum = svadd_f32_x(pt32_sm, vsum, e_lo);
                    vsum = svadd_f32_x(pt32_sm, vsum, e_hi);
                    svst1_f16(pth_sm, p_row + c,
                        svuzp1_f16(svcvt_f16_f32_x(pt32_sm, e_lo),
                                   svcvt_f16_f32_x(pt32_sm, e_hi)));
                }
                for (; c < (size_t)bc; c += VL32_sm) {
                    svbool_t pg32 = svwhilelt_b32_u64(c, (size_t)bc);
                    svfloat32_t vs = svmla_n_f32_x(pg32, vneg_m,
                        svcvt_f32_f16_x(pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(s_row+c)))), scale);
                    svfloat32_t ve = sve_exp_f32(pg32, vs);
                    vsum = svadd_f32_m(pg32, vsum, ve);
                    svst1h_u32(pg32, (uint16_t*)(p_row+c), svreinterpret_u32_f16(svcvt_f16_f32_x(pg32, ve)));
                }
                m_vec[r] = m_new;
                l_vec[r] = alpha * l_vec[r] + svaddv_f32(pt32_sm, vsum);
                scale_f32_sve(O_acc + (size_t)r * d, alpha, (size_t)d);
            }
            if (ft) ft->t_sm += PHASE_US(_t); }

            // ── PV GEMM: P[br, bc] × V[bc, d=64] → O_acc update ─────────
            for (size_t mb = 0; mb < ablocks; ++mb) {
                size_t row_base = mb * 8;
                size_t rows_this = std::min((size_t)8, (size_t)br - row_base);

                { auto _t = PHASE_T0();
                pack_A_fp16_8row(P_tile, bc_padded, pv_Ap,
                                 (size_t)br, row_base, (size_t)bc, 0, rows_this, (size_t)bc);
                if (ft) ft->t_pv_packA += PHASE_US(_t); }

                { auto _t = PHASE_T0();
                if (svcnth() >= 128) // VL=16: use predicated 1VL kernel
                    cache_pv_8x64_vl16(pv_Ap, V + (size_t)j * d, PV_panel,
                                       (int)bc, (int)d);
                else // VL<=8: use existing 2-register kernel
                    sve_cache_pv_8x64(pv_Ap, V + (size_t)j * d, PV_panel,
                                       (int)bc, (int)d);
                for (size_t rr = 0; rr < rows_this; ++rr)
                    acc_f16_f32_n(PV_panel + rr * d, O_acc + (row_base + rr) * (size_t)d, (size_t)d);
                if (ft) ft->t_pv_kern += PHASE_US(_t); }
            }
        }

        // ── Finalize: O[i:i+br] = O_acc / l ─────────────────────────────────
        { auto _t = PHASE_T0();
        const size_t VLh_fin = svcnth(), VL32_fin = svcntw();
        const svbool_t pth_fin = svptrue_b16(), pt32_fin = svptrue_b32();
        for (int r = 0; r < br; ++r) {
            float        inv_l  = 1.f / l_vec[r];
            const float* o_src  = O_acc + (size_t)r * d;
            __fp16*      o_dst  = O + ((size_t)i + r) * d;
            size_t k = 0;
            for (; k + VLh_fin <= (size_t)d; k += VLh_fin) {
                svfloat32_t lo = svmul_n_f32_x(pt32_fin, svld1_f32(pt32_fin, o_src+k), inv_l);
                svfloat32_t hi = svmul_n_f32_x(pt32_fin, svld1_f32(pt32_fin, o_src+k+VL32_fin), inv_l);
                svst1_f16(pth_fin, o_dst + k,
                    svuzp1_f16(svcvt_f16_f32_x(pt32_fin, lo),
                               svcvt_f16_f32_x(pt32_fin, hi)));
            }
            for (; k < (size_t)d; k += VL32_fin) {
                svbool_t pg32 = svwhilelt_b32_u64(k, (size_t)d);
                svst1h_u32(pg32, (uint16_t*)(o_dst+k), svreinterpret_u32_f16(svcvt_f16_f32_x(pg32, svmul_n_f32_x(pg32, svld1_f32(pg32, o_src+k), inv_l))));
            }
        }
        if (ft) ft->t_fin += PHASE_US(_t); }
    }
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

// ── main ─────────────────────────────────────────────────────────────────────
static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [T] [H_q] [H_kv] [d] [n_iter] [Br] [Bc] <causal|noncausal>\n"
        "       defaults: 512 1 1 64 1 0 0 (0 = auto Br/Bc)\n"
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
    const int T      = (pargc > 1) ? std::atoi(argv[1]) : 512;
    const int H_q    = (pargc > 2) ? std::atoi(argv[2]) : 1;
    const int H_kv   = (pargc > 3) ? std::atoi(argv[3]) : 1;
    const int d      = (pargc > 4) ? std::atoi(argv[4]) : 64;
    const int N_ITER = (pargc > 5) ? std::atoi(argv[5]) : 1;
    const int Br_arg = (pargc > 6) ? std::atoi(argv[6]) : 0;  // 0 = auto
    const int Bc_arg = (pargc > 7) ? std::atoi(argv[7]) : 0; // 0 = auto

    if (H_q % H_kv != 0) { fprintf(stderr, "ERROR: H_q(%d) must be multiple of H_kv(%d)\n", H_q, H_kv); return 1; }
    const int group_ratio = H_q / H_kv;
    // Auto-compute Br/Bc from L1/L2 budget (same logic as 11 SPM)
    // L1 budget: Q_tile[Br,d] + O_acc[Br,d] FP32 = Br * 6d bytes
    const size_t L1_bytes = 64 * 1024;
    const size_t L2_bytes = 512 * 1024;
    int Br_auto = (int)(L1_bytes / (size_t)(6 * d));
    Br_auto = (Br_auto / 8) * 8;
    Br_auto = std::max(8, std::min(Br_auto, T));
    const int Br = (Br_arg > 0) ? Br_arg : Br_auto;
    // L2 budget: K_T_knm + scratch(S + P + pv_Ap)
    // K_T_knm = ceil(T/NT)*d*NT*2 (always needed)
    // scratch = 2*Br*Bc*2 + Br*Bc*2 = 3*Br*Bc*2 (S_panel + P_tile + pv_Ap ≈)
    size_t K_knm_bytes = ((T + (int)get_NT() - 1) / (int)get_NT()) * d * get_NT() * 2;
    size_t l2_remain = (L2_bytes > K_knm_bytes) ? L2_bytes - K_knm_bytes : 0;
    int Bc_auto = (int)(l2_remain / ((size_t)Br * 3 * 2));
    Bc_auto = (Bc_auto / (int)get_NT()) * (int)get_NT();
    Bc_auto = std::max((int)get_NT(), std::min(Bc_auto, T));
    const int Bc = (Bc_arg > 0) ? (((Bc_arg + (int)get_NT() - 1) / (int)get_NT()) * (int)get_NT()) : Bc_auto;
    const float scale = 1.f / std::sqrtf((float)d);

    if (d != 64) {
        fprintf(stderr, "ERROR: d=%d; this binary is tuned for d=64\n", d);
        return 1;
    }

    const size_t NT       = get_NT();
    const size_t N_tiles  = ((size_t)T + NT - 1) / NT;
    const size_t ablocks_max = ((size_t)Br + MT - 1) / MT;

    double GFLOP_head = 4.0 * T * (double)d * T * 1e-9;  // QK + PV per head
    double GFLOP_total = GFLOP_head * H_q;

    printf("=== cache_flash_attn (8x64 PV, GQA, %s) ===\n", causal ? "causal" : "noncausal");
    printf("  T=%d  H_q=%d  H_kv=%d  group_ratio=%d  d=%d  Br=%d  Bc=%d  NT=%zu  n_iter=%d\n",
           T, H_q, H_kv, group_ratio, d, Br, Bc, NT, N_ITER);
    printf("  K_T_knm: %.1fKB (L2)  V: read row-major directly (NO prepack!)\n",
           N_tiles * (size_t)d * NT * 2.0 / 1024);
    printf("  Per-i-block scratch (Br=%d): qk_Ap=%.1f S=%.1f P=%.1f pv_Ap=%.1f PV=%.1f O=%.1fKB\n",
           Br,
           ablocks_max * MT * d * 2.0 / 1024,
           ablocks_max * MT * NT * 2.0 / 1024,
           ablocks_max * MT * NT * 2.0 / 1024,
           ablocks_max * MT * NT * 2.0 / 1024,
           ablocks_max * MT * NT * 2.0 / 1024,
           (size_t)Br * d * 4.0 / 1024);
    printf("  GFLOP/head=%.4f\n\n", GFLOP_head);

    // ── Allocate all buffers ─────────────────────────────────────────────────
    AlignedBuffer<__fp16> Q_buf((size_t)H_q  * T * d);
    AlignedBuffer<__fp16> K_buf((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> V_buf((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> O_buf((size_t)H_q  * T * d);

    // Workspace (shared across heads, reused per iter)
    AlignedBuffer<__fp16> K_T     ((size_t)d * T);
    AlignedBuffer<__fp16> K_T_knm (N_tiles * (size_t)d * NT);
    // V_knm REMOVED — sve_cache_pv_8x64 reads V row-major directly
    AlignedBuffer<__fp16> Q_tile  ((size_t)Br * d);
    AlignedBuffer<__fp16> S_panel (ablocks_max * MT * Bc);   // wider for Bc > NT
    AlignedBuffer<__fp16> P_tile  (ablocks_max * MT * Bc);
    AlignedBuffer<__fp16> PV_panel(8 * d);                   // PV output 8×64 (no padding!)
    AlignedBuffer<__fp16> qk_Ap   (ablocks_max * MT * d);
    AlignedBuffer<__fp16> pv_Ap   (8 * Bc);                // per 8-row block, Bc-wide for PV
    AlignedBuffer<float>  O_acc   ((size_t)Br * d);
    AlignedBuffer<float>  m_vec   (Br);
    AlignedBuffer<float>  l_vec   (Br);

    {
        // Same seed pattern as cache_unfused
        { std::mt19937 r1(1); std::uniform_real_distribution<float> d1(-0.1f,0.1f);
          for(size_t i=0;i<(size_t)H_q*T*d;i++) Q_buf.data()[i]=(__fp16)d1(r1); }
        { std::mt19937 r2(2); std::uniform_real_distribution<float> d2(-0.1f,0.1f);
          for(size_t i=0;i<(size_t)H_kv*T*d;i++) K_buf.data()[i]=(__fp16)d2(r2); }
        { std::mt19937 r3(3); std::uniform_real_distribution<float> d3(-0.1f,0.1f);
          for(size_t i=0;i<(size_t)H_kv*T*d;i++) V_buf.data()[i]=(__fp16)d3(r3); }
    }

    // ── Prefill: K transpose+prepack INSIDE ROI ────────────────────────────
    printf("  K_T_knm: %.1fKB to pre-pack\n\n", N_tiles*(size_t)d*NT*2.0/1024);

    // ── Flush cache to ensure cold start (match spm_flash) ──────────────
    flush_dcache_range(Q_buf.data(),      (size_t)H_q*T*d  * sizeof(__fp16));
    flush_dcache_range(K_buf.data(),      (size_t)H_kv*T*d * sizeof(__fp16));
    flush_dcache_range(V_buf.data(),      (size_t)H_kv*T*d * sizeof(__fp16));
    flush_dcache_range(O_buf.data(),      (size_t)H_q*T*d  * sizeof(__fp16));
    flush_dcache_range(K_T.data(),        (size_t)d*T       * sizeof(__fp16));
    flush_dcache_range(K_T_knm.data(),    N_tiles*(size_t)d*NT * sizeof(__fp16));

    // ── COLD run with breakdown (ROI wraps this) ──────────────────────────
    printf("--- [COLD] ---\n");
    FlashTiming cold_ft;
    {
        auto t0 = Clock::now();
        double t_kprep = 0;
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_ratio;
            if (hq % group_ratio == 0) {
                auto _t = PHASE_T0();
                transpose_K(K_buf.data() + (size_t)hkv*T*d, K_T.data(), T, d);
                prepack_B_knm(K_T.data(), K_T_knm.data(), (size_t)d, (size_t)T, (size_t)d, (size_t)T);
                t_kprep += PHASE_US(_t);
            }
            flash_attn_head_nospm(
                Q_buf.data() + (size_t)hq*T*d,
                K_buf.data() + (size_t)hkv*T*d,
                V_buf.data() + (size_t)hkv*T*d,
                O_buf.data() + (size_t)hq*T*d,
                T, d, scale, Br, Bc,
                K_T.data(), K_T_knm.data(),
                Q_tile.data(), S_panel.data(), P_tile.data(), PV_panel.data(),
                qk_Ap.data(), pv_Ap.data(), O_acc.data(), m_vec.data(), l_vec.data(),
                causal, &cold_ft);
        }
        ROI_END();
        double us = us_since(t0);
#ifdef GEM5
        m5_dump_stats(0, 0);
        m5_reset_stats(0, 0);   // reset so WARM runs don't accumulate into ROI
#endif
        double tot = t_kprep + cold_ft.t_packA_q + cold_ft.t_qk + cold_ft.t_sm
                   + cold_ft.t_pv_packA + cold_ft.t_pv_kern + cold_ft.t_fin;
        printf("  COLD: %.2f ms  ->  %.2f GFLOPS\n", us/1000.0, GFLOP_total/(us*1e-6));
        printf("  Breakdown:\n");
        printf("    K trans+prepack: %7.1f us (%5.1f%%)\n", t_kprep, 100*t_kprep/tot);
        printf("    pack_A(Q):       %7.1f us (%5.1f%%)\n", cold_ft.t_packA_q, 100*cold_ft.t_packA_q/tot);
        printf("    QK GEMM:         %7.1f us (%5.1f%%)\n", cold_ft.t_qk, 100*cold_ft.t_qk/tot);
        printf("    softmax:         %7.1f us (%5.1f%%)\n", cold_ft.t_sm, 100*cold_ft.t_sm/tot);
        printf("    PV pack_A(P):    %7.1f us (%5.1f%%)\n", cold_ft.t_pv_packA, 100*cold_ft.t_pv_packA/tot);
        printf("    PV kernel:       %7.1f us (%5.1f%%)\n", cold_ft.t_pv_kern, 100*cold_ft.t_pv_kern/tot);
        printf("    finalize:        %7.1f us (%5.1f%%)\n", cold_ft.t_fin, 100*cold_ft.t_fin/tot);
        printf("    sum:             %7.1f us\n\n", tot);
    }

    // ── WARM runs ─────────────────────────────────────────────────────────────
    if (N_ITER > 0) {
        printf("--- [WARM] %d iterations ---\n", N_ITER);
        double sum_us = 0;
        FlashTiming warm_ft;
        double t_kprep_warm = 0;
        for (int iter = 0; iter < N_ITER; ++iter) {
            auto t0 = Clock::now();
            for (int hq = 0; hq < H_q; ++hq) {
                int hkv = hq / group_ratio;
                if (hq % group_ratio == 0) {
                    auto _t = PHASE_T0();
                    transpose_K(K_buf.data() + (size_t)hkv*T*d, K_T.data(), T, d);
                    prepack_B_knm(K_T.data(), K_T_knm.data(), (size_t)d, (size_t)T, (size_t)d, (size_t)T);
                    t_kprep_warm += PHASE_US(_t);
                }
                flash_attn_head_nospm(
                    Q_buf.data() + (size_t)hq*T*d,
                    K_buf.data() + (size_t)hkv*T*d,
                    V_buf.data() + (size_t)hkv*T*d,
                    O_buf.data() + (size_t)hq*T*d,
                    T, d, scale, Br, Bc,
                    K_T.data(), K_T_knm.data(),
                    Q_tile.data(), S_panel.data(), P_tile.data(), PV_panel.data(),
                    qk_Ap.data(), pv_Ap.data(), O_acc.data(), m_vec.data(), l_vec.data(),
                    causal, &warm_ft);
            }
            sum_us += us_since(t0);
        }
        double avg = sum_us / N_ITER;
        t_kprep_warm /= N_ITER;
        warm_ft.t_packA_q /= N_ITER; warm_ft.t_qk /= N_ITER;
        warm_ft.t_sm /= N_ITER; warm_ft.t_pv_packA /= N_ITER;
        warm_ft.t_pv_kern /= N_ITER; warm_ft.t_fin /= N_ITER;
        double tot = t_kprep_warm + warm_ft.t_packA_q + warm_ft.t_qk + warm_ft.t_sm
                   + warm_ft.t_pv_packA + warm_ft.t_pv_kern + warm_ft.t_fin;
        printf("  WARM avg: %.2f ms  ->  %.2f GFLOPS\n", avg/1000.0, GFLOP_total/(avg*1e-6));
        printf("  Breakdown:\n");
        printf("    K trans+prepack: %7.1f us (%5.1f%%)\n", t_kprep_warm, 100*t_kprep_warm/tot);
        printf("    pack_A(Q):       %7.1f us (%5.1f%%)\n", warm_ft.t_packA_q, 100*warm_ft.t_packA_q/tot);
        printf("    QK GEMM:         %7.1f us (%5.1f%%)\n", warm_ft.t_qk, 100*warm_ft.t_qk/tot);
        printf("    softmax:         %7.1f us (%5.1f%%)\n", warm_ft.t_sm, 100*warm_ft.t_sm/tot);
        printf("    PV pack_A(P):    %7.1f us (%5.1f%%)\n", warm_ft.t_pv_packA, 100*warm_ft.t_pv_packA/tot);
        printf("    PV kernel:       %7.1f us (%5.1f%%)\n", warm_ft.t_pv_kern, 100*warm_ft.t_pv_kern/tot);
        printf("    finalize:        %7.1f us (%5.1f%%)\n", warm_ft.t_fin, 100*warm_ft.t_fin/tot);
        printf("    sum:             %7.1f us\n", tot);
    }

    // ── OUTSIDE ROI: attention mode + logical-output checksum ────────────────
    // O_buf is [H_q*T, d] row-major with no padding: logical region == full buffer.
    printf("ATTENTION_MODE=%s\n", causal ? "causal" : "noncausal");
    { double s=0; for(size_t i=0;i<(size_t)H_q*T*d;i++) s+=(double)(float)O_buf.data()[i]; printf("CHECKSUM: %.9f\n",s); }

    // ── SELF_CHECK (env CF_SELF_CHECK=1): FP64 reference attention ───────────
    if (const char* sc = std::getenv("CF_SELF_CHECK"); sc && sc[0] == '1')
        self_check_attention(Q_buf.data(), K_buf.data(), V_buf.data(), O_buf.data(),
                             T, H_q, H_kv, d, causal);
    return 0;
}
