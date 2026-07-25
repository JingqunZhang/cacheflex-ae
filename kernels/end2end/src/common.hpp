/*
 * common.hpp — shared utilities for bert_bench
 * AlignedBuffer, Clock, ROI macros, fill_random, flush_dcache_range
 */
#pragma once
#include <arm_sve.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

// ----------------------------------------------------------------
// gem5 ROI macros
// ----------------------------------------------------------------
#ifdef GEM5
#  include <gem5/m5ops.h>
#  define ROI_BEGIN() m5_work_begin(0, 0)
#  define ROI_END()   m5_work_end(0, 0)
#else
#  define ROI_BEGIN() do {} while (0)
#  define ROI_END()   do {} while (0)
#endif

// ----------------------------------------------------------------
// Timer
// ----------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
static inline double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// ----------------------------------------------------------------
// Aligned buffer (64-byte aligned for SVE/cache-line efficiency)
// ----------------------------------------------------------------
template <typename T>
struct AlignedBuffer {
    T* ptr; size_t sz;
    explicit AlignedBuffer(size_t n) : sz(n) {
        size_t bytes = n * sizeof(T);
        if (bytes % 64) bytes += 64 - (bytes % 64);
        ptr = static_cast<T*>(std::aligned_alloc(64, bytes));
        if (!ptr) throw std::bad_alloc();
    }
    ~AlignedBuffer() { std::free(ptr); }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return sz; }
};

// ----------------------------------------------------------------
// Fill with uniform random FP16 values
// ----------------------------------------------------------------
static inline void fill_random(__fp16* p, size_t n, float lo, float hi,
                                unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (size_t i = 0; i < n; ++i) p[i] = (__fp16)d(rng);
}

// ----------------------------------------------------------------
// Flush D-cache range to DRAM (simulate cold-cache load)
// In gem5 SE mode this is a no-op (uses GEM5_SE guard)
// ----------------------------------------------------------------
static inline void flush_dcache_range(const void* ptr, size_t bytes) {
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
