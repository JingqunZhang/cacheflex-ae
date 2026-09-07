/*
 * spm_unfused_attn.cpp — SPM Unfused Attention (QK + softmax + PV)
 *
 * Plan B timing: K/V prepack excluded (simulates pre-packed KV cache).
 * SPMCP K/V included in timing (SPM-specific cost).
 *
 * QK GEMM: Q[T,d] × K_T[d,T] → scores[T,T_pad], K_T from SPM per N-tile
 * Softmax: causal softmax on scores[T,T_pad]
 * PV GEMM: P[T,T] × V[T,d] → out[T,d], V from SPM per K-tile
 *
 * Usage: ./spm_unfused_attn [T] [H_q] [H_kv] [d] [n_iter] [MC_qk] [KC_pv] [MC_pv] [phase] <causal|noncausal>
 *        default: 512 1 1 64 1 128 256 128 0  — trailing attention mode is MANDATORY
 */
#include <arm_neon.h>
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

#ifdef GEM5
#  include <gem5/m5ops.h>
#  define ROI_BEGIN() m5_work_begin(0, 0)
#  define ROI_END()   m5_work_end(0, 0)
#else
#  define ROI_BEGIN() do{}while(0)
#  define ROI_END()   do{}while(0)
#endif

#include "kernels_spm.hpp"
#include "11_pv_kernel.hpp"
#include "pv_kernel_8x64_vl16.hpp"
#include "spm_qk_kernel_vl16.hpp"

static inline size_t my_svcnth_early() { size_t n; __asm__("cnth %0":"=r"(n)); return n; }
// VL=16 (svcnth=128): use 1VL=128 to avoid NT=384 > T issues
static inline size_t get_NT() {
    size_t s = my_svcnth_early();
    return (s >= 128) ? s : 3 * s;
}
static constexpr size_t MT = 8;

static void transpose_K(const __fp16* K, __fp16* K_T, int T, int d) {
    for (int r = 0; r < T; ++r)
        for (int c = 0; c < d; ++c)
            K_T[c * T + r] = K[r * d + c];
}

static void prepack_B_knm_local(const __fp16* B, __fp16* Bp, size_t K, size_t N, size_t KC, size_t ldb) {
    if (!ldb) ldb = N;
    const size_t NT = get_NT();
    const size_t N_tiles = (N + NT - 1) / NT;
    const size_t K_tiles = (K + KC - 1) / KC;
    for (size_t kt = 0; kt < K_tiles; ++kt) {
        size_t k0 = kt * KC, kc = std::min(KC, K - k0);
        for (size_t nt = 0; nt < N_tiles; ++nt) {
            size_t n0 = nt * NT, nc = std::min(NT, N - n0);
            __fp16* dst = Bp + (kt * N_tiles + nt) * KC * NT;
            std::memset(dst, 0, KC * NT * sizeof(__fp16));
            for (size_t k = 0; k < kc; ++k) {
                const __fp16* src = B + (k0 + k) * ldb + n0;
                for (size_t n = 0; n < nc; ++n) dst[k * NT + n] = src[n];
            }
        }
    }
}

static inline svfloat32_t sve_exp_f32(svbool_t pg, svfloat32_t x) {
    x = svmax_n_f32_x(pg, x, -88.0f);
    svfloat32_t nf = svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.44269504088896f));
    svint32_t ni = svcvt_s32_f32_x(pg, nf);
    svfloat32_t r = svmls_n_f32_x(pg, x, nf, 6.93147182e-1f);
    r = svmls_n_f32_x(pg, r, nf, -1.90465430e-9f);
    svfloat32_t p = svdup_n_f32(1.0f/120.0f);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f/24.0f), r, p);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f/6.0f), r, p);
    p = svmla_f32_x(pg, svdup_n_f32(0.5f), r, p);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f), r, p);
    p = svmla_f32_x(pg, svdup_n_f32(1.0f), r, p);
    return svscale_f32_x(pg, p, ni);
}

static inline size_t my_svcnth() { size_t n; __asm__("cnth %0":"=r"(n)); return n; }

// Optimized softmax: 2-unrolled exp, 4-unrolled normalize (same as cache_unfused)
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
        for (size_t j = (size_t)T; j < T_pad; j += vl16) {
            svbool_t pg = svwhilelt_b16_u64(j, T_pad);
            svst1_f16(pg, row + j, svdup_f16(neg_inf));
        }
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

static void spmcp_V_range(const __fp16* V, int d, int j_start, int bc, size_t spm_base) {
    __asm__ __volatile__(
        "mov x17, #0x10000\n"
        "mov x10, %x[spm_row]\n"
        "mov x12, %x[src]\n"
        "mov x20, %x[bc]\n"
        "1:\n"
        "lsl x16, x10, #6\n"
        "SPMCP_64_IMM x16, [x12, #0]\n"
        "orr x14, x16, x17\n"
        "SPMCP_64_IMM x14, [x12, #64]\n"
        "add x10, x10, #1\n"
        "add x12, x12, %x[stride]\n"
        "subs x20, x20, #1\n"
        "bne 1b\n"
        : : [src] "r" (V + (size_t)j_start * d),
            [bc] "r" ((size_t)bc),
            [stride] "r" ((size_t)d * sizeof(__fp16)),
            [spm_row] "r" (spm_base)
        : "cc", "memory", "x10", "x12", "x14", "x16", "x17", "x20"
    );
}

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
    const int T      = (pargc > 1) ? std::atoi(argv[1]) : 512;
    const int H_q    = (pargc > 2) ? std::atoi(argv[2]) : 1;
    const int H_kv   = (pargc > 3) ? std::atoi(argv[3]) : 1;
    const int d      = (pargc > 4) ? std::atoi(argv[4]) : 64;
    const int N_ITER = (pargc > 5) ? std::atoi(argv[5]) : 1;
    const size_t MC_qk = (pargc > 6) ? (size_t)std::atoi(argv[6]) : 128;
    const size_t KC_pv = (pargc > 7) ? (size_t)std::atoi(argv[7]) : 256;
    const size_t MC_pv = (pargc > 8) ? (size_t)std::atoi(argv[8]) : 128;
    // phase: 0=all(default), 1=QK only, 2=softmax only, 3=PV only
    const int    phase  = (pargc > 9) ? std::atoi(argv[9]) : 0;

    if (H_q % H_kv != 0) { fprintf(stderr, "ERROR: H_q(%d) must be multiple of H_kv(%d)\n", H_q, H_kv); return 1; }
    const int group_ratio = H_q / H_kv;

    const size_t NT = get_NT();
    const size_t N_tiles = ((size_t)T + NT - 1) / NT;
    const size_t T_pad = N_tiles * NT;
    const float scale = 1.f / std::sqrtf((float)d);
    const double GFLOP = 4.0 * H_q * T * (double)d * T * 1e-9;
    const size_t kc_qk = (size_t)d;
    const size_t K_tiles_pv = ((size_t)T + KC_pv - 1) / KC_pv;
    const size_t max_ablk_qk = (MC_qk + MT - 1) / MT;
    const size_t num_mc_qk = ((size_t)T + MC_qk - 1) / MC_qk;

    const char* phase_names[] = {"all", "QK", "softmax", "PV"};
    printf("=== spm_unfused_attn (8x64 PV, GQA, phase=%s, %s) ===\n",
           phase_names[phase], causal ? "causal" : "noncausal");
    printf("  T=%d H_q=%d H_kv=%d group_ratio=%d d=%d MC_qk=%zu KC_pv=%zu MC_pv=%zu phase=%d\n",
           T, H_q, H_kv, group_ratio, d, MC_qk, KC_pv, MC_pv, phase);
    printf("  scores[%d,%zu]=%.0fKB %s\n", T, T_pad, T*T_pad*2.0/1024,
           T*T_pad*2 > 512*1024 ? "(DRAM!)" : "(L2)");
    printf("  GFLOP=%.4f\n\n", GFLOP);

    AlignedBuffer<__fp16> Q_buf((size_t)H_q * T * d);
    AlignedBuffer<__fp16> K_buf((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> V_buf((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> O_buf((size_t)H_q * T * d);
    AlignedBuffer<__fp16> K_T((size_t)d * T);
    AlignedBuffer<__fp16> K_tile((size_t)d * NT);  // per-N-tile K buffer (no full prepack)
    AlignedBuffer<__fp16> scores((size_t)T * T_pad);
    // Separate buffers for QK (pre-packed Q) and PV (scores rows)
    AlignedBuffer<__fp16> A_cache_qk(num_mc_qk * max_ablk_qk * MT * kc_qk);
    size_t max_ablk_pv = (MC_pv + MT - 1) / MT;
    AlignedBuffer<__fp16> A_cache_pv(max_ablk_pv * MT * KC_pv);
    AlignedBuffer<__fp16> Cpanel(std::max(max_ablk_qk * MT * NT, (size_t)8 * d));

    {
        std::mt19937 rng1(1); std::uniform_real_distribution<float> d1(-0.1f, 0.1f);
        for (size_t i = 0; i < (size_t)H_q*T*d; ++i) Q_buf.data()[i] = (__fp16)d1(rng1);
        std::mt19937 rng2(2); std::uniform_real_distribution<float> d2(-0.1f, 0.1f);
        for (size_t i = 0; i < (size_t)H_kv*T*d; ++i) K_buf.data()[i] = (__fp16)d2(rng2);
        std::mt19937 rng3(3); std::uniform_real_distribution<float> d3(-0.1f, 0.1f);
        for (size_t i = 0; i < (size_t)H_kv*T*d; ++i) V_buf.data()[i] = (__fp16)d3(rng3);
    }
    std::memset(O_buf.data(), 0, (size_t)H_q*T*d*sizeof(__fp16));
    std::memset(scores.data(), 0, (size_t)T*T_pad*sizeof(__fp16));

    // Flush cache to ensure cold start (same as cache variants)
    flush_dcache_range(Q_buf.data(),  (size_t)H_q*T*d  * sizeof(__fp16));
    flush_dcache_range(K_buf.data(),  (size_t)H_kv*T*d * sizeof(__fp16));
    flush_dcache_range(V_buf.data(),  (size_t)H_kv*T*d * sizeof(__fp16));
    flush_dcache_range(O_buf.data(),  (size_t)H_q*T*d  * sizeof(__fp16));
    flush_dcache_range(scores.data(), (size_t)T*T_pad   * sizeof(__fp16));

    // === Helper: run QK GEMM for one head (per-N-tile K transpose+SPMCP) ===
    auto run_qk_head = [&](const __fp16* Q_h, const __fp16* K_h) {
        // Pre-pack Q once
        for (size_t m0 = 0; m0 < (size_t)T; m0 += MC_qk) {
            size_t mc = std::min(MC_qk, (size_t)T - m0);
            pack_A_fp16_8row(Q_h, (size_t)d,
                A_cache_qk.data() + (m0/MC_qk) * max_ablk_qk * MT * kc_qk,
                (size_t)T, m0, (size_t)d, 0, mc, kc_qk);
        }
        // Per-N-tile: transpose K tile → K_tile, SPMCP → SPM, kernel+scatter
        for (size_t nt = 0; nt < N_tiles; ++nt) {
            size_t n0 = nt * NT;
            size_t nc = std::min(NT, (size_t)T - n0);
            // Per-tile K transpose into K_tile[d, NT] (matches cache_unfused's pack_K_tile approach)
            std::memset(K_tile.data(), 0, (size_t)d * NT * sizeof(__fp16));
            for (int k = 0; k < d; ++k)
                for (size_t n = 0; n < nc; ++n)
                    K_tile.data()[(size_t)k * NT + n] = K_h[(n0 + n) * (size_t)d + k];
            // SPMCP K_tile → SPM
            if (my_svcnth() >= 128) // VL=16: use 4-way 1VL SPMCP
                pack_B_tile_to_spm_4way_vl16(K_tile.data(), NT, d, d, 0);
            else
                pack_B_tile_to_spm(K_tile.data(), NT, d, d, 0);
            // Kernel + scatter for all M-blocks
            const size_t VLh = my_svcnth();
            svbool_t pg_full = svptrue_b16();
            svfloat16_t vs_scale = svdup_f16((__fp16)scale);
            for (size_t m0 = 0; m0 < (size_t)T; m0 += MC_qk) {
                size_t mc = std::min(MC_qk, (size_t)T - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                const __fp16* Ap = A_cache_qk.data() + (m0/MC_qk) * max_ablk_qk * MT * kc_qk;
                if (my_svcnth() >= 128) {
                    // VL=16: use 1VL kernel per M-block
                    for (size_t mb = 0; mb < ablocks; ++mb)
                        spm_kernel_1vl_vl16(
                            Ap + mb * MT * kc_qk,
                            Cpanel.data() + mb * MT * NT,
                            (int)kc_qk, (int)NT, kc_qk, 0);
                } else {
                    spm_kernel_mblocks(Ap, Cpanel.data(), (int)kc_qk, (int)ablocks, (size_t)d, 0);
                }
                for (size_t mb = 0; mb < ablocks; ++mb) {
                    const __fp16* tp = Cpanel.data() + mb*MT*NT;
                    for (size_t r = 0; r < MT; ++r) {
                        size_t gr = m0 + mb*MT + r;
                        if (gr >= (size_t)T) break;
                        __fp16* s_row = scores.data() + gr*T_pad + n0;
                        const __fp16* t_row = tp + r*NT;
                        size_t i = 0;
                        for (; i + VLh <= nc; i += VLh)
                            svst1_f16(pg_full, s_row+i,
                                svmul_f16_x(pg_full, svld1_f16(pg_full, t_row+i), vs_scale));
                        if (i < nc) {
                            svbool_t pg = svwhilelt_b16_u64(i, nc);
                            svst1_f16(pg, s_row+i,
                                svmul_f16_x(pg, svld1_f16(pg, t_row+i), vs_scale));
                        }
                    }
                }
            }
        }
    };

    // === Helper: run PV GEMM for one head ===
    auto run_pv_head = [&](const __fp16* V_h, __fp16* O_h) {
        for (size_t kt = 0; kt < K_tiles_pv; ++kt) {
            size_t k0 = kt * KC_pv;
            size_t kc = std::min(KC_pv, (size_t)T - k0);
            // VL=16: V via cache (SPM PV kernel has gem5 bug); VL<=8: V via SPM
            if (my_svcnth() < 128)
                spmcp_V_range(V_h, d, (int)k0, (int)kc, 0);
            for (size_t m0 = 0; m0 < (size_t)T; m0 += MC_pv) {
                size_t mc = std::min(MC_pv, (size_t)T - m0);
                size_t ablocks = (mc + MT - 1) / MT;
                pack_A_fp16_8row(scores.data(), T_pad, A_cache_pv.data(),
                    (size_t)T, m0, (size_t)T, k0, mc, kc);
                for (size_t mb = 0; mb < ablocks; ++mb) {
                    size_t rows = std::min((size_t)8, mc - mb*MT);
                    if (my_svcnth() >= 128) // VL=16: cache PV kernel (workaround)
                        cache_pv_8x64_vl16(A_cache_pv.data() + mb * MT * kc,
                            V_h + k0 * (size_t)d,
                            Cpanel.data(), (int)kc, d);
                    else
                        sve_spm_pv_8x64_vl4(A_cache_pv.data() + mb * MT * kc,
                            Cpanel.data(), (int)kc, d, 0);
                    const size_t VLh = my_svcnth();
                    svbool_t pg_full = svptrue_b16();
                    for (size_t r = 0; r < rows; ++r) {
                        size_t gr = m0 + mb*MT + r;
                        __fp16* o_row = O_h + gr * d;
                        const __fp16* pv = Cpanel.data() + r * d;
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
                                    svadd_f16_x(pg_full, svld1_f16(pg_full, o_row+i), svld1_f16(pg_full, pv+i)));
                            if (i < (size_t)d) {
                                svbool_t pg = svwhilelt_b16_u64(i, (size_t)d);
                                svst1_f16(pg, o_row+i,
                                    svadd_f16_x(pg, svld1_f16(pg, o_row+i), svld1_f16(pg, pv+i)));
                            }
                        }
                    }
                }
            }
        }
    };

    // === Phase-aware execution ===
    // For phase 2/3: pre-run earlier steps without timing
    if (phase >= 2) {
        printf("  [prep] Running QK (untimed)...\n");
        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_ratio;
            run_qk_head(Q_buf.data()+(size_t)hq*T*d, K_buf.data()+(size_t)hkv*T*d);
        }
    }
    if (phase >= 3) {
        printf("  [prep] Running softmax (untimed)...\n");
        if (causal) causal_softmax_fp16(scores.data(), T, T_pad);
        else        noncausal_softmax_fp16(scores.data(), T, T_pad);
    }

    printf("--- ROI ---\n"); fflush(stdout);
    double sum_us = 0;

    // COLD + N_ITER WARM
    for (int iter = -1; iter < N_ITER; ++iter) {
        bool is_cold = (iter < 0);
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        if (is_cold) ROI_BEGIN();
        auto t0 = Clock::now();

        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_ratio;
            const __fp16* Q_h = Q_buf.data() + (size_t)hq  * T * d;
            const __fp16* K_h = K_buf.data() + (size_t)hkv * T * d;
            const __fp16* V_h = V_buf.data() + (size_t)hkv * T * d;
            __fp16* O_h = O_buf.data() + (size_t)hq * T * d;

            if (phase == 0 || phase == 1) run_qk_head(Q_h, K_h);
            if (phase == 0 || phase == 2) {
                if (causal) causal_softmax_fp16(scores.data(), T, T_pad);
                else        noncausal_softmax_fp16(scores.data(), T, T_pad);
            }
            if (phase == 0 || phase == 3) run_pv_head(V_h, O_h);
        }

        double us = us_since(t0);
        if (is_cold) {
            ROI_END();
            printf("  COLD: %.2f ms  ->  %.2f GFLOPS\n", us/1000.0, GFLOP/(us*1e-6));
#ifdef GEM5
            m5_dump_stats(0, 0);
            m5_reset_stats(0, 0);   // reset so WARM runs don't accumulate into ROI
#endif
        } else {
            sum_us += us;
            printf("  iter %d: %.2f ms\n", iter, us / 1000.0);
        }
    }

    double avg = sum_us / N_ITER;
    printf("\n=== WARM avg ===\n");
    printf("  avg: %.2f ms  GFLOPS: %.2f\n", avg/1000.0, GFLOP/(avg*1e-6));

    // ── OUTSIDE ROI: attention mode + logical-output checksum ────────────────
    // O_buf is [H_q*T, d] row-major with no padding: logical region == full buffer.
    printf("ATTENTION_MODE=%s\n", causal ? "causal" : "noncausal");
    { double s=0; for(size_t i=0;i<(size_t)H_q*T*d;i++) s+=(double)(float)O_buf.data()[i]; printf("CHECKSUM: %.9f\n",s); }

    // ── SELF_CHECK (env CF_SELF_CHECK=1): FP64 reference attention ───────────
    if (const char* sc = std::getenv("CF_SELF_CHECK"); sc && sc[0] == '1') {
        if (phase != 0)
            printf("SELF_CHECK SKIP (phase=%d != 0: output is not full attention)\n", phase);
        else
            self_check_attention(Q_buf.data(), K_buf.data(), V_buf.data(), O_buf.data(),
                                 T, H_q, H_kv, d, causal);
    }
    return 0;
}
