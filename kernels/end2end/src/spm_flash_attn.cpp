/*
 * [LANE-FIX applied: fp16<->fp32 widening/narrowing now uses ld1uh/st1h idiom]
 * 11b_flash_spm_jouter_vl4.cpp — SPM Flash Attention with j-outer loop (VL=4, d=64)
 *
 * Per-j-block SPMCP: works for ANY T (not limited to T≤512).
 * j-outer loop: SPMCP K/V once per j-block, reuse across all i-blocks in group.
 * i-blocks grouped so O_acc fits L2.
 *
 * Usage: ./spm_flash_attn [T] [H_q] [H_kv] [d] [n_iter] [Br] [Bc] [group_size] <causal|noncausal>
 *        default: 2048 1 1 64 1 0 0 0 (auto Br/Bc/group_size)
 *        trailing attention mode is MANDATORY
 *        causal mode: block-skipping causal flash attention (skip future KV
 *        blocks, element-wise mask on diagonal blocks)
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

// Include SPM kernels (pack_B_tile_to_spm, spm_kernel for QK)
#include "kernels_spm.hpp"
// Include PV 8x64 SPM kernel
#include "11_pv_kernel.hpp"
// VL=16 specific: 1VL QK kernel + predicated PV kernel
#include "pv_kernel_8x64_vl16.hpp"
#include "spm_qk_kernel_vl16.hpp"

// Clock, AlignedBuffer, us_since provided by common_spm.hpp via kernels_spm.hpp
static inline size_t my_svcnth() { size_t n; __asm__("cnth %0":"=r"(n)); return n; }
// QK uses 3VL for all VLs (proven correct via kernels_spm.hpp)
#if defined(VL_16)
static inline size_t get_NT() { return my_svcnth(); }
#else
static inline size_t get_NT() { return 3 * my_svcnth(); }
#endif
static constexpr size_t MT = 8;

// pack_A_fp16_8row provided by kernels_spm.hpp

// ── Transpose K[T,d] → K_T[d,T] ──
static void transpose_K(const __fp16* K, __fp16* K_T, int T, int d) {
    for (int r = 0; r < T; ++r)
        for (int c = 0; c < d; ++c)
            K_T[c * T + r] = K[r * d + c];
}

// ── SVE exp approximation ──
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

static inline void scale_f32_sve(float* p, float alpha, size_t n) {
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

// Fused: dst[i] = alpha * dst[i] + (float)src[i]  — one L2 round-trip
// Full-width fp16 loads; zip1/zip2 keep the two f32 half-blocks contiguous,
// so per-element semantics match the half-width formulation exactly.
static inline void scale_acc_f16_to_f32(const __fp16* src, float* dst, float alpha, size_t n) {
    const size_t VLh = svcnth(), VL32 = svcntw();
    const svbool_t pth = svptrue_b16(), pt32 = svptrue_b32();
    size_t i = 0;
    for (; i + VLh <= n; i += VLh) {
        svfloat16_t v = svld1_f16(pth, src + i);
        svst1_f32(pt32, dst+i,
            svmla_f32_x(pt32, svcvt_f32_f16_x(pt32, svzip1_f16(v, v)),
                svld1_f32(pt32, dst+i), svdup_n_f32(alpha)));
        svst1_f32(pt32, dst+i+VL32,
            svmla_f32_x(pt32, svcvt_f32_f16_x(pt32, svzip2_f16(v, v)),
                svld1_f32(pt32, dst+i+VL32), svdup_n_f32(alpha)));
    }
    for (; i < n; i += VL32) {
        svbool_t pg32 = svwhilelt_b32_u64(i, n);
        svst1_f32(pg32, dst+i,
            svmla_f32_m(pg32,
                svcvt_f32_f16_x(pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(src+i)))),
                svld1_f32(pg32, dst+i), svdup_n_f32(alpha)));
    }
}

static inline void acc_f16_to_f32(const __fp16* src, float* dst, size_t n) {
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
        svst1_f32(pg32, dst+i, svadd_f32_m(pg32, svld1_f32(pg32, dst+i),
                                             svcvt_f32_f16_x(pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(src+i))))));
    }
}

// ── V SPMCP helper: pack V[j:j+bc, 0:d] to SPM Way0/1 ──
static void prepack_V_to_spm_vl4_range(
    const __fp16* V, size_t T, int d, int j_start, int bc, size_t spm_base_row)
{
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
            [spm_row] "r" (spm_base_row)
        : "cc", "memory", "x10", "x12", "x14", "x16", "x17", "x20"
    );
}

// flush_dcache_range provided by kernels_spm.hpp

// ══════════════════════════════════════════════════════════════════════════
// SELF_CHECK (env CF_SELF_CHECK=1): plain FP64 reference attention, O(T^2*d)
// simple loops, same Q/K/V and same attention mode as the timed run.
// ══════════════════════════════════════════════════════════════════════════
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

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [T] [H_q] [H_kv] [d] [n_iter] [Br] [Bc] [group_size] <causal|noncausal>\n"
        "       defaults: 2048 1 1 64 1 0 0 0 (0 = auto Br/Bc/group_size)\n"
        "       trailing attention-mode argument is MANDATORY (causal or noncausal)\n",
        prog);
}

int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    // MANDATORY trailing attention-mode argument (appended after all positional args)
    bool causal;
    {
        const char* mode = (argc >= 2) ? argv[argc - 1] : nullptr;
        if      (mode && std::strcmp(mode, "causal")    == 0) causal = true;
        else if (mode && std::strcmp(mode, "noncausal") == 0) causal = false;
        else { print_usage(argv[0]); exit(2); }
    }
    const int pargc = argc - 1;  // positional args exclude the trailing mode
    const int T      = (pargc > 1) ? std::atoi(argv[1]) : 2048;
    const int H_q    = (pargc > 2) ? std::atoi(argv[2]) : 1;
    const int H_kv   = (pargc > 3) ? std::atoi(argv[3]) : 1;
    const int d      = (pargc > 4) ? std::atoi(argv[4]) : 64;
    const int N_ITER = (pargc > 5) ? std::atoi(argv[5]) : 1;
    const int Br_arg = (pargc > 6) ? std::atoi(argv[6]) : 0;
    const int Bc_arg = (pargc > 7) ? std::atoi(argv[7]) : 0;
    const int gs_arg = (pargc > 8) ? std::atoi(argv[8]) : 0; // group_size override

    if (H_q % H_kv != 0) { fprintf(stderr, "ERROR: H_q(%d) must be multiple of H_kv(%d)\n", H_q, H_kv); return 1; }
    const int group_ratio = H_q / H_kv;

    const size_t NT = get_NT();
    const float scale = 1.f / std::sqrtf((float)d);

    // ── Auto-compute Br, Bc, group_size ──
    const size_t L1 = 64 * 1024;
    const size_t L2 = 512 * 1024;
    const size_t SPM_3WAY = 192 * 1024;

    // Br: L1 holds Q_tile[Br,d] + O_acc part ≈ Br*6d
    int Br_auto = (int)(L1 / (6 * d));
    Br_auto = (Br_auto / 8) * 8;
    Br_auto = std::max(8, std::min(Br_auto, T));
    const int Br = (Br_arg > 0) ? Br_arg : Br_auto;

    // Bc: SPM capacity constraint
    // VL=16 (1VL): K uses 4-way per NT-block, V uses 2-way per row
    // VL<=8 (3VL): K uses 3-way per NT-block, V uses 2-way per row
#if defined(VL_16)
    const size_t spm_k_per_block = d * 4 * 64;   // 4 ways per NT-block
#else
    const size_t spm_k_per_block = d * 3 * 64;   // 3 ways per NT-block
#endif
    int Bc_auto = (int)NT;  // minimum
    for (int bc = ((T / (int)NT) * (int)NT); bc >= (int)NT; bc -= (int)NT) {
        // Row-space budget: K uses (bc/NT)*d rows, V uses bc rows; each way
        // has 64KB/64B = 1024 rows. A byte budget over-admits (row<<6 then
        // spills into the way bit and aliases K/V).
        size_t rows = ((size_t)bc / NT) * (size_t)d + (size_t)bc;
        if (rows <= 1024) { Bc_auto = bc; break; }
    }
    const int Bc = (Bc_arg > 0) ? (((Bc_arg + (int)NT - 1) / (int)NT) * (int)NT)
                                 : std::max((int)NT, Bc_auto);

    // SPM capacity guard (fail loudly, never silently corrupt).
    if (((size_t)Bc / NT) * (size_t)d + (size_t)Bc > 1024) {
        fprintf(stderr, "FATAL: SPM capacity exceeded: K rows %zu + V rows %d > 1024 (reduce Bc=%d)\n",
                ((size_t)Bc / NT) * (size_t)d, Bc, Bc);
        return 1;
    }

    // Group size: O_acc[group*Br, d] FP32 + m/l + scratch ≤ L2 * 0.85
    size_t scratch = (size_t)Br * Bc * 2 * 2 + (size_t)Br * d * 2 + 8 * Bc * 2 + 8 * d * 2;
    long long l2_budget = (long long)(L2 * 0.85);
    long long l2_avail = l2_budget - (long long)scratch;
    size_t oacc_per = (size_t)Br * d * 4 + (size_t)Br * 8;  // O_acc + m + l
    int n_iblocks = (T + Br - 1) / Br;
    int group_size;
    if (l2_avail <= 0) {
        group_size = 1;  // scratch alone exceeds L2 → 1 i-block per group
    } else {
        group_size = std::min(n_iblocks, (int)((size_t)l2_avail / oacc_per));
        group_size = std::max(1, group_size);
    }
    if (gs_arg > 0) group_size = std::min(gs_arg, n_iblocks);
    int n_groups = (n_iblocks + group_size - 1) / group_size;
    int n_jblocks = (T + Bc - 1) / Bc;

    size_t spm_k_kb = ((size_t)Bc / NT) * d * 3 * 64 / 1024;
    size_t spm_v_kb = (size_t)Bc * 2 * 64 / 1024;

    printf("=== spm_flash_attn (j-outer, GQA, %s) ===\n", causal ? "causal" : "noncausal");
    printf("  T=%d H_q=%d H_kv=%d group_ratio=%d d=%d Br=%d Bc=%d NT=%zu\n",
           T, H_q, H_kv, group_ratio, d, Br, Bc, NT);
    printf("  i-blocks=%d j-blocks=%d groups=%d (group_size=%d)\n",
           n_iblocks, n_jblocks, n_groups, group_size);
    printf("  SPM per j-block: K=%zuKB + V=%zuKB = %zuKB / 192KB\n",
           spm_k_kb, spm_v_kb, spm_k_kb + spm_v_kb);
    printf("  SPMCP calls: %d (n_groups × n_jblocks)\n", n_groups * n_jblocks);
    printf("  O_acc per group: %.1fKB (in L2), scratch: %.1fKB\n",
           group_size * oacc_per / 1024.0, scratch / 1024.0);

    double GFLOP = 4.0 * H_q * T * (double)d * T * 1e-9;
    printf("  GFLOP/iter=%.4f (all %d Q-heads)\n\n", GFLOP, H_q);

    // ── Allocate ──
    AlignedBuffer<__fp16> Q_buf((size_t)H_q  * T * d);
    AlignedBuffer<__fp16> K_buf((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> V_buf((size_t)H_kv * T * d);
    AlignedBuffer<__fp16> O_buf((size_t)H_q  * T * d);

    AlignedBuffer<__fp16> K_T((size_t)d * T);
    // Pre-pack Q for ALL i-blocks (shared across j-blocks)
    size_t ablocks_max = ((size_t)Br + MT - 1) / MT;
    AlignedBuffer<__fp16> qk_Ap_all((size_t)n_iblocks * ablocks_max * MT * d);

    size_t bc_padded_max = (((size_t)Bc + NT - 1) / NT) * NT;  // round UP to NT (handles Bc < NT)
    AlignedBuffer<__fp16> S_tile((size_t)Br * bc_padded_max);
    AlignedBuffer<__fp16> P_tile((size_t)Br * bc_padded_max);
    AlignedBuffer<__fp16> PV_tile(8 * d);
    AlignedBuffer<__fp16> pv_Ap(8 * Bc);

    // Per-group accumulators
    AlignedBuffer<float> O_acc((size_t)group_size * Br * d);
    AlignedBuffer<float> m_vec((size_t)group_size * Br);
    AlignedBuffer<float> l_vec((size_t)group_size * Br);
    AlignedBuffer<float> alpha_vec((size_t)group_size * Br);  // for fused rescale

    // Init random data
    // Same seed pattern as cache_unfused
    { std::mt19937 r1(1); std::uniform_real_distribution<float> d1(-0.1f,0.1f);
      for(size_t i=0;i<(size_t)H_q*T*d;i++) Q_buf.data()[i]=(__fp16)d1(r1); }
    { std::mt19937 r2(2); std::uniform_real_distribution<float> d2(-0.1f,0.1f);
      for(size_t i=0;i<(size_t)H_kv*T*d;i++) K_buf.data()[i]=(__fp16)d2(r2); }
    { std::mt19937 r3(3); std::uniform_real_distribution<float> d3(-0.1f,0.1f);
      for(size_t i=0;i<(size_t)H_kv*T*d;i++) V_buf.data()[i]=(__fp16)d3(r3); }
    std::memset(O_buf.data(), 0, (size_t)H_q*T*d*sizeof(__fp16));

    // Flush cache to ensure cold start (same as cache_flash)
    flush_dcache_range(Q_buf.data(),  (size_t)H_q*T*d  * sizeof(__fp16));
    flush_dcache_range(K_buf.data(),  (size_t)H_kv*T*d * sizeof(__fp16));
    flush_dcache_range(V_buf.data(),  (size_t)H_kv*T*d * sizeof(__fp16));
    flush_dcache_range(O_buf.data(),  (size_t)H_q*T*d  * sizeof(__fp16));

    printf("--- ROI ---\n"); fflush(stdout);
    double sum_us = 0;
    double t_spmcp=0, t_qk=0, t_sm=0, t_pv=0, t_fin=0, t_pack=0;
    auto _now = []{ return Clock::now(); };

    for (int iter = 0; iter < N_ITER; ++iter) {
#ifdef GEM5
        m5_reset_stats(0, 0);
#endif
        ROI_BEGIN();
        auto t0 = Clock::now();

        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_ratio;
            const __fp16* Q_h = Q_buf.data() + (size_t)hq*T*d;
            const __fp16* K_h = K_buf.data() + (size_t)hkv*T*d;
            const __fp16* V_h = V_buf.data() + (size_t)hkv*T*d;
            __fp16* O_h = O_buf.data() + (size_t)hq*T*d;

            // Transpose K (skip if same KV head) + pre-pack Q
            { auto _t = _now();
            if (hq % group_ratio == 0)
                transpose_K(K_h, K_T.data(), T, d);
            for (int ib = 0; ib < n_iblocks; ++ib) {
                int i0 = ib * Br;
                int br = std::min(Br, T - i0);
                pack_A_fp16_8row(Q_h, (size_t)d,
                    qk_Ap_all.data() + (size_t)ib * ablocks_max * MT * d,
                    (size_t)T, (size_t)i0, (size_t)d, 0, (size_t)br, (size_t)d);
            }
            t_pack += us_since(_t); }

            // ── Group loop ──
            for (int g = 0; g < n_groups; ++g) {
                int ib_start = g * group_size;
                int ib_end = std::min(ib_start + group_size, n_iblocks);
                int g_size = ib_end - ib_start;
                // One past the last query row covered by this group (for causal skip)
                const int group_rows_end = std::min(ib_end * Br, T);

                // Init O_acc, m, l for this group
                std::fill_n(O_acc.data(), (size_t)g_size * Br * d, 0.f);
                std::fill_n(m_vec.data(), (size_t)g_size * Br, -1e30f);
                std::fill_n(l_vec.data(), (size_t)g_size * Br, 0.f);

                // ── j-outer loop ──
                for (int j = 0; j < T; j += Bc) {
                    int bc = std::min(Bc, T - j);
                    if (causal) {
                        // Block-skip: KV block starts beyond the group's last query
                        // row (block_col_start > row_block_end) -> entirely future
                        // for every i-block in this group; later j-blocks too.
                        if (j >= group_rows_end) break;
                        // Clamp SPMCP width to the causal frontier of the group
                        bc = std::min(bc, group_rows_end - j);
                    }
                    size_t bblocks = ((size_t)bc + NT - 1) / NT;
                    size_t bc_padded = bblocks * NT;
                    size_t n_bblocks_k = bblocks;  // K NT-blocks for this j-block

                    // SPMCP K and V for this j-block
                    { auto _t = _now();
                    // K_T[:, j:j+bc]: SPMCP each NT-block
                    AlignedBuffer<__fp16> K_pad(d * NT);
                    for (size_t nb = 0; nb < n_bblocks_k; ++nb) {
                        size_t col_start = (size_t)j + nb * NT;
                        size_t nc_k = std::min(NT, (size_t)T - col_start);
                        const __fp16* K_src;
                        size_t K_stride;
                        if (nc_k == NT) {
                            K_src = K_T.data() + col_start;
                            K_stride = (size_t)T;
                        } else {
                            std::memset(K_pad.data(), 0, d * NT * sizeof(__fp16));
                            for (size_t row = 0; row < (size_t)d; ++row)
                                std::memcpy(K_pad.data() + row * NT,
                                            K_T.data() + row * (size_t)T + col_start,
                                            nc_k * sizeof(__fp16));
                            K_src = K_pad.data();
                            K_stride = NT;
                        }
#if defined(VL_16)
                        pack_B_tile_to_spm_4way_vl16(K_src, K_stride,
                                                      (size_t)d, (size_t)d, nb * (size_t)d);
#else
                        pack_B_tile_to_spm(K_src, K_stride,
                                           (size_t)d, (size_t)d, nb * (size_t)d);
#endif
                    }
                    // V[j:j+bc, :]: row-major V, SPMCP to SPM
                    // V layout in SPM: Way0/1 at rows v_base..v_base+bc-1
                    size_t v_spm_base = n_bblocks_k * (size_t)d;  // after K region
                    prepack_V_to_spm_vl4_range(V_h, (size_t)T, d, j, bc, v_spm_base);
                    t_spmcp += us_since(_t); }

                    // ── i-inner loop (within group) ──
                    for (int gi = 0; gi < g_size; ++gi) {
                        int ib = ib_start + gi;
                        int i0 = ib * Br;
                        int br = std::min(Br, T - i0);
                        size_t ablocks = ((size_t)br + MT - 1) / MT;

                        // CAUSAL block-skip: KV block entirely in the future
                        // region for this i-block (block_col_start > row_block_end)
                        if (causal && j >= i0 + br) continue;
                        // CAUSAL diagonal clamp: only the first bc_i columns of
                        // this KV block can be attended by rows i0..i0+br-1;
                        // fully-masked NT sub-blocks are skipped in QK/softmax/PV.
                        int bc_i = bc;
                        if (causal && i0 + br - j < bc) bc_i = i0 + br - j;
                        size_t bblocks_i = ((size_t)bc_i + NT - 1) / NT;
                        size_t bc_padded_i = bblocks_i * NT;

                        const __fp16* qk_Ap = qk_Ap_all.data() +
                            (size_t)ib * ablocks_max * MT * d;
                        float* oacc = O_acc.data() + (size_t)gi * Br * d;
                        float* mvec = m_vec.data() + (size_t)gi * Br;
                        float* lvec = l_vec.data() + (size_t)gi * Br;
                        float* avec = alpha_vec.data() + (size_t)gi * Br;

                        // ── QK GEMM via SPM ──
                        { auto _t = _now();
                        for (size_t mb = 0; mb < ablocks; ++mb) {
                            for (size_t nb = 0; nb < bblocks_i; ++nb) {
                                size_t spm_start = nb * (size_t)d;
#if defined(VL_16)
                                spm_kernel_1vl_vl16(
                                    qk_Ap + mb * 8 * d,
                                    ((__fp16*)S_tile.data()) + mb * 8 * bc_padded_i + nb * NT,
                                    (int)d, (int)bc_padded_i, (size_t)d, spm_start);
#else
                                spm_kernel(qk_Ap + mb * 8 * d,
                                           ((__fp16*)S_tile.data()) + mb * 8 * bc_padded_i + nb * NT,
                                           (int)d, (int)bc_padded_i, (size_t)d, spm_start);
#endif
                            }
                        }
                        // Mask invalid positions
                        size_t last_jt = (size_t)j / NT + bblocks_i - 1;
                        size_t nc_last = std::min(NT, (size_t)T - last_jt * NT);
                        if (nc_last < NT) {
                            const __fp16 neg_inf = (__fp16)(-65504.f);
                            for (size_t r = 0; r < (size_t)br; ++r)
                                for (size_t c = nc_last; c < NT; ++c)
                                    S_tile.data()[r * bc_padded_i + (bblocks_i-1)*NT + c] = neg_inf;
                        }
                        // CAUSAL: element-wise mask on the diagonal block
                        // (col > row -> -inf) before the online-softmax update
                        // (ported from flash_attention/src/spm_flash_attn_causal.cpp,
                        // LLaMA prefill semantics)
                        if (causal) {
                            const __fp16 neg_inf = (__fp16)(-65504.f);
                            for (int r = 0; r < br; ++r) {
                                int qi = i0 + r;              // global query row
                                int start = qi - j + 1;       // first masked column in this j-block
                                if (start < 0) start = 0;
                                for (int c = start; c < bc_i; ++c)
                                    S_tile.data()[(size_t)r * bc_padded_i + c] = neg_inf;
                            }
                        }
                        t_qk += us_since(_t); }

                        // DEBUG: check S_tile after QK
                        // ── Online softmax ──
                        { auto _t = _now();
                        const size_t VLh_sm = svcnth();
                        const size_t VL32_sm = svcntw();
                        const svbool_t pth_sm = svptrue_b16();
                        const svbool_t pt32_sm = svptrue_b32();
                        for (int r = 0; r < br; ++r) {
                            const __fp16* s_row = S_tile.data() + (size_t)r * bc_padded_i;
                            __fp16* p_row = P_tile.data() + (size_t)r * bc_padded_i;

                            // Row max computed in fp16 (exact: fp16 compare is exact and
                            // x -> scale*x is monotone for scale > 0, so scaling the max
                            // afterwards yields bit-identical m_ij).
                            svfloat16_t vmax16 = svdup_n_f16((__fp16)-65504.f);
                            size_t c = 0;
                            for (; c + VLh_sm <= (size_t)bc_i; c += VLh_sm)
                                vmax16 = svmax_f16_m(pth_sm, vmax16, svld1_f16(pth_sm, s_row + c));
                            if (c < (size_t)bc_i) {
                                svbool_t pg = svwhilelt_b16_u64(c, (size_t)bc_i);
                                vmax16 = svmax_f16_m(pg, vmax16, svld1_f16(pg, s_row + c));
                            }
                            float m_ij = scale * (float)svmaxv_f16(pth_sm, vmax16);
                            float m_old = mvec[r];
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
                            for (; c + VLh_sm <= (size_t)bc_i; c += VLh_sm) {
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
                            for (; c < (size_t)bc_i; c += VL32_sm) {
                                svbool_t pg32 = svwhilelt_b32_u64(c, (size_t)bc_i);
                                svfloat32_t vs = svmla_n_f32_x(pg32, vneg_m,
                                    svcvt_f32_f16_x(pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(s_row+c)))), scale);
                                svfloat32_t ve = sve_exp_f32(pg32, vs);
                                vsum = svadd_f32_m(pg32, vsum, ve);
                                svst1h_u32(pg32, (uint16_t*)(p_row+c), svreinterpret_u32_f16(svcvt_f16_f32_x(pg32, ve)));
                            }
                            mvec[r] = m_new;
                            lvec[r] = alpha * lvec[r] + svaddv_f32(pt32_sm, vsum);
                            if (alpha > 1e-30f)
                                scale_f32_sve(oacc + (size_t)r * d, alpha, (size_t)d);
                        }
                        t_sm += us_since(_t); }

                        // ── PV GEMM via SPM ──
                        { auto _t = _now();
                        size_t v_spm_base = n_bblocks_k * (size_t)d;
                        for (size_t mb = 0; mb < ablocks; ++mb) {
                            size_t row_base = mb * 8;
                            size_t rows_this = std::min((size_t)8, (size_t)br - row_base);

                            // Pack P[8, bc_i] from P_tile[br, bc_padded_i]
                            pack_A_fp16_8row(P_tile.data(), bc_padded_i,
                                pv_Ap.data(), (size_t)br, row_base,
                                (size_t)bc_i, 0, rows_this, (size_t)bc_i);

                            // PV via SPM V (V stored at spm rows v_spm_base..v_spm_base+bc-1)
#if defined(VL_16)
                            // VL=16: use cache PV kernel (V from memory, predicated)
                            // V is small (d=64 per row), fits L1. Avoids SPM layout issues.
                            cache_pv_8x64_vl16(pv_Ap.data(),
                                               V_h + (size_t)(j + 0) * d,  // V row pointer
                                               PV_tile.data(),
                                               (int)bc_i, d);
#else
                            sve_spm_pv_8x64_vl4(pv_Ap.data(), PV_tile.data(),
                                                  (int)bc_i, d, v_spm_base);
#endif

                            for (size_t rr = 0; rr < rows_this; ++rr)
                                acc_f16_to_f32(PV_tile.data() + rr * d,
                                               oacc + (row_base + rr) * d, (size_t)d);
                        }
                        t_pv += us_since(_t); }

                    } // end i-inner
                } // end j-outer

                // ── Finalize O for this group ──
                { auto _t = _now();
                for (int gi = 0; gi < g_size; ++gi) {
                    int ib = ib_start + gi;
                    int i0 = ib * Br;
                    int br = std::min(Br, T - i0);
                    const float* oacc = O_acc.data() + (size_t)gi * Br * d;
                    const float* lvec = l_vec.data() + (size_t)gi * Br;
                    const size_t VLh_fin = svcnth(), VL32_fin = svcntw();
                    const svbool_t pth_fin = svptrue_b16(), pt32_fin = svptrue_b32();
                    for (int r = 0; r < br; ++r) {
                        float inv_l = 1.f / lvec[r];
                        __fp16* o_dst = O_h + ((size_t)i0 + r) * d;
                        const float* o_src = oacc + (size_t)r * d;
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
                }
                t_fin += us_since(_t); }
            } // end groups
        } // end hq loop

        double us = us_since(t0);
        ROI_END();
        sum_us += us;
#ifdef GEM5
        m5_dump_stats(0, 0);
#endif
        printf("  iter %d: %.2f ms\n", iter, us / 1000.0);
    }

    double avg = sum_us / N_ITER;
    double tot = t_pack + t_spmcp + t_qk + t_sm + t_pv + t_fin;
    printf("\n=== Summary ===\n");
    printf("  avg: %.2f ms  GFLOPS: %.2f\n", avg/1000.0, GFLOP/(avg*1e-6));
    printf("  Breakdown (avg of %d iters):\n", N_ITER);
    printf("    pack/transpose: %7.1f us (%5.1f%%)\n", t_pack/N_ITER, 100*t_pack/tot);
    printf("    SPMCP:          %7.1f us (%5.1f%%)\n", t_spmcp/N_ITER, 100*t_spmcp/tot);
    printf("    QK GEMM:        %7.1f us (%5.1f%%)\n", t_qk/N_ITER, 100*t_qk/tot);
    printf("    softmax:        %7.1f us (%5.1f%%)\n", t_sm/N_ITER, 100*t_sm/tot);
    printf("    PV GEMM:        %7.1f us (%5.1f%%)\n", t_pv/N_ITER, 100*t_pv/tot);
    printf("    finalize:       %7.1f us (%5.1f%%)\n", t_fin/N_ITER, 100*t_fin/tot);
    printf("    sum:            %7.1f us\n", tot/N_ITER);

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

