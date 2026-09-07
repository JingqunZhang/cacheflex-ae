/*
 * bench_layernorm.cpp — BERT-Base LayerNorm benchmark.
 *
 * Usage:
 *   ./bench_layernorm <model:1> <T> [n_iter]
 */

#include "common.hpp"
#include "layer_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static inline svfloat32_t load_f16_as_f32(
    svbool_t pg32,
    const __fp16* source)
{
    // Load one consecutive FP16 value into the low half of each 32-bit lane
    // before widening.  A regular svld1_f16 uses twice as many lanes as the
    // FP32 loop step and therefore does not preserve the logical indexing.
    return svcvt_f32_f16_z(
        pg32,
        svreinterpret_f16_u32(svld1uh_u32(
            pg32,
            reinterpret_cast<const uint16_t*>(source))));
}

static inline void store_f32_as_f16(
    svbool_t pg32,
    __fp16* destination,
    svfloat32_t values)
{
    // Narrow each FP32 lane and store its low halfword contiguously.  Using
    // svst1_f16 with a b16 predicate would write svcnth() elements while this
    // loop advances by only svcntw() elements, corrupting the next chunk.
    svst1h_u32(
        pg32,
        reinterpret_cast<uint16_t*>(destination),
        svreinterpret_u32_f16(svcvt_f16_f32_z(pg32, values)));
}

static void layernorm_row(
    const __fp16* __restrict__ x,
    const __fp16* __restrict__ gamma,
    const __fp16* __restrict__ beta,
    __fp16* __restrict__ y,
    int d_model,
    float epsilon)
{
    const size_t lanes = svcntw();
    svfloat32_t sum_vec = svdup_f32(0.0f);
    svfloat32_t sq_vec = svdup_f32(0.0f);

    for (size_t d = 0; d < static_cast<size_t>(d_model); d += lanes) {
        const svbool_t pg32 = svwhilelt_b32_u64(d, d_model);
        const svfloat32_t values = load_f16_as_f32(pg32, x + d);
        sum_vec = svadd_f32_m(pg32, sum_vec, values);
        sq_vec = svmla_f32_m(pg32, sq_vec, values, values);
    }

    const svbool_t all32 = svptrue_b32();
    const float sum = svaddv_f32(all32, sum_vec);
    const float sum_sq = svaddv_f32(all32, sq_vec);
    const float mean = sum / d_model;
    const float variance =
        std::max(0.0f, sum_sq / d_model - mean * mean);
    const float inv_std = 1.0f / std::sqrt(variance + epsilon);
    const svfloat32_t mean_vec = svdup_f32(mean);
    const svfloat32_t inv_vec = svdup_f32(inv_std);

    for (size_t d = 0; d < static_cast<size_t>(d_model); d += lanes) {
        const svbool_t pg32 = svwhilelt_b32_u64(d, d_model);
        const svfloat32_t values = load_f16_as_f32(pg32, x + d);
        const svfloat32_t scale = load_f16_as_f32(pg32, gamma + d);
        const svfloat32_t shift = load_f16_as_f32(pg32, beta + d);
        const svfloat32_t normalized = svmul_f32_x(
            pg32, svsub_f32_x(pg32, values, mean_vec), inv_vec);
        const svfloat32_t output =
            svmla_f32_x(pg32, shift, normalized, scale);
        store_f32_as_f16(pg32, y + d, output);
    }
}

static void layernorm(
    const __fp16* x,
    const __fp16* gamma,
    const __fp16* beta,
    __fp16* y,
    int tokens,
    int d_model)
{
    for (int token = 0; token < tokens; ++token) {
        layernorm_row(
            x + static_cast<size_t>(token) * d_model,
            gamma,
            beta,
            y + static_cast<size_t>(token) * d_model,
            d_model,
            1.0e-12f);
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <model:1> <T> [n_iter]\n",
            argv[0]);
        return 1;
    }
    if (std::atoi(argv[1]) != 1) {
        std::fprintf(stderr, "bench_layernorm supports BERT model=1 only\n");
        return 1;
    }

    const int tokens = std::atoi(argv[2]);
    const int n_iter = argc > 3 ? std::atoi(argv[3]) : 1;
    if (tokens <= 0 || n_iter <= 0) {
        std::fprintf(stderr, "T and n_iter must be positive\n");
        return 1;
    }

    constexpr int d_model = BertConfig::D;
    const size_t elements = static_cast<size_t>(tokens) * d_model;
    AlignedBuffer<__fp16> x(elements);
    AlignedBuffer<__fp16> gamma(d_model);
    AlignedBuffer<__fp16> beta(d_model);
    AlignedBuffer<__fp16> y(elements);
    fill_random(x.data(), elements, -0.1f, 0.1f, 1);
    fill_random(gamma.data(), d_model, 0.5f, 1.5f, 2);
    fill_random(beta.data(), d_model, -0.1f, 0.1f, 3);

    layernorm(
        x.data(), gamma.data(), beta.data(), y.data(), tokens, d_model);

    float max_abs_error = 0.0f;
    const int sample_tokens[] = {0, tokens / 2, tokens - 1};
    int previous_token = -1;
    for (int token : sample_tokens) {
        if (token == previous_token) {
            continue;
        }
        previous_token = token;
        const size_t row = static_cast<size_t>(token) * d_model;
        float reference_sum = 0.0f;
        float reference_sum_sq = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            const float value = static_cast<float>(x.data()[row + d]);
            reference_sum += value;
            reference_sum_sq += value * value;
        }
        const float reference_mean = reference_sum / d_model;
        const float reference_variance = std::max(
            0.0f,
            reference_sum_sq / d_model -
                reference_mean * reference_mean);
        const float reference_inv =
            1.0f / std::sqrt(reference_variance + 1.0e-12f);
        for (int d = 0; d < d_model; ++d) {
            const float expected =
                (static_cast<float>(x.data()[row + d]) - reference_mean) *
                    reference_inv * static_cast<float>(gamma.data()[d]) +
                static_cast<float>(beta.data()[d]);
            max_abs_error = std::max(
                max_abs_error,
                std::abs(
                    static_cast<float>(y.data()[row + d]) - expected));
        }
    }
    if (max_abs_error > 5.0e-3f) {
        std::fprintf(
            stderr,
            "LayerNorm validation failed: max_abs_error=%.8f\n",
            max_abs_error);
        return 2;
    }
    std::printf("VALIDATION_MAX_ABS: %.8f\n", max_abs_error);

    double total_us = 0.0;
#ifdef GEM5
    m5_reset_stats(0, 0);
#endif
    ROI_BEGIN();
    for (int iteration = 0; iteration < n_iter; ++iteration) {
        const auto start = Clock::now();
        layernorm(
            x.data(), gamma.data(), beta.data(), y.data(), tokens, d_model);
        total_us += us_since(start);
    }
    ROI_END();
#ifdef GEM5
    m5_dump_stats(0, 0);
#endif

    double checksum = 0.0;
    double checksum_l1 = 0.0;
    for (size_t index = 0; index < elements; ++index) {
        const double value = static_cast<float>(y.data()[index]);
        checksum += value;
        checksum_l1 += std::abs(value);
    }
    std::printf(
        "RESULT kernel=layernorm model=bert T=%d D=%d time_us=%.2f\n",
        tokens,
        d_model,
        total_us / n_iter);
    std::printf("CHECKSUM_LOGICAL: %.10f\n", checksum);
    std::printf("CHECKSUM_LOGICAL_L1: %.10f\n", checksum_l1);
    return 0;
}
