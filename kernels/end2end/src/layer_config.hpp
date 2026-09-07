/*
 * layer_config.hpp — Model configuration for BERT-base and LLaMA-3.2-1B
 *
 * Defines per-model constants and per-GEMM tiling parameters derived from
 * figure1 sweep results and flash_attention sweep results.
 *
 * Every GEMM in the layer has an explicit (MC, KC) entry so you can see
 * exactly what parameters are used for what size.
 */
#pragma once
#include <cstddef>
#include <algorithm>

// ============================================================
// Model configurations
// ============================================================
struct BertConfig {
    static constexpr int D       = 768;
    static constexpr int H       = 12;    // attention heads
    static constexpr int H_kv    = 12;    // same as H for MHA
    static constexpr int d       = 64;    // head dim
    static constexpr int FFN_dim = 3072;  // 4 * D
    static constexpr int G       = 1;     // queries per KV head (MHA)
    static constexpr bool causal = false;
    static constexpr bool use_rope = false;
};

struct LlamaConfig {
    static constexpr int D       = 2048;
    static constexpr int H       = 32;    // query heads
    static constexpr int H_kv    = 8;     // KV heads (GQA)
    static constexpr int d       = 64;    // head dim
    static constexpr int FFN_dim = 8192;  // 4 * D
    static constexpr int G       = 4;     // H / H_kv = queries per KV head
    static constexpr bool causal = true;
    static constexpr bool use_rope = true;
};

// ============================================================
// GEMM tiling — one struct per GEMM call
//
// Constraint: A_panel = MC × KC × 2B ≤ L1 (64KB)  →  MC × KC ≤ 32768
//             B_tile  = KC × NT × 2B ≤ L2 (512KB)  →  KC × 96 × 2 ≤ 512K → KC ≤ 2730
//
// Sources:
//   figure1 sweep:  g1 (256×2048×2048) v1 best MC=64 KC=256
//                   g2 (2048×2048×2048) v1 best MC=512 KC=256 (or MC=128 KC=256)
//                   g5 (512×768×768) v1 best MC=64 KC=256
//   For K=8192 (FFN down): no direct sweep result. KC=256 gives 32 K-tiles.
//     Try KC=512 MC=64 (A_panel=64KB, fits L1). Halves K-tiles to 16.
// ============================================================
struct GemmTileParams {
    size_t MC;   // M-tile (rows per chunk)
    size_t KC;   // K-tile (reduction chunk)
};

// ---- LLaMA tiling per GEMM (cache V1) ----

// QKV/Out proj: [T, 2048] × [2048, N]  where N=2048 (W_Q,W_O) or N=512 (W_K,W_V)
// K=2048, from figure1 g1(256×2048×2048) and g2(2048×2048×2048)
static GemmTileParams llama_proj_tile(size_t T) {
    if (T <= 256)  return {64,  256};   // g1: MC=64 KC=256
    if (T <= 512)  return {128, 256};
    return                {128, 256};   // g2: MC=128 KC=256 (MC=512 possible but less robust)
}

// FFN gate/up: [T, 2048] × [2048, 8192]   K=2048, N=8192
// Same K=2048 as proj, N is larger but doesn't affect MC/KC choice
static GemmTileParams llama_ffn_gate_up_tile(size_t T) {
    return llama_proj_tile(T);  // same K, same constraint
}

// FFN down: [T, 8192] × [8192, 2048]   K=8192!
// K=8192 → KC=256 gives 32 K-tiles (too many RMW passes)
// KC=512 MC=64: A_panel = 64×512×2 = 64KB (exactly fits L1)
static GemmTileParams llama_ffn_down_tile(size_t T) {
    if (T <= 256)  return {64,  512};   // K_tiles = 8192/512 = 16
    return                {64,  512};   // same: MC=64 keeps A_panel=64KB
}

// ---- BERT tiling per GEMM (cache V1) ----

// QKV/Out proj: [T, 768] × [768, 768]
// From figure1 g5 (512×768×768): MC=64 KC=256
// K=768: KC=256 → 3 K-tiles. Could also do KC=768 (single K-tile, no RMW!)
//   A_panel = MC × 768 × 2 → MC ≤ 42 at 64KB. MC=40 KC=768? → 1 K-tile, less overhead
//   But MC=40 is not well-aligned. Stick with figure1 result.
static GemmTileParams bert_proj_tile(size_t T) {
    if (T <= 256)  return {64,  256};
    if (T <= 512)  return {64,  256};   // g5
    return                {128, 256};
}

// FFN W1: [T, 768] × [768, 3072]   K=768, same as proj
static GemmTileParams bert_ffn_w1_tile(size_t T) {
    return bert_proj_tile(T);
}

// FFN W2: [T, 3072] × [3072, 768]   K=3072
// KC=256 → 12 K-tiles. KC=512 MC=64 → 6 K-tiles (A_panel=64KB fits L1)
static GemmTileParams bert_ffn_w2_tile(size_t T) {
    if (T <= 256)  return {64,  512};
    return                {64,  512};
}

// ============================================================
// Flash attention tiling
// From flash_attention sweep results:
//   13c (cache): T=512 best Br=64, Bc=384
//   11b (SPM):   T=512 best Br=40, Bc=480
//
// Bc must be multiple of NT = 3*svcnth() (96 at VL=4, 384 at VL=16).
// Default Bc = 2*NT or 4*NT depending on T.
// Actual Bc comes from sweep args; these are fallbacks only.
// ============================================================
struct FlashTileParams {
    int Br;
    int Bc_tiles;  // Bc = Bc_tiles * NT
};

static FlashTileParams get_flash_tile_cache(int T) {
    // Bc_tiles: number of NT-wide tiles in Bc
    if (T <= 256)       return {64, 2};     // Bc = 2*NT
    else if (T <= 512)  return {64, 4};     // Bc = 4*NT
    else                return {64, 4};     // Bc = 4*NT (T>=1024)
}

static FlashTileParams get_flash_tile_spm(int T) {
    if (T <= 256)       return {40, 2};
    else if (T <= 512)  return {40, 5};
    else                return {40, 5};
}

// ============================================================
// Variant enum
// ============================================================
enum class Variant {
    BASELINE     = 0,   // V1 cache GEMM + unfused attention
    SPM          = 1,   // V3 SPM GEMM + 11b SPM flash attention
    FLASH_CACHE  = 2,   // V1 cache GEMM + 13c cache flash attention
    SPM_GEMM     = 3,   // V3 SPM GEMM + unfused attention (isolate GEMM SPM benefit)
};

static const char* variant_name(Variant v) {
    switch (v) {
        case Variant::BASELINE:    return "baseline";
        case Variant::SPM:         return "spm";
        case Variant::FLASH_CACHE: return "flash_cache";
        case Variant::SPM_GEMM:    return "spm_gemm";
        default:                   return "unknown";
    }
}
