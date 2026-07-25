#!/bin/bash
# build.sh — Build all end-to-end benchmarks
#
# Cache binaries (1-step):
#   bench_kernel       — GEMM V1/V2 + rmsnorm/residual/rope/swiglu_act
#   bench_layernorm    — BERT-Base LayerNorm
#   bench_ffn_seq      — FFN sequential (LLaMA SwiGLU / BERT GELU)
#   cache_flash_attn   — Flash attention (GQA)
#   cache_unfused_attn — Unfused attention (GQA)
#
# SPM binaries (3-step):
#   bench_kernel_spm      — GEMM V3/V4
#   bench_ffn_seq_spm     — FFN sequential SPM
#   spm_flash_attn        — Flash attention SPM
#   spm_unfused_attn      — Unfused attention SPM

: "${CACHEFLEX_ROOT:?source setup_env.sh first}"
set -e
cd "$(dirname "$0")"

MODE="${1:-qemu}"

BENCH_ROOT="$CACHEFLEX_ROOT"
TOOLS="$CACHEFLEX_ROOT/tools"
CROSS_CXX="${CROSS_CXX:-${TOOLS}/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++}"
if [ ! -f "$CROSS_CXX" ]; then echo "Error: compiler not found: $CROSS_CXX"; exit 1; fi

SPM_COMPILER="${SPM_COMPILER:-$CACHEFLEX_ROOT/spm_tools/spm_compiler.py}"
GEM5_INC="${M5_INCLUDE:-$CACHEFLEX_ROOT/gem5/include}"
M5OP="${M5OP_OBJ:-$CACHEFLEX_ROOT/gem5/util/m5/build/arm64/out/m5op.o}"

COMMON="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static -Isrc"
GEM5_DEFS="-DGEM5 -DGEM5_SE -I${GEM5_INC}"
SPM_DEFS="-DVL_4 -mbranch-protection=none"

mkdir -p bin

echo "=== Building end-to-end benchmarks (mode=$MODE) ==="

if [ "$MODE" = "gem5" ]; then
    FLAGS="$COMMON $GEM5_DEFS"

    # Cache binaries (1-step)
    for src in bench_kernel bench_layernorm cache_flash_attn cache_unfused_attn bench_ffn_seq; do
        echo "  $src..."
        $CROSS_CXX $FLAGS src/${src}.cpp $M5OP -lm -o bin/${src}
    done

    # SPM binaries (3-step)
    build_spm() {
        local src="$1" out="$2"
        echo "  $out (3-step SPM)..."
        $CROSS_CXX $FLAGS $SPM_DEFS -S -o /tmp/${out}.s src/${src}
        python3 "$SPM_COMPILER" /tmp/${out}.s /tmp/${out}_enc.s
        $CROSS_CXX -static -o bin/${out} /tmp/${out}_enc.s $M5OP -lm
        rm -f /tmp/${out}.s /tmp/${out}_enc.s
    }

    build_spm bench_kernel_spm.cpp bench_kernel_spm
    build_spm bench_ffn_seq_spm.cpp bench_ffn_seq_spm
    build_spm spm_flash_attn.cpp spm_flash_attn
    build_spm spm_unfused_attn.cpp spm_unfused_attn

else
    FLAGS="$COMMON"

    for src in bench_kernel bench_layernorm cache_flash_attn cache_unfused_attn bench_ffn_seq; do
        echo "  $src..."
        $CROSS_CXX $FLAGS src/${src}.cpp -lm -o bin/${src}
    done

    echo "  (SPM binaries require gem5 mode — skipped)"
fi

echo ""
echo "=== Build complete ==="
ls -la bin/
