#!/bin/bash
# build_vl16.sh — Build VL=16 end-to-end benchmarks for gem5
set -e
cd "$(dirname "$0")"

BENCH_ROOT="$CACHEFLEX_ROOT"
TOOLS="$CACHEFLEX_ROOT/tools"
CROSS_CXX="${CROSS_CXX:-${TOOLS}/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++}"
if [ ! -f "$CROSS_CXX" ]; then echo "Error: compiler not found: $CROSS_CXX"; exit 1; fi

SPM_COMPILER="${SPM_COMPILER:-$CACHEFLEX_ROOT/spm_tools/spm_compiler.py}"
GEM5_INC="${M5_INCLUDE:-$CACHEFLEX_ROOT/gem5/include}"
M5OP="${M5OP_OBJ:-$CACHEFLEX_ROOT/gem5/util/m5/build/arm64/out/m5op.o}"

COMMON="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static -Isrc"
GEM5_DEFS="-DGEM5 -DGEM5_SE -I${GEM5_INC}"
SPM_DEFS="-DVL_16 -mbranch-protection=none"
FLAGS="$COMMON $GEM5_DEFS"
FLASH_FLAGS="$FLAGS -DNO_PHASE_TIMERS -falign-functions=64 -falign-loops=32"

mkdir -p bin/vl16

echo "=== Building VL=16 end-to-end benchmarks ==="

# Cache binaries (1-step, no VL define needed)
for src in bench_kernel bench_layernorm cache_unfused_attn bench_ffn_seq; do
    echo "  $src..."
    $CROSS_CXX $FLAGS src/${src}.cpp $M5OP -lm -o bin/vl16/${src}
done
echo "  cache_flash_attn..."
$CROSS_CXX $FLASH_FLAGS src/cache_flash_attn.cpp $M5OP -lm \
    -o bin/vl16/cache_flash_attn

# SPM binaries (3-step)
build_spm() {
    local src="$1" out="$2" extra_flags="${3:-}"
    echo "  $out (3-step SPM)..."
    $CROSS_CXX $FLAGS $extra_flags $SPM_DEFS -S \
        -o /tmp/${out}_vl16.s src/${src}
    python3 "$SPM_COMPILER" /tmp/${out}_vl16.s /tmp/${out}_vl16_enc.s
    $CROSS_CXX -static -o bin/vl16/${out} /tmp/${out}_vl16_enc.s $M5OP -lm
    rm -f /tmp/${out}_vl16.s /tmp/${out}_vl16_enc.s
}

build_spm bench_kernel_spm.cpp bench_kernel_spm
build_spm bench_ffn_seq_spm.cpp bench_ffn_seq_spm
build_spm spm_flash_attn.cpp spm_flash_attn \
    "-DNO_PHASE_TIMERS -falign-functions=64 -falign-loops=32"
build_spm spm_unfused_attn.cpp spm_unfused_attn

echo ""
echo "=== VL=16 build complete ==="
ls -la bin/vl16/
