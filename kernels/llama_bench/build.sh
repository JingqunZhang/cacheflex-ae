#!/bin/bash
# Build all llama_bench standalone kernels
# QEMU: bin/  (no GEM5 flag, static ARM64 binary)
# gem5: bin_gem5/  (-DGEM5 -DGEM5_SE, linked with m5op.o)

set -e
cd "$(dirname "$0")"

BENCH_ROOT="$(cd .. && pwd)"
TOOLS="$CACHEFLEX_ROOT/tools"
CROSS_CXX="${CROSS_CXX:-${TOOLS}/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++}"

if [ ! -x "$CROSS_CXX" ]; then
    echo "ERROR: Cross-compiler not found: $CROSS_CXX"
    echo "       Set CROSS_CXX env var or install ARM GNU toolchain."
    exit 1
fi

GEM5_INC="$CACHEFLEX_ROOT/gem5/include"
FLAGS="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static"
GEM5_FLAGS="$FLAGS -DGEM5 -DGEM5_SE -I${GEM5_INC}"
M5OP="$CACHEFLEX_ROOT/gem5/util/m5/build/arm64/out/m5op.o"

mkdir -p bin bin_gem5

SRCS=(01_rmsnorm 02_qkv_gqa 03_rope 04_qk_gqa 05_softmax_causal 06_pv_gqa 07_out_proj 08_ffn_swiglu 09_residual_add)

for s in "${SRCS[@]}"; do
    echo "  [QEMU]  $s"
    "$CROSS_CXX" $FLAGS ${s}.cpp -lm -o bin/${s}
    echo "  [gem5]  $s"
    "$CROSS_CXX" $GEM5_FLAGS ${s}.cpp "$M5OP" -lm -o bin_gem5/${s}
done

echo "llama_bench build done."
echo "  QEMU:  qemu-aarch64 -cpu max,sve512=on ./bin/<kernel>"
echo "  gem5:  <gem5.opt> <se.py> ... -c ./bin_gem5/<kernel> -o '<args>'"
