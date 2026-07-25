#!/bin/bash
# build.sh — compile v1_fused and v3_fused benchmarks
#
# Both use fused scatter (no Cpanel), m0-outer loop order.
# V1: B from cache (JIT pack)
# V3: B from SPM (SPMCP)
#
# Usage: bash build.sh

set -e
cd "$(dirname "$0")"

: "${CACHEFLEX_ROOT:?source setup_env.sh first}"
BENCH_ROOT="$CACHEFLEX_ROOT"
TOOLS="$CACHEFLEX_ROOT/tools"
CROSS_CXX="${CROSS_CXX:-${TOOLS}/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++}"
SPM_COMPILER="${SPM_COMPILER:-$CACHEFLEX_ROOT/spm_tools/spm_compiler.py}"
GEM5_INC="$CACHEFLEX_ROOT/gem5/include"
M5OP="${M5OP_OBJ:-$CACHEFLEX_ROOT/gem5/util/m5/build/arm64/out/m5op.o}"

BASE_INC="-I$CACHEFLEX_ROOT/kernels/llama_bench"
SPM_INC="-I$CACHEFLEX_ROOT/kernels/llama_bench_spm"
SELF_INC="-Isrc"

COMMON_FLAGS="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static"
GEM5_DEFS="-DGEM5 -DGEM5_SE -I${GEM5_INC}"
SPM_ASM_FLAGS="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -mbranch-protection=none"

mkdir -p bin_qemu bin

# ---------------------------------------------------------------------------
# Helper: 3-step SPM build
# ---------------------------------------------------------------------------
build_spm_gem5() {
    local src="$1" vl_def="$2" out="$3"
    local asm_f="${out}_tmp.s"
    local enc_f="${out}_tmp_enc.s"
    echo "  [gem5 SPM ${vl_def}]  $src → bin/$out"
    echo "    1/3 C++→ASM"
    "$CROSS_CXX" $SPM_ASM_FLAGS $GEM5_DEFS $SPM_INC $BASE_INC $SELF_INC \
        -D${vl_def} -S -o "${asm_f}" "${src}"
    echo "    2/3 spm_compiler.py"
    python3 "$SPM_COMPILER" "${asm_f}" "${enc_f}"
    echo "    3/3 link → bin/${out}"
    "$CROSS_CXX" -static -o "bin/${out}" "${enc_f}" "$M5OP" -lm
    rm -f "${asm_f}" "${enc_f}"
}

# ---------------------------------------------------------------------------
# V1 fused: cache baseline with fused scatter
# ---------------------------------------------------------------------------
echo "=== V1 fused ==="
echo "  [QEMU]  bin_qemu/v1_fused"
"$CROSS_CXX" $COMMON_FLAGS $BASE_INC $SELF_INC \
    src/v1_fused.cpp -lm -o bin_qemu/v1_fused

echo "  [gem5]  bin/v1_fused"
"$CROSS_CXX" $COMMON_FLAGS $GEM5_DEFS $BASE_INC $SELF_INC \
    src/v1_fused.cpp "$M5OP" -lm -o bin/v1_fused

# ---------------------------------------------------------------------------
# V3 fused: SPM with fused scatter, m0-outer
# ---------------------------------------------------------------------------
echo ""
echo "=== V3 fused ==="
echo "  [QEMU mock]  bin_qemu/v3_fused_mock"
"$CROSS_CXX" $COMMON_FLAGS -DMOCK_SPM $BASE_INC $SELF_INC \
    src/v3_fused.cpp -lm -o bin_qemu/v3_fused_mock

for VL_DEF in VL_4 VL_8 VL_16; do
    vl_num="${VL_DEF#VL_}"
    build_spm_gem5 src/v3_fused.cpp "$VL_DEF" "v3_fused_vl${vl_num}"
done

# ---------------------------------------------------------------------------
# V1 fused prepack: B prepacked outside ROI
# ---------------------------------------------------------------------------
echo ""
echo "=== V1 fused prepack ==="
echo "  [QEMU]  bin_qemu/v1_fused_prepack"
"$CROSS_CXX" $COMMON_FLAGS $BASE_INC $SELF_INC \
    src/v1_fused_prepack.cpp -lm -o bin_qemu/v1_fused_prepack

echo "  [gem5]  bin/v1_fused_prepack"
"$CROSS_CXX" $COMMON_FLAGS $GEM5_DEFS $BASE_INC $SELF_INC \
    src/v1_fused_prepack.cpp "$M5OP" -lm -o bin/v1_fused_prepack

# ---------------------------------------------------------------------------
# V3 fused prepack: SPM with B prepacked outside ROI
# ---------------------------------------------------------------------------
echo ""
echo "=== V3 fused prepack ==="
echo "  [QEMU mock]  bin_qemu/v3_fused_prepack_mock"
"$CROSS_CXX" $COMMON_FLAGS -DMOCK_SPM $BASE_INC $SELF_INC \
    src/v3_fused_prepack.cpp -lm -o bin_qemu/v3_fused_prepack_mock

for VL_DEF in VL_4 VL_8 VL_16; do
    vl_num="${VL_DEF#VL_}"
    build_spm_gem5 src/v3_fused_prepack.cpp "$VL_DEF" "v3_fused_prepack_vl${vl_num}"
done

# ---------------------------------------------------------------------------
# Verify: correctness check (no SPM, no gem5)
# ---------------------------------------------------------------------------
echo ""
echo "=== verify_gemm ==="
echo "  [gem5]  bin/verify_gemm"
"$CROSS_CXX" $COMMON_FLAGS $GEM5_DEFS $BASE_INC $SELF_INC \
    src/verify_gemm.cpp "$M5OP" -lm -o bin/verify_gemm

echo ""
echo "=== Build complete ==="
