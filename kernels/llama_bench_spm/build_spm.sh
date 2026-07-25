#!/bin/bash
# build_spm.sh — 3-step build for llama_bench_spm (CacheFlex SPM, VL=4/8/16)
#
# SPM mnemonics (SPMCP_32_IMM, SPMCP_64_IMM, spm.ld1qd) are CacheFlex ISA
# extensions not native to GCC. Required 3-step build flow:
#   Step 1: C++ → ARM ASM  (SPM opcodes appear as text mnemonics)
#   Step 2: spm_compiler.py → .inst directives (encodes imm6 fields)
#   Step 3: assemble + link → static gem5 binary
#
# Produces bin_gem5/{bench}_{vl4,vl8,vl16} for each benchmark (VL2 is not part of the artifact).
#
# Usage:
#   cd llama_bench_spm/
#   bash build_spm.sh
#
# Override toolchain:
#   CROSS_CXX=/path/to/aarch64-none-linux-gnu-g++ bash build_spm.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BENCH_ROOT="$(cd .. && pwd)"
ARM_TC="$CACHEFLEX_ROOT/tools/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu"

if [ -z "${CROSS_CXX:-}" ]; then
    if [ -x "$ARM_TC/bin/aarch64-none-linux-gnu-g++" ]; then
        CROSS_CXX="$ARM_TC/bin/aarch64-none-linux-gnu-g++"
    else
        CROSS_CXX="aarch64-none-linux-gnu-g++"
    fi
fi

SPM_COMPILER="${SPM_COMPILER:-$CACHEFLEX_ROOT/spm_tools/spm_compiler.py}"
GEM5_INC="$CACHEFLEX_ROOT/gem5/include"
M5OP="$CACHEFLEX_ROOT/gem5/util/m5/build/arm64/out/m5op.o"

# Step 1/3 flags: -S for ASM output, -mbranch-protection=none avoids BTI
# sections that confuse spm_compiler.py
# -DGEM5 -DGEM5_SE for ROI macros; -I for gem5 headers
BASE_FLAGS="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -mbranch-protection=none"
GEM5_FLAGS="$BASE_FLAGS -DGEM5 -DGEM5_SE -I${GEM5_INC}"

mkdir -p bin_gem5

echo "=== Building llama_bench_spm (CacheFlex SPM, VL=4/8/16) ==="
echo "  Compiler : $CROSS_CXX"
echo "  SPM      : $SPM_COMPILER"
echo ""

# Build one benchmark for one VL
# Usage: build_spm_target <src.cpp> <VL_DEFINE> <output_suffix>
build_spm_target() {
    local src="$1"
    local vl_def="$2"   # e.g. VL_4
    local suffix="$3"   # e.g. vl4
    local base="${src%.cpp}"
    local asm_file="${base}_${suffix}.s"
    local enc_file="${base}_${suffix}_enc.s"
    local out_bin="bin_gem5/${base}_${suffix}"

    echo "  [${suffix}] ${src}"

    echo "    [1/3] C++ → ASM ..."
    "$CROSS_CXX" $GEM5_FLAGS -D${vl_def} -S -o "${asm_file}" "${src}"

    echo "    [2/3] spm_compiler.py → encoded ASM ..."
    python3 "$SPM_COMPILER" "${asm_file}" "${enc_file}"

    echo "    [3/3] assemble + link → ${out_bin}"
    "$CROSS_CXX" -static -o "${out_bin}" "${enc_file}" "$M5OP" -lm

    # Clean up intermediate files
    rm -f "${asm_file}" "${enc_file}"
    echo "         → ${out_bin}"
}

BENCHES=(02_qkv_gqa_spm 04_qk_gqa_spm 06_pv_gqa_spm 07_out_proj_spm 08_ffn_swiglu_spm)
VL_DEFS=(VL_4 VL_8 VL_16)
VL_SUFFS=(vl4 vl8 vl16)

for bench in "${BENCHES[@]}"; do
    echo ""
    echo "--- ${bench}.cpp ---"
    for i in "${!VL_DEFS[@]}"; do
        build_spm_target "${bench}.cpp" "${VL_DEFS[$i]}" "${VL_SUFFS[$i]}"
    done
done

# Multi-head persistent SPM evaluation (Method 1)
echo ""
echo "--- 04_qk_gqa_spm_mh.cpp ---"
for i in "${!VL_DEFS[@]}"; do
    build_spm_target "04_qk_gqa_spm_mh.cpp" "${VL_DEFS[$i]}" "${VL_SUFFS[$i]}"
done

echo ""
echo "=== Build complete ==="
echo ""
echo "Binaries in bin_gem5/:"
ls bin_gem5/ | head -40
echo ""
echo "NOTE: QEMU cannot run these binaries — CacheFlex SPM opcodes require gem5."
echo "Run: bash run_spm.sh"
