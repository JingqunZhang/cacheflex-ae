#!/bin/bash
# Build the three Figure 8 driver binaries used for both GEMM sizes at VL=16.
# Build row 4 with experiments/software_alternatives/l2-swpf-non-t-load/build_kernel.sh.
set -eu
cd "$(dirname "$0")/../../.."; ROOT="$PWD"
CROSS_CXX="${CROSS_CXX:-$ROOT/tools/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++}"
SPM_COMPILER="${SPM_COMPILER:-$ROOT/spm_tools/spm_compiler.py}"
M5OP="${M5OP_OBJ:-$ROOT/gem5/util/m5/build/arm64/out/m5op.o}"
GEM5_INC="$ROOT/gem5/include"
SRC="$ROOT/kernels/gemm/software_alternatives"; BIN="$SRC/bin"
COMMON="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static"
DEFS="-DGEM5 -DGEM5_SE -I$GEM5_INC"
INC="-I$ROOT/kernels/llama_bench -I$SRC"
SPM_INC="-I$ROOT/kernels/llama_bench_spm"
SPM_ASM_FLAGS="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -mbranch-protection=none"
OBJD="${CROSS_CXX%g++}objdump"
mkdir -p "$BIN"

# Rows 1 and 2: coherent cache kernel without software prefetch.
echo "[build] v3_l1swpf_none  (L1SWPF, no PF_A/PF_B)"
"$CROSS_CXX" $COMMON $DEFS $INC -DL1SWPF "$SRC/v3_l2pf_nt.cpp" "$M5OP" -lm \
    -o "$BIN/v3_l1swpf_none"
echo "  prfm count (expect 0): $("$OBJD" -d "$BIN/v3_l1swpf_none" 2>/dev/null | grep -ciE 'prfm.*pldl1keep')"

# Row 3: L2-targeted software prefetch.
echo "[build] v3_l2pf_only_b512  (L2PF_ONLY, PF_OFF_BYTES=512)"
"$CROSS_CXX" $COMMON $DEFS $INC -DL2PF_ONLY -DPF_OFF_BYTES=512 "$SRC/v3_l2pf_nt.cpp" "$M5OP" -lm \
    -o "$BIN/v3_l2pf_only_b512"
echo "  prfm pldl2keep count (expect >0): $("$OBJD" -d "$BIN/v3_l2pf_only_b512" 2>/dev/null | grep -ciE 'prfm.*pldl2keep')"

# Rows 5 and 6: CacheFlex kernel.
echo "[build] v3_spm_n0outer_VL_16  (3-step SPM build, -DVL_16)"
ASM="$SRC/v3_spm_n0outer_VL_16_tmp.s"; ENC="$SRC/v3_spm_n0outer_VL_16_tmp_enc.s"
echo "  1/3 C++ -> ASM"
"$CROSS_CXX" $SPM_ASM_FLAGS $DEFS $SPM_INC $INC -DVL_16 -S -o "$ASM" "$SRC/v3_spm_n0outer.cpp"
echo "  2/3 spm_compiler.py"
python3 "$SPM_COMPILER" "$ASM" "$ENC"
N_INST=$(grep -c '\.inst' "$ENC" || true)
echo "      encoded .inst count: $N_INST (must be > 0)"
[ "${N_INST:-0}" -gt 0 ] || { echo "FATAL: no SPM ops encoded"; exit 1; }
echo "  3/3 assemble+link"
"$CROSS_CXX" -static -o "$BIN/v3_spm_n0outer_VL_16" "$ENC" "$M5OP" -lm
rm -f "$ASM" "$ENC"

echo "=== built binaries ==="
ls -la "$BIN/v3_l1swpf_none" "$BIN/v3_l2pf_only_b512" "$BIN/v3_spm_n0outer_VL_16"
echo "=== row-4 binary (built separately) ==="
ls -la "$BIN/v3_l2pin_nt_pf0"
