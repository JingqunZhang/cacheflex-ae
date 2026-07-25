#!/bin/bash
# Build the Figure 8 non-temporal-load kernels.
# The source uses SVE ldnt1h for B reads and scalar prfm pldl2keep for prefetch.
set -eu
cd "$(dirname "$0")/../../.."; ROOT="$PWD"
: "${CACHEFLEX_ROOT:=$ROOT}"
CROSS_CXX="${CROSS_CXX:-$ROOT/tools/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++}"
M5OP="${M5OP_OBJ:-$ROOT/gem5/util/m5/build/arm64/out/m5op.o}"
GEM5_INC="$ROOT/gem5/include"
SRC="$ROOT/kernels/gemm/software_alternatives"
BIN="$SRC/bin"
COMMON="-O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static"
DEFS="-DGEM5 -DGEM5_SE -I$GEM5_INC"
INC="-I$ROOT/kernels/llama_bench -I$SRC"

mkdir -p "$BIN"
# Prefetch distance: 512 bytes.
echo "[build] v3_l2pin_nt (ldnt1h + prfm pldl2keep, PF=512)"
"$CROSS_CXX" $COMMON $DEFS $INC -DL2PIN_NT -DPF_OFF_BYTES=512 \
    "$SRC/v3_l2pf_nt.cpp" "$M5OP" -lm -o "$BIN/v3_l2pin_nt"
# Non-temporal load without software prefetch.
echo "[build] v3_l2pin_nt_pf0 (ldnt1h only, no prefetch)"
"$CROSS_CXX" $COMMON $DEFS $INC -DL2PIN_NT -DPF_OFF_BYTES=0 \
    "$SRC/v3_l2pf_nt.cpp" "$M5OP" -lm -o "$BIN/v3_l2pin_nt_pf0"
echo "[ok] built:"; ls -la "$BIN/v3_l2pin_nt" "$BIN/v3_l2pin_nt_pf0"
# Confirm that the expected instructions were emitted.
OBJD="${CROSS_CXX%g++}objdump"
echo "=== ldnt1h / prfm pldl2keep present in v3_l2pin_nt? ==="
"$OBJD" -d "$BIN/v3_l2pin_nt" 2>/dev/null | grep -ciE "ldnt1h" | sed 's/^/  ldnt1h count: /'
"$OBJD" -d "$BIN/v3_l2pin_nt" 2>/dev/null | grep -iE "prfm.*pldl2keep" | head -1 | sed 's/^/  prfm: /'
