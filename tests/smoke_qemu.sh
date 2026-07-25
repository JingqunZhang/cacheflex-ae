#!/bin/bash
# smoke_qemu.sh — Build cache (non-SPM) kernels and run them under QEMU.
# Validates the cross-compiler, QEMU setup, and attention self-check.
# No gem5 required. ~30 seconds.
#
# Usage:
#   source $CACHEFLEX_ROOT/setup_env.sh
#   bash $CACHEFLEX_ROOT/tests/smoke_qemu.sh
#
# Exit codes: 0 = PASS, 1 = FAIL, 3 = SKIP (toolchain/QEMU not installed yet)

set -e

if [ -z "${CACHEFLEX_ROOT:-}" ]; then
    CACHEFLEX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    source "$CACHEFLEX_ROOT/setup_env.sh" >/dev/null
fi

fail() { echo ""; echo "FAIL: $*"; exit 1; }
# A missing prerequisite tool is a SKIP (exit 3), not a test failure: the
# test cannot run until scripts/setup_tools.sh has fetched the toolchain+QEMU.
skip() { echo ""; echo "SKIP: $*"; exit 3; }

echo "=== [1/4] Environment ==="
[ -x "$CROSS_CXX" ] || skip "cross-compiler not found — run: bash scripts/setup_tools.sh"
[ -x "$QEMU" ]      || skip "QEMU not found — run: bash scripts/setup_tools.sh"
echo "  ok"

echo ""
echo "=== [2/4] Cross-compiling kernels/llama_bench/01_rmsnorm.cpp ==="
BIN="$CACHEFLEX_ROOT/kernels/llama_bench/bin/01_rmsnorm"
mkdir -p "$(dirname "$BIN")"
"$CROSS_CXX" -O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static \
    "$CACHEFLEX_ROOT/kernels/llama_bench/01_rmsnorm.cpp" \
    -lm -o "$BIN"
[ -x "$BIN" ] || fail "binary not built"
echo "  built: $(basename "$BIN")"

echo ""
echo "=== [3/4] Running RMSNorm under QEMU (T=64, D=256, 1 iter) ==="
# QEMU expresses the SVE length in bytes. VL=4 is 4 x 128 bits = 64 bytes.
out=$("$QEMU" -cpu max,sve-default-vector-length=64 "$BIN" 64 256 1 2>&1)
echo "$out" | head -10
echo "$out" | grep -qE "RMSNorm|OK|time|cycles" || fail "no expected output (kernel might have crashed)"

echo ""
echo "=== [4/4] Attention numerical self-check under QEMU ==="
ATTN_DIR="$CACHEFLEX_ROOT/kernels/end2end/bin"
mkdir -p "$ATTN_DIR"
for source in cache_flash_attn cache_unfused_attn; do
    ATTN_BIN="$ATTN_DIR/${source}_qemu_smoke"
    "$CROSS_CXX" -O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static \
        -I"$CACHEFLEX_ROOT/kernels/end2end/src" \
        "$CACHEFLEX_ROOT/kernels/end2end/src/${source}.cpp" \
        -lm -o "$ATTN_BIN"
    [ -x "$ATTN_BIN" ] || fail "$source binary not built"

    if [ "$source" = cache_flash_attn ]; then
        attn_args=(32 4 4 64 1 16 32 causal)
    else
        attn_args=(32 4 4 64 1 32 32 32 0 causal)
    fi
    attn_out=$(CF_SELF_CHECK=1 "$QEMU" \
        -cpu max,sve-default-vector-length=64 \
        "$ATTN_BIN" "${attn_args[@]}" 2>&1)
    echo "  $source"
    echo "$attn_out" | grep -E \
        'ATTENTION_MODE|CHECKSUM|SELF_CHECK' | head -10
    pass_count=$(
        printf '%s\n' "$attn_out" |
            grep -c '^SELF_CHECK PASS$' || true
    )
    [ "$pass_count" -eq 1 ] \
        || fail "$source self-check did not report exactly one PASS"
    if printf '%s\n' "$attn_out" |
        grep -qE '^SELF_CHECK (FAIL|SKIP)'; then
        fail "$source self-check reported FAIL or SKIP"
    fi
    checksum_count=$(
        printf '%s\n' "$attn_out" |
            grep -c '^CHECKSUM:' || true
    )
    [ "$checksum_count" -eq 1 ] \
        || fail "$source did not emit exactly one CHECKSUM receipt"
done

echo ""
echo "========================================"
echo "SMOKE TEST PASSED"
echo "========================================"
