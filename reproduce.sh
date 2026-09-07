#!/usr/bin/env bash
# Generate all paper figures and Table 4 from the supplied measurements.
set -euo pipefail

unset CACHEFLEX_ROOT BEST_JSON GENERATED_DIR RESULTS_DIR OUTPUT_DIR GEM5 GEM5_SE

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/results"
PYTHON_BIN="${PYTHON:-python3}"
mkdir -p "$OUT"
BUILD_OUT="$(mktemp -d "$OUT/.building.XXXXXX")"

cleanup() {
    if [[ "$BUILD_OUT" == "$OUT"/.building.* ]]; then
        rm -rf -- "$BUILD_OUT"
    fi
}
trap cleanup EXIT

echo "[1/5] Figure 2"
RESULTS_DIR="$ROOT/experiments/capacity_motivation/results/reference" \
    OUTPUT_DIR="$BUILD_OUT" \
    "$PYTHON_BIN" "$ROOT/experiments/capacity_motivation/plot_fig2.py"

echo "[2/5] Figure 7"
RESULTS_DIR="$ROOT/experiments/vl_length/results/reference" \
    OUTPUT_DIR="$BUILD_OUT" \
    "$PYTHON_BIN" "$ROOT/experiments/vl_length/plot_fig7.py"

echo "[3/5] Figure 8"
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/reference/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    "$PYTHON_BIN" \
    "$ROOT/experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py"

echo "[4/5] Figure 9"
RESULTS_DIR="$ROOT/experiments/end2end/results/reference" \
    OUTPUT_DIR="$BUILD_OUT" \
    "$PYTHON_BIN" "$ROOT/experiments/end2end/scripts/plot_fig9.py"

echo "[5/5] Table 4"
RESULTS_DIR="$ROOT/experiments/end2end/results/reference" \
    OUTPUT_DIR="$BUILD_OUT" \
    "$PYTHON_BIN" "$ROOT/experiments/end2end/scripts/plot_table4.py"

OUTPUTS=(
    fig2.pdf fig2.png fig2_data.json
    fig7.pdf fig7.png fig7_data.json
    fig8.pdf fig8.png fig8_data.json
    fig9.pdf fig9.png fig9_data.json
    table4_data.json table4.tex table4.md
)
for output in "${OUTPUTS[@]}"; do
    [ -s "$BUILD_OUT/$output" ] || {
        echo "ERROR: missing output: $output" >&2
        exit 1
    }
done
for output in "${OUTPUTS[@]}"; do
    mv -f -- "$BUILD_OUT/$output" "$OUT/$output"
done

trap - EXIT
rmdir "$BUILD_OUT"
echo "Done. Figures and table are in $OUT."
