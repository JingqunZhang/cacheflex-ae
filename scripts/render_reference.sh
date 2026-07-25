#!/usr/bin/env bash
# Render every paper output from the shipped reference results without gem5.
set -euo pipefail

# Keep the public quick path bound to the shipped manifests and references.
unset CACHEFLEX_ROOT BEST_JSON GENERATED_DIR RESULTS_DIR OUTPUT_DIR GEM5_SE

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/generated/reference_outputs"
OUT_PARENT="$ROOT/results/generated"
mkdir -p "$OUT_PARENT"
BUILD_OUT="$(mktemp -d "$OUT_PARENT/.reference_outputs.building.XXXXXX")"

cleanup() {
    if [[ "$BUILD_OUT" == "$OUT_PARENT"/.reference_outputs.building.* ]]; then
        rm -rf -- "$BUILD_OUT"
    fi
}
trap cleanup EXIT

echo "[check] Reference integrity"
python3 "$ROOT/scripts/reference_integrity.py" check --root "$ROOT"

echo "[1/5] Figure 2"
RESULTS_DIR="$ROOT/experiments/capacity_motivation/results/reference" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 "$ROOT/experiments/capacity_motivation/plot_fig2_real.py"

echo "[2/5] Figure 7"
python3 "$ROOT/experiments/vl_length/audit_results.py" \
    "$ROOT/experiments/vl_length/results/reference/results_headh0"
RESULTS_DIR="$ROOT/experiments/vl_length/results/reference" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 "$ROOT/experiments/vl_length/plot_fig7_real.py"

echo "[3/5] Figure 8"
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/reference/best" \
    python3 "$ROOT/experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py" \
    --check
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/reference/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 "$ROOT/experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py"

echo "[4/5] Figure 9"
END_RESULTS="$ROOT/experiments/end2end/results/reference"
RESULTS_DIR="$END_RESULTS" \
    python3 "$ROOT/experiments/end2end/scripts/plot_fig9_real.py" --check
RESULTS_DIR="$END_RESULTS" OUTPUT_DIR="$BUILD_OUT" \
    python3 "$ROOT/experiments/end2end/scripts/plot_fig9_real.py"

echo "[5/5] Table 4 (flash scaling)"
OUTPUT_DIR="$BUILD_OUT" \
    python3 "$ROOT/experiments/end2end/scripts/plot_table4_flash.py" --check

(
    cd "$BUILD_OUT"
    sha256sum \
        fig2_real.pdf fig2_real.png fig2_data.json \
        fig7_real.pdf fig7_real.png fig7_data.json \
        fig8_real.pdf fig8_real.png fig8_data.json \
        fig9_real.pdf fig9_real.png fig9_data.json \
        table4_data.json table4_flash.tex table4_flash.md \
        > OUTPUT_MANIFEST.sha256
)

PREVIOUS="$OUT_PARENT/.reference_outputs.previous.$$"
if [ -e "$OUT" ]; then
    mv "$OUT" "$PREVIOUS"
fi
if ! mv "$BUILD_OUT" "$OUT"; then
    [ ! -e "$PREVIOUS" ] || mv "$PREVIOUS" "$OUT"
    exit 1
fi
BUILD_OUT=""
if [ -e "$PREVIOUS" ]; then
    rm -rf -- "$PREVIOUS"
fi
trap - EXIT

echo "DONE: reference outputs are in $OUT"
