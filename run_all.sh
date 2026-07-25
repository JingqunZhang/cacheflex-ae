#!/bin/bash
# Build the paper kernels, run the four paper experiments in order, and render
# Figure 2, Figure 7, Figure 8, and Figure 9.
set -euo pipefail

# Keep the canonical root workflow bound to this artifact even if the
# caller's shell contains testing overrides.
unset CACHEFLEX_ROOT BEST_JSON GENERATED_DIR RESULTS_DIR OUTPUT_DIR GEM5 GEM5_SE

ROOT="$(cd "$(dirname "$0")" && pwd)"
JOBS=1
MAX_JOBS=16

while [ "$#" -gt 0 ]; do
    case "$1" in
        -j|--jobs)
            shift
            [ "$#" -gt 0 ] || { echo "ERROR: -j requires a value" >&2; exit 2; }
            JOBS="$1"
            ;;
        -h|--help)
            echo "Usage: bash run_all.sh [-j N]  (1 <= N <= 16)"
            echo "Build and run Figure 2, Figure 7, Figure 8, then Figure 9."
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

case "$JOBS" in
    ''|*[!0-9]*) echo "ERROR: -j expects a positive integer" >&2; exit 2 ;;
esac
[ "$JOBS" -ge 1 ] || { echo "ERROR: -j expects a positive integer" >&2; exit 2; }
if [ "$JOBS" -gt "$MAX_JOBS" ]; then
    echo "WARNING: -j $JOBS exceeds the supported limit; capping at $MAX_JOBS" >&2
    JOBS=$MAX_JOBS
fi

cd "$ROOT"
source setup_env.sh
PLOT_PARENT="$ROOT/results/generated"
PLOT_OUT="$ROOT/results/generated/paper_outputs"
mkdir -p "$PLOT_PARENT"

[ -x "$GEM5" ] || {
    echo "ERROR: gem5 is not built. Complete SETUP.md first." >&2
    exit 3
}
[ -f "$M5OP_OBJ" ] || {
    echo "ERROR: the gem5 ROI-marker object is missing. Complete SETUP.md first." >&2
    exit 3
}
cmp -s \
    kernels/gemm/cacheflex/src/kernels_fused.hpp \
    kernels/end2end/src/kernels_fused.hpp || {
    echo "ERROR: Figure 7 and end-to-end fused GEMM cores differ." >&2
    exit 3
}

echo "[reference] Validate and render the expected paper results"
bash scripts/render_reference.sh

echo "[build] Figure 2 and Figure 7 kernels"
bash kernels/gemm/cacheflex/build.sh

echo "[build] Figure 8 kernels"
bash experiments/software_alternatives/l2-swpf-non-t-load/build_kernel.sh
bash experiments/software_alternatives/twosize_ladder/build_bins.sh

echo "[build] Figure 9 kernels"
bash experiments/end2end/build.sh gem5
bash experiments/end2end/build_vl16.sh

echo "[1/4] Figure 2"
bash experiments/capacity_motivation/run.sh -j "$JOBS"

echo "[2/4] Figure 7"
bash experiments/vl_length/run.sh -j "$JOBS"
python3 experiments/vl_length/audit_results.py \
    experiments/vl_length/results/generated/best

echo "[3/4] Figure 8"
bash experiments/software_alternatives/run.sh -j "$JOBS"

echo "[4/4] Figure 9"
bash experiments/end2end/run.sh -j "$JOBS"

BUILD_OUT="$(mktemp -d "$PLOT_PARENT/.paper_outputs.building.XXXXXX")"
cleanup() {
    if [[ "$BUILD_OUT" == "$PLOT_PARENT"/.paper_outputs.building.* ]]; then
        rm -rf -- "$BUILD_OUT"
    fi
}
trap cleanup EXIT

echo "[plot] Figure 2"
RESULTS_DIR="$ROOT/experiments/capacity_motivation/results/generated/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/capacity_motivation/plot_fig2_real.py

echo "[plot] Figure 7 and its energy data"
RESULTS_DIR="$ROOT/experiments/vl_length/results/generated/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/vl_length/plot_fig7_real.py

echo "[plot] Figure 8"
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/generated/best" \
    python3 experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py \
    --check
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/generated/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py

echo "[plot] Figure 9"
END_RESULTS="$ROOT/experiments/end2end/results/generated"
RESULTS_DIR="$END_RESULTS" python3 experiments/end2end/scripts/plot_fig9_real.py --check
RESULTS_DIR="$END_RESULTS" OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/end2end/scripts/plot_fig9_real.py

(
    cd "$BUILD_OUT"
    sha256sum \
        fig2_real.pdf fig2_real.png fig2_data.json \
        fig7_real.pdf fig7_real.png fig7_data.json \
        fig8_real.pdf fig8_real.png fig8_data.json \
        fig9_real.pdf fig9_real.png fig9_data.json \
        > OUTPUT_MANIFEST.sha256
)

PREVIOUS="$PLOT_PARENT/.paper_outputs.previous.$$"
if [ -e "$PLOT_OUT" ]; then
    mv "$PLOT_OUT" "$PREVIOUS"
fi
if ! mv "$BUILD_OUT" "$PLOT_OUT"; then
    [ ! -e "$PREVIOUS" ] || mv "$PREVIOUS" "$PLOT_OUT"
    exit 1
fi
BUILD_OUT=""
if [ -e "$PREVIOUS" ]; then
    rm -rf -- "$PREVIOUS"
fi
trap - EXIT

echo "[compare] Fresh results against the paper reference (5% limit)"
python3 scripts/compare_results.py \
    --reference "$ROOT/results/generated/reference_outputs" \
    --candidate "$PLOT_OUT"

echo "DONE: all paper experiments and plots completed."
echo "Outputs: $PLOT_OUT"
