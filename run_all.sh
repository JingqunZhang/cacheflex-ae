#!/bin/bash
# Build the paper kernels, run the paper experiments, and regenerate every
# reported figure and table.
set -euo pipefail

# Ignore caller-side overrides so this workflow uses the repository settings.
unset CACHEFLEX_ROOT BEST_JSON GENERATED_DIR RESULTS_DIR OUTPUT_DIR GEM5 GEM5_SE
unset RESULT_SUBDIR EXPECTED_CELLS

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
            echo "Build and run Figures 2, 7, 8, 9 and the Table 4 companion cells."
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
PLOT_PARENT="$ROOT/results"
PLOT_OUT="$ROOT/results/rerun"
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

echo "[data] Generate figures from the supplied measurements"
bash reproduce.sh

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

echo "[table] Table 4 companion cells"
BEST_JSON="$ROOT/experiments/end2end/table4_best.json" \
    GENERATED_DIR="$ROOT/experiments/end2end/results/generated" \
    RESULT_SUBDIR=table4 EXPECTED_CELLS=8 \
    bash experiments/end2end/run.sh -j "$JOBS"

BUILD_OUT="$(mktemp -d "$PLOT_PARENT/.rerun.building.XXXXXX")"
cleanup() {
    if [[ "$BUILD_OUT" == "$PLOT_PARENT"/.rerun.building.* ]]; then
        rm -rf -- "$BUILD_OUT"
    fi
}
trap cleanup EXIT

echo "[plot] Figure 2"
RESULTS_DIR="$ROOT/experiments/capacity_motivation/results/generated/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/capacity_motivation/plot_fig2.py

echo "[plot] Figure 7 and its energy data"
RESULTS_DIR="$ROOT/experiments/vl_length/results/generated/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/vl_length/plot_fig7.py

echo "[plot] Figure 8"
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/generated/best" \
    python3 experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py \
    --check
RESULTS_DIR="$ROOT/experiments/software_alternatives/results/generated/best" \
    OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py

echo "[plot] Figure 9"
END_RESULTS="$ROOT/experiments/end2end/results/generated"
RESULTS_DIR="$END_RESULTS" python3 experiments/end2end/scripts/plot_fig9.py --check
RESULTS_DIR="$END_RESULTS" OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/end2end/scripts/plot_fig9.py

echo "[plot] Table 4"
RESULTS_DIR="$END_RESULTS" OUTPUT_DIR="$BUILD_OUT" \
    python3 experiments/end2end/scripts/plot_table4.py

PREVIOUS="$PLOT_PARENT/.rerun.previous.$$"
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

echo "[compare] Fresh results against the paper results (5% limit)"
python3 scripts/compare_results.py \
    --reference "$ROOT/results" \
    --candidate "$PLOT_OUT"

echo "DONE: all paper experiments and plots completed."
echo "Outputs: $PLOT_OUT"
