#!/bin/bash
# Figure 8 experiment driver for two GEMM sizes.
#
# Default mode runs the 12 cells selected by best.json. --sweep searches the
# complete KC-by-MC grids. -j N bounds concurrent gem5 processes.
# Generated results never overwrite results/reference/.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CACHEFLEX_ROOT:-$(cd "$HERE/../.." && pwd)}"
source "$HERE/../../scripts/run_helpers.sh"

RH_HELP_TEXT="Usage: run.sh [--sweep] [-j N] [-h]
  default   : fresh gem5 runs of the 12 cells selected in best.json
              (VL=16) -> results/generated/best/<cell>/
              (existing cell outdirs are moved to <outdir>.prev.<epoch> first)
  --sweep   : full KCxMC grids for both sizes
              (twosize_ladder/controlled_size_headh0.sh w3 / g2048), run
              sequentially -> logs in results/generated/sweep/
  -j N      : bounds the number of concurrent gem5 processes, including
              sweep internals (each ladder invocation receives N as its PAR
              argument; capped at 16 and at nproc)
Requires: source setup_env.sh (CACHEFLEX_ROOT), a built gem5.opt, and the
Figure 8 kernels built by the root run_all.sh workflow.
Plot a finished best run with twosize_ladder/make_fig_ladder_live.py
using RESULTS_DIR=results/generated/best."
rh_parse_args "$@"

BEST="${BEST_JSON:-$HERE/best.json}"
GEN="${GENERATED_DIR:-$HERE/results/generated}"

cells_tsv() {
    python3 "$ROOT/scripts/cell_manifest.py" \
        --experiment software_alternatives --best "$BEST"
}

if [ "$RH_SWEEP" = 1 ]; then
    CTRL="$HERE/twosize_ladder/controlled_size_headh0.sh"
    POOL="$GEN/sweep/pool"
    mkdir -p "$POOL"
    # Run the two internally parallel sweeps sequentially so -j remains a
    # global concurrency bound.
    rh_launch "$GEN/sweep/ladder_w3" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" RESULTS_OUT_ROOT="$POOL" \
        bash "$CTRL" w3    256  2048 2048 "$RH_JOBS"
    if ! rh_wait_all; then exit 1; fi
    RH_PIDS=()
    rh_launch "$GEN/sweep/ladder_g2048" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" RESULTS_OUT_ROOT="$POOL" \
        bash "$CTRL" g2048 2048 2048 2048 "$RH_JOBS"
    rh_wait_all
    exit $?
fi

rh_require_best "$BEST"
if ! rh_materialize_cells 12 6; then exit 3; fi
mkdir -p "$GEN/best"
set -f   # args are word-split below; disable globbing
while IFS=$'\t' read -r name runner bin spm envs args; do
    [ "$envs" = "-" ] && envs=""
    out="$GEN/best/$name"
    rh_launch "$out" "${RH_CLEAN_ENV[@]}" bash "$ROOT/$runner" \
        "$out" "$ROOT/$bin" "$spm" "$envs" $args
done < "$RH_CELLS_FILE"
set +f
rh_wait_all; RC=$?
if ! rh_validate_gem5_cells; then RC=1; fi
rh_compat_layout "$GEN/best"
exit $RC
