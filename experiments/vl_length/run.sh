#!/bin/bash
# run.sh — unified AE run interface for experiments/vl_length (Fig 7).
#
#   default   : FRESH gem5 runs of ONLY the best cells pinned in best.json
#               (42 cells: W1–W7 x VL{4,8,16} x {v1,v3}, each at its
#               pinned KC/MC)
#               -> results/generated/best/<cell>/
#   --sweep   : full KC sweeps via the existing run_kc_sweep_vl{4,8,16}.sh,
#               with their output redirected under results/generated/sweep/
#   -j N      : bounds the TOTAL number of concurrent gem5 processes; in sweep
#               mode the three KC-sweep scripts run SEQUENTIALLY, each with
#               MAX_JOBS=N, so the bound holds globally
#
# Plot-from-best-run (canonical plotter = plot_fig7.py; it expects
# <root>/results_headh0/<cell>.log + <cell>/stats.txt; best mode emits the
# <cell>.log compat links via rh_compat_layout):
#   cd experiments/vl_length
#   P=results/generated/fig7_plotroot; mkdir -p "$P"
#   ln -sfn ../best "$P/results_headh0"
#   RESULTS_DIR="$P" python3 plot_fig7.py
#
# Never overwrites results/reference/ (shipped reference results).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CACHEFLEX_ROOT:-$(cd "$HERE/../.." && pwd)}"
source "$HERE/../../scripts/run_helpers.sh"

RH_HELP_TEXT="Usage: run.sh [--sweep] [-j N] [-h]
  default   : fresh gem5 runs of the 42 best.json cells
              -> results/generated/best/<cell>/
              (existing cell outdirs are moved to <outdir>.prev.<epoch> first)
  --sweep   : full KC sweeps (run_kc_sweep_vl4/vl8/vl16.sh), run SEQUENTIALLY
              -> results/generated/sweep/
  -j N      : bounds the TOTAL number of concurrent gem5 processes, including
              sweep internals (each sweep script gets N as MAX_JOBS; capped
              at 16 and at nproc)
Requires: source setup_env.sh (CACHEFLEX_ROOT) and a built gem5.opt.
Plot a finished best run with plot_fig7.py (see the plot-from-best-run
recipe in the comment atop this run.sh)."
rh_parse_args "$@"

BEST="${BEST_JSON:-$HERE/best.json}"
GEN="${GENERATED_DIR:-$HERE/results/generated}"
GEM5="${GEM5:-$ROOT/gem5/build/ARM/gem5.opt}"
GEM5_SE="${GEM5_SE:-$ROOT/gem5/configs/deprecated/example/se.py}"

# ── canonical flag block; per-cell VL and SPM ports come from best.json ──
GEM5_BASE=(
    --cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz
    --caches --l2cache --l3cache
    --l1d_size=64kB --l1i_size=64kB --l1d_assoc=4 --l1i_assoc=4
    --l2_size=512kB --l2_assoc=8
    --l3_size=4MB --l3_assoc=16
    --cacheline_size=64 --mem-size=16GB --mem-channels=2
    --mem-type=DDR4_2400_8x8
    --l1i-hwp-type=StridePrefetcher
    --l1d-hwp-type=StridePrefetcher
    --l2-hwp-type=AMPMPrefetcher
    --l3-hwp-type=BOPPrefetcher
    -P 'system.cpu[0].icache.prefetcher.degree=4'
    -P 'system.cpu[0].dcache.prefetcher.degree=4'
    -P 'system.cpu[0].icache.tag_latency=2'
    -P 'system.cpu[0].icache.data_latency=2'
    -P 'system.cpu[0].fetchWidth=5'
    -P 'system.cpu[0].decodeWidth=5'
    -P 'system.cpu[0].commitWidth=5'
    -P 'system.cpu[0].renameWidth=5'
    -P 'system.cpu[0].dispatchWidth=8'
    -P 'system.cpu[0].issueWidth=8'
    -P 'system.cpu[0].wbWidth=8'
    -P 'system.cpu[0].numROBEntries=128'
    -P 'system.cpu[0].numIQEntries=80'
    -P 'system.cpu[0].LQEntries=32'
    -P 'system.cpu[0].SQEntries=48'
    -P 'system.cpu[0].numPhysIntRegs=128'
    -P 'system.cpu[0].numPhysFloatRegs=192'
    -P 'system.cpu[0].numPhysVecRegs=192'
    -P 'system.cpu[0].numPhysVecPredRegs=64'
    -P 'system.cpu[0].cacheLoadPorts=2'
    -P 'system.cpu[0].cacheStorePorts=1'
    -P 'system.cpu[0].fuPool.FUList[0].count=2'
    -P 'system.cpu[0].fuPool.FUList[2].count=2'
    -P 'system.cpu[0].fuPool.FUList[5].count=2'
    -P 'system.cpu[0].fuPool.FUList[7].count=2'
    -P 'system.cpu[0].fuPool.FUList[9].count=2'
    -P 'system.tol2bus.frontend_latency=1'
    -P 'system.tol2bus.header_latency=0'
    -P 'system.tol2bus.forward_latency=1'
    -P 'system.tol2bus.response_latency=1'
    -P 'system.tol2bus.width=64'
    -P 'system.l2.tag_latency=8'
    -P 'system.l2.data_latency=8'
    -P 'system.l2.response_latency=8'
    -P 'system.tol3bus.width=32'
    -P 'system.tol3bus.frontend_latency=10'
    -P 'system.tol3bus.forward_latency=10'
    -P 'system.tol3bus.response_latency=10'
    -P 'system.l3.tag_latency=35'
    -P 'system.l3.data_latency=35'
    -P 'system.l3.response_latency=35'
)

cells_tsv() {
    python3 "$ROOT/scripts/cell_manifest.py" \
        --experiment vl_length --best "$BEST"
}

if [ "$RH_SWEEP" = 1 ]; then
    # Full sweep: run the existing KC-sweep scripts through a $0-relative shim
    # so their $(pwd)-relative OUTDIRs (results_kc_sweep*) materialize under
    # results/generated/sweep/ instead of on top of the shipped reference.
    SW="$GEN/sweep"
    mkdir -p "$SW"
    for s in run_kc_sweep_vl4.sh run_kc_sweep_vl8.sh run_kc_sweep_vl16.sh; do
        ln -sf "../../../$s" "$SW/$s"
    done
    # SEQUENTIAL inner invocation: each sweep script has internal parallelism
    # (its MAX_JOBS argument = -j), so running them one after another keeps
    # the TOTAL number of concurrent gem5 processes bounded by -j.
    rh_launch "$SW/kc_sweep_vl4" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" \
        bash "$SW/run_kc_sweep_vl4.sh" "$RH_JOBS"
    if ! rh_wait_all; then exit 1; fi
    RH_PIDS=()
    rh_launch "$SW/kc_sweep_vl8" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" \
        bash "$SW/run_kc_sweep_vl8.sh" "$RH_JOBS"
    if ! rh_wait_all; then exit 1; fi
    RH_PIDS=()
    rh_launch "$SW/kc_sweep_vl16" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" \
        bash "$SW/run_kc_sweep_vl16.sh" "$RH_JOBS"
    rh_wait_all
    exit $?
fi

rh_require_best "$BEST"
if ! rh_materialize_cells 42 5; then exit 3; fi
mkdir -p "$GEN/best"
set -f   # gem5_extra/env are word-split below; disable globbing ([0] patterns)
while IFS=$'\t' read -r name envs bin args extra; do
    [ "$envs" = "-" ] && envs=""
    [ "$extra" = "-" ] && extra=""
    out="$GEN/best/$name"
    rh_launch "$out" "${RH_CLEAN_ENV[@]}" $envs "$GEM5" --outdir="$out" "$GEM5_SE" \
        "${GEM5_BASE[@]}" $extra -c "$ROOT/$bin" -o "$args"
done < "$RH_CELLS_FILE"
set +f
rh_wait_all; RC=$?
if ! rh_validate_gem5_cells; then RC=1; fi
rh_compat_layout "$GEN/best"
exit $RC
