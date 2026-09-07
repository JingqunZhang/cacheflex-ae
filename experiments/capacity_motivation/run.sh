#!/bin/bash
# run.sh — unified AE run interface for experiments/capacity_motivation (Fig 2).
#
#   default   : FRESH gem5 runs of ONLY the best cells pinned in best.json
#               (21 cells: 18 best-of-KC cache points + 3 CacheFlex v3_fused
#               points)                     -> results/generated/best/<cell>/
#   --sweep   : declared capacity x bandwidth x KC grid via the existing scripts
#               run_fig2_cache_n2048_headh0.sh + run_spm_fused_n2048_headh0.sh, with their
#               output redirected under      results/generated/sweep/
#   -j N      : bounds the TOTAL number of concurrent gem5 processes; in sweep
#               mode the two sweep scripts run SEQUENTIALLY and receive N via
#               MAX_JOBS, so the bound holds globally
#
# Plot-from-best-run (canonical plotter = plot_fig2.py; it expects
# <root>/results_fig2_n2048_headh0/fused_*.log and
# <root>/results_spm_fused_n2048_headh0/v3fused_*.log; best mode emits the
# <cell>.log compat links via rh_compat_layout, and the two glob families are
# disjoint, so both aliases may point at the same best/ dir):
#   cd experiments/capacity_motivation
#   P=results/generated/fig2_plotroot; mkdir -p "$P"
#   ln -sfn ../best "$P/results_fig2_n2048_headh0"
#   ln -sfn ../best "$P/results_spm_fused_n2048_headh0"
#   RESULTS_DIR="$P" python3 plot_fig2.py
#
# Never overwrites results/reference/ (shipped reference results).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CACHEFLEX_ROOT:-$(cd "$HERE/../.." && pwd)}"
source "$HERE/../../scripts/run_helpers.sh"

RH_HELP_TEXT="Usage: run.sh [--sweep] [-j N] [-h]
  default   : fresh gem5 runs of the 21 best.json cells
              -> results/generated/best/<cell>/
              (existing cell outdirs are moved to <outdir>.prev.<epoch> first)
  --sweep   : full capacity x bandwidth x KC sweep (run_fig2_cache_n2048_headh0.sh)
              + CacheFlex pinned points (run_spm_fused_n2048_headh0.sh), run
              SEQUENTIALLY -> results/generated/sweep/
  -j N      : bounds the TOTAL number of concurrent gem5 processes, including
              sweep internals (both sweep scripts receive N via MAX_JOBS;
              capped at 16 and at nproc)
Requires: source setup_env.sh (CACHEFLEX_ROOT, GEM5) and a built gem5.opt.
Plot a finished best run with plot_fig2.py (see the plot-from-best-run
recipe in the comment atop this run.sh)."
rh_parse_args "$@"

BEST="${BEST_JSON:-$HERE/best.json}"
GEN="${GENERATED_DIR:-$HERE/results/generated}"
GEM5="${GEM5:-$ROOT/gem5/build/ARM/gem5.opt}"
GEM5_SE="${GEM5_SE:-$ROOT/gem5/configs/deprecated/example/se.py}"

# ── canonical flag block (run_fig2_cache_n2048_headh0.sh / run_spm_fused_n2048_headh0.sh;
#    per-cell l1d size, L1 ports, FU[7]/FU[9], SPM ports and VL come from best.json) ──
GEM5_BASE=(
    --cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz
    --caches --l2cache --l3cache
    --l1i_size=64kB --l1i_assoc=4
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
    -P 'system.cpu[0].fuPool.FUList[0].count=2'
    -P 'system.cpu[0].fuPool.FUList[2].count=2'
    -P 'system.cpu[0].fuPool.FUList[5].count=2'
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
        --experiment capacity_motivation --best "$BEST"
}

if [ "$RH_SWEEP" = 1 ]; then
    # Declared grid: run the existing scripts through a $0-relative shim so that
    # their $(pwd)-relative OUTDIRs materialize under results/generated/sweep/
    # instead of on top of the shipped reference results.
    SW="$GEN/sweep"
    mkdir -p "$SW"
    ln -sfn "../../../../../kernels/gemm/cacheflex/bin" "$SW/bin"   # BIN="$(pwd)/bin/v1_fused"
    ln -sf "../../../run_fig2_cache_n2048_headh0.sh" "$SW/run_fig2_cache_n2048_headh0.sh"
    ln -sf "../../../run_spm_fused_n2048_headh0.sh"  "$SW/run_spm_fused_n2048_headh0.sh"
    # SEQUENTIAL inner invocation: each sweep script self-throttles at
    # MAX_JOBS (forwarded from -j), so running them one after another keeps
    # the TOTAL number of concurrent gem5 processes bounded by -j.
    rh_launch "$SW/fig2_cache_n2048_headh0" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" GEM5="$GEM5" \
        MAX_JOBS="$RH_JOBS" bash "$SW/run_fig2_cache_n2048_headh0.sh"
    if ! rh_wait_all; then exit 1; fi
    RH_PIDS=()
    rh_launch "$SW/spm_fused_n2048_headh0" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" GEM5="$GEM5" \
        MAX_JOBS="$RH_JOBS" bash "$SW/run_spm_fused_n2048_headh0.sh"
    rh_wait_all
    exit $?
fi

rh_require_best "$BEST"
if ! rh_materialize_cells 21 5; then exit 3; fi
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
