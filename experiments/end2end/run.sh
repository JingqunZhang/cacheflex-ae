#!/bin/bash
# run.sh — run a validated end-to-end result set.
#
#   default   : fresh gem5 runs of the cells pinned in best.json, grouped by
#               vector length under results/generated/simulation_results/.
#               Set RESULT_SUBDIR=table4, BEST_JSON=table4_best.json, and
#               EXPECTED_CELLS=8 for the Table 4 companion configurations.
#   --sweep   : full paper sweep via the VL=4 short/mid-sequence sweep,
#               the VL=4 long-sequence sweep, and the VL=16 sweep; output is
#               redirected under results/generated/sweep/
#   -j N      : bounds the TOTAL number of concurrent gem5 processes; in sweep
#               mode the three sweep scripts run SEQUENTIALLY, each receiving N
#               via MAX_JOBS, so the bound holds globally
#
# Plot a complete best-mode run:
#   cd experiments/end2end
#   RESULTS_DIR=results/generated python3 scripts/plot_fig9.py
#
# Binaries: experiments/end2end/bin (VL=4, build.sh) and bin/vl16 (build_vl16.sh).
# Reference results are read-only and are never overwritten.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CACHEFLEX_ROOT:-$(cd "$HERE/../.." && pwd)}"
source "$HERE/../../scripts/run_helpers.sh"

RH_HELP_TEXT="Usage: run.sh [--sweep] [-j N] [-h]
  default   : fresh gem5 runs of the best.json cells, grouped by VL under
              results/generated/simulation_results/{vl4,vl16}/
              (existing cell outdirs are moved to <outdir>.prev.<epoch> first)
  --sweep   : full paper sweep at VL=4 and VL=16, run SEQUENTIALLY
              -> results/generated/sweep/
  -j N      : bounds the TOTAL number of concurrent gem5 processes, including
              sweep internals (sweep scripts receive N via MAX_JOBS; capped
              at 16 and at nproc)
Requires: source setup_env.sh, built gem5.opt, and built kernels
(bash build.sh; bash build_vl16.sh).
Plot a complete run with RESULTS_DIR=results/generated
python3 scripts/plot_fig9.py.
See the plot-from-best-run recipe in the comment atop this run.sh."
rh_parse_args "$@"

BEST="${BEST_JSON:-$HERE/best.json}"
GEN="${GENERATED_DIR:-$HERE/results/generated}"
RESULT_SUBDIR="${RESULT_SUBDIR:-simulation_results}"
EXPECTED_CELLS="${EXPECTED_CELLS:-108}"
GEM5="${GEM5:-$ROOT/gem5/build/ARM/gem5.opt}"
GEM5_SE="${GEM5_SE:-$ROOT/gem5/configs/deprecated/example/se.py}"

case "$RESULT_SUBDIR:$EXPECTED_CELLS" in
    simulation_results:108|table4:8) ;;
    *)
        echo "ERROR: unsupported result set '$RESULT_SUBDIR' ($EXPECTED_CELLS cells)" >&2
        exit 2
        ;;
esac

GEM5_COMMON=(
    --cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz
    --caches --l2cache --l3cache
    --l1d_size=64kB --l1i_size=64kB --l1d_assoc=4 --l1i_assoc=4
    --l2_size=512kB --l2_assoc=8 --l3_size=4MB --l3_assoc=16
    --cacheline_size=64 --mem-size=16GB --mem-channels=2 --mem-type=DDR4_2400_8x8
    --l1i-hwp-type=StridePrefetcher --l1d-hwp-type=StridePrefetcher
    --l2-hwp-type=AMPMPrefetcher --l3-hwp-type=BOPPrefetcher
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
# VL=4 with two SPM load ports and one SPM store port
BLOCK_VL4=(
    -P 'system.cpu[0].spmLoadPorts=2'
    -P 'system.cpu[0].spmStorePorts=1'
    -P 'system.tol2bus.header_latency=0'
    -P 'system.cpu[0].isa[0].sve_vl_se=4'
)
# vl16 block: VL=16 with one SPM load port and one SPM store port
BLOCK_VL16=(
    -P 'system.cpu[0].spmLoadPorts=1'
    -P 'system.cpu[0].spmStorePorts=1'
    -P 'system.tol2bus.header_latency=0'
    -P 'system.cpu[0].isa[0].sve_vl_se=16'
)

cells_tsv() {
    python3 "$ROOT/scripts/cell_manifest.py" \
        --experiment end2end --best "$BEST"
}

if [ "$RH_SWEEP" = 1 ]; then
    # Full sweep: shim the scripts (they `cd $(dirname $0)/..` and use the
    # relative OUTDIRs results/sweep, results/sweep_vl16, and
    # results/sweep_large plus the relative bin/ dirs) so everything lands
    # under results/generated/sweep/.
    SW="$GEN/sweep"
    mkdir -p "$SW/scripts" "$SW/results"
    ln -sfn "../../../bin" "$SW/bin"
    ln -sf "../../../../scripts/sweep_gemm.sh"       "$SW/scripts/sweep_gemm.sh"
    ln -sf "../../../../scripts/sweep_vl16.sh"       "$SW/scripts/sweep_vl16.sh"
    ln -sf "../../../../scripts/sweep_t2048_t4096.sh" "$SW/scripts/sweep_t2048_t4096.sh"
    # SEQUENTIAL inner invocation: each sweep script self-throttles at
    # MAX_JOBS (forwarded from -j), so running them one after another keeps
    # the TOTAL number of concurrent gem5 processes bounded by -j.
    rh_launch "$SW/log_sweep_vl4_short" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" \
        MAX_JOBS="$RH_JOBS" bash "$SW/scripts/sweep_gemm.sh" gem5
    if ! rh_wait_all; then exit 1; fi
    RH_PIDS=()
    rh_launch "$SW/log_sweep_vl4_long" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" \
        MAX_JOBS="$RH_JOBS" bash "$SW/scripts/sweep_t2048_t4096.sh"
    if ! rh_wait_all; then exit 1; fi
    RH_PIDS=()
    rh_launch "$SW/log_sweep_vl16" "${RH_CLEAN_ENV[@]}" CACHEFLEX_ROOT="$ROOT" \
        MAX_JOBS="$RH_JOBS" bash "$SW/scripts/sweep_vl16.sh"
    rh_wait_all
    exit $?
fi

rh_require_best "$BEST"
if ! rh_materialize_cells "$EXPECTED_CELLS" 6 cells_tsv 1,2; then exit 3; fi
mkdir -p "$GEN/$RESULT_SUBDIR/vl4" "$GEN/$RESULT_SUBDIR/vl16"
set -f   # gem5_extra/env/args are word-split below; disable globbing
while IFS=$'\t' read -r name block envs bin args extra; do
    [ "$envs" = "-" ] && envs=""
    [ "$extra" = "-" ] && extra=""
    if [ "$block" = "vl16" ]; then
        out="$GEN/$RESULT_SUBDIR/vl16/$name"
        rh_launch "$out" "${RH_CLEAN_ENV[@]}" $envs "$GEM5" --outdir="$out" "$GEM5_SE" \
            "${GEM5_COMMON[@]}" "${BLOCK_VL16[@]}" $extra -c "$ROOT/$bin" -o "$args"
    else
        out="$GEN/$RESULT_SUBDIR/vl4/$name"
        rh_launch "$out" "${RH_CLEAN_ENV[@]}" $envs "$GEM5" --outdir="$out" "$GEM5_SE" \
            "${GEM5_COMMON[@]}" "${BLOCK_VL4[@]}" $extra -c "$ROOT/$bin" -o "$args"
    fi
done < "$RH_CELLS_FILE"
set +f
rh_wait_all; RC=$?
if ! rh_validate_gem5_cells; then RC=1; fi
rh_compat_layout "$GEN/$RESULT_SUBDIR/vl4"
rh_compat_layout "$GEN/$RESULT_SUBDIR/vl16"
exit $RC
