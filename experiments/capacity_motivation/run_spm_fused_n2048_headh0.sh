#!/bin/bash
# Figure 2 CacheFlex points for M=256, K=N=2048.
set -e
cd "$(dirname "$0")"
if [ "$PWD" != "$CACHEFLEX_ROOT/experiments/capacity_motivation/results/generated/sweep" ]; then
    echo "ERROR: internal sweep helper; use experiments/capacity_motivation/run.sh --sweep [-j N]" >&2
    exit 2
fi

BENCH_ROOT="$CACHEFLEX_ROOT"
GEM5="$CACHEFLEX_ROOT/gem5/build/ARM/gem5.opt"
GEM5_SE="$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py"
BIN_DIR="$CACHEFLEX_ROOT/kernels/gemm/cacheflex/bin"
OUTDIR="$(pwd)/results_spm_fused_n2048_headh0"
mkdir -p "$OUTDIR"

M=256; K=2048; N=2048; NITER=1; MC=256

MAX_JOBS="${MAX_JOBS:-4}"    # run.sh forwards -j through MAX_JOBS
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 1; done; }

move_aside() {
    local path="$1" epoch previous suffix=0
    if [ ! -e "$path" ] && [ ! -L "$path" ]; then
        return
    fi
    epoch=$(date +%s)
    previous="${path}.prev.${epoch}"
    while [ -e "$previous" ] || [ -L "$previous" ]; do
        suffix=$((suffix + 1))
        previous="${path}.prev.${epoch}.${suffix}"
    done
    mv -- "$path" "$previous"
    echo "[PREV] $path -> $previous"
}

wait_for_jobs() {
    local i failures=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "[FAIL] ${LABELS[$i]} (pid ${PIDS[$i]})" >&2
            failures=$((failures + 1))
        fi
    done
    if [ "$failures" -ne 0 ]; then
        echo "ERROR: $failures of ${#PIDS[@]} gem5 runs failed" >&2
        return 1
    fi
}

# Selected KC values: VL4=1024, VL8=512, VL16=320.
declare -A BEST_KC
BEST_KC[4]=1024
BEST_KC[8]=512
BEST_KC[16]=320

TOTAL=0
PIDS=()
LABELS=()

for vl in 4 8 16; do
    kc=${BEST_KC[$vl]}
    throttle

    case $vl in
        4|8)   SPM_LOAD=2; SPM_STORE=1;;
        16)    SPM_LOAD=1; SPM_STORE=1;;
    esac

    label="v3fused_vl${vl}_kc${kc}_mc${MC}"
    logf="$OUTDIR/${label}.log"
    out="$OUTDIR/$label"
    move_aside "$out"
    move_aside "$logf"
    mkdir -p "$out"

    echo "Launching $label ..."

    "$GEM5" --outdir="$out" "$GEM5_SE" \
        --cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz \
        --caches --l2cache --l3cache \
        --l1d_size=64kB --l1i_size=64kB --l1d_assoc=4 --l1i_assoc=4 \
        --l2_size=512kB --l2_assoc=8 \
        --l3_size=4MB --l3_assoc=16 \
        --cacheline_size=64 --mem-size=16GB --mem-channels=2 \
        --mem-type=DDR4_2400_8x8 \
        --l1i-hwp-type=StridePrefetcher \
        --l1d-hwp-type=StridePrefetcher \
        --l2-hwp-type=AMPMPrefetcher \
        --l3-hwp-type=BOPPrefetcher \
        -P 'system.cpu[0].icache.prefetcher.degree=4' \
        -P 'system.cpu[0].dcache.prefetcher.degree=4' \
        -P 'system.cpu[0].icache.tag_latency=2' \
        -P 'system.cpu[0].icache.data_latency=2' \
        -P 'system.cpu[0].fetchWidth=5' \
        -P 'system.cpu[0].decodeWidth=5' \
        -P 'system.cpu[0].commitWidth=5' \
        -P 'system.cpu[0].renameWidth=5' \
        -P 'system.cpu[0].dispatchWidth=8' \
        -P 'system.cpu[0].issueWidth=8' \
        -P 'system.cpu[0].wbWidth=8' \
        -P 'system.cpu[0].numROBEntries=128' \
        -P 'system.cpu[0].numIQEntries=80' \
        -P 'system.cpu[0].LQEntries=32' \
        -P 'system.cpu[0].SQEntries=48' \
        -P 'system.cpu[0].numPhysIntRegs=128' \
        -P 'system.cpu[0].numPhysFloatRegs=192' \
        -P 'system.cpu[0].numPhysVecRegs=192' \
        -P 'system.cpu[0].numPhysVecPredRegs=64' \
        -P 'system.cpu[0].cacheLoadPorts=2' \
        -P 'system.cpu[0].cacheStorePorts=1' \
        -P "system.cpu[0].spmLoadPorts=${SPM_LOAD}" \
        -P "system.cpu[0].spmStorePorts=${SPM_STORE}" \
        -P 'system.cpu[0].fuPool.FUList[0].count=2' \
        -P 'system.cpu[0].fuPool.FUList[2].count=2' \
        -P 'system.cpu[0].fuPool.FUList[5].count=2' \
        -P 'system.cpu[0].fuPool.FUList[7].count=2' \
        -P 'system.cpu[0].fuPool.FUList[9].count=2' \
        -P 'system.tol2bus.frontend_latency=1' -P 'system.tol2bus.header_latency=0' \
        -P 'system.tol2bus.forward_latency=1' \
        -P 'system.tol2bus.response_latency=1' \
        -P 'system.tol2bus.width=64' \
        -P 'system.l2.tag_latency=8' \
        -P 'system.l2.data_latency=8' \
        -P 'system.l2.response_latency=8' \
        -P 'system.tol3bus.width=32' \
        -P 'system.tol3bus.frontend_latency=10' \
        -P 'system.tol3bus.forward_latency=10' \
        -P 'system.tol3bus.response_latency=10' \
        -P 'system.l3.tag_latency=35' \
        -P 'system.l3.data_latency=35' \
        -P 'system.l3.response_latency=35' \
        -P "system.cpu[0].isa[0].sve_vl_se=${vl}" \
        -c "${BIN_DIR}/v3_fused_vl${vl}" \
        -o "$M $K $N $NITER $kc $MC" \
        > "$logf" 2>&1 &
    PIDS+=("$!")
    LABELS+=("$label")
    TOTAL=$((TOTAL + 1))
done

echo "=== Launched $TOTAL fresh runs ==="
if ! wait_for_jobs; then
    exit 1
fi
echo "=== All done ==="
