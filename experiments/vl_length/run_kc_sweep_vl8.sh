#!/bin/bash
# KC sweep at VL=8.
# Internal helper; invoke experiments/vl_length/run.sh --sweep [-j N].

set -e
cd "$(dirname "$0")"
if [ "$PWD" != "$CACHEFLEX_ROOT/experiments/vl_length/results/generated/sweep" ]; then
    echo "ERROR: internal sweep helper; use experiments/vl_length/run.sh --sweep [-j N]" >&2
    exit 2
fi

BENCH_ROOT="$CACHEFLEX_ROOT"
GEM5="$CACHEFLEX_ROOT/gem5/build/ARM/gem5.opt"
GEM5_SE="$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py"
BIN_V1="$CACHEFLEX_ROOT/kernels/gemm/cacheflex/bin/v1_fused"
BIN_V3_DIR="$CACHEFLEX_ROOT/kernels/gemm/cacheflex/bin"
OUTDIR="$(pwd)/results_kc_sweep_vl8"
mkdir -p "$OUTDIR"

MAX_JOBS=${1:-48}
NITER=1

get_spm_ports() {
    local vl=$1
    if [ "$vl" -le 8 ]; then SPM_LP=2; SPM_SP=1
    else SPM_LP=1; SPM_SP=1; fi
}

GEM5_HW_BASE=(
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
    -P 'system.tol2bus.forward_latency=1'
    -P 'system.tol2bus.header_latency=0'
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

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done; }

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

TOTAL=0
PIDS=()
LABELS=()

run() {
    local label="$1" bin="$2" vl="$3" M="$4" K="$5" N="$6" KC="$7" MC="$8"
    local out="$OUTDIR/$label"
    local logf="$OUTDIR/${label}.log"
    throttle
    move_aside "$out"
    move_aside "$logf"
    mkdir -p "$out"
    get_spm_ports "$vl"
    echo "[LAUNCH] $label"
    "$GEM5" --outdir="$out" "$GEM5_SE" \
        "${GEM5_HW_BASE[@]}" \
        -P "system.cpu[0].isa[0].sve_vl_se=${vl}" \
        -P "system.cpu[0].spmLoadPorts=${SPM_LP}" \
        -P "system.cpu[0].spmStorePorts=${SPM_SP}" \
        -c "$bin" -o "$M $K $N $NITER $KC $MC" \
        > "$logf" 2>&1 &
    PIDS+=("$!")
    LABELS+=("$label")
    TOTAL=$((TOTAL + 1))
}

echo "============================================================"
echo "  VL=8 KC sweep, MAX_JOBS=$MAX_JOBS"
echo "============================================================"

for kc in 64 128 256 512 1024;     do run kc${kc}_W1_v1 "$BIN_V1" 8 128 4096 11008 $kc 128; done
for kc in 64 128 256 512 1024;     do run kc${kc}_W2_v1 "$BIN_V1" 8 256 4096 4096 $kc 256; done
for kc in 64 128 256 384 512 1024; do run kc${kc}_W3_v1 "$BIN_V1" 8 256 2048 2048 $kc 256; done
for kc in 64 128 256 384 512;      do run kc${kc}_W4_v1 "$BIN_V1" 8 512 768 768 $kc 512; done
for kc in 64 128 256 384 512 1024; do run kc${kc}_W5_v1 "$BIN_V1" 8 2048 2048 2048 $kc 2048; done
for kc in 32 64;                   do run kc${kc}_W6_v1 "$BIN_V1" 8 2048 64 2048 $kc 1024; done
for kc in 64 128 256;              do run kc${kc}_W7_v1 "$BIN_V1" 8 784 256 1024 $kc 784; done
for kc in 64 128 256 384 512;      do run kc${kc}_W1_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 128 4096 11008 $kc 128; done
for kc in 64 128 256 384 512;      do run kc${kc}_W2_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 256 4096 4096 $kc 256; done
for kc in 64 128 256 384 512;      do run kc${kc}_W3_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 256 2048 2048 $kc 256; done
for kc in 64 128 256 384 512;      do run kc${kc}_W4_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 512 768 768 $kc 256; done
for kc in 64 128 256 384 512;      do run kc${kc}_W5_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 2048 2048 2048 $kc 1024; done
for kc in 32 64;                   do run kc${kc}_W6_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 2048 64 2048 $kc 1024; done
for kc in 64 128 256 384 512;      do run kc${kc}_W7_v3 "$BIN_V3_DIR/v3_fused_vl8" 8 784 256 1024 $kc 784; done
if ! wait_for_jobs; then
    exit 1
fi
echo "VL8 KC SWEEP DONE"
