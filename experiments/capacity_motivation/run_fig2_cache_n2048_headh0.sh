#!/bin/bash
# Figure 2 cache sweep for M=256, K=N=2048, FP16.
# Kernel: v1_fused, MC=256.
# L1 configurations:
#   64KB/4w(256sets), 256KB/4w(1024sets), 512KB/4w(2048sets)
# Port configurations:
#   BW=2: LD=2,ST=1, FU[7]=2,FU[9]=2
#   BW=4: LD=4,ST=2, FU[7]=4,FU[9]=2
# KC values:
#   VL=4:  1024, 2048
#   VL=8:  768, 1024
#   VL=16: 512, 768
# Total: 3 capacities × 2 port configurations × 6 VL-KC pairs = 36 runs.

set -e
cd "$(dirname "$0")"
if [ "$PWD" != "$CACHEFLEX_ROOT/experiments/capacity_motivation/results/generated/sweep" ]; then
    echo "ERROR: internal sweep helper; use experiments/capacity_motivation/run.sh --sweep [-j N]" >&2
    exit 2
fi

BENCH_ROOT="$CACHEFLEX_ROOT"
GEM5="$GEM5"
GEM5_SE="$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py"
BIN="$(pwd)/bin/v1_fused"
OUTDIR="$(pwd)/results_fig2_n2048_headh0"
mkdir -p "$OUTDIR"

MAX_JOBS="${MAX_JOBS:-48}"   # run.sh forwards -j through MAX_JOBS
M=256; K=2048; N=2048; NITER=1; MC=256

# L1 configs: "size_kB assoc"
declare -a L1_CONFIGS=(
    "64 4"
    "256 4"
    "512 4"
)

# Bandwidth configs: "ld_ports st_ports fu7 fu9"
declare -a BW_CONFIGS=(
    "2 1 2 2"
    "4 2 4 2"
)

get_kc_values() {
    case $1 in
        2)  echo "1024";;
        4)  echo "1024 2048";;
        8)  echo "768 1024";;
        16) echo "512 768";;
    esac
}

throttle() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 1; done
}

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

for vl in 4 8 16; do
    KC_VALUES=($(get_kc_values $vl))
    for l1_cfg in "${L1_CONFIGS[@]}"; do
        read -r l1_kb l1_assoc <<< "$l1_cfg"
        for bw_cfg in "${BW_CONFIGS[@]}"; do
            read -r ld_ports st_ports fu7 fu9 <<< "$bw_cfg"
            for kc in "${KC_VALUES[@]}"; do
                [ "$kc" -gt "$K" ] && continue

                label="fused_vl${vl}_l1${l1_kb}k${l1_assoc}w_ld${ld_ports}st${st_ports}_kc${kc}_mc${MC}"
                logf="$OUTDIR/${label}.log"
                out="$OUTDIR/$label"
                throttle
                move_aside "$out"
                move_aside "$logf"
                mkdir -p "$out"
                TOTAL=$((TOTAL + 1))

                "$GEM5" --outdir="$out" "$GEM5_SE" \
                    --cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz \
                    --caches --l2cache --l3cache \
                    --l1d_size=${l1_kb}kB --l1i_size=64kB \
                    --l1d_assoc=${l1_assoc} --l1i_assoc=4 \
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
                    -P "system.cpu[0].cacheLoadPorts=${ld_ports}" \
                    -P "system.cpu[0].cacheStorePorts=${st_ports}" \
                    -P 'system.cpu[0].fuPool.FUList[0].count=2' \
                    -P 'system.cpu[0].fuPool.FUList[2].count=2' \
                    -P 'system.cpu[0].fuPool.FUList[5].count=2' \
                    -P "system.cpu[0].fuPool.FUList[7].count=${fu7}" \
                    -P "system.cpu[0].fuPool.FUList[9].count=${fu9}" \
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
                    -c "$BIN" \
                    -o "$M $K $N $NITER $kc $MC" \
                    > "$logf" 2>&1 &
                PIDS+=("$!")
                LABELS+=("$label")
            done
        done
    done
done

echo "=== Launched $TOTAL fresh runs ==="
echo "    MAX_JOBS=$MAX_JOBS  outdir: $OUTDIR"
echo "Waiting..."
if ! wait_for_jobs; then
    exit 1
fi
echo "=== All done ==="
