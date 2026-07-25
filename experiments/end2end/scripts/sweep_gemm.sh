#!/bin/bash
# sweep_gemm.sh — VL=4 short/mid-sequence sweep for the paper experiment
#
# Cache GEMM:
#   V1 (loop=0): MC ∈ {256,384,512,768}  KC ∈ {256,384,512,768,1024}
#   V2 (loop=1): MC ∈ {64,128,256}       KC ∈ {256,512}
# SPM GEMM (V3/V4):
#   MC ∈ {64,128}  KC ∈ {512,768,1024}
# Flash attention:
#   Cache: Br ∈ {32,40,64}  Bc ∈ {288,384,480,576}
#   SPM:   Br ∈ {32,40,64}  Bc ∈ {288,384,480,576}  (672,768 removed: exceed SPM row space)
# FFN: same MC/KC as GEMM, ×(seq+fused)
#
# Usage:
# Internal helper; use experiments/end2end/run.sh --sweep [-j N].

set -e
cd "$(dirname "$0")/.."

MODE="${1:-native}"
BIN="bin/bench_kernel"
BIN_SPM="bin/bench_kernel_spm"
FFN_SEQ="bin/bench_ffn_seq"
FFN_SEQ_SPM="bin/bench_ffn_seq_spm"
FLASH_BIN="bin/cache_flash_attn"
UNFUSED_BIN="bin/cache_unfused_attn"
SPM_FLASH="bin/spm_flash_attn"
SPM_UNFUSED="bin/spm_unfused_attn"
OUTDIR="results/sweep"
NITER="${NITER:-1}"
mkdir -p "$OUTDIR"

BENCH_ROOT="$CACHEFLEX_ROOT"
GEM5_BIN="${GEM5_BIN:-$CACHEFLEX_ROOT/gem5/build/ARM/gem5.opt}"
GEM5_SCRIPT="${GEM5_SCRIPT:-$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py}"
# Canonical VL=4 configuration.
GEM5_CONFIG="--cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz \
    --caches --l2cache --l3cache \
    --l1d_size=64kB --l1i_size=64kB --l1d_assoc=4 --l1i_assoc=4 \
    --l2_size=512kB --l2_assoc=8 --l3_size=4MB --l3_assoc=16 \
    --cacheline_size=64 --mem-size=16GB --mem-channels=2 --mem-type=DDR4_2400_8x8 \
    --l1i-hwp-type=StridePrefetcher --l1d-hwp-type=StridePrefetcher \
    --l2-hwp-type=AMPMPrefetcher --l3-hwp-type=BOPPrefetcher \
    -P system.cpu[0].icache.prefetcher.degree=4 \
    -P system.cpu[0].dcache.prefetcher.degree=4 \
    -P system.cpu[0].icache.tag_latency=2 -P system.cpu[0].icache.data_latency=2 \
    -P system.cpu[0].fetchWidth=5 -P system.cpu[0].decodeWidth=5 \
    -P system.cpu[0].commitWidth=5 -P system.cpu[0].renameWidth=5 \
    -P system.cpu[0].dispatchWidth=8 -P system.cpu[0].issueWidth=8 -P system.cpu[0].wbWidth=8 \
    -P system.cpu[0].numROBEntries=128 -P system.cpu[0].numIQEntries=80 \
    -P system.cpu[0].LQEntries=32 -P system.cpu[0].SQEntries=48 \
    -P system.cpu[0].numPhysIntRegs=128 -P system.cpu[0].numPhysFloatRegs=192 \
    -P system.cpu[0].numPhysVecRegs=192 -P system.cpu[0].numPhysVecPredRegs=64 \
    -P system.cpu[0].cacheLoadPorts=2 -P system.cpu[0].cacheStorePorts=1 \
    -P system.cpu[0].spmLoadPorts=2 -P system.cpu[0].spmStorePorts=1 \
    -P system.cpu[0].fuPool.FUList[0].count=2 -P system.cpu[0].fuPool.FUList[2].count=2 \
    -P system.cpu[0].fuPool.FUList[5].count=2 -P system.cpu[0].fuPool.FUList[7].count=2 \
    -P system.cpu[0].fuPool.FUList[9].count=2 \
    -P system.tol2bus.frontend_latency=1 -P system.tol2bus.header_latency=0 \
    -P system.tol2bus.forward_latency=1 \
    -P system.tol2bus.response_latency=1 -P system.tol2bus.width=64 \
    -P system.l2.tag_latency=8 -P system.l2.data_latency=8 -P system.l2.response_latency=8 \
    -P system.tol3bus.width=32 -P system.tol3bus.frontend_latency=10 \
    -P system.tol3bus.forward_latency=10 -P system.tol3bus.response_latency=10 \
    -P system.l3.tag_latency=35 -P system.l3.data_latency=35 -P system.l3.response_latency=35 \
    -P system.cpu[0].isa[0].sve_vl_se=4"
MAX_JOBS="${MAX_JOBS:-48}"
job_count=0
PIDS=()

run_one() {
    local name="$1" binary="$2"; shift 2
    if [ ! -f "$binary" ]; then
        echo "ERROR: missing binary: $binary" >&2
        return 1
    fi
    if [ "$MODE" = "gem5" ]; then
        local outdir="${OUTDIR}/${name}"
        if [ -e "$outdir" ]; then
            local prev="${outdir}.prev.$(date +%s)"
            while [ -e "$prev" ]; do prev="$prev.1"; done
            mv "$outdir" "$prev"
        fi
        mkdir -p "$outdir"
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done
        $GEM5_BIN --outdir="$outdir" $GEM5_SCRIPT $GEM5_CONFIG \
            -c "$binary" -o "$*" > "${outdir}/stdout.log" 2>&1 &
        PIDS+=("$!")
        job_count=$((job_count + 1))
    else
        echo "  $name"
        "$binary" "$@" > "${OUTDIR}/${name}.log" 2>"${OUTDIR}/${name}.err"
    fi
}

# ============================================================
# Sweep helpers
# ============================================================

# Cache GEMM: V1 (MC large) + V2 (MC small)
sweep_cache_gemm() {
    local tag="$1" mid="$2" T="$3" K="$4" N="$5"
    # Fused V1 (m0-outer): MC up to 2048, KC up to 1024
    for mc in 256 512 768 1024 2048; do
        [ $mc -gt $T ] && continue
        for kc in 256 384 512 768 1024; do
            [ $kc -gt $K ] && continue
            run_one "${tag}_T${T}_cache_K${K}_N${N}_mc${mc}_kc${kc}" \
                "$BIN" gemm $mid $T $K $N $mc $kc $NITER
        done
    done
}

# SPM GEMM: V3+V4, MC ∈ {64,128}, KC ∈ {512,768,1024}
sweep_spm_gemm() {
    local tag="$1" mid="$2" T="$3" K="$4" N="$5"
    # SPM fused V3: MC up to 1024, KC from 512
    for mc in 64 128 256 512 1024; do
        [ $mc -gt $T ] && continue
        for kc in 512 768 1024; do
            [ $kc -gt $K ] && continue
            run_one "${tag}_T${T}_spm_K${K}_N${N}_mc${mc}_kc${kc}" \
                "$BIN_SPM" gemm $mid $T $K $N $mc $kc $NITER
        done
    done
}

# FFN cache: sequential only (no fused)
sweep_cache_ffn() {
    local tag="$1" mid="$2" T="$3" K="$4"
    for mc in 256 512 768 1024 2048; do
        [ $mc -gt $T ] && continue
        for kc in 256 384 512 768 1024; do
            [ $kc -gt $K ] && continue
            run_one "${tag}_T${T}_ffn_seq_mc${mc}_kc${kc}" \
                "$FFN_SEQ" $mid $T $mc $kc $NITER
        done
    done
}

# FFN SPM: sequential only (no fused)
sweep_spm_ffn() {
    local tag="$1" mid="$2" T="$3" K="$4"
    for mc in 64 128 256 512 1024; do
        [ $mc -gt $T ] && continue
        for kc in 512 768 1024; do
            [ $kc -gt $K ] && continue
            run_one "${tag}_T${T}_ffn_seq_spm_mc${mc}_kc${kc}" \
                "$FFN_SEQ_SPM" $mid $T $mc $kc $NITER
        done
    done
}

# Flash attention
sweep_flash() {
    local tag="$1" mid="$2" T="$3" Hq="$4" Hkv="$5" mode="$6"
    # Cache: Br ∈ {32,40,64}, Bc ∈ {288,384,480,576}
    for br in 32 40 64; do
        [ $br -gt $T ] && continue
        for bc in 288 384 480 576; do
            run_one "${tag}_T${T}_cflash_br${br}_bc${bc}" \
                "$FLASH_BIN" $T $Hq $Hkv 64 $NITER $br $bc "$mode"
        done
    done
    # SPM: Br ∈ {32,40,64}, Bc ∈ {288,384,480,576}  (672,768 removed: exceed SPM row space)
    for br in 32 40 64; do
        [ $br -gt $T ] && continue
        for bc in 288 384 480 576; do
            run_one "${tag}_T${T}_sflash_br${br}_bc${bc}" \
                "$SPM_FLASH" $T $Hq $Hkv 64 $NITER $br $bc 0 "$mode"
        done
    done
}

# ============================================================
# Main sweep
# ============================================================
echo "=== Full Sweep (MAX_JOBS=$MAX_JOBS) ==="

for T in 256 1024; do
    echo "========== T=$T =========="

    # ── LLaMA (model=0) ──
    echo "--- LLaMA cache GEMM ---"
    sweep_cache_gemm llama 0 $T 2048 2048    # Q/Out proj
    sweep_cache_gemm llama 0 $T 2048 512     # K/V proj
    sweep_cache_gemm llama 0 $T 8192 2048    # FFN down

    echo "--- LLaMA SPM GEMM ---"
    sweep_spm_gemm llama 0 $T 2048 2048
    sweep_spm_gemm llama 0 $T 2048 512
    sweep_spm_gemm llama 0 $T 8192 2048

    echo "--- LLaMA FFN ---"
    sweep_cache_ffn llama 0 $T 2048
    sweep_spm_ffn llama 0 $T 2048

    echo "--- LLaMA attention ---"
    sweep_flash llama 0 $T 32 8 noncausal

    # ── BERT (model=1) ──
    echo "--- BERT cache GEMM ---"
    sweep_cache_gemm bert 1 $T 768 768       # QKVO proj
    sweep_cache_gemm bert 1 $T 3072 768      # FFN W2

    echo "--- BERT SPM GEMM ---"
    sweep_spm_gemm bert 1 $T 768 768
    sweep_spm_gemm bert 1 $T 3072 768

    echo "--- BERT FFN ---"
    sweep_cache_ffn bert 1 $T 768
    sweep_spm_ffn bert 1 $T 768

    echo "--- BERT attention ---"
    sweep_flash bert 1 $T 12 12 noncausal

    # ── Direct runs (no sweep) ──
    echo "--- Direct runs ---"
    run_one "llama_T${T}_rmsnorm"    "$BIN" rmsnorm 0 $T $NITER
    run_one "llama_T${T}_residual"   "$BIN" residual 0 $T $NITER
    run_one "llama_T${T}_rope"       "$BIN" rope 0 $T $NITER
    run_one "llama_T${T}_unfused"    "$UNFUSED_BIN" $T 32 8 64 $NITER 128 256 128 0 noncausal
    run_one "llama_T${T}_spm_unfused" "$SPM_UNFUSED" $T 32 8 64 $NITER 128 256 128 0 noncausal
    run_one "bert_T${T}_layernorm"   "bin/bench_layernorm" 1 $T $NITER
    run_one "bert_T${T}_residual"    "$BIN" residual 1 $T $NITER
    run_one "bert_T${T}_unfused"     "$UNFUSED_BIN" $T 12 12 64 $NITER 128 256 128 0 noncausal
    run_one "bert_T${T}_spm_unfused" "$SPM_UNFUSED" $T 12 12 64 $NITER 128 256 128 0 noncausal

    echo ""
done

if [ "$MODE" = "gem5" ]; then
    echo "Launched $job_count gem5 jobs (max $MAX_JOBS parallel). Waiting..."
    failures=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done
    if [ "$failures" -ne 0 ]; then
        echo "ERROR: $failures VL=4 sweep jobs failed" >&2
        exit 1
    fi
fi
echo "=== Sweep complete. Total runs: $job_count ==="
