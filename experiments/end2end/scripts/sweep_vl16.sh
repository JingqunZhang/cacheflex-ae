#!/bin/bash
# sweep_vl16.sh — VL=16 (2048-bit SVE) end-to-end layer sweep
#
# GEMM: 8×3VL kernel (NT=384), B prepack outside ROI, C padded to NT
# Flash Attn: VL=16 1VL QK (NT=128) + predicated PV
#
# Cache GEMM: MC ∈ {256,512,1024}  KC ∈ {128,256,341}
# SPM GEMM:   MC ∈ {256,512}   KC ∈ {256,320,341}  (KC_max=341 for 3VL VL=16)
# Flash attn:  Br ∈ {32,64}     Bc ∈ {128,256,384}  (multiples of NT=128)
#
# Total: ~320 jobs, MAX_JOBS=72
# Usage: bash scripts/sweep_vl16.sh

set -e
cd "$(dirname "$0")/.."

BENCH_ROOT="$CACHEFLEX_ROOT"
GEM5_BIN="${GEM5_BIN:-$CACHEFLEX_ROOT/gem5/build/ARM/gem5.opt}"
GEM5_SCRIPT="$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py"
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
-P system.cpu[0].spmLoadPorts=1 -P system.cpu[0].spmStorePorts=1 \
-P system.cpu[0].fuPool.FUList[0].count=2 -P system.cpu[0].fuPool.FUList[2].count=2 \
-P system.cpu[0].fuPool.FUList[5].count=2 -P system.cpu[0].fuPool.FUList[7].count=2 \
-P system.cpu[0].fuPool.FUList[9].count=2 \
-P system.tol2bus.frontend_latency=1 -P system.tol2bus.forward_latency=1 \
-P system.tol2bus.header_latency=0 \
-P system.tol2bus.response_latency=1 -P system.tol2bus.width=64 \
-P system.l2.tag_latency=8 -P system.l2.data_latency=8 -P system.l2.response_latency=8 \
-P system.tol3bus.width=32 -P system.tol3bus.frontend_latency=10 \
-P system.tol3bus.forward_latency=10 -P system.tol3bus.response_latency=10 \
-P system.l3.tag_latency=35 -P system.l3.data_latency=35 -P system.l3.response_latency=35 \
-P system.cpu[0].isa[0].sve_vl_se=16"

OUTDIR="results/sweep_vl16"
MAX_JOBS="${MAX_JOBS:-72}"
BINDIR="bin/vl16"
job_count=0
PIDS=()

mkdir -p "$OUTDIR"

run_one() {
    local name="$1" binary="$2"; shift 2
    if [ ! -f "$binary" ]; then echo "ERROR: missing binary: $binary" >&2; return 1; fi
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
}

# ── Cache GEMM: MC ∈ {256,512,1024}  KC ∈ {128,256,341} ──
# Drop MC=2048 (marginal gain, doubles sim time)
# Drop KC=192 (close to 128/256, not insightful)
cache_gemm() {
    local tag="$1" mid="$2" T="$3" K="$4" N="$5"
    for mc in 256 512 1024; do
        [ $mc -gt $T ] && continue
        for kc in 128 256 341; do
            [ $kc -gt $K ] && continue
            run_one "${tag}_T${T}_cache_K${K}_N${N}_mc${mc}_kc${kc}" \
                "$BINDIR/bench_kernel" gemm $mid $T $K $N $mc $kc 1
        done
    done
}

# Figure 9 consumes the sequential FFN implementations, so keep their search
# grids explicit rather than inferring them from individual GEMM cells.
cache_ffn() {
    local tag="$1" mid="$2" T="$3"
    for mc in 256 512 1024; do
        [ $mc -gt $T ] && continue
        for kc in 128 256 341; do
            run_one "${tag}_T${T}_ffn_seq_mc${mc}_kc${kc}" \
                "$BINDIR/bench_ffn_seq" $mid $T $mc $kc 1
        done
    done
}

spm_ffn() {
    local tag="$1" mid="$2" T="$3"
    for mc in 256 512; do
        [ $mc -gt $T ] && continue
        for kc in 256 320 341; do
            run_one "${tag}_T${T}_ffn_seq_spm_mc${mc}_kc${kc}" \
                "$BINDIR/bench_ffn_seq_spm" $mid $T $mc $kc 1
        done
    done
}

# ── SPM GEMM: MC ∈ {256,512}  KC ∈ {256,320,341} ──
# KC_max = 341 for VL=16 3VL layout (floor(1024/3))
# MC=256 often optimal for SPM (smaller SPMCP overhead)
spm_gemm() {
    local tag="$1" mid="$2" T="$3" K="$4" N="$5"
    for mc in 256 512; do
        [ $mc -gt $T ] && continue
        for kc in 256 320 341; do
            [ $kc -gt $K ] && continue
            run_one "${tag}_T${T}_spm_K${K}_N${N}_mc${mc}_kc${kc}" \
                "$BINDIR/bench_kernel_spm" gemm $mid $T $K $N $mc $kc 1
        done
    done
}

echo "=== VL=16 Sweep (cache + SPM, NT_gemm=384, NT_attn=128) ==="

for T in 256 1024 2048 4096; do
    echo "========== T=$T =========="

    # ── LLaMA (model=0) ──
    echo "--- LLaMA GEMM ---"
    cache_gemm llama 0 $T 2048 2048      # Q/Out proj
    cache_gemm llama 0 $T 2048 512       # K/V proj
    cache_gemm llama 0 $T 2048 8192      # FFN gate/up
    cache_gemm llama 0 $T 8192 2048      # FFN down

    spm_gemm llama 0 $T 2048 2048
    spm_gemm llama 0 $T 2048 512
    spm_gemm llama 0 $T 2048 8192
    spm_gemm llama 0 $T 8192 2048
    cache_ffn llama 0 $T
    spm_ffn llama 0 $T

    echo "--- LLaMA Attention ---"
    # VL=16 flash attn uses NT=128. Sweep Bc as multiples of 128.
    for br in 32 64; do
        for bc in 128 256 384; do
            [ $bc -gt $T ] && continue
            run_one "llama_T${T}_cflash_br${br}_bc${bc}" \
                "$BINDIR/cache_flash_attn" $T 32 8 64 1 $br $bc noncausal
            run_one "llama_T${T}_sflash_br${br}_bc${bc}" \
                "$BINDIR/spm_flash_attn" $T 32 8 64 1 $br $bc 0 noncausal
        done
    done
    # Unfused attention (no Bc sweep needed)
    run_one "llama_T${T}_unfused" "$BINDIR/cache_unfused_attn" $T 32 8 64 1 128 256 128 0 noncausal
    run_one "llama_T${T}_spm_unfused" "$BINDIR/spm_unfused_attn" $T 32 8 64 1 128 256 128 0 noncausal

    echo "--- LLaMA Misc ---"
    run_one "llama_T${T}_rmsnorm"  "$BINDIR/bench_kernel" rmsnorm 0 $T 1
    run_one "llama_T${T}_residual" "$BINDIR/bench_kernel" residual 0 $T 1
    run_one "llama_T${T}_rope"     "$BINDIR/bench_kernel" rope 0 $T 1

    # ── BERT (model=1) ──
    echo "--- BERT GEMM ---"
    cache_gemm bert 1 $T 768 768         # QKVO proj
    cache_gemm bert 1 $T 768 3072        # FFN W1
    cache_gemm bert 1 $T 3072 768        # FFN W2

    spm_gemm bert 1 $T 768 768
    spm_gemm bert 1 $T 768 3072
    spm_gemm bert 1 $T 3072 768
    cache_ffn bert 1 $T
    spm_ffn bert 1 $T

    echo "--- BERT Attention ---"
    for br in 32 64; do
        for bc in 128 256 384; do
            [ $bc -gt $T ] && continue
            run_one "bert_T${T}_cflash_br${br}_bc${bc}" \
                "$BINDIR/cache_flash_attn" $T 12 12 64 1 $br $bc noncausal
            run_one "bert_T${T}_sflash_br${br}_bc${bc}" \
                "$BINDIR/spm_flash_attn" $T 12 12 64 1 $br $bc 0 noncausal
        done
    done
    run_one "bert_T${T}_unfused" "$BINDIR/cache_unfused_attn" $T 12 12 64 1 128 256 128 0 noncausal
    run_one "bert_T${T}_spm_unfused" "$BINDIR/spm_unfused_attn" $T 12 12 64 1 128 256 128 0 noncausal

    echo "--- BERT Misc ---"
    run_one "bert_T${T}_layernorm"  "$BINDIR/bench_layernorm" 1 $T 1
    run_one "bert_T${T}_residual" "$BINDIR/bench_kernel" residual 1 $T 1

    echo ""
done

echo "Launched $job_count gem5 jobs (max $MAX_JOBS parallel). Waiting..."
failures=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
if [ "$failures" -ne 0 ]; then
    echo "ERROR: $failures VL16 gem5 jobs failed" >&2
    exit 1
fi
echo "=== VL=16 sweep complete. Total runs: $job_count ==="
