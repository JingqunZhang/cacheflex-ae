#!/bin/bash
# sweep_t2048_t4096.sh — Targeted sweep for T=2048 and T=4096
#
# Per-kernel GEMMs and the sequential FFN implementations are both swept.
# Figure 9 consumes the sequential FFN cells pinned in best.json.
#
# Unique GEMM shapes:
#   LLaMA: 2048×2048 (Q/Out ×2), 2048×512 (K/V ×2), 2048×8192 (gate/up ×2), 8192×2048 (down ×1)
#   BERT:  768×768 (QKVO ×4), 768×3072 (W1 ×1), 3072×768 (W2 ×1)
#
# MC ∈ {1024, 2048}, KC: cache → {512, 768, 1024}, SPM → {768, 1024}
#
# Usage: bash scripts/sweep_t2048_t4096.sh

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
-P system.cpu[0].spmLoadPorts=2 -P system.cpu[0].spmStorePorts=1 \
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
-P system.cpu[0].isa[0].sve_vl_se=4"

OUTDIR="results/sweep_large"
MAX_JOBS="${MAX_JOBS:-72}"
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

echo "=== Targeted Sweep: T=2048,4096  MC∈{1024,2048}  KC∈{512,768,1024} ==="

for T in 2048 4096; do
    echo "========== T=$T =========="

    # ── LLaMA (model=0) ──
    # Unique shapes: 2048×2048 (Q/Out), 2048×512 (K/V), 2048×8192 (gate/up), 8192×2048 (down)

    echo "--- LLaMA cache GEMM ---"
    for mc in 1024 2048; do
        [ $mc -gt $T ] && continue
        for kc in 512 768 1024; do
            # Q/Out proj (K=2048, N=2048) — ×2 in layer
            run_one "llama_T${T}_cache_K2048_N2048_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 0 $T 2048 2048 $mc $kc 1
            # K/V proj (K=2048, N=512) — ×2 in layer
            run_one "llama_T${T}_cache_K2048_N512_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 0 $T 2048 512 $mc $kc 1
            # FFN gate/up (K=2048, N=8192) — ×2 in layer
            run_one "llama_T${T}_cache_K2048_N8192_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 0 $T 2048 8192 $mc $kc 1
            # FFN down (K=8192, N=2048) — ×1 in layer
            run_one "llama_T${T}_cache_K8192_N2048_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 0 $T 8192 2048 $mc $kc 1
        done
    done

    echo "--- LLaMA SPM GEMM ---"
    for mc in 1024 2048; do
        [ $mc -gt $T ] && continue
        for kc in 768 1024; do
            run_one "llama_T${T}_spm_K2048_N2048_mc${mc}_kc${kc}" \
                bin/bench_kernel_spm gemm 0 $T 2048 2048 $mc $kc 1
            run_one "llama_T${T}_spm_K2048_N512_mc${mc}_kc${kc}" \
                bin/bench_kernel_spm gemm 0 $T 2048 512 $mc $kc 1
            run_one "llama_T${T}_spm_K2048_N8192_mc${mc}_kc${kc}" \
                bin/bench_kernel_spm gemm 0 $T 2048 8192 $mc $kc 1
            run_one "llama_T${T}_spm_K8192_N2048_mc${mc}_kc${kc}" \
                bin/bench_kernel_spm gemm 0 $T 8192 2048 $mc $kc 1
        done
    done

    echo "--- LLaMA FFN sequence ---"
    for mc in 1024 2048; do
        [ $mc -gt $T ] && continue
        for kc in 512 768 1024; do
            run_one "llama_T${T}_ffn_seq_mc${mc}_kc${kc}" \
                bin/bench_ffn_seq 0 $T $mc $kc 1
        done
        for kc in 768 1024; do
            run_one "llama_T${T}_ffn_seq_spm_mc${mc}_kc${kc}" \
                bin/bench_ffn_seq_spm 0 $T $mc $kc 1
        done
    done

    echo "--- LLaMA Attention ---"
    for br in 32 64; do
        for bc in 384 576; do
            run_one "llama_T${T}_cflash_br${br}_bc${bc}" \
                bin/cache_flash_attn $T 32 8 64 1 $br $bc noncausal
            run_one "llama_T${T}_sflash_br${br}_bc${bc}" \
                bin/spm_flash_attn $T 32 8 64 1 $br $bc 0 noncausal
        done
    done
    run_one "llama_T${T}_unfused" bin/cache_unfused_attn $T 32 8 64 1 128 256 128 0 noncausal
    run_one "llama_T${T}_spm_unfused" bin/spm_unfused_attn $T 32 8 64 1 128 256 128 0 noncausal

    echo "--- LLaMA Misc ---"
    run_one "llama_T${T}_rmsnorm" bin/bench_kernel rmsnorm 0 $T 1
    run_one "llama_T${T}_residual" bin/bench_kernel residual 0 $T 1
    run_one "llama_T${T}_rope" bin/bench_kernel rope 0 $T 1

    # ── BERT (model=1) ──
    # Unique shapes: 768×768 (QKVO ×4), 768×3072 (W1 ×1), 3072×768 (W2 ×1)

    echo "--- BERT cache GEMM ---"
    for mc in 1024 2048; do
        [ $mc -gt $T ] && continue
        # QKVO proj (K=768, N=768) — ×4 in layer
        for kc in 512 768; do
            run_one "bert_T${T}_cache_K768_N768_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 1 $T 768 768 $mc $kc 1
        done
        # FFN W1 (K=768, N=3072) — ×1 in layer
        for kc in 512 768; do
            run_one "bert_T${T}_cache_K768_N3072_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 1 $T 768 3072 $mc $kc 1
        done
        # FFN W2 (K=3072, N=768) — ×1 in layer
        for kc in 512 768 1024; do
            run_one "bert_T${T}_cache_K3072_N768_mc${mc}_kc${kc}" \
                bin/bench_kernel gemm 1 $T 3072 768 $mc $kc 1
        done
    done

    echo "--- BERT SPM GEMM ---"
    for mc in 1024 2048; do
        [ $mc -gt $T ] && continue
        run_one "bert_T${T}_spm_K768_N768_mc${mc}_kc768" \
            bin/bench_kernel_spm gemm 1 $T 768 768 $mc 768 1
        run_one "bert_T${T}_spm_K768_N3072_mc${mc}_kc768" \
            bin/bench_kernel_spm gemm 1 $T 768 3072 $mc 768 1
        for kc in 768 1024; do
            run_one "bert_T${T}_spm_K3072_N768_mc${mc}_kc${kc}" \
                bin/bench_kernel_spm gemm 1 $T 3072 768 $mc $kc 1
        done
    done

    echo "--- BERT FFN sequence ---"
    for mc in 1024 2048; do
        [ $mc -gt $T ] && continue
        for kc in 512 768 1024; do
            run_one "bert_T${T}_ffn_seq_mc${mc}_kc${kc}" \
                bin/bench_ffn_seq 1 $T $mc $kc 1
        done
        for kc in 768 1024; do
            run_one "bert_T${T}_ffn_seq_spm_mc${mc}_kc${kc}" \
                bin/bench_ffn_seq_spm 1 $T $mc $kc 1
        done
    done

    echo "--- BERT Attention ---"
    for br in 32 64; do
        for bc in 384 576; do
            run_one "bert_T${T}_cflash_br${br}_bc${bc}" \
                bin/cache_flash_attn $T 12 12 64 1 $br $bc noncausal
            run_one "bert_T${T}_sflash_br${br}_bc${bc}" \
                bin/spm_flash_attn $T 12 12 64 1 $br $bc 0 noncausal
        done
    done
    run_one "bert_T${T}_unfused" bin/cache_unfused_attn $T 12 12 64 1 128 256 128 0 noncausal
    run_one "bert_T${T}_spm_unfused" bin/spm_unfused_attn $T 12 12 64 1 128 256 128 0 noncausal

    echo "--- BERT Misc ---"
    run_one "bert_T${T}_layernorm" bin/bench_layernorm 1 $T 1
    run_one "bert_T${T}_residual" bin/bench_kernel residual 1 $T 1

    echo ""
done

echo "Launched $job_count gem5 jobs (max $MAX_JOBS parallel). Waiting..."
failures=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
if [ "$failures" -ne 0 ]; then
    echo "ERROR: $failures large-T gem5 jobs failed" >&2
    exit 1
fi
echo "=== Sweep complete. Total runs: $job_count ==="
