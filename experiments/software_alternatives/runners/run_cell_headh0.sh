#!/bin/bash
# Figure 8 single-core cell runner.
# Usage: run_cell_headh0.sh <outdir> <binary> <spmflag 0|1> <envstr> <M> <N> <K> <KC> <MC>
#   outdir: absolute, or relative to repo root
#   envstr: e.g. "CF_SPM_SHARED_BW=1" or "" for none
set -u
cd "$(dirname "$0")/../../.."; ROOT="$PWD"
GEM5="${GEM5:-$ROOT/gem5/build/ARM/gem5.opt}"
SE="${GEM5_SE:-$ROOT/gem5/configs/deprecated/example/se.py}"
out="$1"; bin="$2"; spmf="$3"; envs="$4"; M="$5"; N="$6"; K="$7"; kc="$8"; mc="$9"
mkdir -p "$out"
spm="-P system.cpu[0].spmLoadPorts=1 -P system.cpu[0].spmStorePorts=1"
unset CF_L1_PIN_KEEP CF_L2_PIN_KEEP CF_NT_BYPASS CF_NT_CLEANRESP
unset CF_RANGE_PROBE CF_SELF_CHECK CF_SPM_HDR_CHARGE CF_SPM_LINE_GRAIN
unset CF_SPM_SHARED_BUS CF_SPM_SHARED_BW
unset CF_BELADY CF_BELADY_L1 CF_BELADY_L2 CF_BELADY_TRACE
unset CF_BELADY_L1_TRACE CF_BELADY_L2_TRACE CF_BELADY_LOG_VIRTUAL
printf '%s\n' "${envs:-none}" > "$out/RUN_ENV.txt"
printf '%s\n' "${0#"$ROOT/"}" > "$out/RUNNER_USED.txt"
printf '%s\n' "$bin" > "$out/BINARY_USED.txt"
sha256sum "$bin" "$GEM5" > "$out/BINARIES.sha256"
# Drivers parse workload arguments as M K N; forward M, N, K accordingly.
env $envs "$GEM5" --outdir="$out" "$SE" \
  --cpu-type=DerivO3CPU --sys-clock=1.5GHz --cpu-clock=2.5GHz --caches --l2cache --l3cache \
  --l1d_size=64kB --l1i_size=64kB --l1d_assoc=4 --l1i_assoc=4 \
  --l2_size=512kB --l2_assoc=8 --l3_size=4MB --l3_assoc=16 \
  --cacheline_size=64 --mem-size=16GB --mem-channels=2 --mem-type=DDR4_2400_8x8 \
  --l1i-hwp-type=StridePrefetcher --l1d-hwp-type=StridePrefetcher \
  --l2-hwp-type=AMPMPrefetcher --l3-hwp-type=BOPPrefetcher \
  -P 'system.cpu[0].icache.prefetcher.degree=4' -P 'system.cpu[0].dcache.prefetcher.degree=4' \
  -P 'system.cpu[0].icache.tag_latency=2' -P 'system.cpu[0].icache.data_latency=2' \
  -P 'system.cpu[0].fetchWidth=5' -P 'system.cpu[0].decodeWidth=5' -P 'system.cpu[0].commitWidth=5' \
  -P 'system.cpu[0].renameWidth=5' -P 'system.cpu[0].dispatchWidth=8' -P 'system.cpu[0].issueWidth=8' \
  -P 'system.cpu[0].wbWidth=8' -P 'system.cpu[0].numROBEntries=128' -P 'system.cpu[0].numIQEntries=80' \
  -P 'system.cpu[0].LQEntries=32' -P 'system.cpu[0].SQEntries=48' \
  -P 'system.cpu[0].numPhysIntRegs=128' -P 'system.cpu[0].numPhysFloatRegs=192' \
  -P 'system.cpu[0].numPhysVecRegs=192' -P 'system.cpu[0].numPhysVecPredRegs=64' \
  -P 'system.cpu[0].cacheLoadPorts=2' -P 'system.cpu[0].cacheStorePorts=1' $spm \
  -P 'system.cpu[0].fuPool.FUList[0].count=2' -P 'system.cpu[0].fuPool.FUList[2].count=2' \
  -P 'system.cpu[0].fuPool.FUList[5].count=2' -P 'system.cpu[0].fuPool.FUList[7].count=2' \
  -P 'system.cpu[0].fuPool.FUList[9].count=2' \
  -P 'system.tol2bus.width=64' -P 'system.tol2bus.frontend_latency=1' -P 'system.tol2bus.header_latency=0' \
  -P 'system.tol2bus.forward_latency=1' -P 'system.tol2bus.response_latency=1' \
  -P 'system.l2.tag_latency=8' -P 'system.l2.data_latency=8' -P 'system.l2.response_latency=8' \
  -P 'system.tol3bus.width=32' -P 'system.tol3bus.frontend_latency=10' \
  -P 'system.tol3bus.forward_latency=10' -P 'system.tol3bus.response_latency=10' \
  -P 'system.l3.tag_latency=35' -P 'system.l3.data_latency=35' -P 'system.l3.response_latency=35' \
  -P 'system.cpu[0].isa[0].sve_vl_se=16' \
  -c "$bin" -o "$M $K $N 1 $kc $mc" > "$out/stdout.txt" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  printf "[%s] gem5 failed (exit %s)\n" "$(basename "$out")" "$rc" >&2
  exit "$rc"
fi
g=$(grep -oE '[0-9.]+ GFLOPS' "$out/stdout.txt" | tail -1 | grep -oE '^[0-9.]+' || true)
kus=$(grep -oE "kernel       :\s+[0-9.]+" "$out/stdout.txt" | grep -oE "[0-9.]+" || true)
cs=$(grep -ihE "checksum" "$out/stdout.txt" | tail -1 | grep -oE '[-0-9.]+$' || true)
if [ -z "$g" ] || [ -z "$cs" ] || [ ! -s "$out/stats.txt" ] ||
   ! grep -q 'Exiting @ tick' "$out/stdout.txt" ||
   grep -Eiq 'panic:|fatal:|segmentation fault|segfault|core dumped|simulated exit code not 0|validation failed|\[DBG\]|assert(ion)?[^[:cntrl:]]{0,256}failed' "$out/stdout.txt"; then
  printf "[%s] invalid gem5 result receipt\n" "$(basename "$out")" >&2
  exit 1
fi
printf "[%s] %8s GF  kernel_us=%-9s cksum=%s\n" \
  "$(basename "$out")" "$g" "${kus:-?}" "${cs:-?}" | tee "$out/.line"
