#!/bin/bash
# smoke_gem5.sh — gem5 SPM smoke gate. Runs ONE small SPM GEMM cell under the
# CacheFlex gem5 fork and verifies (1) the SPM datapath engaged
# (system.l2.spmReads > 0) and (2) functional correctness using three
# bit-exact logical-output receipts.
#
# Reference cell: W7 = 784x256x1024 FP16 GEMM, v3_fused_vl16 (SPM, VL=16),
# KC=256 MC=784, NITER=1 — the fastest pinned W7 cell
# (~2 min wall on the reference host).
# The expected values below are for the logical M-by-N output only. They were
# independently matched by the cache baseline, the SPM mock, and gem5 SPM
# executions using the same deterministic inputs, arguments, and VL.
#
# Usage:
#   source $CACHEFLEX_ROOT/setup_env.sh
#   bash $CACHEFLEX_ROOT/tests/smoke_gem5.sh
#
# Exit codes: 0 = PASS, 1 = FAIL, 3 = SKIP (prerequisite not built yet)

set -e

if [ -z "${CACHEFLEX_ROOT:-}" ]; then
    CACHEFLEX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    source "$CACHEFLEX_ROOT/setup_env.sh" >/dev/null
fi

GEM5="${GEM5:-$CACHEFLEX_ROOT/gem5/build/ARM/gem5.opt}"
GEM5_SE="${GEM5_CONFIG:-$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py}"
BIN="$CACHEFLEX_ROOT/kernels/gemm/cacheflex/bin/v3_fused_vl16"
REBUILD_HINT='cd gem5 && scons build/ARM/gem5.opt -j4   (SETUP.md step 5)'

EXPECT_SUM="6.97402781248"
EXPECT_L1="34227.6789363"
EXPECT_WEIGHTED_L1="51316.5992603"

fail() { echo ""; echo "FAIL: $*"; exit 1; }
skip() { echo ""; echo "SKIP: $*"; exit 3; }

echo "=== [1/4] gem5 binary ==="
[ -f "$GEM5" ] || skip "gem5 not found at $GEM5 — build it: $REBUILD_HINT"
# --build-info exercises the embedded Python interpreter, so it catches the
# common case of a gem5.opt copied from another machine: the binary embeds the
# BUILD host's libpython and fails to start anywhere else.
if ! "$GEM5" --build-info >/dev/null 2>&1; then
    echo ""
    echo "  gem5 exists but does not execute on this host. Most common cause:"
    echo "  the binary was built on another machine and links that host's"
    echo "  libpython (inherent to gem5 builds, not a defect)."
    echo "  Rebuild on THIS machine: $REBUILD_HINT"
    fail "gem5 binary present but not executable"
fi
echo "  ok: $GEM5 executes"

echo ""
echo "=== [2/4] SPM kernel binary ==="
[ -x "$BIN" ] || skip "v3_fused_vl16 not found — build it: bash kernels/gemm/cacheflex/build.sh"
echo "  ok: $BIN"

echo ""
echo "=== [3/4] Running one SPM cell (W7 784x256x1024, VL16, ~2 min) ==="
OUTDIR="$CACHEFLEX_ROOT/tests/out_smoke_gem5"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"

# Canonical flag set plus the VL16 cell parameters
# (sve_vl_se=16, spmLoad/StorePorts=1)
# so the result is directly comparable with the reference run.
GEM5_FLAGS=(
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
    -P 'system.cpu[0].isa[0].sve_vl_se=16'
    -P 'system.cpu[0].spmLoadPorts=1'
    -P 'system.cpu[0].spmStorePorts=1'
)

if ! "$GEM5" --outdir="$OUTDIR" "$GEM5_SE" "${GEM5_FLAGS[@]}" \
        -c "$BIN" -o "784 256 1024 1 256 784" > "$LOG" 2>&1; then
    tail -20 "$LOG"
    fail "gem5 run exited non-zero (full log: $LOG)"
fi
echo "  run complete (log: $LOG)"

echo ""
echo "=== [4/4] Verifying SPM activity + checksum ==="
STATS="$OUTDIR/stats.txt"
[ -f "$STATS" ] || fail "no stats.txt produced (log: $LOG)"

SPM_READS=$(grep -m1 '^system\.l2\.spmReads' "$STATS" | awk '{print $2}' || true)
[ -n "$SPM_READS" ] || fail "system.l2.spmReads not in stats.txt — is this gem5 built from the CacheFlex fork in this archive?"
[ "$SPM_READS" -gt 0 ] 2>/dev/null \
    || fail "system.l2.spmReads = $SPM_READS (expected > 0): SPM datapath did not engage"
echo "  spmReads: $SPM_READS  (> 0 ok)"

GOT_SUM=$(
    awk '$1 == "CHECKSUM_LOGICAL:" { print $2; exit }' "$LOG" || true
)
GOT_L1=$(
    awk '$1 == "CHECKSUM_LOGICAL_L1:" { print $2; exit }' "$LOG" || true
)
GOT_WEIGHTED_L1=$(
    awk '$1 == "CHECKSUM_LOGICAL_WEIGHTED_L1:" { print $2; exit }' \
        "$LOG" || true
)
[ -n "$GOT_SUM" ] && [ -n "$GOT_L1" ] && [ -n "$GOT_WEIGHTED_L1" ] \
    || fail "logical output receipts are missing (log: $LOG)"
[ "$GOT_SUM" = "$EXPECT_SUM" ] \
    || fail "logical sum mismatch — expected $EXPECT_SUM, got $GOT_SUM"
[ "$GOT_L1" = "$EXPECT_L1" ] \
    || fail "logical L1 mismatch — expected $EXPECT_L1, got $GOT_L1"
[ "$GOT_WEIGHTED_L1" = "$EXPECT_WEIGHTED_L1" ] \
    || fail "weighted logical L1 mismatch — expected $EXPECT_WEIGHTED_L1, got $GOT_WEIGHTED_L1"
echo "  logical output receipts: exact match"

echo ""
echo "========================================"
echo "SMOKE TEST PASSED (gem5 SPM)"
echo "  spmReads  : $SPM_READS (expected > 0)"
echo "  checksum  : $GOT_SUM"
echo "  L1        : $GOT_L1"
echo "  weighted  : $GOT_WEIGHTED_L1"
echo "========================================"
