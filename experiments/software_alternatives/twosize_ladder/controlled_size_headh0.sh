#!/bin/bash
# Usage: controlled_size_headh0.sh <tag> <M> <N> <K> <PAR>
# Figure 8 sweep for one GEMM size. Existing cells are moved aside.
#
# Cache-side rows use KC={64,128,256,512}; CacheFlex rows use
# KC={64,128,256}; all rows use MC={8,16,32,64}. The no-prefetch row reuses
# the selected cache tile, and the shared-bandwidth row reuses the selected
# CacheFlex KC.
set -euo pipefail
ROOT="${CACHEFLEX_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
TL=$ROOT/experiments/software_alternatives/twosize_ladder
: "${RESULTS_OUT_ROOT:?internal sweep helper; use experiments/software_alternatives/run.sh --sweep [-j N]}"
RUN=$ROOT/experiments/software_alternatives/runners/run_cell_headh0.sh
RUNOFF=$TL/run_cell_hwpfoff_headh0.sh
BIN=$ROOT/kernels/gemm/software_alternatives/bin
PARSE=$TL/parse_cell.py
SELECT=$TL/select_exact_best.py
NTENV="CF_NT_BYPASS=1 CF_NT_CLEANRESP=1 CF_L2_PIN_KEEP=1"
L2PFENV="CF_NT_BYPASS=1"
SHBWENV="CF_SPM_SHARED_BW=1 CF_SPM_HDR_CHARGE=1"
tag=$1; M=$2; N=$3; K=$4; PAR=${5:-24}
OUT=$RESULTS_OUT_ROOT/${tag}_hh0full
mkdir -p "$OUT"
ln -sfn "${tag}_hh0full" "$RESULTS_OUT_ROOT/$tag"
INVOCATION_ID="${tag}-$(date +%s)-$$"
# CacheFlex is limited to KC<=341 by the VL16 SPM row budget.
KCS="64 128 256 512"; KCS_SPM="64 128 256"; MCS="8 16 32 64"
PIDS=()
LABELS=()

sem(){ while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 1; done; }

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

launch_cell() {
    local d="$1" label="$2"
    shift 2
    sem
    move_aside "$d"
    mkdir -p "$d"
    printf '%s\n' "$INVOCATION_ID" > "$d/SWEEP_INVOCATION_ID"
    "$@" >/dev/null 2>&1 &
    PIDS+=("$!")
    LABELS+=("$label")
}

wait_for_jobs() {
    local phase="$1" i failures=0 total="${#PIDS[@]}"
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "[FAIL] $phase: ${LABELS[$i]} (pid ${PIDS[$i]})" >&2
            failures=$((failures + 1))
        fi
    done
    PIDS=()
    LABELS=()
    if [ "$failures" -ne 0 ]; then
        echo "ERROR: $failures of $total cells failed in $phase" >&2
        return 1
    fi
}

csv_axis() {
    local axis="$1"
    printf '%s\n' "${axis// /,}"
}

bestKCMC() {
    local prefix="$1" kcs="$2" mcs="$3"
    python3 "$SELECT" \
        --out "$OUT" \
        --prefix "$prefix" \
        --kcs "$(csv_axis "$kcs")" \
        --mcs "$(csv_axis "$mcs")" \
        --invocation-id "$INVOCATION_ID" \
        --parser "$PARSE"
}

bestMC() {
    local prefix="$1" kc="$2" mcs="$3" winner
    winner=$(bestKCMC "$prefix" "$kc" "$mcs")
    printf '%s\n' "${winner#* }"
}

echo "[ctrl $tag] PHASE 1 cache and CacheFlex rows, PAR=$PAR  $(date +%H:%M:%S)"
for kc in $KCS; do for mc in $MCS; do
  d="$OUT/cacheopt_kc${kc}_mc${mc}"
  launch_cell "$d" "cacheopt_kc${kc}_mc${mc}" bash "$RUN" "$d" "$BIN/v3_l1swpf_none" 0 "" $M $N $K $kc $mc
done; done
for kc in $KCS_SPM; do for mc in $MCS; do
  d="$OUT/cacheflex_kc${kc}_mc${mc}"
  launch_cell "$d" "cacheflex_kc${kc}_mc${mc}" bash "$RUN" "$d" "$BIN/v3_spm_n0outer_VL_16" 1 "" $M $N $K $kc $mc
done; done
if ! wait_for_jobs "phase 1"; then
  exit 1
fi
read -r KCc MCc <<< "$(bestKCMC cacheopt "$KCS" "$MCS")"
read -r KCs MCs <<< "$(bestKCMC cacheflex "$KCS_SPM" "$MCS")"
echo "[ctrl $tag] row1 cacheopt KC$KCc/MC$MCc ; row6 cacheflex KC$KCs/MC$MCs  $(date +%H:%M:%S)"

echo "[ctrl $tag] PHASE 2 software alternatives and shared-bandwidth row  $(date +%H:%M:%S)"
for kc in $KCS; do for mc in $MCS; do
  d="$OUT/pldl2keep_kc${kc}_mc${mc}"
  launch_cell "$d" "pldl2keep_kc${kc}_mc${mc}" bash "$RUN" "$d" "$BIN/v3_l2pf_only_b512" 0 "$L2PFENV" $M $N $K $kc $mc
  d="$OUT/nt_kc${kc}_mc${mc}"
  launch_cell "$d" "nt_kc${kc}_mc${mc}" bash "$RUN" "$d" "$BIN/v3_l2pin_nt_pf0" 0 "$NTENV" $M $N $K $kc $mc
done; done
for mc in $MCS; do
  d="$OUT/shbw_kc${KCs}_mc${mc}"
  launch_cell "$d" "shbw_kc${KCs}_mc${mc}" bash "$RUN" "$d" "$BIN/v3_spm_n0outer_VL_16" 1 "$SHBWENV" $M $N $K $KCs $mc
done
if ! wait_for_jobs "phase 2"; then
  exit 1
fi
read -r KC3 MC3 <<< "$(bestKCMC pldl2keep "$KCS" "$MCS")"
read -r KC4 MC4 <<< "$(bestKCMC nt "$KCS" "$MCS")"
MC5=$(bestMC shbw "$KCs" "$MCS")
echo "[ctrl $tag] row3 pldl2keep KC$KC3/MC$MC3 ; row4 nt KC$KC4/MC$MC4 ; row5 shbw KC$KCs/MC$MC5  $(date +%H:%M:%S)"

echo "[ctrl $tag] PHASE 3 no-prefetch row"
d="$OUT/pfoff_kc${KCc}_mc${MCc}"
move_aside "$d"
mkdir -p "$d"
printf '%s\n' "$INVOCATION_ID" > "$d/SWEEP_INVOCATION_ID"
if ! bash "$RUNOFF" "$d" "$BIN/v3_l1swpf_none" 0 "" $M $N $K $KCc $MCc >/dev/null 2>&1; then
  echo "[FAIL] phase 3: pfoff_kc${KCc}_mc${MCc}" >&2
  exit 1
fi
# Validate the derived cell with the same strict receipt and provenance gate.
read -r PF_KC PF_MC <<< "$(bestKCMC pfoff "$KCc" "$MCc")"
if [ "$PF_KC" != "$KCc" ] || [ "$PF_MC" != "$MCc" ]; then
    echo "ERROR: derived pfoff selector returned an undeclared tile" >&2
    exit 2
fi

move_aside "$OUT/chosen.txt"
cat > "$OUT/chosen.txt" <<EOF
row1_cacheopt    ${tag}/cacheopt_kc${KCc}_mc${MCc}
row2_pfoff       ${tag}/pfoff_kc${KCc}_mc${MCc}
row3_pldl2keep   ${tag}/pldl2keep_kc${KC3}_mc${MC3}
row4_pldl2keepnt ${tag}/nt_kc${KC4}_mc${MC4}
row5_spm_shbw    ${tag}/shbw_kc${KCs}_mc${MC5}
row6_cacheflex   ${tag}/cacheflex_kc${KCs}_mc${MCs}
EOF
echo "[ctrl $tag] selected rows:"; cat "$OUT/chosen.txt"
while read rn cell; do
  python3 "$PARSE" "$RESULTS_OUT_ROOT/$cell/stdout.txt" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print('  %-18s gf=%s total=%s cksum=%s'%('$rn',d['gflops'],d['total'],d['checksum']))" 2>/dev/null
done < "$OUT/chosen.txt"
echo "DONE_CTRL $tag $(date +%H:%M:%S)"
