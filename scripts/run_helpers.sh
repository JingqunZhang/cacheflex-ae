#!/bin/bash
# run_helpers.sh — shared CLI + parallel-control helpers for experiments/<name>/run.sh
#
# Usage in an experiment's run.sh:
#   source "$(dirname "$0")/../../scripts/run_helpers.sh"
#   rh_parse_args "$@"          # sets RH_SWEEP (0/1), RH_JOBS (int)
#   rh_launch <outdir> <cmd...> # backgrounded cell, bounded by RH_JOBS
#   rh_wait_all                 # waits; returns nonzero if any cell failed
#   rh_compat_layout <bestdir>  # best mode only, after rh_wait_all: add
#                               # <cell>.log -> <cell>/stdout.log compat links
#
# Contract (AE unified interface):
#   default        = best mode (fresh run of best.json cells), JOBS=1.
#                    FRESH means: an existing cell outdir is moved aside to
#                    <outdir>.prev.<epoch> before the cell is (re)run — old
#                    output is never silently reused. Each experiment validates
#                    its exact best-cell count before creating output.
#   --sweep        = experiment's declared tuning grid
#   -j N | --jobs N = bound on the TOTAL number of concurrent gem5 processes,
#                    including the internal parallelism of sweep sub-scripts
#                    (capped at 16 and at nproc). run.sh sweep branches must invoke
#                    inner sweep scripts SEQUENTIALLY and forward N as their
#                    MAX_JOBS/PAR so the bound holds globally.
#   -h | --help    = print RH_HELP_TEXT (experiment sets it) and exit 0

RH_SWEEP=0
RH_JOBS=1
RH_FAILS=0
RH_PIDS=()
RH_OUTS=()
RH_MAX_JOBS=16
RH_CELLS_FILE=""

# Start every cell from a clean experiment-control environment. A caller's
# shell must never silently enable a gate in an unrelated cell.
RH_CLEAN_ENV=(
    env
    -u CF_L1_PIN_KEEP
    -u CF_L2_PIN_KEEP
    -u CF_NT_BYPASS
    -u CF_NT_CLEANRESP
    -u CF_RANGE_PROBE
    -u CF_SELF_CHECK
    -u CF_SPM_HDR_CHARGE
    -u CF_SPM_LINE_GRAIN
    -u CF_SPM_SHARED_BUS
    -u CF_SPM_SHARED_BW
    -u CF_BELADY
    -u CF_BELADY_L1
    -u CF_BELADY_L2
    -u CF_BELADY_TRACE
    -u CF_BELADY_L1_TRACE
    -u CF_BELADY_L2_TRACE
    -u CF_BELADY_LOG_VIRTUAL
)

rh_parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --sweep) RH_SWEEP=1 ;;
            -j|--jobs)
                shift
                [ $# -gt 0 ] || { echo "ERROR: -j/--jobs requires a value" >&2; exit 2; }
                RH_JOBS="$1" ;;
            -j*) RH_JOBS="${1#-j}" ;;
            -h|--help)
                echo "${RH_HELP_TEXT:-Usage: run.sh [--sweep] [-j N] [-h]
  default   : fresh run of the best configs from best.json (no sweep), JOBS=1;
              existing cell outdirs are moved to <outdir>.prev.<epoch> first
  --sweep   : run the declared tuning grid (results -> results/generated/sweep/)
  -j N      : bound the TOTAL number of concurrent gem5 processes, including
              sweep-script internals (capped at 16 and at nproc)}"
                exit 0 ;;
            *) echo "ERROR: unknown option '$1' (see --help)" >&2; exit 2 ;;
        esac
        shift
    done
    case "$RH_JOBS" in (*[!0-9]*|'') echo "ERROR: -j expects a positive integer" >&2; exit 2;; esac
    [ "$RH_JOBS" -ge 1 ] || { echo "ERROR: -j expects >= 1" >&2; exit 2; }
    local np cap
    np=$(nproc)
    cap=$RH_MAX_JOBS
    if [ "$np" -lt "$cap" ]; then
        cap=$np
    fi
    if [ "$RH_JOBS" -gt "$cap" ]; then
        echo "WARNING: -j $RH_JOBS exceeds the supported limit ($cap); capping at $cap" >&2
        RH_JOBS=$cap
    fi
}

# Require the experiment's best.json before a default (best-mode) run.
rh_require_best() {   # rh_require_best <path-to-best.json>
    if [ ! -f "$1" ]; then
        echo "ERROR: $1 not found — best-mode needs the shipped best.json." >&2
        echo "       A sweep is NOT started automatically; run with --sweep explicitly." >&2
        exit 3
    fi
}

# Materialize and validate a best.json-derived TSV before any output directory
# is created or any gem5 cell is launched. Process substitution alone cannot
# propagate a failed JSON producer to the parent shell.
rh_materialize_cells() {
    # rh_materialize_cells <expected-rows> <columns> [producer] [key-fields]
    # key-fields is a comma-separated list of TSV columns; it defaults to the
    # first column. End-to-end uses "1,2" because the same workload name may
    # legitimately occur once in each VL block.
    local expected="$1" columns="$2" producer="${3:-cells_tsv}"
    local key_fields="${4:-1}" tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/cacheflex-ae-cells.XXXXXX")" || {
        echo "ERROR: cannot create a temporary cell list" >&2
        return 3
    }
    if ! "$producer" > "$tmp"; then
        echo "ERROR: failed to parse the best configuration" >&2
        rm -f -- "$tmp"
        return 3
    fi
    if ! awk -F '\t' -v expected="$expected" -v columns="$columns" \
        -v key_fields="$key_fields" '
        BEGIN {
            key_count = split(key_fields, key_columns, ",")
        }
        NF != columns {
            printf "ERROR: malformed best cell row %d (expected %d fields, got %d)\n",
                   NR, columns, NF > "/dev/stderr"
            bad = 1
        }
        $1 == "" {
            printf "ERROR: empty best cell name at row %d\n", NR > "/dev/stderr"
            bad = 1
        }
        {
            for (field = 1; field <= NF; field++) {
                if ($field == "") {
                    printf "ERROR: empty field %d in best cell row %d\n",
                           field, NR > "/dev/stderr"
                    bad = 1
                }
            }
        }
        {
            key = ""
            for (part = 1; part <= key_count; part++) {
                column = key_columns[part]
                if (column < 1 || column > columns) {
                    printf "ERROR: invalid uniqueness-key column %s\n",
                           column > "/dev/stderr"
                    bad = 1
                    continue
                }
                key = key (part == 1 ? "" : "/") $column
            }
            if (seen[key]++) {
                printf "ERROR: duplicate best cell key: %s\n",
                       key > "/dev/stderr"
                bad = 1
            }
        }
        END {
            if (NR != expected) {
                printf "ERROR: expected %d best cells, found %d\n",
                       expected, NR > "/dev/stderr"
                bad = 1
            }
            exit bad
        }
    ' "$tmp"; then
        rm -f -- "$tmp"
        return 3
    fi
    RH_CELLS_FILE="$tmp"
    trap 'rm -f -- "$RH_CELLS_FILE"' EXIT
}

rh__slot_wait() {
    while [ "$(jobs -rp | wc -l)" -ge "$RH_JOBS" ]; do
        # Reap one slot here, but count its status only in rh_wait_all().
        # Bash retains a child's status for a later `wait <pid>`; counting it
        # in both places made one failed cell appear twice in the summary.
        wait -n || true
    done
}

# rh_launch <outdir> <cmd...>  — each cell gets its own outdir; stdout+stderr
# captured to <outdir>/stdout.log.
# Default (fresh) semantics: an existing outdir is moved aside to
# <outdir>.prev.<epoch> before the run — old output is never silently reused.
rh_launch() {
    local out="$1"; shift
    if [ -e "$out" ]; then
        local prev="$out.prev.$(date +%s)"
        while [ -e "$prev" ]; do prev="$prev.1"; done
        mv "$out" "$prev"
        echo "[prev] $out -> $prev"
    fi
    mkdir -p "$out"
    rh__slot_wait
    echo "[run ] $out"
    # Per-cell wall-clock cap prevents one hung simulation from blocking the
    # full run. Override with RH_CELL_TIMEOUT (seconds); default 24h —
    # the largest cells (vl4 llama T=4096 FFN) legitimately run for many
    # hours under a fully loaded -j run.
    ( timeout "${RH_CELL_TIMEOUT:-86400}" "$@" > "$out/stdout.log" 2>&1 ) &
    RH_PIDS+=($!)
    RH_OUTS+=("$out")
}

rh_wait_all() {
    local p
    for p in "${RH_PIDS[@]}"; do
        wait "$p" || RH_FAILS=$((RH_FAILS + 1))
    done
    if [ "$RH_FAILS" -gt 0 ]; then
        echo "FAILED cells: $RH_FAILS" >&2
        return 1
    fi
    return 0
}

# Validate the gem5 result produced by every rh_launch in a default best run.
# gem5 may return host status 0 even when the simulated program exits nonzero,
# so the shell status alone is not a sufficient success receipt.
rh_validate_gem5_cells() {
    local out log fail=0
    for out in "${RH_OUTS[@]}"; do
        log="$out/stdout.log"
        if [ -s "$out/stdout.txt" ]; then
            log="$out/stdout.txt"
        fi
        if [ ! -s "$out/config.ini" ] || [ ! -s "$out/stats.txt" ] || [ ! -s "$log" ]; then
            echo "FAILED cell receipt: $out (missing/empty config.ini, stats.txt, or result log)" >&2
            fail=1
            continue
        fi
        if ! grep -q 'Begin Simulation Statistics' "$out/stats.txt"; then
            echo "FAILED cell receipt: $out (stats.txt has no simulation-statistics dump)" >&2
            fail=1
        fi
        if ! grep -q 'Exiting @ tick' "$log"; then
            echo "FAILED cell receipt: $out (no clean gem5 exit)" >&2
            fail=1
        fi
        if grep -Eiq \
            'panic:|fatal:|segmentation fault|segfault|core dumped|simulated exit code not 0|validation failed|SELF_CHECK (FAIL|SKIP)|\[DBG\]|assert(ion)?[^[:cntrl:]]{0,256}failed' \
            "$log"; then
            echo "FAILED cell receipt: $out (crash, nonzero exit, validation, or debug marker)" >&2
            fail=1
        fi
    done
    return "$fail"
}

# rh_compat_layout <bestdir> — post-step for best mode (call after rh_wait_all).
# For every cell dir <bestdir>/<cell>/ that has a stdout.log, create a sibling
# RELATIVE symlink <bestdir>/<cell>.log -> <cell>/stdout.log so the generated
# tree matches the reference layout the canonical plotters expect
# (<dir>/<cell>.log next to <dir>/<cell>/stats.txt).
rh_compat_layout() {
    local base="$1" d n
    [ -d "$base" ] || return 0
    for d in "$base"/*/; do
        [ -d "$d" ] || continue
        n="$(basename "$d")"
        case "$n" in (*.prev.*) continue ;; esac   # skip moved-aside runs
        [ -f "$base/$n/stdout.log" ] || continue
        ln -sfn "$n/stdout.log" "$base/$n.log"
    done
    return 0
}
