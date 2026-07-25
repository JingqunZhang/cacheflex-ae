#!/usr/bin/env python3
"""Compute cache + DRAM energy from gem5 ROI statistics.

The per-access constants below are the frozen values used for the paper
figures.  The reusable ``parse_stats`` and ``compute_energy`` functions are
imported by the Figure 8 plotter.

Energy breakdown:
  - Cache dynamic: L1/L2/L3 read + write per-access energy
  - SPM dynamic: spmReads + spmWrites (CacheFlex only)
  - Cache idle (leakage): idle_power × ROI_time
  - DRAM: gem5 totalEnergy (includes dynamic + background)

Output: per-workload energy breakdown (cache_dynamic, cache_idle, spm, dram).
"""

import math
import os
import re

# ── Per-access energy from RTL (nJ) ──
E = {
    'l1_read':    0.3609,
    'l1_write':   0.4824,
    'l2_read':    1.1205,
    'l2_write':   1.4409,
    'l3_read':    5.121,
    'l3_write':   6.8205,
    'spm_read':   0.5328,
    'spm_write':  0.5526,
}

# Idle power (mW)
IDLE_L1  = 12.0
IDLE_L2_BL = 26.0   # baseline: cache-only L2
IDLE_L2_CF = 31.0   # CacheFlex: L2 + SPM structure
IDLE_L3  = 51.0

STATS_MARKER = re.compile(
    r"^-+\s+(Begin|End) Simulation Statistics\s+-+\s*$", re.MULTILINE
)

# gem5 omits unobserved vector-stat subentries instead of printing an explicit
# zero.  These are the only absent counters accepted by this energy model.
OPTIONAL_ZERO_STATS = frozenset(
    {
        "system.cpu.dcache.ReadSharedReq.accesses::total",
        "system.cpu.dcache.WriteLineReq.accesses::total",
    }
)


def _first_complete_stats_block(text, stats_path):
    markers = list(STATS_MARKER.finditer(text))
    begin_count = sum(marker.group(1) == "Begin" for marker in markers)
    end_count = len(markers) - begin_count
    if (
        not markers
        or begin_count != text.count("Begin Simulation Statistics")
        or end_count != text.count("End Simulation Statistics")
    ):
        raise ValueError(f"{stats_path}: malformed simulation-statistics markers")

    blocks = []
    block_start = None
    for marker in markers:
        if marker.group(1) == "Begin":
            if block_start is not None:
                raise ValueError(f"{stats_path}: nested statistics begin marker")
            block_start = marker.end()
        else:
            if block_start is None:
                raise ValueError(f"{stats_path}: unmatched statistics end marker")
            blocks.append(text[block_start:marker.start()])
            block_start = None
    if block_start is not None or not blocks:
        raise ValueError(f"{stats_path}: no complete statistics block")
    return blocks[0]


def _stat_value(roi, name, stats_path, *, positive=False):
    matches = re.findall(
        rf"^{re.escape(name)}[ \t]+(\S+)", roi, re.MULTILINE
    )
    if not matches:
        if name in OPTIONAL_ZERO_STATS:
            return 0.0
        raise ValueError(f"{stats_path}: missing required ROI statistic {name}")
    if len(matches) != 1:
        raise ValueError(
            f"{stats_path}: ROI statistic {name} occurs {len(matches)} times"
        )
    try:
        value = float(matches[0])
    except ValueError as error:
        raise ValueError(
            f"{stats_path}: ROI statistic {name} is not numeric: "
            f"{matches[0]!r}"
        ) from error
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        constraint = "positive" if positive else "non-negative"
        raise ValueError(
            f"{stats_path}: ROI statistic {name} must be finite and "
            f"{constraint}, got {value!r}"
        )
    return value


def parse_stats(stats_path):
    """Extract the first post-reset ROI dump from gem5 stats.txt."""
    with open(stats_path, encoding="utf-8", errors="replace") as stream:
        txt = stream.read()
    roi = _first_complete_stats_block(txt, stats_path)

    r = {}

    # Timing
    r['simTicks'] = _stat_value(roi, 'simTicks', stats_path, positive=True)
    r['simFreq'] = _stat_value(roi, 'simFreq', stats_path, positive=True)
    r['roi_time_s'] = r['simTicks'] / r['simFreq']

    # L1 dcache
    # CPTOSPMReq source reads are classified by gem5 as ReadSharedReq.
    r['l1_reads'] = (
        _stat_value(
            roi, 'system.cpu.dcache.ReadReq.accesses::total', stats_path
        )
        + _stat_value(
            roi, 'system.cpu.dcache.ReadSharedReq.accesses::total', stats_path
        )
    )
    r['l1_writes'] = (
        _stat_value(
            roi, 'system.cpu.dcache.WriteReq.accesses::total', stats_path
        )
        + _stat_value(
            roi, 'system.cpu.dcache.WriteLineReq.accesses::total', stats_path
        )
    )

    # In this paper's runs, L2 overallAccesses contains the read-class command
    # accesses and excludes WritebackDirty.  The latter is therefore charged
    # separately at the write-access energy.  This is a workload/configuration
    # invariant checked when the canonical reference set is sealed.
    r['l2_reads'] = _stat_value(
        roi, 'system.l2.overallAccesses::total', stats_path
    )
    r['l2_writes'] = _stat_value(
        roi, 'system.l2.WritebackDirty.accesses::total', stats_path
    )

    # L3 uses the same command-counter accounting.
    r['l3_reads'] = _stat_value(
        roi, 'system.l3.overallAccesses::total', stats_path
    )
    r['l3_writes'] = _stat_value(
        roi, 'system.l3.WritebackDirty.accesses::total', stats_path
    )

    # SPM
    r['spm_reads'] = _stat_value(roi, 'system.l2.spmReads', stats_path)
    r['spm_writes'] = _stat_value(roi, 'system.l2.spmWrites', stats_path)

    # DRAM: sum all ranks totalEnergy (pJ)
    r['dram_energy_pJ'] = sum(
        _stat_value(
            roi,
            f'system.mem_ctrls{c}.dram.rank{rk}.totalEnergy',
            stats_path,
        )
        for c in [0, 1] for rk in [0, 1])

    return r


def compute_energy(stats, is_cacheflex=False, vl=4):
    """Compute energy breakdown in mJ.

    SPM read energy correction: gem5 counts spm.ld1qd instructions, but
    each instruction loads VL×128b = VL×16B. The per-access energy is for
    one 64B cacheline read. So we multiply spmReads by (VL*16/64) = VL/4.
      VL=4:  ×1 (64B per ld1qd = 1 cacheline)
      VL=8:  ×2 (128B per ld1qd = 2 cachelines)
      VL=16: ×4 (256B per ld1qd = 4 cachelines)
    """
    required = (
        'l1_reads', 'l1_writes', 'l2_reads', 'l2_writes',
        'l3_reads', 'l3_writes', 'spm_reads', 'spm_writes',
        'roi_time_s', 'dram_energy_pJ',
    )
    for field in required:
        if field not in stats:
            raise ValueError(f"missing parsed energy field {field}")
        value = stats[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"energy field {field} must be finite")
        if value < 0 or (field == 'roi_time_s' and value <= 0):
            raise ValueError(f"energy field {field} is outside its valid range")
    if vl not in (4, 8, 16):
        raise ValueError(f"unsupported vector length for energy model: {vl!r}")

    # Cache dynamic (nJ → mJ: /1e6)
    cache_dyn_nJ = (
        stats['l1_reads']  * E['l1_read']  +
        stats['l1_writes'] * E['l1_write'] +
        stats['l2_reads']  * E['l2_read']  +
        stats['l2_writes'] * E['l2_write'] +
        stats['l3_reads']  * E['l3_read']  +
        stats['l3_writes'] * E['l3_write']
    )
    cache_dyn_mJ = cache_dyn_nJ / 1e6

    # SPM dynamic (nJ → mJ) — corrected for multi-cacheline loads
    spm_read_multiplier = vl // 4  # 1 for VL=4, 2 for VL=8, 4 for VL=16
    spm_dyn_nJ = (
        stats['spm_reads']  * E['spm_read'] * spm_read_multiplier +
        stats['spm_writes'] * E['spm_write']
    )
    spm_dyn_mJ = spm_dyn_nJ / 1e6

    # Cache idle / leakage (mW × s = mJ)
    idle_l2 = IDLE_L2_CF if is_cacheflex else IDLE_L2_BL
    idle_total_mW = IDLE_L1 + idle_l2 + IDLE_L3
    cache_idle_mJ = idle_total_mW * stats['roi_time_s']

    # DRAM (pJ → mJ: /1e9)
    dram_mJ = stats['dram_energy_pJ'] / 1e9

    result = {
        'cache_dyn_mJ': cache_dyn_mJ,
        'spm_dyn_mJ':   spm_dyn_mJ,
        'cache_idle_mJ': cache_idle_mJ,
        'dram_mJ':       dram_mJ,
        'total_mJ':      cache_dyn_mJ + spm_dyn_mJ + cache_idle_mJ + dram_mJ,
    }
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError(
            f"computed energy is not finite and non-negative: {result!r}"
        )
    return result


# ── Workload definitions: (tag, M, K, N, v1_best_config, v3_best_config) ──
WORKLOADS = [
    ('W1', 128,  4096, 11008, 'W1_v1_vl4',  'W1_v3_vl4'),
    ('W2', 256,  4096,  4096, 'W2_v1_vl4',  'W2_v3_vl4'),
    ('W3', 256,  2048,  2048, 'W3_v1_vl4',  'W3_v3_vl4'),
    ('W4', 512,   768,   768, 'W4_v1_vl4',  'W4_v3_vl4'),
    ('W5', 2048, 2048,  2048, 'W5_v1_vl4',  'W5_v3_vl4'),
    ('W6', 2048,   64,  2048, 'W6_v1_vl4',  'W6_v3_vl4'),
    ('W7', 784,   256,  1024, 'W7_v1_vl4',  'W7_v3_vl4'),
]

# Input root: RESULTS_DIR env var override (e.g. results/generated/best after
# a best-mode run.sh); the default is the canonical vl_length reference set.
RESULT_DIR = os.environ.get(
    'RESULTS_DIR',
    os.path.expandvars(
        '$CACHEFLEX_ROOT/experiments/vl_length/results/reference/results_headh0'))


if __name__ == '__main__':
    print(f'{"ID":<4} {"Var":<4} {"Cache Dyn":>10} {"SPM Dyn":>10} '
          f'{"Cache Idle":>11} {"DRAM":>10} {"Total":>10}  (all mJ)')
    print('-' * 70)

    results = []
    for tag, M, K, N, v1_cfg, v3_cfg in WORKLOADS:
        for var, cfg, is_cf in [('BL', v1_cfg, False), ('CF', v3_cfg, True)]:
            stats_path = os.path.join(RESULT_DIR, cfg, 'stats.txt')
            if not os.path.exists(stats_path):
                print(f'{tag:<4} {var:<4} MISSING: {stats_path}')
                continue

            stats = parse_stats(stats_path)
            eng = compute_energy(stats, is_cacheflex=is_cf, vl=4)
            results.append((tag, var, eng))

            print(f'{tag:<4} {var:<4} {eng["cache_dyn_mJ"]:>10.2f} {eng["spm_dyn_mJ"]:>10.2f} '
                  f'{eng["cache_idle_mJ"]:>11.2f} {eng["dram_mJ"]:>10.2f} {eng["total_mJ"]:>10.2f}')

        # Print gain
        bl = [r for r in results if r[0] == tag and r[1] == 'BL']
        cf = [r for r in results if r[0] == tag and r[1] == 'CF']
        if bl and cf:
            bl_total = bl[-1][2]['total_mJ']
            cf_total = cf[-1][2]['total_mJ']
            pct = (cf_total - bl_total) / bl_total * 100
            print(f'     {"":4} {"reduction:":>10} {pct:>+.1f}%')
        print()
