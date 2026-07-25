#!/usr/bin/env python3
"""Render Figure 7 from the canonical gem5 result cells.

Layout, colors, and annotations match the publication figure.
Data provenance: perf = pinned GFLOPS per (W, variant, VL);
energy = RTL per-access model applied to the SAME run's stats.txt.
"""
import glob
import json
import math
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIGDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.abspath(os.environ.get("OUTPUT_DIR", FIGDIR))
# Results root: CLI arg or RESULTS_DIR env var override; default = script dir
# (where the results_* compat symlinks / dirs live, e.g. results/reference).
if len(sys.argv) > 1:
    R = os.path.abspath(sys.argv[1])
elif os.environ.get('RESULTS_DIR'):
    R = os.path.abspath(os.environ['RESULTS_DIR'])
else:
    R = FIGDIR
WS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']
VLS = [4, 8, 16]
PEAK = {4: 320.0, 8: 640.0, 16: 1280.0}
GP = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*GFLOPS',
    re.I,
)
GFLOPS_WORD = re.compile(r'\bGFLOPS\b', re.I)
LOGICAL_CHECKSUM = re.compile(
    r'^\s*CHECKSUM_LOGICAL:\s*'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$',
    re.I | re.M,
)
LOGICAL_CHECKSUM_L1 = re.compile(
    r'^\s*CHECKSUM_LOGICAL_L1:\s*'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$',
    re.I | re.M,
)
LOGICAL_CHECKSUM_WEIGHTED = re.compile(
    r'^\s*CHECKSUM_LOGICAL_WEIGHTED_L1:\s*'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$',
    re.I | re.M,
)
CLEAN_EXIT = 'Exiting @ tick'
BAD_LOG = re.compile(
    r'panic:|fatal:|segmentation fault|segfault|core dumped|'
    r'simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|'
    r'\[DBG\]|'
    r'assert(?:ion)?[^\n]{0,256}failed',
    re.I,
)
STATS_BEGIN = '---------- Begin Simulation Statistics'
STATS_END = '---------- End Simulation Statistics'

E = {'l1_read': 0.3609, 'l1_write': 0.4824, 'l2_read': 1.1205, 'l2_write': 1.4409,
     'l3_read': 5.121, 'l3_write': 6.8205, 'spm_read': 0.5328, 'spm_write': 0.5526}
IDLE_L1, IDLE_L2_BL, IDLE_L2_CF, IDLE_L3 = 12.0, 26.0, 31.0, 51.0

RESULT_SUBDIR = ('results_headh0'
                 if os.path.isdir(os.path.join(R, 'results_headh0'))
                 else '.')
SOURCE_CELLS = []

def candidates(w, var, vl):
    return [os.path.join(R, RESULT_SUBDIR, f'{w}_{var}_vl{vl}')]

def invalid(path, reason):
    raise SystemExit(f'ERROR: invalid Figure 7 input {path}: {reason}')

def gflops_of(rd):
    lf = rd + '.log'
    if not os.path.isfile(lf) or os.path.getsize(lf) == 0:
        invalid(lf, 'missing or empty log')
    with open(lf, errors='ignore') as handle:
        text = handle.read()
    if CLEAN_EXIT not in text:
        invalid(lf, 'clean gem5 exit missing')
    if BAD_LOG.search(text):
        invalid(lf, 'crash, nonzero exit, validation, or debug marker present')
    if len(GFLOPS_WORD.findall(text)) != 1:
        invalid(lf, 'expected exactly one GFLOPS receipt')
    match = GP.search(text)
    if match is None:
        invalid(lf, 'GFLOPS receipt is not numeric')
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0:
        invalid(lf, f'GFLOPS must be finite and positive, got {value!r}')
    return value


def logical_checksum_of(rd):
    log_path = rd + '.log'
    with open(log_path, errors='ignore') as handle:
        text = handle.read()
    matches = LOGICAL_CHECKSUM.findall(text)
    l1_matches = LOGICAL_CHECKSUM_L1.findall(text)
    weighted_matches = LOGICAL_CHECKSUM_WEIGHTED.findall(text)
    if (
        len(matches) != 1
        or len(l1_matches) != 1
        or len(weighted_matches) != 1
    ):
        invalid(
            log_path,
            'expected one logical-output sum, one L1 checksum, and one '
            f'weighted checksum, found {len(matches)}, {len(l1_matches)}, '
            f'and {len(weighted_matches)}',
        )
    value = float(matches[0])
    value_l1 = float(l1_matches[0])
    value_weighted = float(weighted_matches[0])
    if not math.isfinite(value):
        invalid(log_path, f'logical-output checksum is not finite: {value!r}')
    if not math.isfinite(value_l1) or value_l1 <= 0:
        invalid(
            log_path,
            f'logical-output L1 checksum must be finite and positive: '
            f'{value_l1!r}',
        )
    if not math.isfinite(value_weighted) or value_weighted <= 0:
        invalid(
            log_path,
            f'weighted logical-output checksum is not finite and positive: '
            f'{value_weighted!r}',
        )
    weighted_bound_tolerance = (
        1.0e-6 * max(value_l1, value_weighted, 1.0)
    )
    if (
        value_weighted < value_l1 - weighted_bound_tolerance
        or value_weighted > 2.0 * value_l1 + weighted_bound_tolerance
    ):
        invalid(
            log_path,
            'weighted logical L1 checksum is outside its valid '
            '[L1, 2*L1] bounds',
        )
    return value, value_l1, value_weighted

def best_run(w, var, vl):
    best, bg = None, -1
    for rd in candidates(w, var, vl):
        g = gflops_of(rd)
        if g is not None and g > bg:
            best, bg = rd, g
    return best, bg

def energy_parts(rd, is_cf, vl):
    """-> (cache_mJ incl. idle, spm_mJ, dram_mJ) from the run's ROI stats."""
    p = os.path.join(rd, 'stats.txt')
    if not os.path.isfile(p) or os.path.getsize(p) == 0:
        invalid(p, 'missing or empty stats file')
    with open(p, errors='ignore') as handle:
        text = handle.read()
    parts = text.split(STATS_BEGIN)
    if len(parts) < 2:
        invalid(p, 'ROI statistics begin marker missing')
    roi_parts = parts[1].split(STATS_END, 1)
    if len(roi_parts) != 2:
        invalid(p, 'ROI statistics end marker missing')
    roi = roi_parts[0]  # m5_reset -> m5_dump; later dumps are post-ROI

    def g(name, *, required=True):
        matches = re.findall(
            rf'^{re.escape(name)}\s+(\S+)', roi, re.MULTILINE
        )
        if not matches:
            if not required:
                return 0.0
            invalid(p, f'missing ROI statistic {name}')
        if len(matches) != 1:
            invalid(p, f'expected one ROI statistic {name}, found {len(matches)}')
        try:
            value = float(matches[0])
        except ValueError:
            invalid(p, f'ROI statistic {name} is not numeric: {matches[0]!r}')
        if not math.isfinite(value) or value < 0:
            invalid(
                p,
                f'ROI statistic {name} must be finite and non-negative, '
                f'got {value!r}',
            )
        return value

    sim_ticks = g('simTicks')
    sim_freq = g('simFreq')
    if sim_ticks <= 0 or sim_freq <= 0:
        invalid(
            p,
            f'simTicks and simFreq must be positive, got '
            f'{sim_ticks!r} and {sim_freq!r}',
        )

    l1_read_req = g('system.cpu.dcache.ReadReq.accesses::total')
    # A zero-valued ReadSharedReq stat is omitted by the baseline gem5 dump.
    # CacheFlex must report it because CPTOSPM source reads use this command.
    l1_read_shared = g(
        'system.cpu.dcache.ReadSharedReq.accesses::total', required=is_cf
    )
    l1_write_req = g('system.cpu.dcache.WriteReq.accesses::total')
    l1_write_line = g('system.cpu.dcache.WriteLineReq.accesses::total')
    l2_reads = g('system.l2.overallAccesses::total')
    l2_writes = g('system.l2.WritebackDirty.accesses::total')
    l3_reads = g('system.l3.overallAccesses::total')
    l3_writes = g('system.l3.WritebackDirty.accesses::total')
    spm_reads = g('system.l2.spmReads')
    spm_writes = g('system.l2.spmWrites')
    dram_energy = [
        g(f'system.mem_ctrls{controller}.dram.rank{rank}.totalEnergy')
        for controller in (0, 1)
        for rank in (0, 1)
    ]

    cache = (((l1_read_req + l1_read_shared) * E['l1_read']) +
             ((l1_write_req + l1_write_line) * E['l1_write']) +
             l2_reads * E['l2_read'] +
             l2_writes * E['l2_write'] +
             l3_reads * E['l3_read'] +
             l3_writes * E['l3_write']) / 1e6
    spm = (spm_reads * E['spm_read'] * (vl // 4) +
           spm_writes * E['spm_write']) / 1e6
    t = sim_ticks / sim_freq
    idle = (IDLE_L1 + (IDLE_L2_CF if is_cf else IDLE_L2_BL) + IDLE_L3) * t
    dram = sum(dram_energy) / 1e9
    result = (cache + idle, spm, dram)
    if any(not math.isfinite(value) or value < 0 for value in result):
        invalid(p, f'computed energy is not finite and non-negative: {result!r}')
    return result

def build_data():
    """DATA[vl] = [(bl_gf, cf_gf, bl_cache, bl_dram, cf_cache, cf_spm, cf_dram)]"""
    result_dir = os.path.join(R, RESULT_SUBDIR)
    if not os.path.isdir(result_dir):
        raise SystemExit(f'ERROR: missing results dir: {result_dir}')
    data = {}
    for vl in VLS:
        rows = []
        for w in WS:
            r1, g1 = best_run(w, 'v1', vl)
            r3, g3 = best_run(w, 'v3', vl)
            if r1 is None or r3 is None:
                var = 'v1' if r1 is None else 'v3'
                raise SystemExit(f'ERROR: missing run {w}_{var}_vl{vl} under '
                                 f'{result_dir}')
            c1, c1_l1, c1_weighted = logical_checksum_of(r1)
            c3, c3_l1, c3_weighted = logical_checksum_of(r3)
            checksum_tolerance = 0.015 * max(abs(c1), abs(c3)) + 1.0e-9
            if abs(c1 - c3) > checksum_tolerance:
                invalid(
                    result_dir,
                    f'{w}/VL{vl} baseline and CacheFlex logical checksums '
                    f'differ: {c1!r} vs {c3!r}',
                )
            l1_tolerance = 0.015 * max(c1_l1, c3_l1) + 1.0e-9
            if abs(c1_l1 - c3_l1) > l1_tolerance:
                invalid(
                    result_dir,
                    f'{w}/VL{vl} baseline and CacheFlex logical L1 '
                    f'checksums differ: {c1_l1!r} vs {c3_l1!r}',
                )
            weighted_tolerance = (
                0.015 * max(abs(c1_weighted), abs(c3_weighted)) + 1.0e-9
            )
            if abs(c1_weighted - c3_weighted) > weighted_tolerance:
                invalid(
                    result_dir,
                    f'{w}/VL{vl} baseline and CacheFlex weighted logical '
                    f'checksums differ: {c1_weighted!r} vs '
                    f'{c3_weighted!r}',
                )
            for configuration, result_path, gflops in (
                ('baseline', r1, g1),
                ('cacheflex', r3, g3),
            ):
                relative = os.path.relpath(result_path, R)
                SOURCE_CELLS.append({
                    "workload": w,
                    "vector_length": vl,
                    "configuration": configuration,
                    "cell": os.path.basename(result_path),
                    "result_path": relative,
                    "log_path": relative + ".log",
                    "stats_path": os.path.join(relative, "stats.txt"),
                    "gflops": gflops,
                    "logical_checksum": (
                        c1 if configuration == 'baseline' else c3
                    ),
                    "logical_checksum_l1": (
                        c1_l1 if configuration == 'baseline' else c3_l1
                    ),
                    "logical_checksum_weighted": (
                        c1_weighted
                        if configuration == 'baseline'
                        else c3_weighted
                    ),
                })
            bc, bs, bd = energy_parts(r1, False, vl)
            cc, cs, cd = energy_parts(r3, True, vl)
            rows.append((g1, g3, bc, bd, cc, cs, cd))
        data[vl] = rows
    return data

# ── Figure style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'text.usetex': False, 'font.size': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.bbox': None, 'savefig.pad_inches': 0,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': '#555555',
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'text.color': '#333333',
})
C_BL, C_CF, C_GAIN = '#4878A8', '#E8853B', '#C44E52'
C_CACHE, C_SPM, C_DRAM = '#6BAF6B', '#8C6BB1', '#A0A0A0'
ids = WS
n = len(ids)
PAPER_WIDTH_IN = 734.34 / 72.0
PAPER_HEIGHT_IN = 320.874 / 72.0


def write_data(DATA):
    os.makedirs(OUTDIR, exist_ok=True)
    payload = {
        "schema_version": 1,
        "figure": "Figure 7",
        "units": {"performance": "GFLOPS", "energy": "mJ"},
        "workloads": WS,
        "vector_lengths": VLS,
        "source_cells": SOURCE_CELLS,
        "data": {},
    }
    for vl in VLS:
        rows = {}
        speedups = []
        energy_ratios = []
        for workload, row in zip(WS, DATA[vl]):
            bl_gf, cf_gf, bl_cache, bl_dram, cf_cache, cf_spm, cf_dram = row
            bl_total = bl_cache + bl_dram
            cf_total = cf_cache + cf_spm + cf_dram
            speedup = cf_gf / bl_gf
            energy_ratio = cf_total / bl_total
            speedups.append(speedup)
            energy_ratios.append(energy_ratio)
            rows[workload] = {
                "baseline_gflops": bl_gf,
                "cacheflex_gflops": cf_gf,
                "baseline_utilization_pct": bl_gf / PEAK[vl] * 100.0,
                "cacheflex_utilization_pct": cf_gf / PEAK[vl] * 100.0,
                "speedup": speedup,
                "baseline_energy_mJ": {
                    "cache": bl_cache,
                    "dram": bl_dram,
                    "total": bl_total,
                },
                "cacheflex_energy_mJ": {
                    "cache": cf_cache,
                    "spm": cf_spm,
                    "dram": cf_dram,
                    "total": cf_total,
                },
                "energy_reduction_pct": (1.0 - energy_ratio) * 100.0,
            }
        payload["data"][str(vl)] = {
            "rows": rows,
            "gmean_speedup": float(np.exp(np.mean(np.log(speedups)))),
            "gmean_energy_reduction_pct": float(
                (1.0 - np.exp(np.mean(np.log(energy_ratios)))) * 100.0
            ),
        }
    with open(os.path.join(OUTDIR, "fig7_data.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def render(DATA, out_base):
    fig, axes = plt.subplots(
        2, 3, figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN),
        gridspec_kw={'wspace': 0.20, 'hspace': 0.18}
    )
    w = 0.36
    for column, vl in enumerate(VLS):
        ax_perf, ax_energy = axes[0, column], axes[1, column]
        peak, d = PEAK[vl], DATA[vl]
        bl_perf = np.array([r[0] for r in d]); cf_perf = np.array([r[1] for r in d])
        bl_cache = np.array([r[2] for r in d]); bl_dram = np.array([r[3] for r in d])
        cf_cache = np.array([r[4] for r in d]); cf_spm = np.array([r[5] for r in d])
        cf_dram = np.array([r[6] for r in d])
        bl_total = bl_cache + bl_dram
        cf_total = cf_cache + cf_spm + cf_dram
        spd_ratio = cf_perf / bl_perf
        e_red = (cf_total - bl_total) / bl_total * 100
        gmean_spd = np.exp(np.mean(np.log(spd_ratio)))
        gmean_e_ratio = np.exp(np.mean(np.log(cf_total / bl_total)))
        x = np.arange(n + 1)
        bl_util = bl_perf / peak * 100; cf_util = cf_perf / peak * 100
        gm_bl_u = np.exp(np.mean(np.log(bl_util)))
        gm_cf_u = np.exp(np.mean(np.log(cf_util)))
        all_bl = np.append(bl_util, gm_bl_u); all_cf = np.append(cf_util, gm_cf_u)
        all_spd = np.append(spd_ratio, gmean_spd)
        ax_perf.bar(x - w/2, all_bl, w, color=C_BL, edgecolor='white', linewidth=0.4)
        ax_perf.bar(x + w/2, all_cf, w, color=C_CF, edgecolor='white', linewidth=0.4)
        for i in range(len(x)):
            ax_perf.text(x[i], max(all_bl[i], all_cf[i]) + 1.5,
                         f'{all_spd[i]:.2f}x', ha='center', va='bottom',
                         fontsize=6.5, fontweight='bold', color=C_GAIN)
        sep_x = (x[n-1] + x[-1]) / 2
        ax_perf.axvline(x=sep_x, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        if column == 0:
            ax_perf.set_ylabel('SIMD Util. (%)', fontsize=9, fontweight='bold',
                               labelpad=3.3)
        ax_perf.set_ylim(0, 108); ax_perf.set_xticks(x)
        ax_perf.set_xticklabels(ids + ['Gmean'], fontsize=8)
        ax_perf.set_title(f'VL={vl} ({vl * 128}-bit)', fontsize=9,
                          fontweight='bold', pad=6)
        ax_perf.set_axisbelow(True)
        ax_perf.grid(axis='y', color='#B0B0B0', linestyle='-',
                     linewidth=0.4, alpha=0.28)
        ax_perf.spines['top'].set_visible(False); ax_perf.spines['right'].set_visible(False)
        ax_perf.spines['left'].set_color('#555555')
        ax_perf.spines['bottom'].set_color('#555555')
        ax_perf.spines['left'].set_linewidth(0.6)
        ax_perf.spines['bottom'].set_linewidth(0.6)
        ax_perf.tick_params(axis='both', labelsize=8, colors='#333333',
                            width=0.6, pad=3.0)
        ax_perf.set_xlim(-0.6, x[-1] + 0.6)
        bl_cache_n = bl_cache / bl_total; bl_dram_n = bl_dram / bl_total
        cf_cache_n = cf_cache / bl_total; cf_spm_n = cf_spm / bl_total
        cf_dram_n = cf_dram / bl_total
        gm_bl_c = np.mean(bl_cache_n); gm_bl_d = 1.0 - gm_bl_c
        gm_cf_c = np.mean(cf_cache_n); gm_cf_s = np.mean(cf_spm_n); gm_cf_d = np.mean(cf_dram_n)
        # Keep the mean component proportions, but make the Gmean stack's
        # total equal the geometric-mean ratio used by the label and JSON.
        gm_cf_sum = gm_cf_c + gm_cf_s + gm_cf_d
        if gm_cf_sum <= 0:
            raise SystemExit(f'ERROR: non-positive Gmean energy components at VL={vl}')
        gm_scale = gmean_e_ratio / gm_cf_sum
        gm_cf_c *= gm_scale; gm_cf_s *= gm_scale; gm_cf_d *= gm_scale
        all_bl_c = np.append(bl_cache_n, gm_bl_c); all_bl_d = np.append(bl_dram_n, gm_bl_d)
        all_cf_c = np.append(cf_cache_n, gm_cf_c); all_cf_s = np.append(cf_spm_n, gm_cf_s)
        all_cf_d = np.append(cf_dram_n, gm_cf_d)
        cf_cs = all_cf_c + all_cf_s; cf_d_bot = 1.0 - all_cf_d; gap_h = cf_d_bot - cf_cs
        all_e_red = np.append(e_red, (gmean_e_ratio - 1) * 100)
        ax_energy.bar(x - w/2, all_bl_c, w, color=C_CACHE, edgecolor='white', linewidth=0.4)
        ax_energy.bar(x - w/2, all_bl_d, w, bottom=all_bl_c, color=C_DRAM, edgecolor='white', linewidth=0.4)
        ax_energy.bar(x + w/2, all_cf_c, w, color=C_CACHE, edgecolor='white', linewidth=0.4)
        ax_energy.bar(x + w/2, all_cf_s, w, bottom=all_cf_c, color=C_SPM, edgecolor='white', linewidth=0.4)
        bars_gap = ax_energy.bar(x + w/2, gap_h, w, bottom=cf_cs,
                                 color='white', edgecolor='#CCCCCC', linewidth=0.4)
        for bar in bars_gap:
            bar.set_hatch('///')
        ax_energy.bar(x + w/2, all_cf_d, w, bottom=cf_d_bot, color=C_DRAM, edgecolor='white', linewidth=0.4)
        for i in range(len(x)):
            rh = gap_h[i]; mid = cf_cs[i] + rh / 2
            if rh > 0.10:
                ax_energy.text(x[i] + w/2, mid, f'{all_e_red[i]:.0f}%', ha='center',
                               va='center', fontsize=6.5, fontweight='bold', rotation=90,
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                         edgecolor='none', alpha=0.8))
            elif rh > 0.04:
                ax_energy.text(x[i] + w/2, mid, f'{all_e_red[i]:.0f}%', ha='center',
                               va='center', fontsize=5.5, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.05', facecolor='white',
                                         edgecolor='none', alpha=0.8))
        ax_energy.axvline(x=sep_x, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        if column == 0:
            ax_energy.set_ylabel('Norm. Energy', fontsize=9, fontweight='bold',
                                 labelpad=3.3)
        ax_energy.set_ylim(0, 1.15); ax_energy.set_xticks(x)
        ax_energy.set_xticklabels(ids + ['Gmean'], fontsize=8)
        ax_energy.set_axisbelow(True)
        ax_energy.grid(axis='y', color='#B0B0B0', linestyle='-',
                       linewidth=0.4, alpha=0.28)
        ax_energy.spines['top'].set_visible(False); ax_energy.spines['right'].set_visible(False)
        ax_energy.spines['left'].set_color('#555555')
        ax_energy.spines['bottom'].set_color('#555555')
        ax_energy.spines['left'].set_linewidth(0.6)
        ax_energy.spines['bottom'].set_linewidth(0.6)
        ax_energy.tick_params(axis='both', labelsize=8, colors='#333333',
                              width=0.6, pad=3.0)
        ax_energy.set_xlim(-0.6, x[-1] + 0.6)
    fig.legend(
        handles=[
            Patch(facecolor=C_BL, edgecolor='white', linewidth=0.4, label='Baseline'),
            Patch(facecolor=C_CF, edgecolor='white', linewidth=0.4, label='CacheFlex'),
            Patch(facecolor=C_CACHE, edgecolor='white', linewidth=0.4, label='Cache'),
            Patch(facecolor=C_SPM, edgecolor='white', linewidth=0.4, label='SPM'),
            Patch(facecolor=C_DRAM, edgecolor='white', linewidth=0.4, label='DRAM'),
            Patch(facecolor='white', edgecolor='#CCCCCC', linewidth=0.4,
                  hatch='///', label='Reduced'),
        ],
        loc='upper center', bbox_to_anchor=(0.459, 1.006), ncol=6,
        frameon=False, fontsize=8, handlelength=1.5, handletextpad=0.4,
        columnspacing=1.4
    )
    fig.text(0.7142, 0.971, 'speedup', color=C_GAIN, fontsize=8,
             fontweight='bold', ha='left', va='center')
    fig.subplots_adjust(left=0.050, right=0.997, top=0.856, bottom=0.051,
                        wspace=0.20, hspace=0.18)
    fig.savefig(out_base + '.pdf', bbox_inches=None, pad_inches=0)
    fig.savefig(out_base + '.png', dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f'Saved: {out_base}.png / .pdf')

_data = build_data()
write_data(_data)
os.makedirs(OUTDIR, exist_ok=True)
render(_data, os.path.join(OUTDIR, 'fig7_real'))
