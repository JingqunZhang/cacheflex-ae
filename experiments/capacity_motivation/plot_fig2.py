#!/usr/bin/env python3
"""Fig 2 rendered from the canonical real runs (no hardcoded data).

Style uses the broken y-axis and %-of-peak labels of the publication figure.
Data: best-of-KC GFLOPS per (L1 capacity, bandwidth, VL) from
results_fig2_n2048_headh0/ (cache) and results_spm_fused_n2048_headh0/ (CacheFlex); paper shape 256x2048x2048.
Missing runs are listed, not invented.
"""
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

FIGDIR = Path(__file__).resolve().parent
OUTDIR = Path(os.environ.get("OUTPUT_DIR", FIGDIR)).resolve()
# Keep the publication asset's physical canvas so fig2.pdf can replace
# images/f2.pdf without changing the paper layout.
PAPER_WIDTH_IN = 677.376 / 72.0
PAPER_HEIGHT_IN = 357.3785 / 72.0
# Results root: CLI arg or RESULTS_DIR env var override; default = script dir
# (where the results_* compat symlinks / dirs live, e.g. results/reference).
if len(sys.argv) > 1:
    R = Path(sys.argv[1]).resolve()
elif os.environ.get('RESULTS_DIR'):
    R = Path(os.environ['RESULTS_DIR']).resolve()
else:
    R = FIGDIR
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
GP = re.compile(rf'(?<![\w.])({NUMBER})\s*GFLOPS\b')
GFLOPS_TOKEN = re.compile(r'\bGFLOPS\b')
LOGICAL_CHECKSUM = re.compile(
    rf'(?m)^\s*CHECKSUM_LOGICAL:\s*({NUMBER})\s*$'
)
LOGICAL_CHECKSUM_L1 = re.compile(
    rf'(?m)^\s*CHECKSUM_LOGICAL_L1:\s*({NUMBER})\s*$'
)
LOGICAL_CHECKSUM_WEIGHTED = re.compile(
    rf'(?m)^\s*CHECKSUM_LOGICAL_WEIGHTED_L1:\s*({NUMBER})\s*$'
)
BAD_LOG = re.compile(
    rb"(?:\bpanic:|\bfatal:|segmentation fault|\bsegfault\b|"
    rb"core dumped|\bassert(?:ion)?\b[^\r\n]{0,256}\bfailed\b|"
    rb"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
    rb"\[DBG\])",
    re.IGNORECASE,
)
CLEAN_EXIT = b'Exiting @ tick'
STATS_BEGIN = re.compile(
    r'(?m)^-+\s+Begin Simulation Statistics\s+-+\s*$'
)
STATS_END = re.compile(
    r'(?m)^-+\s+End Simulation Statistics\s+-+\s*$'
)
VLS = [4, 8, 16]
peaks = [320, 640, 1280]
vl_labels = ['512-bit (VL=4)', '1024-bit (VL=8)', '2048-bit (VL=16)']
BEST_PATH = FIGDIR / 'best.json'
if not BEST_PATH.is_file():
    raise SystemExit(f'ERROR: missing fixed selection: {BEST_PATH}')
_best_entries = json.loads(BEST_PATH.read_text(encoding='utf-8'))['cells']
SELECTED_NAMES = [entry['name'] for entry in _best_entries]
if len(SELECTED_NAMES) != len(set(SELECTED_NAMES)):
    raise SystemExit(f'ERROR: duplicate cell name in {BEST_PATH}')
SOURCE_CELLS = []
OUTPUT_CHECKSUMS = []


def _relative_result(dirname, name):
    return name if dirname == '.' else f'{dirname}/{name}'


def _require_first_roi_stats(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f'ERROR: missing/empty stats file: {path}')
    text = path.read_text(encoding='utf-8', errors='replace')
    begin = STATS_BEGIN.search(text)
    if begin is None:
        raise SystemExit(f'ERROR: no ROI statistics begin marker: {path}')
    end = STATS_END.search(text, begin.end())
    if end is None:
        raise SystemExit(f'ERROR: incomplete first ROI statistics block: {path}')
    nested_begin = STATS_BEGIN.search(text, begin.end(), end.start())
    if nested_begin is not None:
        raise SystemExit(f'ERROR: malformed first ROI statistics block: {path}')
    block = text[begin.end():end.start()]
    for key in ('simTicks', 'simFreq'):
        values = re.findall(rf'(?m)^{key}\s+(\S+)', block)
        if len(values) != 1:
            raise SystemExit(
                f'ERROR: first ROI block in {path} must contain exactly one '
                f'{key}; found {len(values)}'
            )
        try:
            value = float(values[0])
        except ValueError:
            raise SystemExit(
                f'ERROR: non-numeric {key} in first ROI block: {path}'
            ) from None
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(
                f'ERROR: {key} must be finite and positive in first ROI '
                f'block: {path}'
            )


def gflops(path):
    path = Path(path)
    stats = path.with_suffix('') / 'stats.txt'
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f'ERROR: missing/empty result log: {path}')
    data = path.read_bytes()
    if CLEAN_EXIT not in data:
        raise SystemExit(f'ERROR: no clean gem5 exit in result log: {path}')
    if BAD_LOG.search(data):
        raise SystemExit(
            f'ERROR: crash, nonzero exit, validation, self-check, debug, or '
            f'assertion marker in result log: {path}'
        )
    text = data.decode('utf-8', errors='replace')
    matches = GP.findall(text)
    token_count = len(GFLOPS_TOKEN.findall(text))
    if token_count != 1 or len(matches) != 1:
        raise SystemExit(
            f'ERROR: result log {path} must contain exactly one finite '
            f'numeric GFLOPS value; found {token_count} GFLOPS markers and '
            f'{len(matches)} numeric values'
        )
    value = float(matches[0])
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(
            f'ERROR: GFLOPS must be finite and positive in result log: {path}'
        )
    _require_first_roi_stats(stats)
    logical = LOGICAL_CHECKSUM.findall(text)
    logical_l1 = LOGICAL_CHECKSUM_L1.findall(text)
    logical_weighted = LOGICAL_CHECKSUM_WEIGHTED.findall(text)
    if (
        len(logical) != 1
        or len(logical_l1) != 1
        or len(logical_weighted) != 1
    ):
        raise SystemExit(
            f'ERROR: result log {path} must contain exactly one logical '
            f'sum, one L1 checksum, and one weighted checksum; found '
            f'{len(logical)}, {len(logical_l1)}, and '
            f'{len(logical_weighted)}'
        )
    checksum = float(logical[0])
    checksum_l1 = float(logical_l1[0])
    checksum_weighted = float(logical_weighted[0])
    if not math.isfinite(checksum):
        raise SystemExit(
            f'ERROR: logical checksum must be finite in result log: {path}'
        )
    if not math.isfinite(checksum_l1) or checksum_l1 <= 0:
        raise SystemExit(
            f'ERROR: logical L1 checksum must be finite and positive in '
            f'result log: {path}'
        )
    if not math.isfinite(checksum_weighted) or checksum_weighted <= 0:
        raise SystemExit(
            f'ERROR: weighted logical checksum must be finite and positive '
            f'in result log: {path}'
        )
    weighted_bound_tolerance = (
        1.0e-6 * max(checksum_l1, checksum_weighted, 1.0)
    )
    if (
        checksum_weighted < checksum_l1 - weighted_bound_tolerance
        or checksum_weighted
        > 2.0 * checksum_l1 + weighted_bound_tolerance
    ):
        raise SystemExit(
            f'ERROR: weighted logical L1 checksum is outside its valid '
            f'[L1, 2*L1] bounds in result log: {path}'
        )
    return value, checksum, checksum_l1, checksum_weighted

def best_cache(dirname, vl, cap, ld):
    """Read the unique cache cell selected for this configuration."""
    pattern = re.compile(
        rf'fused_vl{vl}_l1{cap}k4w_ld{ld}st\d+_kc\d+_mc\d+'
    )
    names = [name for name in SELECTED_NAMES if pattern.fullmatch(name)]
    if len(names) != 1:
        raise SystemExit(
            f'ERROR: {BEST_PATH} must select exactly one cache cell for '
            f'VL={vl}, L1={cap}KB, load_ports={ld}; found {names}'
        )
    value, checksum, checksum_l1, checksum_weighted = gflops(
        R / dirname / f'{names[0]}.log'
    )
    OUTPUT_CHECKSUMS.append(
        (names[0], checksum, checksum_l1, checksum_weighted)
    )
    SOURCE_CELLS.append({
        'configuration': 'cache',
        'vector_length': vl,
        'l1_capacity_kib': cap,
        'load_ports': ld,
        'cell': names[0],
        'result_path': _relative_result(dirname, names[0]),
        'gflops': value,
        'logical_checksum': checksum,
        'logical_checksum_l1': checksum_l1,
        'logical_checksum_weighted': checksum_weighted,
    })
    return value

def best_spm(dirname, vl):
    """Read the unique CacheFlex cell selected for this vector length."""
    pattern = re.compile(rf'v3fused_vl{vl}_kc\d+_mc\d+')
    names = [name for name in SELECTED_NAMES if pattern.fullmatch(name)]
    if len(names) != 1:
        raise SystemExit(
            f'ERROR: {BEST_PATH} must select exactly one CacheFlex cell for '
            f'VL={vl}; found {names}'
        )
    value, checksum, checksum_l1, checksum_weighted = gflops(
        R / dirname / f'{names[0]}.log'
    )
    OUTPUT_CHECKSUMS.append(
        (names[0], checksum, checksum_l1, checksum_weighted)
    )
    SOURCE_CELLS.append({
        'configuration': 'cacheflex',
        'vector_length': vl,
        'cell': names[0],
        'result_path': _relative_result(dirname, names[0]),
        'gflops': value,
        'logical_checksum': checksum,
        'logical_checksum_l1': checksum_l1,
        'logical_checksum_weighted': checksum_weighted,
    })
    return value

def build(cache_dir, spm_dir):
    """-> configs list in template order; None entries mark missing data."""
    for d in (cache_dir, spm_dir):
        if not (R / d).is_dir():
            raise SystemExit(f'ERROR: missing results dir: {R / d}')
    series = []
    for cap, color in [(64, '#4878A8'), (256, '#E8853B'), (512, '#6BAF6B')]:
        for ld, bw, hatch in [(2, 320, None), (4, 640, '///')]:
            vals = [best_cache(cache_dir, vl, cap, ld) for vl in VLS]
            series.append((f'{cap} KB, {bw} GB/s', vals, color, hatch))
    series.append(('CacheFlex (64KB L1; 320(L1)+320(SPM) GB/s)',
                   [best_spm(spm_dir, vl) for vl in VLS], '#8C6BB1', 'xx'))
    (
        reference_name,
        reference_sum,
        reference_l1,
        reference_weighted,
    ) = OUTPUT_CHECKSUMS[0]
    for name, value_sum, value_l1, value_weighted in OUTPUT_CHECKSUMS[1:]:
        sum_tolerance = (
            0.015 * max(abs(reference_sum), abs(value_sum)) + 1.0e-9
        )
        if abs(reference_sum - value_sum) > sum_tolerance:
            raise SystemExit(
                f'ERROR: Figure 2 logical checksum mismatch: '
                f'{reference_name}={reference_sum!r} vs {name}={value_sum!r}'
            )
        l1_tolerance = 0.015 * max(reference_l1, value_l1) + 1.0e-9
        if abs(reference_l1 - value_l1) > l1_tolerance:
            raise SystemExit(
                f'ERROR: Figure 2 logical L1 checksum mismatch: '
                f'{reference_name}={reference_l1!r} vs {name}={value_l1!r}'
            )
        weighted_tolerance = (
            0.015
            * max(abs(reference_weighted), abs(value_weighted))
            + 1.0e-9
        )
        if abs(reference_weighted - value_weighted) > weighted_tolerance:
            raise SystemExit(
                f'ERROR: Figure 2 weighted logical checksum mismatch: '
                f'{reference_name}={reference_weighted!r} vs '
                f'{name}={value_weighted!r}'
            )
    series.append(('Theoretical Peak', list(map(float, peaks)), '#A0A0A0', None))
    return series

def write_data(configs):
    payload = {
        "schema_version": 1,
        "figure": "Figure 2",
        "units": {"performance": "GFLOPS"},
        "vector_lengths": VLS,
        "theoretical_peak_gflops": peaks,
        "series": [
            {
                "label": label,
                "gflops": values,
                "percent_of_peak": [
                    round(value / peaks[index] * 100.0, 6)
                    for index, value in enumerate(values)
                ],
            }
            for label, values, _, _ in configs
            if not label.startswith("Theoretical")
        ],
        "source_cells": SOURCE_CELLS,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "fig2_data.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def render(configs, out_base):
    missing = [(lbl, VLS[i]) for lbl, vals, _, _ in configs
               for i, v in enumerate(vals) if v is None]
    if missing:
        print(f'[{out_base}] waiting for {len(missing)} runs:',
              ', '.join(f'{l}@VL{v}' for l, v in missing[:8]),
              '...' if len(missing) > 8 else '')
        return False
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 14, 'axes.labelsize': 15,
                         'legend.fontsize': 14, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'axes.linewidth': 0.6,
                         'figure.dpi': 300, 'savefig.dpi': 300,
                         'savefig.bbox': None, 'savefig.pad_inches': 0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': '#555555',
                         'xtick.color': '#333333', 'ytick.color': '#333333',
                         'text.color': '#333333'})
    n_configs = len(configs)
    x_groups = np.arange(len(vl_labels))
    # These limits and the 0.111 width reproduce the publication asset's
    # group centers and contiguous eight-bar groups exactly.
    bar_width = 0.111
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN))
    gs = GridSpec(
        2, 1, figure=fig, height_ratios=[0.90, 3.0], hspace=0.079,
        left=0.072865, right=0.996811, bottom=0.114, top=0.994
    )
    ax_top, ax_bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in [ax_top, ax_bot]:
        for j, (label, vals, color, hatch) in enumerate(configs):
            offset = (j - (n_configs - 1) / 2) * bar_width
            if label.startswith('Theoretical'):
                alpha, ec, lw = 0.35, 'white', 0.3
            elif hatch == '///':
                alpha, ec, lw = 0.5, 'white', 0.3
            else:
                alpha, ec, lw = 1.0, 'white', 0.3
            ax.bar(x_groups + offset, vals, bar_width,
                   label=label if ax is ax_bot else None, color=color,
                   alpha=alpha, edgecolor=ec, linewidth=lw, hatch=hatch, zorder=3)
    ax_bot.set_ylim(150, 660); ax_top.set_ylim(1240, 1310)
    ax_bot.set_xlim(-0.588, 2.588); ax_top.set_xlim(-0.588, 2.588)
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_top.yaxis.set_major_locator(ticker.FixedLocator([1280]))
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)
    ax_bot.tick_params(axis='both', pad=3.1)
    ax_top.tick_params(axis='y', pad=3.1)
    ax_top.set_zorder(2)
    ax_top.patch.set_visible(False)
    d = 0.008
    kw = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d*3, +d*3), **kw)
    ax_top.plot((1-d, 1+d), (-d*3, +d*3), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1-d, 1+d), **kw)
    ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kw)
    for ax in [ax_top, ax_bot]:
        ax.set_axisbelow(True)
        ax.grid(True, which='major', color='#B0B0B0', alpha=0.28,
                linewidth=0.4, axis='y')
    for j, (label, vals, color, hatch) in enumerate(configs):
        offset = (j - (n_configs - 1) / 2) * bar_width
        for i, v in enumerate(vals):
            if label.startswith('Theoretical'):
                tgt = ax_top if v > 650 else ax_bot
                tgt.text(x_groups[i] + offset, v + (3 if v > 660 else 5),
                         f'{v:.0f}', ha='center', va='bottom', fontsize=10,
                         color='#555555', fontstyle='italic')
            else:
                ax_bot.text(x_groups[i] + offset, min(v, 640) + 8,
                            f'{v/peaks[i]*100:.0f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color=color, rotation=90)
    ax_bot.set_xticks(x_groups); ax_bot.set_xticklabels(vl_labels)
    ax_bot.set_xlabel('SIMD Width', fontweight='bold', labelpad=3.5)
    ax_top.set_xticks(x_groups); ax_top.set_xticklabels([])
    fig.text(0.0142, 0.554, 'GFLOPS', ha='center', va='center',
             rotation='vertical', fontsize=15, fontweight='bold',
             color='#000000')
    h, l = ax_bot.get_legend_handles_labels()
    legend = ax_top.legend(
        h, l, loc='upper left', bbox_to_anchor=(0.023, 1.04),
        frameon=False, ncol=2, columnspacing=0.45,
        handletextpad=0.35, labelspacing=0.465, fontsize=14
    )
    legend.set_zorder(20)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTDIR / f'{out_base}.png'), dpi=300,
                bbox_inches=None, pad_inches=0)
    plt.savefig(str(OUTDIR / f'{out_base}.pdf'),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f'Saved: {out_base}.png / .pdf')
    return True

if (R / 'results_fig2_n2048_headh0').is_dir():
    cache_dir, spm_dir = ('results_fig2_n2048_headh0',
                          'results_spm_fused_n2048_headh0')
else:
    # A default AE run writes both disjoint filename families into one flat
    # results/generated/best directory.
    cache_dir = spm_dir = '.'
_configs = build(cache_dir, spm_dir)
_ok = render(_configs, 'fig2')
if not _ok:
    raise SystemExit(f'ERROR: missing runs under {R} (listed above) — figure not rendered')
write_data(_configs)
