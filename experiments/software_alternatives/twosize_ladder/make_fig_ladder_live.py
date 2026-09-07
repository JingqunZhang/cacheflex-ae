#!/usr/bin/env python3
"""Render Figure 8 from gem5 result cells.

The figure shows runtime and energy for 256x2048x2048 and 2048x2048x2048,
each normalized to its cache baseline. The 12 cells are selected by best.json.
Times and checksums come from stdout.txt; energy comes from stats.txt.

Runtime rows show speedup over the cache baseline; energy rows show percentage
change. Missing or invalid inputs abort before outputs are written.

Default outputs (in experiments/software_alternatives/):
  fig8.{pdf,png}
  fig8_data.json
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import configparser
import os, sys, json, re, importlib.util, shlex
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'text.usetex': False,
    'font.size': 7,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': None,
    'savefig.pad_inches': 0,
})

# Plot colors.
C_STAGE   = '#A0A0A0'   # runtime: data staging (pack)
C_COMPUTE = '#6BAF6B'   # runtime: compute (kernel + C)
C_CACHE   = '#6BAF6B'   # energy: cache dyn + idle
C_SPM     = '#8C6BB1'   # energy: SPM dynamic
C_DRAM    = '#A0A0A0'   # energy: DRAM
C_GAIN    = '#C44E52'   # red speedup / reduction labels
C_HI      = '#EAF5EC'   # CacheFlex / SPM row highlight

TL   = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.abspath(os.environ.get("OUTPUT_DIR", os.path.dirname(TL)))
ROOT = os.environ.get("CACHEFLEX_ROOT",
                      str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                               if (p / "setup_env.sh").exists())))
# --check validates the selected result directory without writing outputs.
CHECK_ONLY = '--check' in sys.argv[1:]
POSITIONAL = [argument for argument in sys.argv[1:] if argument != '--check']
if len(POSITIONAL) > 1:
    raise SystemExit(
        "usage: make_fig_ladder_live.py [RESULTS_DIR] [--check]"
    )
if POSITIONAL:
    DATA = os.path.abspath(POSITIONAL[0])
elif os.environ.get("RESULTS_DIR"):
    DATA = os.path.abspath(os.environ["RESULTS_DIR"])
else:
    DATA = TL
BEST_JSON = os.path.abspath(
    os.environ.get(
        "BEST_JSON",
        os.path.join(os.path.dirname(TL), 'best.json'),
    )
)

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

pc = _load(os.path.join(TL, "parse_cell.py"), "parse_cell")
ce = _load(os.path.join(ROOT, "scripts/energy_model.py"),
           "compute_energy")

ROW_LABELS = {
    'row1_cacheopt':    'HW PF (best)',
    'row2_pfoff':       'no-PF',
    'row3_pldl2keep':   'SW-L2PF',
    'row4_pldl2keepnt': 'L2 Pin + NT loads',
    'row5_spm_shbw':    'CacheFlex: shared L2 BW',
    'row6_cacheflex':   'CacheFlex: addl. L2 BW',
}
ROW_ORDER  = list(ROW_LABELS)
CF_ROWS    = {'row5_spm_shbw', 'row6_cacheflex'}
SIZES      = [('w3', '256×2048×2048'), ('g2048', '2048×2048×2048')]
SIZE_ORDER = ['w3', 'g2048']
# Fixed output canvas keeps the PDF size stable across Matplotlib versions.
PAPER_WIDTH_IN = 296.85 / 72.0
PAPER_HEIGHT_IN = 391.84 / 72.0

MISSING = []
BAD_RESULT = re.compile(
    r"panic:|fatal:|segmentation fault|segfault|core dumped|"
    r"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
    r"\[DBG\]|"
    r"assert(?:ion)?[^\n]{0,256}failed",
    re.I,
)


def _miss(msg):
    MISSING.append(msg)
    print(f"MISSING: {msg}")


ROW_PREFIX = {
    'cacheopt': 'row1_cacheopt',
    'pfoff': 'row2_pfoff',
    'pldl2keep': 'row3_pldl2keep',
    'nt': 'row4_pldl2keepnt',
    'shbw': 'row5_spm_shbw',
    'cacheflex': 'row6_cacheflex',
}
ROW_FAMILY = {row: family for family, row in ROW_PREFIX.items()}
ROW_ENV = {
    'row1_cacheopt': set(),
    'row2_pfoff': set(),
    'row3_pldl2keep': {'CF_NT_BYPASS=1'},
    'row4_pldl2keepnt': {
        'CF_NT_BYPASS=1',
        'CF_NT_CLEANRESP=1',
        'CF_L2_PIN_KEEP=1',
    },
    'row5_spm_shbw': {
        'CF_SPM_SHARED_BW=1',
        'CF_SPM_HDR_CHARGE=1',
    },
    'row6_cacheflex': set(),
}


def _load_best_cells():
    """Load the Figure 8 selection manifest."""
    try:
        with open(BEST_JSON, encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        _miss(f"cannot load BEST_JSON {BEST_JSON}: {error}")
        return []
    cells = payload.get('cells')
    if not isinstance(cells, list) or len(cells) != 12:
        count = len(cells) if isinstance(cells, list) else 'non-list'
        _miss(f"BEST_JSON must contain exactly 12 cells, found {count}")
        return []
    names = [entry.get('name') for entry in cells if isinstance(entry, dict)]
    if (
        len(names) != 12
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 12
    ):
        _miss("BEST_JSON cell names must be 12 distinct non-empty strings")
        return []
    return cells


BEST_CELLS = _load_best_cells()
BEST_BY_NAME = {
    entry['name']: entry for entry in BEST_CELLS
    if isinstance(entry, dict) and isinstance(entry.get('name'), str)
}


def _manifest_selection(tag):
    """Resolve one tag's six cells from best.json."""
    selected = []
    prefix = tag + '_'
    for entry in BEST_CELLS:
        name = entry['name']
        if not name.startswith(prefix):
            continue
        family = name[len(prefix):].split('_kc', 1)[0]
        row = ROW_PREFIX.get(family)
        if row is None:
            _miss(
                f"BEST_JSON {name}: unknown Figure 8 family {family!r}"
            )
            continue
        selected.append((row, name))
    return selected


def _validate_selection(tag, selected):
    """Require one distinct, correctly named cell for every Figure 8 row."""
    problems = []
    rows = [row for row, _ in selected]
    cells = [cell for _, cell in selected]
    for row in ROW_ORDER:
        count = rows.count(row)
        if count != 1:
            problems.append(f"{row} appears {count} times (expected once)")
    unknown = sorted(set(rows) - set(ROW_ORDER))
    if unknown:
        problems.append(f"unknown rows: {unknown}")
    if len(cells) != len(set(cells)):
        problems.append("the same cell is assigned to multiple rows")

    data_root = os.path.realpath(DATA)
    for row, cell in selected:
        if row not in ROW_FAMILY:
            continue
        normalized = os.path.normpath(cell)
        resolved = os.path.realpath(os.path.join(DATA, normalized))
        try:
            inside = os.path.commonpath((data_root, resolved)) == data_root
        except ValueError:
            inside = False
        if os.path.isabs(cell) or normalized.startswith("..") or not inside:
            problems.append(f"{row}: unsafe cell path {cell!r}")
            continue

        basename = os.path.basename(normalized)
        local = basename[len(tag) + 1:] if basename.startswith(tag + "_") else basename
        family = local.split("_kc", 1)[0]
        expected_family = ROW_FAMILY[row]
        if family != expected_family:
            problems.append(
                f"{row}: cell {cell!r} has family {family!r}, "
                f"expected {expected_family!r}"
            )

    if problems:
        for problem in problems:
            _miss(f"{tag} selection: {problem}")
        return []
    return selected


def _selected_cells(tag):
    """Use only the cells selected in best.json."""
    ambiguous = [
        path
        for path in (
            os.path.join(DATA, tag, 'chosen.txt'),
            os.path.join(DATA, f'{tag}_hh0full', 'chosen.txt'),
        )
        if os.path.isfile(path)
    ]
    if ambiguous:
        for path in ambiguous:
            _miss(
                f"{os.path.relpath(path, DATA)} is not a canonical Figure 8 "
                "input; paper rendering is owned exclusively by best.json"
            )
        return []
    return _validate_selection(tag, _manifest_selection(tag))


def _path_matches_manifest(recorded, expected):
    """Accept an exact relative path or a scrubbed/absolute path with that suffix."""
    normalized = os.path.normpath(recorded).replace(os.sep, '/')
    expected = os.path.normpath(expected).replace(os.sep, '/')
    return normalized == expected or normalized.endswith('/' + expected)


def _check_manifest_receipts(celldir, entry):
    """Bind a result directory to its exact manifest binary/runner/arguments."""
    relative = os.path.relpath(celldir, DATA)
    problems = []
    for filename, key in (
        ('BINARY_USED.txt', 'binary'),
        ('RUNNER_USED.txt', 'runner'),
    ):
        path = os.path.join(celldir, filename)
        if not os.path.isfile(path):
            problems.append(f"{relative}/{filename} is missing")
            continue
        recorded = open(
            path, encoding='utf-8', errors='ignore'
        ).read().strip()
        if not _path_matches_manifest(recorded, entry[key]):
            problems.append(
                f"{relative}/{filename}={recorded!r}, "
                f"expected {entry[key]!r}"
            )

    config_path = os.path.join(celldir, 'config.ini')
    config = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        loaded = config.read(config_path)
    except configparser.Error as error:
        problems.append(f"{relative}/config.ini is not parseable: {error}")
        loaded = []
    if loaded:
        try:
            argv = shlex.split(
                config.get('system.cpu.workload', 'cmd', fallback='')
            )
        except ValueError as error:
            problems.append(f"{relative}/config.ini workload is invalid: {error}")
            argv = []
        expected_public = shlex.split(entry['args'])
        expected_args = [
            expected_public[0],
            expected_public[2],
            expected_public[1],
            '1',
            expected_public[3],
            expected_public[4],
        ] if len(expected_public) == 5 else []
        if (
            not argv
            or not _path_matches_manifest(argv[0], entry['binary'])
            or argv[1:] != expected_args
        ):
            problems.append(
                f"{relative}/config.ini workload does not match "
                f"{entry['binary']} {expected_args}"
            )

    hashes = os.path.join(celldir, 'BINARIES.sha256')
    if not os.path.isfile(hashes):
        problems.append(f"{relative}/BINARIES.sha256 is missing")
    else:
        rows = []
        for line in open(hashes, encoding='utf-8', errors='ignore'):
            match = re.fullmatch(r'([0-9a-f]{64})\s+(.+)', line.strip())
            if match:
                rows.append(match.groups())
        if len(rows) != 2:
            problems.append(
                f"{relative}/BINARIES.sha256 must contain exactly two hashes"
            )
        elif not any(
            _path_matches_manifest(path, entry['binary'])
            for _, path in rows
        ):
            problems.append(
                f"{relative}/BINARIES.sha256 has no workload-binary receipt"
            )
    return problems


def _check_interconnect_config(celldir, is_cf):
    path = os.path.join(celldir, 'config.ini')
    if not os.path.exists(path):
        return f"{os.path.relpath(celldir, DATA)}/config.ini"
    config = configparser.ConfigParser(strict=False)
    try:
        loaded = config.read(path)
    except configparser.Error as error:
        return f"{os.path.relpath(celldir, DATA)}/config.ini ({error})"
    if not loaded:
        return f"{os.path.relpath(celldir, DATA)}/config.ini (not parseable)"

    expected = {
        ('system.tol2bus', 'frontend_latency'): '1',
        ('system.tol2bus', 'header_latency'): '0',
        ('system.tol2bus', 'forward_latency'): '1',
        ('system.tol2bus', 'response_latency'): '1',
        ('system.cpu.dcache', 'mshrs'): '8',
        ('system.cpu.isa', 'sve_vl_se'): '16',
    }
    expected.update({
        ('system.cpu', 'spmloadports'): '1',
        ('system.cpu', 'spmstoreports'): '1',
    })
    bad = []
    for (section, key), value in expected.items():
        actual = config.get(section, key, fallback=None)
        if actual != value:
            bad.append(f'{section}.{key}={actual}')
    if bad:
        return f"{os.path.relpath(celldir, DATA)}/config.ini ({', '.join(bad)})"
    return None


def _check_environment_receipt(celldir, row):
    path = os.path.join(celldir, 'RUN_ENV.txt')
    relative = os.path.relpath(celldir, DATA)
    if not os.path.isfile(path):
        return f"{relative}/RUN_ENV.txt"
    text = open(path, encoding='utf-8', errors='ignore').read().strip()
    try:
        actual = set() if text in ('', 'none') else set(shlex.split(text))
    except ValueError as error:
        return f"{relative}/RUN_ENV.txt ({error})"
    expected = ROW_ENV[row]
    if actual != expected:
        return (
            f"{relative}/RUN_ENV.txt (got {sorted(actual)}, "
            f"expected {sorted(expected)})"
        )
    return None


def build_live():
    data = {}
    for tag, label in SIZES:
        selected = _selected_cells(tag)
        if not selected:
            continue
        rows = {}
        for rname, cell in selected:
            if rname not in ROW_LABELS:
                print(f"NOTE: unknown row '{rname}' in {tag}/chosen.txt — skipped")
                continue
            is_cf = rname in CF_ROWS
            row = dict(label=ROW_LABELS[rname], is_cf=is_cf, cell=cell,
                       pack_us=None, kernel_us=None, c_us=None, total_us=None,
                       gflops=None, checksum=None, checksum_l1=None,
                       checksum_weighted=None,
                       cache_mJ=None, spm_mJ=None, dram_mJ=None, total_mJ=None,
                       spm_reads=None, spm_writes=None, roi_time_s=None)
            celldir = os.path.join(DATA, cell)
            manifest_entry = BEST_BY_NAME.get(cell)
            if manifest_entry is None:
                _miss(f"{cell}: not present in best.json")
            else:
                for receipt_problem in _check_manifest_receipts(
                    celldir, manifest_entry
                ):
                    _miss(receipt_problem)
            config_problem = _check_interconnect_config(celldir, is_cf)
            if config_problem:
                _miss(f"{config_problem} — not a canonical result")
            environment_problem = _check_environment_receipt(celldir, rname)
            if environment_problem:
                _miss(f"{environment_problem} — row environment not verified")

            # Parse time, performance, and checksum fields.
            so = os.path.join(celldir, 'stdout.txt')
            if not os.path.exists(so):
                _miss(f"{cell}/stdout.txt (row {rname})")
            else:
                log_text = open(so, encoding='utf-8', errors='ignore').read()
                if 'Exiting @ tick' not in log_text:
                    _miss(f"{cell}/stdout.txt has no clean gem5 exit (row {rname})")
                if BAD_RESULT.search(log_text):
                    _miss(f"{cell}/stdout.txt has a crash, nonzero exit, "
                          f"validation, or debug marker (row {rname})")
                try:
                    rt = pc.parse(so)
                except (OSError, ValueError) as error:
                    _miss(
                        f"{cell}/stdout.txt has an invalid ROI receipt "
                        f"(row {rname}): {error}"
                    )
                else:
                    row.update(
                        pack_us=rt['pack_B'] + rt['pack_A'],
                        kernel_us=rt['kernel'],
                        c_us=rt['C_write'] + rt['C_RMW'],
                        total_us=rt['total'], gflops=rt['gflops'],
                        checksum=rt['checksum'],
                        checksum_l1=rt['checksum_l1'],
                        checksum_weighted=rt['checksum_weighted'])

            # Compute energy from the gem5 statistics.
            st_path = os.path.join(celldir, 'stats.txt')
            if not os.path.exists(st_path):
                _miss(f"{cell}/stats.txt (row {rname}) — energy fields null")
            else:
                try:
                    st = ce.parse_stats(st_path)
                    eng = ce.compute_energy(st, is_cacheflex=is_cf, vl=16)
                except (OSError, ValueError) as error:
                    _miss(
                        f"{cell}/stats.txt has invalid ROI energy data "
                        f"(row {rname}): {error}"
                    )
                else:
                    row.update(
                        cache_mJ=eng['cache_dyn_mJ'] + eng['cache_idle_mJ'],
                        spm_mJ=eng['spm_dyn_mJ'], dram_mJ=eng['dram_mJ'],
                        total_mJ=eng['total_mJ'],
                        spm_reads=st['spm_reads'], spm_writes=st['spm_writes'],
                        roi_time_s=st['roi_time_s'])
            rows[rname] = row
        for rname in ROW_ORDER:
            if rname not in rows:
                _miss(f"{tag}: row {rname} absent from chosen.txt")
        checksums = [
            (
                rname,
                rows[rname]['checksum'],
                rows[rname]['checksum_l1'],
                rows[rname]['checksum_weighted'],
            )
            for rname in ROW_ORDER
            if (
                rname in rows
                and rows[rname]['checksum'] is not None
                and rows[rname]['checksum_l1'] is not None
                and rows[rname]['checksum_weighted'] is not None
            )
        ]
        if len(checksums) == len(ROW_ORDER):
            (
                reference_name,
                reference_value,
                reference_l1,
                reference_weighted,
            ) = checksums[0]
            for rname, value, value_l1, value_weighted in checksums[1:]:
                tolerance = max(abs(reference_value), abs(value)) * 0.015 + 1e-9
                if abs(reference_value - value) > tolerance:
                    _miss(
                        f"{tag}: checksum mismatch {reference_name}="
                        f"{reference_value} vs {rname}={value}"
                    )
                l1_tolerance = max(reference_l1, value_l1) * 0.015 + 1e-9
                if abs(reference_l1 - value_l1) > l1_tolerance:
                    _miss(
                        f"{tag}: L1 checksum mismatch {reference_name}="
                        f"{reference_l1} vs {rname}={value_l1}"
                    )
                weighted_tolerance = (
                    max(abs(reference_weighted), abs(value_weighted))
                    * 0.015
                    + 1e-9
                )
                if abs(reference_weighted - value_weighted) > weighted_tolerance:
                    _miss(
                        f"{tag}: weighted checksum mismatch {reference_name}="
                        f"{reference_weighted} vs {rname}={value_weighted}"
                    )
        data[tag] = dict(label=label, rows=rows)
    return data


def _style(ax, ys, ylabels, R, xlim, xlabel=None):
    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.6, alpha=0.7, zorder=2)
    ax.grid(axis='x', linestyle='--', alpha=0.3); ax.set_axisbelow(True)
    ax.set_xlim(*xlim); ax.set_ylim(-0.5, len(ROW_ORDER) - 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=0, labelsize=6.5)
    ax.set_yticks(ys); ax.set_yticklabels(ylabels, fontsize=6.5)
    for tick, rn in zip(ax.get_yticklabels(), ROW_ORDER):
        if rn in R and R[rn]['is_cf']:
            tick.set_fontweight('bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=6.8)


def _missing_row(ax, y):
    ax.text(0.02, y, 'missing', va='center', fontsize=6.0,
            color='#888888', style='italic', zorder=4)


def draw_runtime(ax, R, title, ys, ylabels, xlabel=None):
    base_row = R.get('row1_cacheopt')
    base = base_row['total_us'] if (base_row and base_row['total_us']) else None
    for y, rn in zip(ys, ROW_ORDER):
        r = R.get(rn)
        if r and r['is_cf']:
            ax.axhspan(y - 0.5, y + 0.5, color=C_HI, zorder=0)
        if r is None or r['total_us'] is None or base is None:
            _missing_row(ax, y)
            continue
        pack_n = r['pack_us'] / base
        comp_n = ((r['kernel_us'] or 0) + (r['c_us'] or 0)) / base
        total_n = r['total_us'] / base
        left = 0.0
        for w, c in [(pack_n, C_STAGE), (comp_n, C_COMPUTE)]:
            ax.barh(y, w, left=left, color=c, height=0.66,
                    edgecolor='black', linewidth=0.7, zorder=3)
            left += w
        ax.text(total_n + 0.03, y, f'{base / r["total_us"]:.2f}$\\times$',
                va='center', fontsize=6.3, fontweight='bold',
                color=C_GAIN, zorder=4)
    _style(ax, ys, ylabels, R, (0, 1.62), xlabel)
    ax.set_title(title, fontsize=7.2, fontweight='bold', pad=3)


def draw_energy(ax, R, title, ys, ylabels, xlabel=None):
    base_row = R.get('row1_cacheopt')
    base = base_row['total_mJ'] if (base_row and base_row['total_mJ']) else None
    for y, rn in zip(ys, ROW_ORDER):
        r = R.get(rn)
        if r and r['is_cf']:
            ax.axhspan(y - 0.5, y + 0.5, color=C_HI, zorder=0)
        if r is None or r['total_mJ'] is None or base is None:
            _missing_row(ax, y)
            continue
        c_n = r['cache_mJ'] / base; s_n = r['spm_mJ'] / base; d_n = r['dram_mJ'] / base
        tot_n = r['total_mJ'] / base
        left = 0.0
        for w, c in [(c_n, C_CACHE), (s_n, C_SPM), (d_n, C_DRAM)]:
            if w <= 0:
                continue
            ax.barh(y, w, left=left, color=c, height=0.66,
                    edgecolor='black', linewidth=0.7, zorder=3)
            left += w
        if tot_n < 1.0:
            ax.barh(y, 1.0 - tot_n, left=tot_n, color='white', height=0.66,
                    edgecolor='black', linewidth=0.7, hatch='...', zorder=3)
        ered = (r['total_mJ'] - base) / base * 100
        ax.text(max(tot_n, 1.0) + 0.03, y, f'{ered:+.0f}%',
                va='center', fontsize=6.3, fontweight='bold', color=C_GAIN, zorder=4)
    _style(ax, ys, ylabels, R, (0, 1.34), xlabel)
    ax.set_title(title, fontsize=7.2, fontweight='bold', pad=3)


def main():
    data = build_live()
    if MISSING:
        print("\nInvalid or missing inputs; no output files were written:")
        for item in MISSING:
            print(f"  - {item}")
        raise SystemExit(
            f"ERROR: {len(MISSING)} invalid/missing input(s) under {DATA}"
        )
    if CHECK_ONLY:
        print(
            "FIG8_CHECK_PASS: 12 exact best.json cells with matching "
            "configuration and provenance receipts"
        )
        return
    os.makedirs(OUT, exist_ok=True)

    for tag in SIZE_ORDER:
        if tag not in data:
            continue
        rows = data[tag]['rows']
        baseline_time = rows['row1_cacheopt']['total_us']
        baseline_energy = rows['row1_cacheopt']['total_mJ']
        for row in rows.values():
            row['normalized_runtime'] = row['total_us'] / baseline_time
            row['speedup_vs_cacheopt'] = baseline_time / row['total_us']
            row['normalized_energy'] = row['total_mJ'] / baseline_energy
            row['energy_change_pct'] = (
                row['total_mJ'] / baseline_energy - 1.0
            ) * 100.0
            row['energy_reduction_pct'] = (
                1.0 - row['total_mJ'] / baseline_energy
            ) * 100.0
    payload = {
        'schema_version': 1,
        'figure': 'Figure 8',
        'units': {
            'time': 'us',
            'energy': 'mJ',
            'gflops': 'GFLOP/s',
            'normalized_values': 'ratio',
            'percentage_values': 'percent',
        },
        **data,
    }
    out_json = os.path.join(OUT, 'fig8_data.json')
    with open(out_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    print('wrote', out_json)

    for tag, label in SIZES:
        if tag not in data:
            continue
        r1 = data[tag]['rows'].get('row1_cacheopt')
        print(f"\n=== {label} ===")
        for rn in ROW_ORDER:
            r = data[tag]['rows'].get(rn)
            if not r:
                continue
            if r['total_us'] is None:
                print(f"  {r['label']:26s} cell={r['cell']}  ** missing **")
                continue
            sp = (r1['total_us'] / r['total_us']) if (r1 and r1['total_us']) else 0
            if r['total_mJ'] is not None and r1 and r1['total_mJ']:
                de = (r['total_mJ'] - r1['total_mJ']) / r1['total_mJ'] * 100
                es = (f"E={r['total_mJ']:7.3f}mJ  C={r['cache_mJ']:6.3f} "
                      f"S={r['spm_mJ']:6.3f} D={r['dram_mJ']:6.3f}  dE={de:+6.1f}%")
            else:
                es = "E=missing"
            print(f"  {r['label']:26s} gf={r['gflops']:7.2f} spd={sp:5.2f}x | {es}"
                  f"   [{r['cell']}]")

    sizes = [s for s in SIZE_ORDER if s in data]
    if not sizes:
        raise SystemExit("no sizes with data — figure not drawn")
    ys = np.arange(len(ROW_ORDER))[::-1]
    ylabels = [ROW_LABELS[r] for r in ROW_ORDER]

    # The working canvas leaves enough room for the fixed output box.
    fig = plt.figure(figsize=(3.45, 6.1472))
    _gs   = fig.add_gridspec(2, 1, hspace=0.45)   # gap between runtime & energy groups
    _gtop = _gs[0].subgridspec(2, 1, hspace=0.45)  # (a),(b) runtime
    _gbot = _gs[1].subgridspec(2, 1, hspace=0.45)  # (c),(d) energy
    axes  = [fig.add_subplot(_gtop[0]), fig.add_subplot(_gtop[1]),
             fig.add_subplot(_gbot[0]), fig.add_subplot(_gbot[1])]

    def rows_of(tag):
        return data[tag]['rows'] if tag in data else {}

    draw_runtime(axes[0], rows_of('w3'),
                 "(a) Runtime — 256×2048×2048", ys, ylabels)
    draw_runtime(axes[1], rows_of('g2048'),
                 "(b) Runtime — 2048×2048×2048", ys, ylabels,
                 xlabel='Normalized end-to-end runtime')
    draw_energy(axes[2], rows_of('w3'),
                "(c) Energy — 256×2048×2048", ys, ylabels)
    draw_energy(axes[3], rows_of('g2048'),
                "(d) Energy — 2048×2048×2048", ys, ylabels,
                xlabel='Normalized memory-system energy')

    axes[0].legend(handles=[
        Patch(fc=C_STAGE, ec='black', linewidth=0.7, label='data staging (pack)'),
        Patch(fc=C_COMPUTE, ec='black', linewidth=0.7, label='compute'),
    ], loc='lower left', bbox_to_anchor=(0.0, 1.18), ncol=2, frameon=False,
       fontsize=6.2, handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    axes[2].legend(handles=[
        Patch(fc=C_CACHE, ec='black', linewidth=0.7, label='Cache'),
        Patch(fc=C_SPM, ec='black', linewidth=0.7, label='SPM'),
        Patch(fc=C_DRAM, ec='black', linewidth=0.7, label='DRAM'),
        Patch(fc='white', ec='black', linewidth=0.7, hatch='...', label='Reduced'),
    ], loc='lower left', bbox_to_anchor=(0.0, 1.18), ncol=4, frameon=False,
       fontsize=6.2, handlelength=1.0, handletextpad=0.3, columnspacing=0.7)

    base = os.path.join(OUT, 'fig8')
    # Avoid a version-dependent MediaBox from bbox_inches="tight".
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    if tight.width > PAPER_WIDTH_IN or tight.height > PAPER_HEIGHT_IN:
        raise SystemExit(
            "ERROR: Figure 8 content exceeds the fixed paper canvas "
            f"({tight.width * 72:.3f}x{tight.height * 72:.3f} pt content; "
            f"{PAPER_WIDTH_IN * 72:.3f}x{PAPER_HEIGHT_IN * 72:.3f} pt canvas)"
        )
    paper_box = Bbox.from_bounds(
        tight.x0 - (PAPER_WIDTH_IN - tight.width) / 2,
        tight.y0 - (PAPER_HEIGHT_IN - tight.height) / 2,
        PAPER_WIDTH_IN,
        PAPER_HEIGHT_IN,
    )
    fig.savefig(base + '.pdf', bbox_inches=paper_box, pad_inches=0)
    fig.savefig(base + '.png', dpi=300, bbox_inches=paper_box, pad_inches=0)
    plt.close(fig)
    print('\nwrote', base + '.pdf/.png')

if __name__ == '__main__':
    main()
