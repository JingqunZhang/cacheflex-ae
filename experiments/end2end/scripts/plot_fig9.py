#!/usr/bin/env python3
"""Validate canonical end-to-end receipts and generate Figure 9 data.

The paper grid is fixed:
  * VL=4  -> simulation_results/vl4
  * VL=16 -> simulation_results/vl16
  * T in {256, 4096}
  * models in {LLaMA-3.2-1B, BERT-base}

Every regex family must resolve to exactly one cell in the matching best.json
block, and that exact result must be present and parseable.  There is
deliberately no fallback or post-hoc fastest-cell selection.  BEST_JSON may
override the experiment-local best.json for validation/testing.

A normal render also writes fig9_data.json beside the figure.  The JSON uses
the same parsed cells as the plot and records the component values and source
cell names needed to trace every plotted total.
"""

import argparse
import json
import math
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(os.environ.get("OUTPUT_DIR", BASE))
PAPER_T = (256, 4096)
PAPER_VL = (4, 16)
MODELS = ("llama", "bert")
LAYERNORM_MAX_ABS_LIMIT = 5.0e-3
POOL_BY_VL = {
    4: os.path.join("simulation_results", "vl4"),
    16: os.path.join("simulation_results", "vl16"),
}
CFGS = ("BL", "CF", "BL+FA", "CF+FA")
CLEAN_EXIT = "Exiting @ tick"
STATS_MARKER = re.compile(
    r"^-+\s+(Begin|End) Simulation Statistics\s+-+\s*$", re.MULTILINE
)
BAD_LOG = re.compile(
    r"panic:|fatal:|segmentation fault|segfault|core dumped|"
    r"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
    r"\[DBG\]|"
    r"assert(?:ion)?[^\n]{0,256}failed",
    re.IGNORECASE,
)

E = {
    "l1_read": 0.3609,
    "l1_write": 0.4824,
    "l2_read": 1.1205,
    "l2_write": 1.4409,
    "l3_read": 5.121,
    "l3_write": 6.8205,
    "spm_read": 0.5328,
    "spm_write": 0.5526,
}
IDLE_L1, IDLE_L2_BL, IDLE_L2_CF, IDLE_L3 = 12.0, 26.0, 31.0, 51.0

# These two request classes are vector-stat subentries.  gem5 omits an
# unobserved subentry rather than printing an explicit zero.  Every other
# counter used by the energy model must be present exactly once.
OPTIONAL_ZERO_STATS = frozenset(
    {
        "system.cpu.dcache.ReadSharedReq.accesses::total",
        "system.cpu.dcache.WriteLineReq.accesses::total",
    }
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the fixed paper Figure 9 from canonical result pools."
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        help="result root containing simulation_results/{vl4,vl16}/ "
        "(default: RESULTS_DIR or experiments/end2end/results/reference)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only validate the complete paper grid; do not write output files",
    )
    return parser.parse_args()


def result_root(cli_root):
    if cli_root:
        return os.path.abspath(cli_root)
    if os.environ.get("RESULTS_DIR"):
        return os.path.abspath(os.environ["RESULTS_DIR"])
    return os.path.join(BASE, "results", "reference")


def best_json_path():
    override = os.environ.get("BEST_JSON")
    return os.path.abspath(override) if override else os.path.join(BASE, "best.json")


def load_allowed_cells(path):
    """Return best.json cell names grouped by their exact VL block."""
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"cannot load BEST_JSON {path}: {error}"]

    cells = manifest.get("cells")
    if not isinstance(cells, list):
        return {}, [f"BEST_JSON {path} has no list-valued 'cells' field"]

    allowed = {vl: [] for vl in PAPER_VL}
    errors = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"BEST_JSON {path} cells[{index}] is not an object")
            continue
        block = cell.get("block")
        name = cell.get("name")
        for vl in PAPER_VL:
            if block == f"vl{vl}":
                if not isinstance(name, str) or not name:
                    errors.append(
                        f"BEST_JSON {path} cells[{index}] in block {block!r} "
                        "has no non-empty name"
                    )
                else:
                    # Keep a list rather than a set so duplicate manifest entries
                    # correctly fail the one-cell-per-family invariant.
                    allowed[vl].append(name)
                break
    return allowed, errors


def required_patterns():
    """Yield every regex family consumed by Figure 9."""
    for vl in PAPER_VL:
        for sequence_length in PAPER_T:
            for model in MODELS:
                table = kernel_table(model, sequence_length)
                for kind in ("proj", "ffn"):
                    for _, cache_pattern, spm_pattern in table[kind]:
                        yield vl, cache_pattern
                        yield vl, spm_pattern
                for key in ("attn_unf", "attn_flash"):
                    _, cache_pattern, spm_pattern = table[key]
                    yield vl, cache_pattern
                    yield vl, spm_pattern
                for _, pattern in table["other"]:
                    yield vl, pattern


def resolve_best_families(allowed, path):
    """Resolve every regex family to exactly one best.json cell name."""
    selected = {}
    errors = []
    for vl, pattern in required_patterns():
        key = (vl, pattern)
        if key in selected:
            continue
        matches = [
            name for name in allowed.get(vl, []) if re.fullmatch(pattern, name)
        ]
        if len(matches) != 1:
            rendered = ", ".join(repr(name) for name in matches) or "none"
            errors.append(
                f"BEST_JSON selection error: block 'vl{vl}' family "
                f"/{pattern}/ must match exactly 1 cell, found {len(matches)} "
                f"({rendered}) in {path}"
            )
            continue
        selected[key] = matches[0]
    return selected, errors


def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def first_complete_stats_block(text):
    """Return the first complete statistics block, rejecting malformed dumps."""
    markers = list(STATS_MARKER.finditer(text))
    begin_count = sum(marker.group(1) == "Begin" for marker in markers)
    end_count = len(markers) - begin_count
    if (
        not markers
        or begin_count != text.count("Begin Simulation Statistics")
        or end_count != text.count("End Simulation Statistics")
    ):
        return None

    blocks = []
    block_start = None
    for marker in markers:
        if marker.group(1) == "Begin":
            if block_start is not None:
                return None
            block_start = marker.end()
        else:
            if block_start is None:
                return None
            blocks.append(text[block_start : marker.start()])
            block_start = None
    if block_start is not None or not blocks:
        return None
    return blocks[0]


def stat_value(roi, name, *, positive=False):
    """Read one finite stat value; only documented zero subentries may be absent."""
    matches = re.findall(
        rf"^{re.escape(name)}[ \t]+(\S+)", roi, re.MULTILINE
    )
    if not matches:
        if name in OPTIONAL_ZERO_STATS:
            return 0.0
        raise ValueError(f"missing required ROI stat {name}")
    if len(matches) != 1:
        raise ValueError(f"ROI stat {name} occurs {len(matches)} times")
    try:
        value = float(matches[0])
    except ValueError as error:
        raise ValueError(f"ROI stat {name} is not numeric") from error
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"ROI stat {name} is outside its valid range")
    return value


def roi_energy(stats_path, cacheflex_system, vl):
    """Return first-ROI memory energy, or None for an invalid stats file."""
    try:
        text = read_text(stats_path)
    except OSError:
        return None
    roi = first_complete_stats_block(text)
    if roi is None or vl not in PAPER_VL:
        return None
    try:
        sim_freq = stat_value(roi, "simFreq", positive=True)
        sim_ticks = stat_value(roi, "simTicks", positive=True)

        def get(name):
            return stat_value(roi, name)

        cache_dynamic = (
            (
                get("system.cpu.dcache.ReadReq.accesses::total")
                + get("system.cpu.dcache.ReadSharedReq.accesses::total")
            )
            * E["l1_read"]
            + (
                get("system.cpu.dcache.WriteReq.accesses::total")
                + get("system.cpu.dcache.WriteLineReq.accesses::total")
            )
            * E["l1_write"]
            + get("system.l2.overallAccesses::total") * E["l2_read"]
            + get("system.l2.WritebackDirty.accesses::total") * E["l2_write"]
            + get("system.l3.overallAccesses::total") * E["l3_read"]
            + get("system.l3.WritebackDirty.accesses::total") * E["l3_write"]
        ) / 1e6

        # gem5 counts vector SPM load instructions.  Each spm.ld1qd transfers
        # VL*16 bytes, while the RTL energy constant is per 64-byte access.
        spm = (
            get("system.l2.spmReads") * E["spm_read"] * (vl // 4)
            + get("system.l2.spmWrites") * E["spm_write"]
        ) / 1e6
        roi_seconds = sim_ticks / sim_freq
        idle = (
            IDLE_L1
            + (IDLE_L2_CF if cacheflex_system else IDLE_L2_BL)
            + IDLE_L3
        ) * roi_seconds
        dram = sum(
            get(f"system.mem_ctrls{controller}.dram.rank{rank}.totalEnergy")
            for controller in (0, 1)
            for rank in (0, 1)
        ) / 1e9
    except ValueError:
        return None
    return {"cache": cache_dynamic + idle, "spm": spm, "dram": dram}


def roi_time_us(stats_path):
    """Return the first complete ROI's simulated execution time."""
    try:
        text = read_text(stats_path)
    except OSError:
        return None
    roi = first_complete_stats_block(text)
    if roi is None:
        return None
    try:
        sim_freq = stat_value(roi, "simFreq", positive=True)
        sim_ticks = stat_value(roi, "simTicks", positive=True)
    except ValueError:
        return None
    value = sim_ticks / sim_freq * 1.0e6
    return value if math.isfinite(value) and value > 0 else None


def parse_time_us(log_text):
    for pattern, multiplier in (
        (r"time_us=(\S+)", 1.0),
        (r"FIRST_ROI:\s+(\S+)\s+ms", 1000.0),
        (r"COLD:\s+(\S+)\s+ms", 1000.0),
        (r"avg:\s+(\S+)\s+ms", 1000.0),
    ):
        matches = re.findall(pattern, log_text)
        if matches:
            if len(matches) != 1:
                return None
            try:
                value = float(matches[0]) * multiplier
            except ValueError:
                return None
            return value if math.isfinite(value) and value > 0 else None
    return None


def optional_receipt_value(log_text, label):
    matches = re.findall(
        rf"^\s*{re.escape(label)}:\s*(.*?)\s*$",
        log_text,
        re.MULTILINE,
    )
    if not matches:
        return True, None
    if len(matches) != 1 or not matches[0]:
        return False, None
    try:
        value = float(matches[0])
    except ValueError:
        return False, None
    return (True, value) if math.isfinite(value) else (False, None)


def checksum_value(log_text):
    """Return (valid, value); a missing checksum is valid for non-attention cells."""
    return optional_receipt_value(log_text, "CHECKSUM")


def parse_checksum(log_text):
    """Return the single finite benchmark checksum, or None."""
    valid, value = checksum_value(log_text)
    return value if valid else None


def selected_roi(pool_dir, pattern, cacheflex_system, vl, selected):
    """Return the complete ROI selected by best.json, never a post-hoc fastest cell."""
    name = selected.get((vl, pattern))
    if name is None:
        return None
    stats_path = os.path.join(pool_dir, name, "stats.txt")
    log_path = os.path.join(pool_dir, name, "stdout.log")
    if not (
        os.path.isfile(stats_path)
        and os.path.getsize(stats_path) > 0
        and os.path.isfile(log_path)
        and os.path.getsize(log_path) > 0
    ):
        return None
    log_text = read_text(log_path)
    clean_exit_count = log_text.count(CLEAN_EXIT)
    if clean_exit_count != 1 or BAD_LOG.search(log_text):
        return None
    logged_time_us = parse_time_us(log_text)
    time_us = roi_time_us(stats_path)
    checksum_valid, checksum = checksum_value(log_text)
    logical_valid, logical_checksum = optional_receipt_value(
        log_text, "CHECKSUM_LOGICAL"
    )
    logical_l1_valid, logical_checksum_l1 = optional_receipt_value(
        log_text, "CHECKSUM_LOGICAL_L1"
    )
    validation_valid, validation = optional_receipt_value(
        log_text, "VALIDATION_MAX_ABS"
    )
    energy = roi_energy(stats_path, cacheflex_system, vl)
    if (
        time_us is None
        or logged_time_us is None
        or abs(time_us - logged_time_us) > max(10.0, 0.005 * time_us)
        or not checksum_valid
        or not logical_valid
        or not logical_l1_valid
        or not validation_valid
        or energy is None
    ):
        return None
    is_attention = bool(
        re.search(r"_(?:spm_)?unfused$", name)
        or re.search(r"_[cs]flash_br\d+_bc\d+$", name)
    )
    if is_attention:
        modes = re.findall(
            r"^ATTENTION_MODE=(causal|noncausal)\s*$",
            log_text,
            re.MULTILINE,
        )
        if modes != ["noncausal"]:
            return None
    return (
        time_us,
        energy,
        name,
        checksum,
        logical_checksum,
        logical_checksum_l1,
        validation,
    )


def kernel_table(model, sequence_length):
    t = sequence_length
    if model == "llama":
        return {
            "proj": [
                (
                    2,
                    rf"llama_T{t}_cache_K2048_N2048_mc\d+_kc\d+",
                    rf"llama_T{t}_spm_K2048_N2048_mc\d+_kc\d+",
                ),
                (
                    2,
                    rf"llama_T{t}_cache_K2048_N512_mc\d+_kc\d+",
                    rf"llama_T{t}_spm_K2048_N512_mc\d+_kc\d+",
                ),
            ],
            "ffn": [
                (
                    1,
                    rf"llama_T{t}_ffn_seq_mc\d+_kc\d+",
                    rf"llama_T{t}_ffn_seq_spm_mc\d+_kc\d+",
                ),
                (
                    1,
                    rf"llama_T{t}_cache_K8192_N2048_mc\d+_kc\d+",
                    rf"llama_T{t}_spm_K8192_N2048_mc\d+_kc\d+",
                ),
            ],
            "attn_unf": (1, rf"llama_T{t}_unfused", rf"llama_T{t}_spm_unfused"),
            "attn_flash": (
                1,
                rf"llama_T{t}_cflash_br\d+_bc\d+",
                rf"llama_T{t}_sflash_br\d+_bc\d+",
            ),
            "other": [
                (2, rf"llama_T{t}_rmsnorm"),
                (2, rf"llama_T{t}_residual"),
                (1, rf"llama_T{t}_rope"),
            ],
        }
    return {
        "proj": [
            (
                4,
                rf"bert_T{t}_cache_K768_N768_mc\d+_kc\d+",
                rf"bert_T{t}_spm_K768_N768_mc\d+_kc\d+",
            )
        ],
        "ffn": [
            (
                1,
                rf"bert_T{t}_ffn_seq_mc\d+_kc\d+",
                rf"bert_T{t}_ffn_seq_spm_mc\d+_kc\d+",
            ),
            (
                1,
                rf"bert_T{t}_cache_K3072_N768_mc\d+_kc\d+",
                rf"bert_T{t}_spm_K3072_N768_mc\d+_kc\d+",
            ),
        ],
        "attn_unf": (1, rf"bert_T{t}_unfused", rf"bert_T{t}_spm_unfused"),
        "attn_flash": (
            1,
            rf"bert_T{t}_cflash_br\d+_bc\d+",
            rf"bert_T{t}_sflash_br\d+_bc\d+",
        ),
        "other": [
            (2, rf"bert_T{t}_layernorm"),
            (2, rf"bert_T{t}_residual"),
        ],
    }


def layer_data(pool_dir, model, sequence_length, vl, selected, missing):
    """Return time and energy components for the four paper configurations."""
    table = kernel_table(model, sequence_length)
    output = {}
    for config, use_spm, use_flash in (
        ("BL", False, False),
        ("CF", True, False),
        ("BL+FA", False, True),
        ("CF+FA", True, True),
    ):
        time_components = {"proj": 0.0, "ffn": 0.0, "attn": 0.0, "other": 0.0}
        energy_components = {"cache": 0.0, "spm": 0.0, "dram": 0.0}
        sources = []
        complete = True

        def add(kind, count, pattern):
            nonlocal complete
            match = selected_roi(
                pool_dir, pattern, use_spm, vl, selected
            )
            if match is None:
                chosen = selected.get((vl, pattern))
                suffix = f" (best.json cell: {chosen})" if chosen else ""
                missing.add(
                    f"VL={vl}, T={sequence_length}, model={model}, "
                    f"config={config}: {pattern}{suffix}"
                )
                complete = False
                return
            (
                time_us,
                energy,
                cell_name,
                checksum,
                logical_checksum,
                logical_checksum_l1,
                validation,
            ) = match
            if kind == "attn" and checksum is None:
                missing.add(
                    f"VL={vl}, T={sequence_length}, model={model}, "
                    f"config={config}: {cell_name} has no single parseable "
                    "CHECKSUM line"
                )
                complete = False
                return
            if (
                cell_name.endswith("_layernorm")
                and (
                    validation is None
                    or validation < 0
                    or validation > LAYERNORM_MAX_ABS_LIMIT
                    or logical_checksum is None
                )
            ):
                missing.add(
                    f"VL={vl}, T={sequence_length}, model={model}, "
                    f"config={config}: {cell_name} has no valid "
                    "VALIDATION_MAX_ABS receipt"
                )
                complete = False
                return
            if (
                (
                    cell_name.endswith("_rmsnorm")
                    or cell_name.endswith("_residual")
                )
                and validation is not None
                and (validation < 0 or validation > LAYERNORM_MAX_ABS_LIMIT)
            ):
                missing.add(
                    f"VL={vl}, T={sequence_length}, model={model}, "
                    f"config={config}: {cell_name} has an out-of-range "
                    "VALIDATION_MAX_ABS receipt"
                )
                complete = False
                return
            time_components[kind] += time_us * count
            for component in energy_components:
                energy_components[component] += energy[component] * count
            sources.append(
                {
                    "component": kind,
                    "multiplicity": count,
                    "cell": cell_name,
                    "time_us_per_invocation": time_us,
                    "energy_mJ_per_invocation": dict(energy),
                    "checksum": checksum,
                    "checksum_logical": logical_checksum,
                    "checksum_logical_l1": logical_checksum_l1,
                    "validation_max_abs": validation,
                }
            )

        for kind in ("proj", "ffn"):
            for count, cache_pattern, spm_pattern in table[kind]:
                add(kind, count, spm_pattern if use_spm else cache_pattern)
        count, cache_pattern, spm_pattern = (
            table["attn_flash"] if use_flash else table["attn_unf"]
        )
        add("attn", count, spm_pattern if use_spm else cache_pattern)
        for count, pattern in table["other"]:
            add("other", count, pattern)
        output[config] = (
            {
                "time_components_us": time_components,
                "energy_components_mJ": energy_components,
                "sources": sources,
            }
            if complete
            else None
        )

    def attention_source(config):
        entry = output.get(config)
        if entry is None:
            return None
        matches = [
            source for source in entry["sources"] if source["component"] == "attn"
        ]
        return matches[0] if len(matches) == 1 else None

    def matching_source(config, pattern):
        entry = output.get(config)
        if entry is None:
            return None
        matches = [
            source
            for source in entry["sources"]
            if re.fullmatch(pattern, source["cell"])
        ]
        return matches[0] if len(matches) == 1 else None

    numerical_pair_failed = False
    for _, cache_pattern, spm_pattern in table["proj"] + table["ffn"]:
        cache_source = matching_source("BL", cache_pattern)
        spm_source = matching_source("CF", spm_pattern)
        if cache_source is None or spm_source is None:
            continue
        cache_kc_match = re.search(r"_kc(\d+)$", cache_source["cell"])
        spm_kc_match = re.search(r"_kc(\d+)$", spm_source["cell"])
        if cache_kc_match is None or spm_kc_match is None:
            missing.add(
                f"VL={vl}, T={sequence_length}, model={model}: "
                "GEMM/FFN cell name has no KC receipt"
            )
            numerical_pair_failed = True
            continue
        same_kc = cache_kc_match.group(1) == spm_kc_match.group(1)
        relative_tolerance = 1.0e-6 if same_kc else 0.015
        standalone_gemm = cache_source["checksum_logical"] is not None
        if standalone_gemm:
            cache_sum = cache_source["checksum_logical"]
            spm_sum = spm_source["checksum_logical"]
            cache_l1 = cache_source["checksum_logical_l1"]
            spm_l1 = spm_source["checksum_logical_l1"]
            sum_tolerance = (
                relative_tolerance * max(cache_l1, spm_l1, 1.0) + 1.0e-9
            )
            l1_tolerance = (
                relative_tolerance * max(cache_l1, spm_l1, 1.0) + 1.0e-9
            )
            mismatch = (
                abs(cache_sum - spm_sum) > sum_tolerance
                or abs(cache_l1 - spm_l1) > l1_tolerance
            )
            rendered_values = (
                f"sum {cache_sum} vs {spm_sum}, "
                f"L1 {cache_l1} vs {spm_l1}"
            )
        else:
            cache_sum = cache_source["checksum"]
            spm_sum = spm_source["checksum"]
            if cache_sum is None or spm_sum is None:
                continue
            tolerance = (
                relative_tolerance
                * max(abs(cache_sum), abs(spm_sum), 1.0)
                + 1.0e-9
            )
            mismatch = abs(cache_sum - spm_sum) > tolerance
            rendered_values = f"{cache_sum} vs {spm_sum}"
        if mismatch:
            missing.add(
                f"VL={vl}, T={sequence_length}, model={model}: "
                f"GEMM/FFN checksum mismatch "
                f"{cache_source['cell']} vs {spm_source['cell']}: "
                f"{rendered_values}"
            )
            numerical_pair_failed = True

    if numerical_pair_failed:
        for config in output:
            output[config] = None

    for cache_config, spm_config in (("BL", "CF"), ("BL+FA", "CF+FA")):
        cache_source = attention_source(cache_config)
        spm_source = attention_source(spm_config)
        if cache_source is None or spm_source is None:
            continue
        cache_sum = cache_source["checksum_logical"]
        cache_l1 = cache_source["checksum_logical_l1"]
        spm_sum = spm_source["checksum_logical"]
        spm_l1 = spm_source["checksum_logical_l1"]
        if None in (cache_sum, cache_l1, spm_sum, spm_l1):
            cache_sum = cache_source["checksum"]
            spm_sum = spm_source["checksum"]
            if cache_sum is None or spm_sum is None:
                missing.add(
                    f"VL={vl}, T={sequence_length}, model={model}: "
                    "attention pair lacks numerical receipts"
                )
                output[cache_config] = None
                output[spm_config] = None
                continue
            tolerance = 0.015 * max(abs(cache_sum), abs(spm_sum), 1.0) + 1.0e-9
            if abs(cache_sum - spm_sum) > tolerance:
                missing.add(
                    f"VL={vl}, T={sequence_length}, model={model}: attention "
                    f"checksum mismatch {cache_config}={cache_sum} "
                    f"vs {spm_config}={spm_sum}"
                )
                output[cache_config] = None
                output[spm_config] = None
            continue
        sum_tolerance = 0.015 * max(cache_l1, spm_l1, 1.0) + 1.0e-9
        l1_tolerance = 0.015 * max(cache_l1, spm_l1, 1.0) + 1.0e-9
        if (
            abs(cache_sum - spm_sum) > sum_tolerance
            or abs(cache_l1 - spm_l1) > l1_tolerance
        ):
            missing.add(
                f"VL={vl}, T={sequence_length}, model={model}: attention "
                f"checksum mismatch {cache_config}=({cache_sum}, {cache_l1}) "
                f"vs {spm_config}=({spm_sum}, {spm_l1})"
            )
            output[cache_config] = None
            output[spm_config] = None
    return output


def validate_paper_grid(results_root):
    missing = set()
    manifest_path = best_json_path()
    allowed, manifest_errors = load_allowed_cells(manifest_path)
    selected, selection_errors = resolve_best_families(allowed, manifest_path)
    missing.update(manifest_errors)
    missing.update(selection_errors)
    pools = {}
    for vl, pool_name in POOL_BY_VL.items():
        pool_dir = os.path.join(results_root, pool_name)
        if not os.path.isdir(pool_dir):
            missing.add(f"VL={vl}: missing canonical pool {pool_dir}")
        else:
            pools[vl] = pool_dir

    data = {}
    for vl in PAPER_VL:
        if vl not in pools:
            continue
        for sequence_length in PAPER_T:
            for model in MODELS:
                data[(vl, sequence_length, model)] = layer_data(
                    pools[vl], model, sequence_length, vl, selected, missing
                )
    return data, sorted(missing)


def report_missing(results_root, missing):
    print(
        "ERROR: Figure 9 paper grid is incomplete or not parseable.",
        file=sys.stderr,
    )
    print(f"Results root: {results_root}", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print(
        "Required pools: simulation_results/vl4 and simulation_results/vl16; "
        "required T values: 256 and 4096. Only the exact best.json selections "
        "were considered; no fallback was used.",
        file=sys.stderr,
    )


def write_data(data, output_dir):
    """Write the exact parsed values and cell provenance consumed by Figure 9."""
    records = []
    for model in MODELS:
        for sequence_length in PAPER_T:
            for vl in PAPER_VL:
                layer = data[(vl, sequence_length, model)]
                baseline_time_us = sum(
                    layer["BL"]["time_components_us"].values()
                )
                baseline_energy_mj = sum(
                    layer["BL"]["energy_components_mJ"].values()
                )
                configurations = {}
                for config in CFGS:
                    entry = layer[config]
                    total_time_us = sum(entry["time_components_us"].values())
                    total_energy_mj = sum(
                        entry["energy_components_mJ"].values()
                    )
                    configurations[config] = {
                        "time_us": {
                            "components": dict(entry["time_components_us"]),
                            "total": total_time_us,
                        },
                        "energy_mJ": {
                            "components": dict(entry["energy_components_mJ"]),
                            "total": total_energy_mj,
                        },
                        "speedup_vs_BL": baseline_time_us / total_time_us,
                        "energy_reduction_vs_BL_pct": (
                            1.0 - total_energy_mj / baseline_energy_mj
                        )
                        * 100.0,
                        "source_cells": list(entry["sources"]),
                    }
                records.append(
                    {
                        "model": model,
                        "sequence_length": sequence_length,
                        "vector_length": vl,
                        "source_pool": POOL_BY_VL[vl],
                        "configurations": configurations,
                    }
                )

    payload = {
        "schema_version": 1,
        "figure": "Figure 9",
        "units": {"time": "us", "energy": "mJ"},
        "records": records,
    }
    output_path = os.path.join(output_dir, "fig9_data.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return output_path


def render(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_path = write_data(data, OUTPUT_DIR)
    from render_fig9 import render as render_figure

    render_figure(data_path, OUTPUT_DIR)
    print(f"Saved {data_path}")


def main():
    args = parse_args()
    results = result_root(args.results_root)
    data, missing = validate_paper_grid(results)
    if missing:
        report_missing(results, missing)
        return 1
    if args.check:
        print(
            "OK: Figure 9 paper grid is complete and parseable "
            "(T=256/4096, VL=4/16, LLaMA/BERT)."
        )
        return 0
    render(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
