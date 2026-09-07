#!/usr/bin/env python3
"""Generate Table IV from the supplied FlashAttention measurements."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_DIR", BASE / "results" / "reference")
).resolve()
ENDPOINT_ROOT = RESULTS_ROOT / "simulation_results"
COMPANION_ROOT = RESULTS_ROOT / "table4"

# (VL, T) -> (pool, cache cell, SPM cell)
CELLS: dict[tuple[int, int], tuple[str, str, str]] = {
    (4, 256): (
        "endpoint",
        "llama_T256_cflash_br64_bc576",
        "llama_T256_sflash_br64_bc384",
    ),
    (4, 1024): (
        "companion",
        "llama_T1024_cflash_br64_bc576",
        "llama_T1024_sflash_br64_bc576",
    ),
    (4, 2048): (
        "companion",
        "llama_T2048_cflash_br64_bc576",
        "llama_T2048_sflash_br64_bc576",
    ),
    (4, 4096): (
        "endpoint",
        "llama_T4096_cflash_br64_bc576",
        "llama_T4096_sflash_br64_bc576",
    ),
    (16, 256): (
        "endpoint",
        "llama_T256_cflash_br64_bc256",
        "llama_T256_sflash_br32_bc256",
    ),
    (16, 1024): (
        "companion",
        "llama_T1024_cflash_br64_bc384",
        "llama_T1024_sflash_br64_bc384",
    ),
    (16, 2048): (
        "companion",
        "llama_T2048_cflash_br64_bc384",
        "llama_T2048_sflash_br64_bc384",
    ),
    (16, 4096): (
        "endpoint",
        "llama_T4096_cflash_br64_bc384",
        "llama_T4096_sflash_br32_bc384",
    ),
}

SEQUENCE_LENGTHS = (256, 1024, 2048, 4096)
VECTOR_LENGTHS = (4, 16)
CACHE_MODE = {(4, 1024), (16, 256), (16, 1024)}
EXPECTED_REPORTED_GAIN = {
    (4, 256): 3.2,
    (4, 1024): 0.0,
    (4, 2048): 7.4,
    (4, 4096): 9.7,
    (16, 256): 0.0,
    (16, 1024): 0.0,
    (16, 2048): 18.3,
    (16, 4096): 15.2,
}
KV_BYTES_PER_TOKEN = 256
L2_BYTES = 512 * 1024
CHECK_TOLERANCE_PP = 0.5

CLEAN_EXIT = "Exiting @ tick"
BAD_LOG_RE = re.compile(
    r"panic:|fatal:|segmentation fault|\bsegfault\b|core dumped|"
    r"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
    r"\[DBG\]|assert(?:ion)?[^\n]{0,256}failed",
    re.IGNORECASE,
)
STATS_MARKER = re.compile(
    r"^-+\s+(Begin|End) Simulation Statistics\s+-+\s*$", re.MULTILINE
)
COLD_RE = re.compile(r"^\s*COLD:\s*(\S+)\s+ms\b", re.MULTILINE)
ITER0_RE = re.compile(r"^\s*iter\s+0:\s*(\S+)\s+ms\b", re.MULTILINE)
AVG_RE = re.compile(r"^\s*avg:\s*(\S+)\s+ms\b", re.MULTILINE)
CHECKSUM_RE = re.compile(r"^\s*CHECKSUM:\s*(\S+)\s*$", re.MULTILINE)
MODE_RE = re.compile(r"^ATTENTION_MODE=(causal|noncausal)\s*$", re.MULTILINE)
CELL_RE = re.compile(
    r"^llama_T(?P<t>\d+)_(?P<kind>[cs]flash)_br(?P<br>\d+)_bc(?P<bc>\d+)$"
)


class RenderError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert that measured and selected values match Table IV",
    )
    return parser.parse_args()


def output_dir() -> Path:
    override = os.environ.get("OUTPUT_DIR")
    if override:
        return Path(override).resolve()
    return BASE.parents[1] / "results"


def pool_dir(pool: str, vl: int) -> Path:
    root = ENDPOINT_ROOT if pool == "endpoint" else COMPANION_ROOT
    return root / f"vl{vl}"


def cell_dir(pool: str, vl: int, name: str) -> Path:
    return pool_dir(pool, vl) / name


def read_required(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise RenderError(f"missing/empty file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def one_finite(matches: list[str], label: str, *, positive: bool) -> float:
    if len(matches) != 1:
        raise RenderError(f"{label}: expected exactly one receipt, found {len(matches)}")
    try:
        value = float(matches[0])
    except ValueError as error:
        raise RenderError(f"{label}: non-numeric receipt {matches[0]!r}") from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise RenderError(f"{label}: receipt is outside its valid range")
    return value


def first_stats_block(text: str, label: str) -> str:
    markers = list(STATS_MARKER.finditer(text))
    begin_count = sum(marker.group(1) == "Begin" for marker in markers)
    end_count = len(markers) - begin_count
    if (
        not markers
        or begin_count != text.count("Begin Simulation Statistics")
        or end_count != text.count("End Simulation Statistics")
    ):
        raise RenderError(f"{label}: malformed statistics markers")

    blocks: list[str] = []
    start = None
    for marker in markers:
        if marker.group(1) == "Begin":
            if start is not None:
                raise RenderError(f"{label}: nested statistics blocks")
            start = marker.end()
        else:
            if start is None:
                raise RenderError(f"{label}: unmatched statistics end marker")
            blocks.append(text[start : marker.start()])
            start = None
    if start is not None or not blocks:
        raise RenderError(f"{label}: incomplete statistics block")
    return blocks[0]


def stats_value(block: str, name: str, label: str) -> float:
    matches = re.findall(rf"^{re.escape(name)}[ \t]+(\S+)", block, re.MULTILINE)
    return one_finite(matches, f"{label}/{name}", positive=True)


def roi_time_us(path: Path, label: str) -> float:
    block = first_stats_block(read_required(path), label)
    ticks = stats_value(block, "simTicks", label)
    frequency = stats_value(block, "simFreq", label)
    return ticks / frequency * 1.0e6


def parse_config(path: Path, vl: int, name: str, is_spm: bool) -> None:
    label = f"VL={vl}/{name}"
    config_path = path / "config.ini"
    read_required(config_path)
    config = configparser.ConfigParser(interpolation=None, strict=True)
    try:
        with config_path.open(encoding="utf-8") as stream:
            config.read_file(stream)
    except (OSError, configparser.Error) as error:
        raise RenderError(f"{label}: unparseable config.ini: {error}") from error

    expected = {
        ("system", "cache_line_size"): "64",
        ("system.cpu_clk_domain", "clock"): "400",
        ("system.cpu.isa", "sve_vl_se"): str(vl),
        ("system.cpu", "fetchWidth"): "5",
        ("system.cpu", "commitWidth"): "5",
        ("system.cpu", "dispatchWidth"): "8",
        ("system.cpu", "issueWidth"): "8",
        ("system.cpu", "wbWidth"): "8",
        ("system.cpu", "numIQEntries"): "80",
        ("system.l2", "size"): str(L2_BYTES),
        ("system.l2", "assoc"): "8",
        ("system.l3", "size"): str(4 * 1024 * 1024),
        ("system.l3", "assoc"): "16",
        ("system.cpu.dcache", "size"): str(64 * 1024),
        ("system.cpu.dcache", "assoc"): "4",
        ("system.cpu.icache", "size"): str(64 * 1024),
        ("system.cpu.icache", "assoc"): "4",
        ("system.cpu", "cacheLoadPorts"): "2",
        ("system.cpu", "cacheStorePorts"): "1",
        ("system.cpu", "spmLoadPorts"): "2" if vl == 4 else "1",
        ("system.cpu", "spmStorePorts"): "1",
        ("system.cpu", "numROBEntries"): "128",
        ("system.cpu", "LQEntries"): "32",
        ("system.cpu", "SQEntries"): "48",
        ("system.cpu", "numPhysIntRegs"): "128",
        ("system.cpu", "numPhysFloatRegs"): "192",
        ("system.cpu", "numPhysVecRegs"): "192",
    }
    for (section, option), wanted in expected.items():
        got = config.get(section, option, fallback=None)
        got = got.strip() if got is not None else None
        if got != wanted:
            raise RenderError(
                f"{label}: {section}.{option}={got!r}, expected {wanted!r}"
            )

    identity = CELL_RE.fullmatch(name)
    if identity is None:
        raise RenderError(f"{label}: cell name does not encode its workload")
    expected_kind = "sflash" if is_spm else "cflash"
    if identity.group("kind") != expected_kind:
        raise RenderError(f"{label}: cell kind does not match selected path")

    command = config.get("system.cpu.workload", "cmd", fallback="")
    try:
        argv = shlex.split(command)
    except ValueError as error:
        raise RenderError(f"{label}: malformed workload command: {error}") from error
    if not argv:
        raise RenderError(f"{label}: empty workload command")
    binary = Path(argv[0]).name
    binary_prefix = "spm_flash_attn" if is_spm else "cache_flash_attn"
    if binary != binary_prefix and not binary.startswith(binary_prefix + "_"):
        raise RenderError(
            f"{label}: workload binary {binary!r} is not a {binary_prefix} binary"
        )
    expected_args = [
        identity.group("t"),
        "32",
        "8",
        "64",
        "1",
        identity.group("br"),
        identity.group("bc"),
    ]
    if is_spm:
        expected_args.append("0")
    expected_args.append("noncausal")
    if argv[1:] != expected_args:
        raise RenderError(
            f"{label}: workload args {argv[1:]!r}, expected {expected_args!r}"
        )
    executable = config.get("system.cpu.workload", "executable", fallback="")
    if Path(executable).name != binary:
        raise RenderError(
            f"{label}: executable {executable!r} disagrees with cmd binary {argv[0]!r}"
        )


def parse_cell(pool: str, vl: int, name: str, is_spm: bool) -> dict[str, object]:
    path = cell_dir(pool, vl, name)
    label = f"VL={vl}/{name}"
    if not path.is_dir():
        raise RenderError(f"missing cell directory: {path}")
    parse_config(path, vl, name, is_spm)

    log_text = read_required(path / "stdout.log")
    if BAD_LOG_RE.search(log_text):
        raise RenderError(f"{label}: crash or validation-failure marker in stdout.log")
    exit_count = log_text.count(CLEAN_EXIT)
    if exit_count != 1:
        raise RenderError(f"{label}: expected one clean exit, found {exit_count}")
    modes = MODE_RE.findall(log_text)
    if modes != ["noncausal"]:
        raise RenderError(f"{label}: expected one ATTENTION_MODE=noncausal receipt")
    checksum = one_finite(
        CHECKSUM_RE.findall(log_text), f"{label}/CHECKSUM", positive=False
    )

    if is_spm:
        logged_ms = one_finite(AVG_RE.findall(log_text), f"{label}/avg", positive=True)
        iter0_ms = one_finite(
            ITER0_RE.findall(log_text), f"{label}/iter 0", positive=True
        )
        if not math.isclose(logged_ms, iter0_ms, rel_tol=5.0e-3, abs_tol=0.01):
            raise RenderError(f"{label}: avg and iter-0 receipts disagree")
    else:
        logged_ms = one_finite(
            COLD_RE.findall(log_text), f"{label}/COLD", positive=True
        )

    stats_us = roi_time_us(path / "stats.txt", label)
    logged_us = logged_ms * 1000.0
    if abs(stats_us - logged_us) > max(10.0, 0.005 * stats_us):
        raise RenderError(
            f"{label}: log time {logged_us:.3f} us disagrees with "
            f"first ROI {stats_us:.3f} us"
        )
    return {"time_us": logged_us, "roi_time_us": stats_us, "checksum": checksum}


def validate_companion_identity() -> None:
    expected = {
        (f"vl{vl}", name)
        for (vl, _), (pool, cache_name, spm_name) in CELLS.items()
        if pool == "companion"
        for name in (cache_name, spm_name)
    }
    actual = {
        (block.name, cell.name)
        for block in COMPANION_ROOT.glob("vl*")
        if block.is_dir()
        for cell in block.iterdir()
        if cell.is_dir() and ".prev." not in cell.name
    }
    if actual != expected:
        raise RenderError(
            "companion cell set mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def collect(check: bool) -> dict[str, object]:
    validate_companion_identity()
    records: list[dict[str, object]] = []
    for vl in VECTOR_LENGTHS:
        for sequence_length in SEQUENCE_LENGTHS:
            pool, cache_name, spm_name = CELLS[(vl, sequence_length)]
            cache = parse_cell(pool, vl, cache_name, is_spm=False)
            spm = parse_cell(pool, vl, spm_name, is_spm=True)
            if not math.isclose(
                float(cache["checksum"]),
                float(spm["checksum"]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise RenderError(
                    f"VL={vl}/T={sequence_length}: cache/SPM checksums disagree"
                )

            cache_time_us = float(cache["time_us"])
            spm_time_us = float(spm["time_us"])
            measured_gain = (cache_time_us - spm_time_us) / spm_time_us * 100.0
            selected_mode = "cache" if (vl, sequence_length) in CACHE_MODE else "spm"
            reported_gain = 0.0 if selected_mode == "cache" else round(measured_gain, 1)
            expected_gain = EXPECTED_REPORTED_GAIN[(vl, sequence_length)]
            if (
                check
                and selected_mode == "spm"
                and abs(measured_gain - expected_gain) > CHECK_TOLERANCE_PP
            ):
                raise RenderError(
                    f"VL={vl}/T={sequence_length}: measured {measured_gain:+.3f}% "
                    f"does not reproduce {expected_gain:+.1f}%"
                )
            if check and reported_gain != expected_gain:
                raise RenderError(
                    f"VL={vl}/T={sequence_length}: selected value "
                    f"{reported_gain:+.1f}% does not match {expected_gain:+.1f}%"
                )

            records.append(
                {
                    "vector_length": vl,
                    "sequence_length": sequence_length,
                    "kv_bytes": sequence_length * KV_BYTES_PER_TOKEN,
                    "kv_l2_pct": sequence_length * KV_BYTES_PER_TOKEN / L2_BYTES * 100.0,
                    "cache_time_us": cache_time_us,
                    "spm_time_us": spm_time_us,
                    "measured_gain_pct": measured_gain,
                    "selected_mode": selected_mode,
                    "reported_gain_pct": reported_gain,
                    "source_pool": (
                        f"simulation_results/vl{vl}"
                        if pool == "endpoint"
                        else f"table4/vl{vl}"
                    ),
                    "source_cells": {"cache": cache_name, "spm": spm_name},
                }
            )

    return {
        "schema_version": 1,
        "table": "Table IV",
        "metric": "(cache_time_us - spm_time_us) / spm_time_us",
        "units": {"time": "us", "gain": "percent"},
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "l2_bytes": L2_BYTES,
        "records": records,
    }


def record_map(data: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(record["vector_length"]), int(record["sequence_length"])): record
        for record in data["records"]  # type: ignore[index]
    }


def format_kv(kv_bytes: int) -> str:
    return f"{kv_bytes // 1024}\\,KB"


def latex_gain(record: dict[str, object]) -> str:
    if record["selected_mode"] == "cache":
        return "0"
    return f"${float(record['reported_gain_pct']):+.1f}\\%$"


def markdown_gain(record: dict[str, object]) -> str:
    if record["selected_mode"] == "cache":
        return "0 (cache mode)"
    return f"{float(record['reported_gain_pct']):+.1f}%"


def to_latex(data: dict[str, object]) -> str:
    by_key = record_map(data)
    lines = [
        "% Auto-generated by plot_table4.py -- do not edit by hand.",
        "\\begin{tabular}{rrrll}",
        "\\toprule",
        "$T$ & $K{+}V$ & $K{+}V$/L2 & VL\\,=\\,4 & VL\\,=\\,16 \\\\",
        "\\midrule",
    ]
    for sequence_length in SEQUENCE_LENGTHS:
        first = by_key[(4, sequence_length)]
        lines.append(
            f"{sequence_length} & {format_kv(int(first['kv_bytes']))} & "
            f"{float(first['kv_l2_pct']):.0f}\\% & "
            f"{latex_gain(first)} & {latex_gain(by_key[(16, sequence_length)])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def to_markdown(data: dict[str, object]) -> str:
    by_key = record_map(data)
    lines = [
        "| T | K+V | K+V/L2 | VL=4 | VL=16 |",
        "|---:|---:|---:|:---|:---|",
    ]
    for sequence_length in SEQUENCE_LENGTHS:
        first = by_key[(4, sequence_length)]
        lines.append(
            f"| {sequence_length} | {int(first['kv_bytes']) // 1024} KB | "
            f"{float(first['kv_l2_pct']):.0f}% | {markdown_gain(first)} | "
            f"{markdown_gain(by_key[(16, sequence_length)])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        data = collect(args.check)
    except (RenderError, OSError) as error:
        print(f"TABLE4_FAIL: {error}", file=sys.stderr)
        return 1

    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "table4_data.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (out / "table4.tex").write_text(to_latex(data), encoding="utf-8")
    (out / "table4.md").write_text(to_markdown(data), encoding="utf-8")

    print("Table IV: CacheFlex FlashAttention gain on LLaMA")
    print(to_markdown(data), end="")
    if args.check:
        print(
            "TABLE4_CHECK_PASS: 5 measured SPM selections match within "
            f"{CHECK_TOLERANCE_PP} pp; 3 selections remain in cache mode"
        )
    print(f"wrote {out / 'table4_data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
