#!/usr/bin/env python3
"""Render Table 4 (tab:flash_scaling) from the shipped reference results.

Table 4 reports the measured CacheFlex SPM performance change of the
FlashAttention kernel (CF+FA vs. BL+FA) on LLaMA across per-head K+V
working-set sizes, at VL=4 and VL=16.  The gain for each (VL, T) point is

    gain = (t_cflash_cold - t_sflash_cold) / t_sflash_cold

where t_cflash_cold is the cold-cache single-iteration time of the
cache-resident FlashAttention baseline (BL+FA, cache_flash_attn) and
t_sflash_cold is the cold-cache time of the SPM CacheFlex variant
(CF+FA, spm_flash_attn).

The T=256 and T=4096 endpoints are the exact same flash cells that
Figure 9 consumes, read straight from the frozen end-to-end reference
(experiments/end2end/results/reference/simulation_results).  The
intermediate T=1024 and T=2048 points live in a small, self-contained
companion pool (experiments/end2end/table4/reference) so that Table 4 can
be regenerated with no gem5 simulation while leaving the frozen Figure 9
108-cell contract untouched.

This script never simulates.  It parses the source-emitted cold-run
timing receipts, validates that every cell recorded the paper CPU
configuration, and (with --check) asserts that the rendered gains match
the values printed in the paper.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # experiments/end2end

# Endpoints (T=256, 4096) are read from the frozen Figure 9 reference; the
# intermediate points (T=1024, 2048) from the self-contained Table 4 pool.
FIG9_REFERENCE = BASE / "results" / "reference" / "simulation_results"
TABLE4_REFERENCE = BASE / "table4" / "reference"

# Exact cell selection behind each submitted-version Table 4 entry.  "fig9" cells
# come from the shared Figure 9 reference; "table4" cells from the
# companion pool.  Every pair is LLaMA cache_flash_attn vs spm_flash_attn.
CELLS: dict[tuple[str, int], tuple[str, str, str]] = {
    ("vl4", 256): ("fig9", "llama_T256_cflash_br64_bc576", "llama_T256_sflash_br64_bc384"),
    ("vl4", 1024): ("table4", "llama_T1024_cflash_br64_bc576", "llama_T1024_sflash_br32_bc576"),
    ("vl4", 2048): ("table4", "llama_T2048_cflash_br64_bc576", "llama_T2048_sflash_br64_bc576"),
    ("vl4", 4096): ("fig9", "llama_T4096_cflash_br64_bc576", "llama_T4096_sflash_br64_bc576"),
    ("vl16", 256): ("fig9", "llama_T256_cflash_br64_bc256", "llama_T256_sflash_br32_bc256"),
    ("vl16", 1024): ("table4", "llama_T1024_cflash_br64_bc384", "llama_T1024_sflash_br64_bc384"),
    ("vl16", 2048): ("table4", "llama_T2048_cflash_br64_bc384", "llama_T2048_sflash_br64_bc384"),
    ("vl16", 4096): ("fig9", "llama_T4096_cflash_br64_bc384", "llama_T4096_sflash_br32_bc384"),
}

# Submitted-version Table 4 gains (percent), for the --check reproduction
# gate: it certifies the frozen reference against the submitted version.
# Equal-treatment results are documented in docs/CLAIMS.md point 3.
PAPER_GAIN = {
    ("vl4", 256): 8, ("vl4", 1024): 3, ("vl4", 2048): 10, ("vl4", 4096): 14,
    ("vl16", 256): 9, ("vl16", 1024): -1, ("vl16", 2048): 22, ("vl16", 4096): 17,
}

SEQUENCE_LENGTHS = (256, 1024, 2048, 4096)
# fp16 K and V, head_dim = 64  ->  2 tensors * 64 lanes * 2 bytes = 256 B/token.
KV_BYTES_PER_TOKEN = 256
# Tolerance (percentage points) between the freshly parsed gain and the
# submitted-version integer value (that table rounds to whole percent).
CHECK_TOLERANCE_PP = 1.5

CLEAN_EXIT = "Exiting @ tick"
# The frozen Figure 9 reference intentionally ships a few vl4 T=4096 flash
# baselines that reached their timing receipt but were stopped before the
# clean-exit line (mirrors reference_integrity.LEGACY_INCOMPLETE_EXIT_CELLS).
# Their cold timing receipt is still authoritative.
LEGACY_NO_CLEAN_EXIT = {
    ("vl4", "llama_T4096_cflash_br64_bc576"),
}
BAD_LOG_RE = re.compile(
    r"(?i)panic:|fatal:|segmentation fault|\bsegfault\b|core dumped|"
    r"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
    r"\[DBG\]"
)
BLOCK_VL = {"vl4": "4", "vl16": "16"}
BLOCK_SPM_LOAD_PORTS = {"vl4": "2", "vl16": "1"}
EXPECTED_L2_SIZE = "524288"

COLD_RE = re.compile(r"(?mi)^\s*COLD:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
ITER0_RE = re.compile(r"(?mi)^\s*iter\s+0:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
AVG_RE = re.compile(r"(?mi)^\s*avg:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")


class RenderError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the rendered gains reproduce the submitted-version Table 4",
    )
    return parser.parse_args()


def pool_root(source: str) -> Path:
    return FIG9_REFERENCE if source == "fig9" else TABLE4_REFERENCE


def cell_dir(vl: str, source: str, name: str) -> Path:
    return pool_root(source) / vl / name


def read_log(path: Path, allow_incomplete: bool = False) -> str:
    log = path / "stdout.log"
    if not log.is_file() or log.stat().st_size == 0:
        raise RenderError(f"missing/empty stdout.log: {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if BAD_LOG_RE.search(text):
        raise RenderError(f"crash/validation marker in log: {log}")
    if CLEAN_EXIT not in text and not allow_incomplete:
        raise RenderError(f"no clean gem5 exit: {log}")
    return text


def cold_ms(text: str, label: str) -> float:
    """Cold-cache, single-iteration time in ms.

    cache_flash_attn prints an explicit COLD receipt; spm_flash_attn prints
    the first (and only) iteration as 'iter 0:' / 'avg:'.  Both denote the
    same cold single-iteration measurement.
    """
    for pattern in (COLD_RE, ITER0_RE, AVG_RE):
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            if value <= 0:
                raise RenderError(f"non-positive timing receipt in {label}")
            return value
    raise RenderError(f"no COLD/iter0/avg timing receipt in {label}")


def check_config(path: Path, vl: str, binary: str, label: str) -> list[str]:
    problems: list[str] = []
    config_path = path / "config.ini"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        return [f"{label}: missing/empty config.ini"]
    config = configparser.ConfigParser(interpolation=None, strict=True)
    try:
        with config_path.open(encoding="utf-8") as stream:
            config.read_file(stream)
    except (OSError, configparser.Error) as error:
        return [f"{label}: unparseable config.ini: {error}"]

    checks = {
        ("system.cpu.isa", "sve_vl_se"): BLOCK_VL[vl],
        ("system.l2", "size"): EXPECTED_L2_SIZE,
        ("system.cpu", "spmloadports"): BLOCK_SPM_LOAD_PORTS[vl],
        ("system.cpu", "spmstoreports"): "1",
        ("system.cpu", "numROBEntries"): "128",
        ("system.cpu", "LQEntries"): "32",
        ("system.cpu", "SQEntries"): "48",
    }
    for (section, option), wanted in checks.items():
        got = config.get(section, option, fallback=None)
        if got is not None:
            got = got.strip()
        if got != wanted:
            problems.append(
                f"{label}: {section}.{option}={got!r}, expected {wanted!r}"
            )

    command = config.get("system.cpu.workload", "cmd", fallback="")
    actual_binary = Path(command.split()[0]).name if command.split() else None
    if actual_binary != binary:
        problems.append(
            f"{label}: workload binary {actual_binary!r}, expected {binary!r}"
        )
    return problems


def render(check: bool) -> dict[str, object]:
    problems: list[str] = []
    rows: list[dict[str, object]] = []
    gains: dict[tuple[str, int], float] = {}

    for (vl, seq), (source, cflash, sflash) in CELLS.items():
        label = f"{vl}/T{seq}"
        cflash_dir = cell_dir(vl, source, cflash)
        sflash_dir = cell_dir(vl, source, sflash)
        cflash_text = read_log(
            cflash_dir, allow_incomplete=(vl, cflash) in LEGACY_NO_CLEAN_EXIT
        )
        sflash_text = read_log(
            sflash_dir, allow_incomplete=(vl, sflash) in LEGACY_NO_CLEAN_EXIT
        )
        cflash_ms = cold_ms(cflash_text, f"{label} BL+FA {cflash}")
        sflash_ms = cold_ms(sflash_text, f"{label} CF+FA {sflash}")
        gain = (cflash_ms - sflash_ms) / sflash_ms * 100.0
        gains[(vl, seq)] = gain
        problems += check_config(cflash_dir, vl, "cache_flash_attn", f"{label} BL+FA")
        problems += check_config(sflash_dir, vl, "spm_flash_attn", f"{label} CF+FA")

    for seq in SEQUENCE_LENGTHS:
        rows.append(
            {
                "T": seq,
                "kv_bytes": seq * KV_BYTES_PER_TOKEN,
                # Full-precision gains; the submitted-version table rounds to
                # whole percent, so round once for display (see signed()).
                "vl4_gain_pct": round(gains[("vl4", seq)], 2),
                "vl16_gain_pct": round(gains[("vl16", seq)], 2),
            }
        )

    if check:
        for (vl, seq), gain in gains.items():
            published = PAPER_GAIN[(vl, seq)]
            if abs(round(gain) - published) > CHECK_TOLERANCE_PP:
                problems.append(
                    f"{vl}/T{seq}: rendered gain {gain:+.1f}% does not match "
                    f"submitted-version {published:+d}% "
                    f"(tol {CHECK_TOLERANCE_PP} pp)"
                )
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        raise RenderError(f"Table 4 render found {len(problems)} problem(s)")

    return {
        "table": "Table 4 (tab:flash_scaling)",
        "metric": "(t_BL+FA_cold - t_CF+FA_cold) / t_CF+FA_cold, LLaMA",
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "rows": rows,
    }


def format_kv(kv_bytes: int) -> str:
    kb = kv_bytes // 1024
    return f"{kb}\\,KB"


def signed(pct: float) -> str:
    return f"{pct:+.0f}\\%"


def to_latex(data: dict[str, object]) -> str:
    lines = [
        "% Auto-generated by plot_table4_flash.py -- do not edit by hand.",
        "\\begin{tabular}{r r r r}",
        "\\toprule",
        "$T$ & $K{+}V$ & VL\\,=\\,4 & VL\\,=\\,16 \\\\",
        "\\midrule",
    ]
    for row in data["rows"]:  # type: ignore[index]
        lines.append(
            f"{row['T']} & {format_kv(int(row['kv_bytes']))} & "
            f"{signed(float(row['vl4_gain_pct']))} & "
            f"{signed(float(row['vl16_gain_pct']))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def to_markdown(data: dict[str, object]) -> str:
    lines = [
        "| T | K+V | VL=4 | VL=16 |",
        "|---:|---:|---:|---:|",
    ]
    for row in data["rows"]:  # type: ignore[index]
        kb = int(row["kv_bytes"]) // 1024
        lines.append(
            f"| {row['T']} | {kb} KB | "
            f"{float(row['vl4_gain_pct']):+.0f}% | "
            f"{float(row['vl16_gain_pct']):+.0f}% |"
        )
    return "\n".join(lines) + "\n"


def output_dir() -> Path:
    override = os.environ.get("OUTPUT_DIR")
    if override:
        return Path(override)
    return BASE / "results" / "generated" / "reference_outputs"


def main() -> int:
    args = parse_args()
    try:
        data = render(args.check)
    except RenderError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "table4_data.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (out / "table4_flash.tex").write_text(to_latex(data), encoding="utf-8")
    (out / "table4_flash.md").write_text(to_markdown(data), encoding="utf-8")

    print("Table 4 (tab:flash_scaling): CF+FA vs BL+FA on LLaMA")
    print(to_markdown(data), end="")
    if args.check:
        print(
            "TABLE4_CHECK_PASS: 8 flash pairs reproduce the "
            f"submitted-version gains within {CHECK_TOLERANCE_PP} pp"
        )
    print(f"wrote {out / 'table4_data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
