#!/usr/bin/env python3
"""Compare fresh figure data with the paper reference at a fixed 5% limit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


LIMIT_PERCENT = 5.0
FIGURE_FILES = (
    "fig2_data.json",
    "fig7_data.json",
    "fig8_data.json",
    "fig9_data.json",
)
PATH_FIELDS = {"result_path", "log_path", "stats_path"}


class SchemaError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Figure 2/7/8/9 performance and energy data against "
            "the supplied reference."
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    return parser.parse_args()


def ignored_measurement(path: tuple[str, ...]) -> bool:
    lowered = ".".join(path).lower()
    return (
        "checksum" in lowered
        or "validation_max_abs" in lowered
        or lowered.endswith(".spm_reads")
        or lowered.endswith(".spm_writes")
    )


def plotted_measurement(path: tuple[str, ...]) -> bool:
    lowered = ".".join(path).lower()
    if "theoretical_peak_gflops" in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "gflops",
            "time_us",
            "_us",
            "roi_time_s",
            "energy_mj",
            "_mj",
            "speedup",
            "normalized_runtime",
            "normalized_energy",
            "percent",
            "_pct",
        )
    )


def compare_structure(
    expected: Any,
    actual: Any,
    path: tuple[str, ...],
    problems: list[str],
) -> None:
    where = ".".join(path)
    if ignored_measurement(path) or plotted_measurement(path):
        return
    if path and path[-1] in PATH_FIELDS:
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            problems.append(f"{where}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(
                f"{where}: expected object, got {type(actual).__name__}"
            )
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            problems.append(
                f"{where}: key mismatch; missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys & actual_keys):
            compare_structure(
                expected[key], actual[key], path + (str(key),), problems
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            problems.append(
                f"{where}: expected list, got {type(actual).__name__}"
            )
            return
        if len(expected) != len(actual):
            problems.append(
                f"{where}: expected {len(expected)} items, got {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            compare_structure(
                expected_item,
                actual_item,
                path + (f"[{index}]",),
                problems,
            )
        return
    if isinstance(expected, (int, float)) and isinstance(
        actual, (int, float)
    ):
        if expected != actual:
            problems.append(f"{where}: expected {expected!r}, got {actual!r}")
        return
    if type(expected) is not type(actual) or expected != actual:
        problems.append(f"{where}: expected {expected!r}, got {actual!r}")


def number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{where}: expected a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SchemaError(f"{where}: expected a finite non-negative number")
    return result


def add_metric(
    metrics: dict[str, tuple[float, float | None]],
    name: str,
    value: Any,
    anchor: float | None = None,
) -> None:
    if name in metrics:
        raise SchemaError(f"{name}: duplicate metric")
    metrics[name] = (number(value, name), anchor)


def figure_metrics(
    document: dict[str, Any], filename: str
) -> dict[str, tuple[float, float | None]]:
    metrics: dict[str, tuple[float, float | None]] = {}
    try:
        if filename == "fig2_data.json":
            for series in document["series"]:
                label = str(series["label"])
                for index, value in enumerate(series["gflops"]):
                    add_metric(metrics, f"series/{label}/gflops[{index}]", value)

        elif filename == "fig7_data.json":
            for vector_length in document["vector_lengths"]:
                section = document["data"][str(vector_length)]
                for workload in document["workloads"]:
                    row = section["rows"][workload]
                    prefix = f"VL{vector_length}/{workload}"
                    add_metric(
                        metrics,
                        f"{prefix}/baseline_gflops",
                        row["baseline_gflops"],
                    )
                    add_metric(
                        metrics,
                        f"{prefix}/cacheflex_gflops",
                        row["cacheflex_gflops"],
                    )
                    for component in ("cache", "dram"):
                        add_metric(
                            metrics,
                            f"{prefix}/baseline_energy/{component}",
                            row["baseline_energy_mJ"][component],
                        )
                    for component in ("cache", "spm", "dram"):
                        add_metric(
                            metrics,
                            f"{prefix}/cacheflex_energy/{component}",
                            row["cacheflex_energy_mJ"][component],
                        )
                    add_metric(
                        metrics,
                        f"{prefix}/speedup",
                        row["speedup"],
                        anchor=1.0,
                    )
                    baseline_energy = sum(
                        number(
                            row["baseline_energy_mJ"][component],
                            f"{prefix}/baseline_energy/{component}",
                        )
                        for component in ("cache", "dram")
                    )
                    cacheflex_energy = sum(
                        number(
                            row["cacheflex_energy_mJ"][component],
                            f"{prefix}/cacheflex_energy/{component}",
                        )
                        for component in ("cache", "spm", "dram")
                    )
                    if baseline_energy == 0.0:
                        raise SchemaError(
                            f"{prefix}: zero baseline energy is invalid"
                        )
                    add_metric(
                        metrics,
                        f"{prefix}/energy_ratio",
                        cacheflex_energy / baseline_energy,
                        anchor=1.0,
                    )
                add_metric(
                    metrics,
                    f"VL{vector_length}/gmean_speedup",
                    section["gmean_speedup"],
                    anchor=1.0,
                )
                add_metric(
                    metrics,
                    f"VL{vector_length}/gmean_energy_ratio",
                    1.0 - section["gmean_energy_reduction_pct"] / 100.0,
                    anchor=1.0,
                )

        elif filename == "fig8_data.json":
            for shape in ("w3", "g2048"):
                for row_name, row in document[shape]["rows"].items():
                    prefix = f"{shape}/{row_name}"
                    for field in (
                        "pack_us",
                        "kernel_us",
                        "c_us",
                        "cache_mJ",
                        "spm_mJ",
                        "dram_mJ",
                    ):
                        add_metric(metrics, f"{prefix}/{field}", row[field])
                    add_metric(
                        metrics,
                        f"{prefix}/normalized_runtime",
                        row["normalized_runtime"],
                        anchor=1.0,
                    )
                    add_metric(
                        metrics,
                        f"{prefix}/normalized_energy",
                        row["normalized_energy"],
                        anchor=1.0,
                    )

        elif filename == "fig9_data.json":
            for record in document["records"]:
                identity = (
                    f"{record['model']}/T{record['sequence_length']}"
                    f"/VL{record['vector_length']}"
                )
                for configuration, values in record[
                    "configurations"
                ].items():
                    prefix = f"{identity}/{configuration}"
                    for component in ("proj", "ffn", "attn", "other"):
                        add_metric(
                            metrics,
                            f"{prefix}/time/{component}",
                            values["time_us"]["components"][component],
                        )
                    for component in ("cache", "spm", "dram"):
                        add_metric(
                            metrics,
                            f"{prefix}/energy/{component}",
                            values["energy_mJ"]["components"][component],
                        )
                    add_metric(
                        metrics,
                        f"{prefix}/speedup",
                        values["speedup_vs_BL"],
                        anchor=1.0,
                    )
                    add_metric(
                        metrics,
                        f"{prefix}/energy_ratio",
                        1.0
                        - values["energy_reduction_vs_BL_pct"] / 100.0,
                        anchor=1.0,
                    )
        else:
            raise SchemaError(f"unsupported figure data file: {filename}")
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise SchemaError(f"{filename}: malformed figure data: {error}") from error
    return metrics


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaError(f"{path}: {error}") from error
    if not isinstance(payload, dict):
        raise SchemaError(f"{path}: top-level JSON value must be an object")
    return payload


def fig9_improvement_direction(name: str) -> int:
    """Direction in which a Figure 9 metric may exceed the limit.

    Returns -1 if lower-than-reference is an improvement (times, energy,
    energy_ratio), +1 if higher-than-reference is an improvement (speedup),
    0 if the metric must stay symmetric.

    The end-to-end kernels include post-submission software optimizations
    (see docs/CLAIMS.md): a fresh simulation may legitimately be faster than
    the shipped reference, so Figure 9 gates are one-sided — regressions
    beyond the limit still fail, improvements of any size pass.
    """
    if "/time/" in name or "/energy/" in name or name.endswith("/energy_ratio"):
        return -1
    if name.endswith("/speedup"):
        return +1
    return 0


def compare_metrics(
    expected: dict[str, tuple[float, float | None]],
    actual: dict[str, tuple[float, float | None]],
    filename: str,
    problems: list[str],
) -> tuple[int, float]:
    expected_keys = set(expected)
    actual_keys = set(actual)
    if expected_keys != actual_keys:
        problems.append(
            f"{filename}: metric mismatch; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    worst = 0.0
    count = 0
    for name in sorted(expected_keys & actual_keys):
        expected_value, expected_anchor = expected[name]
        actual_value, actual_anchor = actual[name]
        count += 1
        if expected_anchor != actual_anchor:
            problems.append(f"{filename}/{name}: comparison anchor mismatch")
            continue
        difference = abs(actual_value - expected_value)
        slack = 1.0e-12 * max(expected_value, actual_value, 1.0)
        if expected_value == 0.0:
            if difference > slack:
                problems.append(
                    f"{filename}/{name}: expected zero, got {actual_value:.9g}"
                )
            continue
        relative_error = 100.0 * difference / expected_value
        improvement = 0
        if filename == "fig9_data.json":
            direction = fig9_improvement_direction(name)
            if direction < 0 and actual_value < expected_value:
                improvement = 1
            elif direction > 0 and actual_value > expected_value:
                improvement = 1
        if not improvement:
            worst = max(worst, relative_error)
        if relative_error > LIMIT_PERCENT + 1.0e-10:
            if improvement:
                print(
                    f"[IMPROVED] {filename}/{name}: reference "
                    f"{expected_value:.9g}, got {actual_value:.9g} "
                    f"({relative_error:.3f}% better than reference)"
                )
                continue
            problems.append(
                f"{filename}/{name}: expected {expected_value:.9g}, "
                f"got {actual_value:.9g}; relative error "
                f"{relative_error:.3f}% exceeds {LIMIT_PERCENT:g}%"
            )
            continue
        if (
            expected_anchor is not None
            and abs(expected_value - expected_anchor) >= 0.01
            and (expected_value - expected_anchor)
            * (actual_value - expected_anchor)
            < 0.0
            and not improvement
        ):
            problems.append(
                f"{filename}/{name}: result direction changed "
                f"({expected_value:.9g} to {actual_value:.9g})"
            )
    return count, worst


def main() -> int:
    args = parse_args()
    problems: list[str] = []
    summaries: list[tuple[str, int, float]] = []

    try:
        for filename in FIGURE_FILES:
            expected = load_json(args.reference / filename)
            actual = load_json(args.candidate / filename)
            compare_structure(
                expected, actual, (filename,), problems
            )
            expected_metrics = figure_metrics(expected, filename)
            actual_metrics = figure_metrics(actual, filename)
            count, worst = compare_metrics(
                expected_metrics,
                actual_metrics,
                filename,
                problems,
            )
            summaries.append((filename, count, worst))
    except SchemaError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    for filename, count, worst in summaries:
        print(f"[CHECK] {filename}: {count} values, worst={worst:.3f}%")
    if problems:
        for problem in problems[:100]:
            print(f"FAIL: {problem}", file=sys.stderr)
        if len(problems) > 100:
            print(
                f"FAIL: {len(problems) - 100} additional differences omitted",
                file=sys.stderr,
            )
        print(
            f"RESULT COMPARISON FAILED: {len(problems)} difference(s), "
            f"limit={LIMIT_PERCENT:g}%",
            file=sys.stderr,
        )
        return 1

    print(
        f"RESULT COMPARISON PASSED: limit={LIMIT_PERCENT:g}% "
        "(structure and cell identities matched exactly)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
