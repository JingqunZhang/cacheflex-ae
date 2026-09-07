#!/usr/bin/env python3
"""Validate the canonical Figure 7 result set.

Checks every expected cell for:
  * the canonical tol2bus latency configuration and expected SVE VL;
  * a completed log, config.ini, and stats.txt;
  * workload dimensions, binary family, and checksum consistency;
  * SPM activity only for the CacheFlex variant.

Usage:
  python3 audit_results.py [RESULTS_DIR]

RESULTS_DIR defaults to results/reference/results_headh0.
"""
from pathlib import Path
import math
import re
import sys


HERE = Path(__file__).resolve().parent
RESULTS = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else HERE / "results" / "reference" / "results_headh0"
)

DIMS = {
    "W1": (128, 4096, 11008),
    "W2": (256, 4096, 4096),
    "W3": (256, 2048, 2048),
    "W4": (512, 768, 768),
    "W5": (2048, 2048, 2048),
    "W6": (2048, 64, 2048),
    "W7": (784, 256, 1024),
}
EXPECTED_BUS = {
    "frontend_latency": "1",
    "header_latency": "0",
    "forward_latency": "1",
    "response_latency": "1",
    "width": "64",
}
EXPECTED = [
    (workload, variant, vl)
    for workload in DIMS
    for variant in ("v1", "v3")
    for vl in (4, 8, 16)
]


def section(text, name):
    match = re.search(
        rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[)",
        text,
        re.S | re.M,
    )
    return match.group(1) if match else ""


def value(text, key):
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.M)
    return match.group(1).strip() if match else None


def command(config_text):
    for name in re.findall(r"^\[([^\]]+)\]", config_text, re.M):
        candidate = value(section(config_text, name), "cmd")
        if candidate:
            return candidate.split()
    return []


def checksum(log_text):
    """Return the three unique logical-output invariants."""
    values = []
    for receipt in (
        "CHECKSUM_LOGICAL",
        "CHECKSUM_LOGICAL_L1",
        "CHECKSUM_LOGICAL_WEIGHTED_L1",
    ):
        matches = re.findall(
            rf"(?m)^\s*{receipt}:\s*([0-9.eE+-]+)\s*$",
            log_text,
        )
        if len(matches) != 1:
            return None
        try:
            result = float(matches[0])
        except ValueError:
            return None
        if not math.isfinite(result):
            return None
        values.append(result)
    if values[1] <= 0 or values[2] <= 0:
        return None
    bound_tolerance = 1.0e-6 * max(values[1], values[2], 1.0)
    if (
        values[2] < values[1] - bound_tolerance
        or values[2] > 2.0 * values[1] + bound_tolerance
    ):
        return None
    return tuple(values)


def validate_cell(workload, variant, vl):
    label = f"{workload}_{variant}_vl{vl}"
    run_dir = RESULTS / label
    log_path = RESULTS / f"{label}.log"
    problems = []

    if not run_dir.is_dir():
        return [f"{label}: missing result directory"]
    for required in ("config.ini", "stats.txt"):
        if not (run_dir / required).is_file():
            problems.append(f"missing {required}")
    if not log_path.is_file():
        problems.append("missing log")
    if problems:
        return [f"{label}: {', '.join(problems)}"]

    config_text = (run_dir / "config.ini").read_text(errors="ignore")
    bus = section(config_text, "system.tol2bus")
    for key, expected in EXPECTED_BUS.items():
        actual = value(bus, key)
        if actual != expected:
            problems.append(f"tol2bus.{key}={actual}, expected {expected}")

    isa = section(config_text, "system.cpu.isa")
    if value(isa, "sve_vl_se") != str(vl):
        problems.append(f"unexpected SVE VL {value(isa, 'sve_vl_se')}")

    args = command(config_text)
    expected_dims = tuple(map(str, DIMS[workload]))
    if len(args) < 4:
        problems.append("workload command missing")
    else:
        if tuple(args[1:4]) != expected_dims:
            problems.append(
                f"workload dimensions {tuple(args[1:4])}, expected {expected_dims}"
            )
        binary = Path(args[0]).name
        if not binary.startswith(f"{variant}_"):
            problems.append(f"binary {binary} does not match {variant}")

    log_text = log_path.read_text(errors="ignore")
    if "GFLOPS" not in log_text:
        problems.append("GFLOPS receipt missing")
    if checksum(log_text) is None:
        problems.append(
            "logical sum/L1/weighted checksum missing, ambiguous, or invalid"
        )
    if "[DBG]" in log_text:
        problems.append("debug trace present")
    if re.search(
        r"panic:|fatal:|segmentation fault|segfault|core dumped|"
        r"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
        r"assert(?:ion)?[^\n]{0,256}failed",
        log_text,
        re.I,
    ):
        problems.append("crash marker present")
    if "Exiting @ tick" not in log_text:
        problems.append("clean gem5 exit missing")

    stats_text = (run_dir / "stats.txt").read_text(errors="ignore")
    match = re.search(r"^system\.l2\.spmReads\s+(\d+)", stats_text, re.M)
    spm_reads = int(match.group(1)) if match else 0
    if variant == "v1" and spm_reads:
        problems.append(f"baseline has {spm_reads} SPM reads")
    if variant == "v3" and not spm_reads:
        problems.append("CacheFlex has no SPM reads")
    if stats_text.count("Begin Simulation Statistics") < 2:
        problems.append("ROI statistics block missing")

    return [f"{label}: {', '.join(problems)}"] if problems else []


def main():
    if not RESULTS.is_dir():
        raise SystemExit(f"ERROR: result directory not found: {RESULTS}")

    failures = []
    for workload, variant, vl in EXPECTED:
        failures.extend(validate_cell(workload, variant, vl))

    for workload in DIMS:
        for vl in (4, 8, 16):
            left_path = RESULTS / f"{workload}_v1_vl{vl}.log"
            right_path = RESULTS / f"{workload}_v3_vl{vl}.log"
            if not left_path.is_file() or not right_path.is_file():
                continue
            left = checksum(
                left_path.read_text(errors="ignore")
            )
            right = checksum(
                right_path.read_text(errors="ignore")
            )
            if left is None or right is None:
                continue
            for field, left_value, right_value in zip(
                ("sum", "L1", "weighted"), left, right
            ):
                tolerance = (
                    max(abs(left_value), abs(right_value)) * 0.015 + 1e-9
                )
                if abs(left_value - right_value) > tolerance:
                    failures.append(
                        f"{workload}/VL{vl}: {field} checksum mismatch "
                        f"{left_value} vs {right_value}"
                    )

    if failures:
        print(f"AUDIT FAILED: {len(failures)} issue(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"AUDIT PASS: {len(EXPECTED)} canonical cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
