#!/usr/bin/env python3
"""Select the fastest cell from one exact, freshly generated sweep grid.

The caller supplies the complete KC and MC domains.  Candidate discovery by
glob is deliberately forbidden: every declared cell must exist, every
candidate-like non-archive entry must be declared, and every cell must carry
the current sweep invocation identifier plus a complete gem5 receipt.
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path
import re
import sys


REQUIRED_RECEIPTS = (
    "stdout.txt",
    "stats.txt",
    "BINARY_USED.txt",
    "RUNNER_USED.txt",
    "BINARIES.sha256",
    "RUN_ENV.txt",
    "SWEEP_INVOCATION_ID",
)
BAD_OUTPUT = re.compile(
    r"panic:|fatal:|segmentation fault|segfault|core dumped|"
    r"simulated exit code not 0|validation failed|\[DBG\]|"
    r"assert(?:ion)?[^\n]{0,256}failed",
    re.IGNORECASE,
)


def fail(message):
    raise ValueError(message)


def parse_axis(value, name):
    tokens = value.split(",")
    if not tokens or any(not token for token in tokens):
        fail(f"{name}: expected a non-empty comma-separated integer list")
    try:
        numbers = [int(token) for token in tokens]
    except ValueError as error:
        fail(f"{name}: all entries must be integers ({value!r})")
    if any(number <= 0 for number in numbers):
        fail(f"{name}: all entries must be positive ({value!r})")
    if len(numbers) != len(set(numbers)):
        fail(f"{name}: duplicate entries are not allowed ({value!r})")
    return numbers


def load_parser(path):
    spec = importlib.util.spec_from_file_location("fig8_parse_cell", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load cell parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "parse", None)):
        fail(f"cell parser has no callable parse(): {path}")
    return module.parse


def require_regular_nonempty(path, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label}: missing regular file {path}")
    if path.stat().st_size == 0:
        fail(f"{label}: empty receipt {path}")


def validate_cell(cell, invocation_id, parse_cell):
    label = cell.name
    if cell.is_symlink() or not cell.is_dir():
        fail(f"{label}: expected a real directory, not a symlink")
    for receipt in REQUIRED_RECEIPTS:
        require_regular_nonempty(cell / receipt, label)

    recorded_id = (cell / "SWEEP_INVOCATION_ID").read_text(
        encoding="utf-8"
    ).strip()
    if recorded_id != invocation_id:
        fail(
            f"{label}: stale or foreign invocation marker "
            f"{recorded_id!r}; expected {invocation_id!r}"
        )

    stdout_path = cell / "stdout.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    if stdout.count("Exiting @ tick") != 1:
        fail(
            f"{label}: expected exactly one successful gem5 exit receipt, "
            f"found {stdout.count('Exiting @ tick')}"
        )
    bad = BAD_OUTPUT.search(stdout)
    if bad:
        fail(f"{label}: failure marker in stdout: {bad.group(0)!r}")

    result = parse_cell(str(stdout_path))
    gflops = result.get("gflops")
    if (
        isinstance(gflops, bool)
        or not isinstance(gflops, (int, float))
        or not math.isfinite(gflops)
        or gflops <= 0
    ):
        fail(f"{label}: invalid GFLOPS value {gflops!r}")
    return float(gflops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--kcs", required=True)
    parser.add_argument("--mcs", required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--parser", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.prefix):
        fail(f"unsafe candidate prefix: {args.prefix!r}")
    if not args.invocation_id or "\n" in args.invocation_id:
        fail("invalid sweep invocation identifier")

    out = Path(args.out)
    if out.is_symlink() or not out.is_dir():
        fail(f"sweep output is not a real directory: {out}")
    kcs = parse_axis(args.kcs, "KCS")
    mcs = parse_axis(args.mcs, "MCS")
    parse_cell = load_parser(args.parser)

    ordered = [
        (kc, mc, f"{args.prefix}_kc{kc}_mc{mc}")
        for kc in kcs
        for mc in mcs
    ]
    expected = {name for _, _, name in ordered}

    extras = []
    candidate_stem = f"{args.prefix}_kc"
    archive_suffix = re.compile(r"\.prev\.[0-9]+(?:\.[0-9]+)?$")
    for entry in os.scandir(out):
        name = entry.name
        if not name.startswith(candidate_stem):
            continue
        # move_aside() creates only this exact archive suffix.  A looser
        # substring exception would let malformed live candidates escape the
        # undeclared-entry gate.
        if archive_suffix.search(name):
            continue
        if name not in expected:
            extras.append(name)
    if extras:
        fail(
            f"{args.prefix}: undeclared candidate entries present: "
            + ", ".join(sorted(extras))
        )

    best = None
    for kc, mc, name in ordered:
        cell = out / name
        if not cell.exists() and not cell.is_symlink():
            fail(f"{args.prefix}: missing declared candidate {name}")
        gflops = validate_cell(cell, args.invocation_id, parse_cell)
        # Strict '>' plus the declared KC-major/MC-minor order gives a stable
        # tie-break independent of filesystem enumeration order.
        if best is None or gflops > best[0]:
            best = (gflops, kc, mc)

    if best is None:
        fail(f"{args.prefix}: empty candidate grid")
    print(f"{best[1]} {best[2]}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: exact Figure 8 sweep selection failed: {error}", file=sys.stderr)
        sys.exit(2)
