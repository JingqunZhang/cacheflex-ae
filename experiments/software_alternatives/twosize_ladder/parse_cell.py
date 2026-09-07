#!/usr/bin/env python3
"""Extract one complete Figure 8 phase-time receipt from a gem5 stdout log.

Every consumed field must occur exactly once and be a finite number.  Invalid
or ambiguous logs raise ``ValueError`` so callers cannot silently plot a
partial receipt.
"""
import json
import math
import re
import sys


FIELDS = {
    "checksum": r"^\s*CHECKSUM:\s*(\S+)\s*$",
    "checksum_l1": r"^\s*CHECKSUM_LOGICAL_L1:\s*(\S+)\s*$",
    "checksum_weighted": (
        r"^\s*CHECKSUM_LOGICAL_WEIGHTED_L1:\s*(\S+)\s*$"
    ),
    "pack_B": r"^\s*pack_B/SPMCP\s*:\s*(\S+)\s*us\b",
    "pack_A": r"^\s*pack_A\s*:\s*(\S+)\s*us\b",
    "kernel": r"^\s*kernel\s*:\s*(\S+)\s*us\b",
    "C_write": r"^\s*C_write\(k0=0\)\s*:\s*(\S+)\s*us\b",
    "C_RMW": r"^\s*C_RMW \(k0>0\)\s*:\s*(\S+)\s*us\b",
    "total": r"^\s*total\s*:\s*(\S+)\s*us\b",
    "gflops": r"(\S+)\s*GFLOPS\b",
}
PHASE_FIELDS = ("pack_B", "pack_A", "kernel", "C_write", "C_RMW")


def _one_finite(text, field, pattern):
    matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(
            f"{field}: expected exactly one receipt, found {len(matches)}"
        )
    try:
        value = float(matches[0])
    except ValueError as error:
        raise ValueError(f"{field}: not numeric: {matches[0]!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{field}: must be finite, got {value!r}")
    return value

def parse(path):
    with open(path, encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    result = {
        field: _one_finite(text, field, pattern)
        for field, pattern in FIELDS.items()
    }

    for field in PHASE_FIELDS:
        if result[field] < 0:
            raise ValueError(
                f"{field}: phase time must be non-negative, got {result[field]!r}"
            )
    for field in ("total", "gflops"):
        if result[field] <= 0:
            raise ValueError(
                f"{field}: must be positive, got {result[field]!r}"
            )
    if result["checksum_l1"] <= 0:
        raise ValueError(
            "checksum_l1: must be positive, "
            f"got {result['checksum_l1']!r}"
        )
    if result["checksum_weighted"] <= 0:
        raise ValueError(
            "checksum_weighted: must be positive, "
            f"got {result['checksum_weighted']!r}"
        )
    checksum_tolerance = (
        1.0e-6
        * max(result["checksum_l1"], result["checksum_weighted"], 1.0)
    )
    if (
        result["checksum_weighted"]
        < result["checksum_l1"] - checksum_tolerance
        or result["checksum_weighted"]
        > 2.0 * result["checksum_l1"] + checksum_tolerance
    ):
        raise ValueError(
            "checksum_weighted: outside valid [checksum_l1, "
            "2*checksum_l1] bounds"
        )

    phase_total = sum(result[field] for field in PHASE_FIELDS)
    tolerance = max(1.0, result["total"] * 1.0e-5)
    if abs(phase_total - result["total"]) > tolerance:
        raise ValueError(
            f"phase sum {phase_total:.6g} us does not match "
            f"total {result['total']:.6g} us"
        )
    return result

if __name__ == '__main__':
    print(json.dumps(parse(sys.argv[1]), indent=2))
