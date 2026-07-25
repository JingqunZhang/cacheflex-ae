#!/usr/bin/env python3
"""Validate or regenerate the checksums for the shipped reference results.

This is the single owner of results/reference/MANIFEST.sha256.  Its file
selection matches scripts/pack_artifact.sh: canonical reference pools only,
with heavyweight gem5 metadata omitted from the public artifact.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

import cell_manifest


CANONICAL_POOLS = {
    "capacity_motivation": (
        "results_fig2_n2048_headh0",
        "results_spm_fused_n2048_headh0",
    ),
    "vl_length": ("results_headh0",),
    "software_alternatives": ("best",),
    "end2end": ("simulation_results/vl4", "simulation_results/vl16"),
}

EXPECTED_CELL_COUNTS = {
    "capacity_motivation": 21,
    "vl_length": 42,
    "software_alternatives": 12,
    "end2end": 108,
}

CONFIG_LOCATIONS = {
    "cpu_type": ("system.cpu", "type"),
    "system_clock_period": ("system.clk_domain", "clock"),
    "cpu_clock_period": ("system.cpu_clk_domain", "clock"),
    "fetch_width": ("system.cpu", "fetchWidth"),
    "decode_width": ("system.cpu", "decodeWidth"),
    "commit_width": ("system.cpu", "commitWidth"),
    "rename_width": ("system.cpu", "renameWidth"),
    "dispatch_width": ("system.cpu", "dispatchWidth"),
    "issue_width": ("system.cpu", "issueWidth"),
    "writeback_width": ("system.cpu", "wbWidth"),
    "rob_entries": ("system.cpu", "numROBEntries"),
    "iq_entries": ("system.cpu", "numIQEntries"),
    "lq_entries": ("system.cpu", "LQEntries"),
    "sq_entries": ("system.cpu", "SQEntries"),
    "physical_int_registers": ("system.cpu", "numPhysIntRegs"),
    "physical_float_registers": ("system.cpu", "numPhysFloatRegs"),
    "physical_vector_registers": ("system.cpu", "numPhysVecRegs"),
    "physical_predicate_registers": (
        "system.cpu",
        "numPhysVecPredRegs",
    ),
    "cache_line_size": ("system", "cache_line_size"),
    "dcache_size": ("system.cpu.dcache", "size"),
    "dcache_assoc": ("system.cpu.dcache", "assoc"),
    "dcache_tag_latency": ("system.cpu.dcache", "tag_latency"),
    "dcache_data_latency": ("system.cpu.dcache", "data_latency"),
    "icache_size": ("system.cpu.icache", "size"),
    "icache_assoc": ("system.cpu.icache", "assoc"),
    "icache_tag_latency": ("system.cpu.icache", "tag_latency"),
    "icache_data_latency": ("system.cpu.icache", "data_latency"),
    "l2_size": ("system.l2", "size"),
    "l2_assoc": ("system.l2", "assoc"),
    "l2_tag_latency": ("system.l2", "tag_latency"),
    "l2_data_latency": ("system.l2", "data_latency"),
    "l2_response_latency": ("system.l2", "response_latency"),
    "l3_size": ("system.l3", "size"),
    "l3_assoc": ("system.l3", "assoc"),
    "l3_tag_latency": ("system.l3", "tag_latency"),
    "l3_data_latency": ("system.l3", "data_latency"),
    "l3_response_latency": ("system.l3", "response_latency"),
    "mshrs": ("system.cpu.dcache", "mshrs"),
    "header_latency": ("system.tol2bus", "header_latency"),
    "tol2_width": ("system.tol2bus", "width"),
    "tol2_frontend_latency": ("system.tol2bus", "frontend_latency"),
    "tol2_forward_latency": ("system.tol2bus", "forward_latency"),
    "tol2_response_latency": ("system.tol2bus", "response_latency"),
    "tol3_width": ("system.tol3bus", "width"),
    "tol3_frontend_latency": ("system.tol3bus", "frontend_latency"),
    "tol3_forward_latency": ("system.tol3bus", "forward_latency"),
    "tol3_response_latency": ("system.tol3bus", "response_latency"),
    "vl": ("system.cpu.isa", "sve_vl_se"),
    "cache_load_ports": ("system.cpu", "cacheloadports"),
    "cache_store_ports": ("system.cpu", "cachestoreports"),
    "spm_load_ports": ("system.cpu", "spmloadports"),
    "spm_store_ports": ("system.cpu", "spmstoreports"),
    "simd_alu_units": ("system.cpu.fuPool.FUList07", "count"),
    "simd_mult_units": ("system.cpu.fuPool.FUList09", "count"),
    "integer_alu_units": ("system.cpu.fuPool.FUList00", "count"),
    "integer_mult_units": ("system.cpu.fuPool.FUList02", "count"),
    "floating_point_units": ("system.cpu.fuPool.FUList05", "count"),
    "l1i_prefetcher": ("system.cpu.icache.prefetcher", "type"),
    "l1i_prefetch_degree": ("system.cpu.icache.prefetcher", "degree"),
    "l1i_prefetch_queue": ("system.cpu.icache.prefetcher", "queue_size"),
    "l1d_prefetcher": ("system.cpu.dcache.prefetcher", "type"),
    "l1d_prefetch_degree": ("system.cpu.dcache.prefetcher", "degree"),
    "l1d_prefetch_queue": ("system.cpu.dcache.prefetcher", "queue_size"),
    "l2_prefetcher": ("system.l2.prefetcher", "type"),
    "l2_prefetch_queue": ("system.l2.prefetcher", "queue_size"),
    "l3_prefetcher": ("system.l3.prefetcher", "type"),
    "l3_prefetch_degree": ("system.l3.prefetcher", "degree"),
    "l3_prefetch_queue": ("system.l3.prefetcher", "queue_size"),
    "memory_controller_0": ("system.mem_ctrls0", "type"),
    "memory_controller_1": ("system.mem_ctrls1", "type"),
    "dram_0_type": ("system.mem_ctrls0.dram", "type"),
    "dram_1_type": ("system.mem_ctrls1.dram", "type"),
    "dram_0_tck": ("system.mem_ctrls0.dram", "tCK"),
    "dram_1_tck": ("system.mem_ctrls1.dram", "tCK"),
    "dram_0_device_bus_width": (
        "system.mem_ctrls0.dram",
        "device_bus_width",
    ),
    "dram_1_device_bus_width": (
        "system.mem_ctrls1.dram",
        "device_bus_width",
    ),
    "dram_0_devices_per_rank": (
        "system.mem_ctrls0.dram",
        "devices_per_rank",
    ),
    "dram_1_devices_per_rank": (
        "system.mem_ctrls1.dram",
        "devices_per_rank",
    ),
    "dram_0_ranks_per_channel": (
        "system.mem_ctrls0.dram",
        "ranks_per_channel",
    ),
    "dram_1_ranks_per_channel": (
        "system.mem_ctrls1.dram",
        "ranks_per_channel",
    ),
}

PAPER_CONFIG: dict[str, str | None] = {
    "cpu_type": "BaseO3CPU",
    "system_clock_period": "667",
    "cpu_clock_period": "400",
    "fetch_width": "5",
    "decode_width": "5",
    "commit_width": "5",
    "rename_width": "5",
    "dispatch_width": "8",
    "issue_width": "8",
    "writeback_width": "8",
    "rob_entries": "128",
    "iq_entries": "80",
    "lq_entries": "32",
    "sq_entries": "48",
    "physical_int_registers": "128",
    "physical_float_registers": "192",
    "physical_vector_registers": "192",
    "physical_predicate_registers": "64",
    "cache_line_size": "64",
    "dcache_size": "65536",
    "dcache_assoc": "4",
    "dcache_tag_latency": "2",
    "dcache_data_latency": "2",
    "icache_size": "65536",
    "icache_assoc": "4",
    "icache_tag_latency": "2",
    "icache_data_latency": "2",
    "l2_size": "524288",
    "l2_assoc": "8",
    "l2_tag_latency": "8",
    "l2_data_latency": "8",
    "l2_response_latency": "8",
    "l3_size": "4194304",
    "l3_assoc": "16",
    "l3_tag_latency": "35",
    "l3_data_latency": "35",
    "l3_response_latency": "35",
    "mshrs": "8",
    "header_latency": "0",
    "tol2_width": "64",
    "tol2_frontend_latency": "1",
    "tol2_forward_latency": "1",
    "tol2_response_latency": "1",
    "tol3_width": "32",
    "tol3_frontend_latency": "10",
    "tol3_forward_latency": "10",
    "tol3_response_latency": "10",
    "cache_load_ports": "2",
    "cache_store_ports": "1",
    "simd_alu_units": "2",
    "simd_mult_units": "2",
    "integer_alu_units": "2",
    "integer_mult_units": "2",
    "floating_point_units": "2",
    "l1i_prefetcher": "StridePrefetcher",
    "l1i_prefetch_degree": "4",
    "l1i_prefetch_queue": "32",
    "l1d_prefetcher": "StridePrefetcher",
    "l1d_prefetch_degree": "4",
    "l1d_prefetch_queue": "32",
    "l2_prefetcher": "AMPMPrefetcher",
    "l2_prefetch_queue": "32",
    "l3_prefetcher": "BOPPrefetcher",
    "l3_prefetch_degree": "1",
    "l3_prefetch_queue": "32",
    "memory_controller_0": "MemCtrl",
    "memory_controller_1": "MemCtrl",
    "dram_0_type": "DRAMInterface",
    "dram_1_type": "DRAMInterface",
    "dram_0_tck": "833",
    "dram_1_tck": "833",
    "dram_0_device_bus_width": "8",
    "dram_1_device_bus_width": "8",
    "dram_0_devices_per_rank": "8",
    "dram_1_devices_per_rank": "8",
    "dram_0_ranks_per_channel": "2",
    "dram_1_ranks_per_channel": "2",
}

BLOCK_CONFIG = {
    "vl4": {
        "vl": "4",
        "spm_load_ports": "2",
        "spm_store_ports": "1",
    },
    "vl16": {
        "vl": "16",
        "spm_load_ports": "1",
        "spm_store_ports": "1",
    },
}

GEM5_EXTRA_KEYS = {
    "system.cpu[0].dcache.mshrs": "mshrs",
    "system.tol2bus.header_latency": "header_latency",
    "system.cpu[0].isa[0].sve_vl_se": "vl",
    "system.cpu[0].cacheLoadPorts": "cache_load_ports",
    "system.cpu[0].cacheStorePorts": "cache_store_ports",
    "system.cpu[0].spmLoadPorts": "spm_load_ports",
    "system.cpu[0].spmStorePorts": "spm_store_ports",
    "system.cpu[0].fuPool.FUList[7].count": "simd_alu_units",
    "system.cpu[0].fuPool.FUList[9].count": "simd_mult_units",
}

ATTENTION_BINARIES = {
    "cache_flash_attn",
    "spm_flash_attn",
    "cache_unfused_attn",
    "spm_unfused_attn",
}
# Exact compatibility allowlists for the frozen Figure 9 reference profile.
# The eight unfused cells were regenerated (2026-07) with the declared
# best.json arguments and the shipped binaries after an audit found the
# original cells were produced with a truncated command line; they now
# carry full modern argv and are validated by the standard path (see
# experiments/end2end/UNFUSED_REGENERATION.md).
LEGACY_MODELESS_ATTENTION_CELLS = frozenset(
    {
        ("vl4", "llama_T256_cflash_br64_bc576"),
        ("vl4", "llama_T256_sflash_br64_bc384"),
        ("vl4", "llama_T4096_cflash_br64_bc576"),
        ("vl4", "llama_T4096_sflash_br64_bc576"),
        ("vl16", "llama_T256_cflash_br64_bc256"),
        ("vl16", "llama_T256_sflash_br32_bc256"),
        ("vl16", "llama_T4096_cflash_br64_bc384"),
        ("vl16", "llama_T4096_sflash_br32_bc384"),
    }
)
LEGACY_INCOMPLETE_EXIT_CELLS = frozenset(
    {
        ("vl4", "llama_T4096_cflash_br64_bc576"),
    }
)
CACHE_START_POLICY = (
    b"CACHE_START_POLICY=evict-v1 bytes=8388608 line=64 order=bitrev17"
)
LAYERNORM_MAX_ABS_LIMIT = 5.0e-3

PRUNED_DIRS = {"fs", "__pycache__", "bin", "bin_gem5", "bin_qemu"}
PRUNED_FILES = {"citations.bib", "config.json"}
PRUNED_SUFFIXES = (".pyc", ".o", "_enc.s", ".trace")
BAD_LOG = re.compile(
    rb"(?:\bpanic:|\bfatal:|segmentation fault|\bsegfault\b|"
    rb"core dumped|\bassert(?:ion)?\b[^\r\n]{0,256}\bfailed\b|"
    rb"simulated exit code not 0|validation failed|SELF_CHECK (?:FAIL|SKIP)|"
    rb"\[DBG\])",
    re.IGNORECASE,
)
CLEAN_EXIT = b"Exiting @ tick"
HOST_PATH_MARKERS = tuple(
    b"/" + component + b"/"
    for component in (b"data", b"home", b"Users")
)
WINDOWS_USER_PATH = re.compile(rb"(?i)(?:[A-Z]:[\\/])Users[\\/]")
NONANONYMOUS_GEM5_HOST = re.compile(
    rb"(?m)^gem5 executing on (?!anonymous-host(?:[,\s]|$))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or regenerate canonical reference-result manifests."
    )
    parser.add_argument(
        "mode",
        choices=("check", "write-manifests", "privacy"),
        help=(
            "check reference manifests/cells; write-manifests atomically "
            "regenerates every reference manifest; privacy scans archive "
            "member paths read from stdin"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="artifact repository root",
    )
    return parser.parse_args()


def is_packaged_file(path: Path, pool: Path) -> bool:
    relative = path.relative_to(pool)
    if any(component in PRUNED_DIRS for component in relative.parts[:-1]):
        return False
    if path.name in PRUNED_FILES or path.name.startswith("m5out"):
        return False
    return not path.name.endswith(PRUNED_SUFFIXES)


def reference_files(root: Path, experiment: str) -> list[tuple[str, Path]]:
    reference = root / "experiments" / experiment / "results" / "reference"
    files: list[tuple[str, Path]] = []
    for pool_name in CANONICAL_POOLS[experiment]:
        pool = reference / pool_name
        if not pool.is_dir():
            continue
        for path in pool.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and is_packaged_file(path, pool)
            ):
                relative = "./" + path.relative_to(reference).as_posix()
                files.append((relative, path))
    return sorted(files)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def manifest_entries(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    problems: list[str] = []
    if not path.is_file():
        return entries, [f"missing manifest: {path}"]
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (\./.+)", line)
        if not match:
            problems.append(f"{path}:{line_number}: malformed manifest line")
            continue
        checksum, relative = match.groups()
        if relative in entries:
            problems.append(f"{path}:{line_number}: duplicate path {relative}")
        entries[relative] = checksum
    return entries, problems


def check_pool_layout(root: Path) -> list[str]:
    problems: list[str] = []
    for experiment, pools in CANONICAL_POOLS.items():
        reference = root / "experiments" / experiment / "results" / "reference"
        if not reference.is_dir():
            problems.append(f"missing reference directory: {reference}")
            continue
        for pool_name in pools:
            pool = reference / pool_name
            if not pool.is_dir() or not any(pool.iterdir()):
                problems.append(f"missing/empty canonical pool: {pool}")
    return problems


def check_shared_kernel_source(root: Path) -> list[str]:
    """Keep the end-to-end and Figure 7 fused GEMM core byte-identical."""
    paths = (
        root / "kernels/gemm/cacheflex/src/kernels_fused.hpp",
        root / "kernels/end2end/src/kernels_fused.hpp",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        return [
            "missing shared fused-GEMM source: "
            + ", ".join(str(path.relative_to(root)) for path in missing)
        ]
    if paths[0].read_bytes() != paths[1].read_bytes():
        return [
            "Figure 7 and end-to-end fused-GEMM source copies differ: "
            f"{paths[0].relative_to(root)} vs {paths[1].relative_to(root)}"
        ]
    return []


def load_best_cells(
    root: Path, experiment: str, problems: list[str]
) -> list[dict[str, object]] | None:
    """Load and validate the experiment-specific, frozen best-cell schema."""
    path = root / "experiments" / experiment / "best.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        problems.append(f"{experiment}: invalid best.json: {error}")
        return None
    if not isinstance(payload, dict):
        problems.append(f"{experiment}: best.json root is not an object")
        return None
    if payload.get("experiment") != experiment:
        problems.append(
            f"{experiment}: best.json experiment={payload.get('experiment')!r}"
        )
    cells = payload.get("cells")
    if not isinstance(cells, list):
        problems.append(f"{experiment}: best.json 'cells' is not a list")
        return None
    wanted_count = EXPECTED_CELL_COUNTS[experiment]
    if len(cells) != wanted_count:
        problems.append(
            f"{experiment}: best.json has {len(cells)} cells, "
            f"expected {wanted_count}"
        )

    valid = True
    identities: list[object] = []
    for index, cell in enumerate(cells):
        label = f"{experiment}: best.json cell {index}"
        if not isinstance(cell, dict):
            problems.append(f"{label} is not an object")
            valid = False
            continue
        try:
            cell_manifest.EMITTERS[experiment](cell, index)
        except (cell_manifest.ManifestError, IndexError) as error:
            problems.append(f"{label} violates the runner schema: {error}")
            valid = False
        for key in ("name", "binary", "args"):
            value = cell.get(key)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{label} has invalid {key!r}")
                valid = False
        name = cell.get("name")
        if isinstance(name, str) and (
            Path(name).name != name or name in {".", ".."}
        ):
            problems.append(f"{label} has unsafe name {name!r}")
            valid = False
        raw_args = cell.get("args")
        if isinstance(raw_args, str):
            try:
                args = shlex.split(raw_args)
            except ValueError as error:
                problems.append(f"{label} has malformed args: {error}")
                valid = False
                args = []
        else:
            args = []
        if not args:
            problems.append(f"{label} has empty args")
            valid = False

        if experiment == "end2end":
            block = cell.get("block")
            if not isinstance(block, str) or block not in BLOCK_CONFIG:
                problems.append(f"{label} has invalid block {block!r}")
                valid = False
            identity: object = (
                block if isinstance(block, str) else f"<invalid-{index}>",
                name if isinstance(name, str) else f"<invalid-{index}>",
            )
        else:
            if "block" in cell:
                problems.append(f"{label} unexpectedly contains 'block'")
                valid = False
            identity = (
                name if isinstance(name, str) else f"<invalid-{index}>"
            )

        if experiment == "software_alternatives":
            spm = cell.get("spm")
            if type(spm) is not int or spm not in {0, 1}:
                problems.append(f"{label} has invalid spm flag {spm!r}")
                valid = False
            runner = cell.get("runner")
            if not isinstance(runner, str) or not runner.strip():
                problems.append(f"{label} has invalid 'runner'")
                valid = False

        binary = cell.get("binary")
        if isinstance(binary, str) and Path(binary).name in ATTENTION_BINARIES:
            if not args or args[-1] not in {"causal", "noncausal"}:
                problems.append(
                    f"{label} attention args lack causal/noncausal mode"
                )
                valid = False
        identities.append(identity)

    if len(set(identities)) != len(identities):
        problems.append(f"{experiment}: duplicate best.json cell identity")
        valid = False

    if experiment == "end2end":
        for block in BLOCK_CONFIG:
            count = sum(
                isinstance(cell, dict) and cell.get("block") == block
                for cell in cells
            )
            if count != 54:
                problems.append(
                    f"{experiment}: best.json block {block} has {count} cells, "
                    "expected 54"
                )

    return cells if valid else None


def uses_legacy_end_reference_profile(root: Path) -> bool:
    """Recognize the explicit compatibility profile for the frozen End data."""
    path = root / "experiments" / "end2end" / "best.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("reference_validation_profile")
        == "legacy-measured-v1"
    )


def is_legacy_modeless_attention(cell: dict[str, object]) -> bool:
    return (
        str(cell.get("block")),
        str(cell.get("name")),
    ) in LEGACY_MODELESS_ATTENTION_CELLS


def is_legacy_incomplete_exit(cell: dict[str, object]) -> bool:
    return (
        str(cell.get("block")),
        str(cell.get("name")),
    ) in LEGACY_INCOMPLETE_EXIT_CELLS


def legacy_recorded_workload_args(
    cell: dict[str, object], expected: list[str]
) -> list[str]:
    """Return the exact argv shape emitted by the selected legacy binary."""
    binary = Path(str(cell["binary"])).name
    if binary == "cache_flash_attn":
        return expected[:-1]
    if binary == "spm_flash_attn":
        return expected[:-2]
    if binary in {"cache_unfused_attn", "spm_unfused_attn"}:
        return expected[:5]
    return expected


def gem5_spm_defaults(root: Path, problems: list[str]) -> dict[str, str]:
    """Derive the SPM-port values used when a runner omits explicit overrides."""
    source = root / "gem5" / "src" / "cpu" / "o3" / "BaseO3CPU.py"
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        problems.append(f"cannot read gem5 SPM-port defaults: {error}")
        return {}
    defaults: dict[str, str] = {}
    for source_name, key in (
        ("spmLoadPorts", "spm_load_ports"),
        ("spmStorePorts", "spm_store_ports"),
    ):
        matches = re.findall(
            rf"^\s*{source_name}\s*=\s*Param\.Unsigned\(\s*([0-9]+)\s*,",
            text,
            re.MULTILINE,
        )
        if len(matches) != 1:
            problems.append(
                f"{source.relative_to(root)}: expected one numeric default "
                f"for {source_name}, found {len(matches)}"
            )
        else:
            defaults[key] = str(int(matches[0]))
    return defaults


def gem5_extra_params(
    experiment: str, cell: dict[str, object], problems: list[str]
) -> dict[str, str]:
    label = f"{experiment}/{cell.get('name', '<unnamed>')}"
    value = cell.get("gem5_extra", "")
    if not isinstance(value, str):
        problems.append(f"{label}: gem5_extra is not a string")
        return {}
    try:
        tokens = shlex.split(value)
    except ValueError as error:
        problems.append(f"{label}: malformed gem5_extra: {error}")
        return {}
    params: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        assignment = None
        if token == "-P":
            index += 1
            if index >= len(tokens):
                problems.append(f"{label}: gem5_extra ends with bare -P")
                break
            assignment = tokens[index]
        elif token.startswith("-P") and len(token) > 2:
            assignment = token[2:]
        if assignment is not None:
            if "=" not in assignment:
                problems.append(
                    f"{label}: malformed gem5 parameter {assignment!r}"
                )
            else:
                raw_key, raw_value = assignment.split("=", 1)
                key = GEM5_EXTRA_KEYS.get(raw_key)
                if key is not None:
                    raw_value = raw_value.strip()
                    if re.fullmatch(r"[0-9]+", raw_value) is None:
                        problems.append(
                            f"{label}: {raw_key} has non-integer "
                            f"value {raw_value!r}"
                        )
                    elif key in params:
                        problems.append(
                            f"{label}: duplicate gem5 parameter {raw_key}"
                        )
                    else:
                        params[key] = str(int(raw_value))
        index += 1
    return params


def capacity_l1d_config(
    cell: dict[str, object], problems: list[str]
) -> dict[str, str] | None:
    """Extract the capacity row's explicit L1D geometry in config.ini units."""
    label = f"capacity_motivation/{cell.get('name', '<unnamed>')}"
    try:
        tokens = shlex.split(str(cell.get("gem5_extra", "")))
    except ValueError as error:
        problems.append(f"{label}: malformed gem5_extra: {error}")
        return None
    values: dict[str, str] = {}
    for option, key in (
        ("--l1d_size", "dcache_size"),
        ("--l1d_assoc", "dcache_assoc"),
    ):
        matches = [
            token.split("=", 1)[1]
            for token in tokens
            if token.startswith(option + "=")
        ]
        if len(matches) != 1:
            problems.append(
                f"{label}: expected one {option} setting, found {len(matches)}"
            )
            return None
        value = matches[0]
        if key == "dcache_assoc":
            if re.fullmatch(r"[0-9]+", value) is None:
                problems.append(f"{label}: malformed L1D associativity {value!r}")
                return None
            values[key] = str(int(value))
            continue
        size = re.fullmatch(r"([0-9]+)([kM]?B)", value)
        if size is None:
            problems.append(f"{label}: malformed L1D size {value!r}")
            return None
        amount, unit = size.groups()
        multiplier = {"B": 1, "kB": 1024, "MB": 1024 * 1024}[unit]
        values[key] = str(int(amount) * multiplier)
    return values


def expected_cell_config(
    experiment: str,
    cell: dict[str, object],
    spm_defaults: dict[str, str],
    problems: list[str],
) -> dict[str, str | None] | None:
    label = f"{experiment}/{cell['name']}"
    expected = dict(PAPER_CONFIG)
    params = gem5_extra_params(experiment, cell, problems)

    if experiment == "capacity_motivation":
        geometry = capacity_l1d_config(cell, problems)
        if geometry is None:
            return None
        expected.update(geometry)
        for key in (
            "cache_load_ports",
            "cache_store_ports",
            "simd_alu_units",
            "simd_mult_units",
        ):
            if key not in params:
                problems.append(f"{label}: gem5_extra must specify {key}")
                return None
            expected[key] = params[key]

    if (
        experiment == "software_alternatives"
        and "_pfoff_" in str(cell["name"])
    ):
        for key in (
            "l1i_prefetcher",
            "l1i_prefetch_degree",
            "l1i_prefetch_queue",
            "l1d_prefetcher",
            "l1d_prefetch_degree",
            "l1d_prefetch_queue",
            "l2_prefetcher",
            "l2_prefetch_queue",
            "l3_prefetcher",
            "l3_prefetch_degree",
            "l3_prefetch_queue",
        ):
            expected[key] = None

    if experiment == "end2end":
        expected.update(BLOCK_CONFIG[cell["block"]])
        for key, value in params.items():
            if key in expected and expected[key] != value:
                problems.append(
                    f"{label}: gem5_extra {key}={value}, "
                    f"expected {expected[key]}"
                )
        return expected

    if experiment == "software_alternatives":
        expected["vl"] = "16"
        expected.update({"spm_load_ports": "1", "spm_store_ports": "1"})
        return expected

    for fixed in ("mshrs", "header_latency"):
        if fixed in params and params[fixed] != expected[fixed]:
            problems.append(
                f"{label}: gem5_extra {fixed}={params[fixed]}, "
                f"expected {expected[fixed]}"
            )
    vl = params.get("vl")
    if vl not in {"4", "8", "16"}:
        problems.append(
            f"{label}: gem5_extra must specify VL 4, 8, or 16"
        )
        return None
    expected["vl"] = vl

    explicit_ports = {
        key: params[key]
        for key in ("spm_load_ports", "spm_store_ports")
        if key in params
    }
    if len(explicit_ports) == 2:
        expected.update(explicit_ports)
    elif explicit_ports:
        problems.append(
            f"{label}: gem5_extra must specify both SPM load/store ports"
        )
        return None
    elif experiment == "capacity_motivation" and not str(
        cell["name"]
    ).startswith("v3fused_"):
        if set(spm_defaults) != {"spm_load_ports", "spm_store_ports"}:
            problems.append(f"{label}: cannot derive SPM-port defaults")
            return None
        expected.update(spm_defaults)
    else:
        problems.append(
            f"{label}: gem5_extra must specify both SPM load/store ports"
        )
        return None
    return expected


def cell_pool(experiment: str, cell: dict[str, object]) -> str:
    if experiment == "capacity_motivation":
        if str(cell["name"]).startswith("v3fused_"):
            return "results_spm_fused_n2048_headh0"
        return "results_fig2_n2048_headh0"
    if experiment == "vl_length":
        return "results_headh0"
    if experiment == "software_alternatives":
        return "best"
    return f"simulation_results/{cell['block']}"


def expected_workload_args(
    experiment: str, cell: dict[str, object], problems: list[str]
) -> list[str] | None:
    label = f"{experiment}/{cell['name']}"
    try:
        args = shlex.split(str(cell["args"]))
    except ValueError as error:
        problems.append(f"{label}: malformed best.json args: {error}")
        return None
    if experiment != "software_alternatives":
        return args
    if len(args) != 5:
        problems.append(
            f"{label}: Figure 8 best.json args have {len(args)} fields, expected 5"
        )
        return None
    # Figure 8 best.json stores M N K KC MC; the microkernel takes
    # M K N n_iter KC MC.
    m, n, k, kc, mc = args
    return [m, k, n, "1", kc, mc]


def single_finite_receipt(
    log: bytes,
    receipt: bytes,
    label: str,
    problems: list[str],
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float | None:
    matches = re.findall(
        rb"(?m)^\s*" + re.escape(receipt) + rb":\s*(\S+)\s*$",
        log,
    )
    display = receipt.decode("ascii")
    if len(matches) != 1:
        problems.append(
            f"{label}: expected one {display} receipt, found {len(matches)}"
        )
        return None
    try:
        value = float(matches[0])
    except ValueError:
        problems.append(f"{label}: malformed {display} receipt")
        return None
    if not math.isfinite(value):
        problems.append(f"{label}: non-finite {display} receipt")
        return None
    if minimum is not None and (
        value <= minimum if strict_minimum else value < minimum
    ):
        comparison = ">" if strict_minimum else ">="
        problems.append(
            f"{label}: {display}={value!r}, expected {comparison} {minimum}"
        )
        return None
    return value


def validate_result_receipts(
    experiment: str,
    cell: dict[str, object],
    log: bytes,
    *,
    legacy_end_reference: bool = False,
) -> list[str]:
    """Validate source-emitted correctness and ROI-policy receipts."""
    name = str(cell["name"])
    label = f"{experiment}/{name}"
    problems: list[str] = []

    if experiment in {"capacity_motivation", "vl_length"}:
        single_finite_receipt(log, b"CHECKSUM_LOGICAL", label, problems)
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_WEIGHTED_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )
        return problems

    if experiment == "software_alternatives":
        single_finite_receipt(log, b"CHECKSUM", label, problems)
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_WEIGHTED_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )
        return problems

    if experiment != "end2end":
        return problems

    binary = Path(str(cell["binary"])).name
    try:
        args = shlex.split(str(cell["args"]))
    except ValueError as error:
        return [f"{label}: malformed best.json args: {error}"]

    if legacy_end_reference:
        modes = re.findall(rb"(?m)^ATTENTION_MODE=(\S+)\s*$", log)
        wanted_mode = args[-1].encode("ascii") if args else b""
        if binary in ATTENTION_BINARIES:
            modeless = is_legacy_modeless_attention(cell)
            if modeless and modes:
                problems.append(
                    f"{label}: legacy modeless cell unexpectedly has "
                    f"ATTENTION_MODE receipts {modes!r}"
                )
            elif not modeless and modes != [wanted_mode]:
                problems.append(
                    f"{label}: ATTENTION_MODE receipts {modes!r}, "
                    f"expected {[wanted_mode]!r}"
                )
            if is_legacy_incomplete_exit(cell):
                checksums = re.findall(
                    rb"(?m)^\s*CHECKSUM:\s*(\S+)\s*$", log
                )
                if len(checksums) > 1:
                    problems.append(
                        f"{label}: malformed optional legacy CHECKSUM receipt"
                    )
                elif checksums:
                    try:
                        value = float(checksums[0])
                    except ValueError:
                        value = math.nan
                    if not math.isfinite(value):
                        problems.append(
                            f"{label}: non-finite optional legacy CHECKSUM"
                        )
            else:
                single_finite_receipt(log, b"CHECKSUM", label, problems)
        if binary == "bench_layernorm":
            validation = single_finite_receipt(
                log,
                b"VALIDATION_MAX_ABS",
                label,
                problems,
                minimum=0.0,
            )
            if (
                validation is not None
                and validation > LAYERNORM_MAX_ABS_LIMIT
            ):
                problems.append(
                    f"{label}: VALIDATION_MAX_ABS={validation!r}, "
                    f"limit is {LAYERNORM_MAX_ABS_LIMIT}"
                )
            single_finite_receipt(
                log, b"CHECKSUM_LOGICAL", label, problems
            )
        for receipt in (
            b"CHECKSUM_LOGICAL",
            b"CHECKSUM_LOGICAL_L1",
            b"CHECKSUM_LOGICAL_WEIGHTED_L1",
        ):
            matches = re.findall(
                rb"(?m)^\s*" + re.escape(receipt) + rb":\s*(\S+)\s*$",
                log,
            )
            if matches and len(matches) != 1:
                problems.append(
                    f"{label}: malformed optional {receipt.decode()} receipt"
                )
            elif matches:
                try:
                    value = float(matches[0])
                except ValueError:
                    value = math.nan
                if not math.isfinite(value):
                    problems.append(
                        f"{label}: non-finite optional {receipt.decode()} receipt"
                    )
        return problems

    if binary in ATTENTION_BINARIES:
        policy_lines = re.findall(
            rb"(?m)^" + re.escape(CACHE_START_POLICY) + rb"\s*$",
            log,
        )
        roi_headers = re.findall(rb"(?m)^--- \[FIRST_ROI\] ---\s*$", log)
        roi_times = re.findall(
            rb"(?m)^\s*FIRST_ROI:\s+(\S+)\s+ms(?:\s+->.*)?$",
            log,
        )
        if len(policy_lines) != 1:
            problems.append(
                f"{label}: expected one CACHE_START_POLICY receipt, "
                f"found {len(policy_lines)}"
            )
        if len(roi_headers) != 1:
            problems.append(
                f"{label}: expected one FIRST_ROI header, "
                f"found {len(roi_headers)}"
            )
        if len(roi_times) != 1:
            problems.append(
                f"{label}: expected one FIRST_ROI timing receipt, "
                f"found {len(roi_times)}"
            )
        else:
            try:
                roi_time = float(roi_times[0])
            except ValueError:
                roi_time = math.nan
            if not math.isfinite(roi_time) or roi_time <= 0:
                problems.append(f"{label}: invalid FIRST_ROI timing receipt")
        header_position = log.find(b"--- [FIRST_ROI] ---")
        policy_position = log.find(CACHE_START_POLICY)
        timing_position = log.find(b"FIRST_ROI:")
        if (
            header_position >= 0
            and policy_position >= 0
            and timing_position >= 0
            and not header_position < policy_position < timing_position
        ):
            problems.append(
                f"{label}: FIRST_ROI policy/timing receipts are out of order"
            )
        modes = re.findall(rb"(?m)^ATTENTION_MODE=(\S+)\s*$", log)
        wanted_mode = args[-1].encode("ascii") if args else b""
        if modes != [wanted_mode]:
            problems.append(
                f"{label}: ATTENTION_MODE receipts {modes!r}, "
                f"expected {[wanted_mode]!r}"
            )
        single_finite_receipt(log, b"CHECKSUM", label, problems)
        single_finite_receipt(log, b"CHECKSUM_LOGICAL", label, problems)
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )

    if binary in {"bench_ffn_seq", "bench_ffn_seq_spm"}:
        single_finite_receipt(log, b"CHECKSUM", label, problems)
        single_finite_receipt(log, b"CHECKSUM_LOGICAL", label, problems)
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )

    if (
        binary in {"bench_kernel", "bench_kernel_spm"}
        and args[:1] == ["gemm"]
    ):
        single_finite_receipt(
            log, b"CHECKSUM_LOGICAL", label, problems
        )
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )

    if binary == "bench_layernorm":
        results = re.findall(
            rb"(?m)^RESULT kernel=layernorm model=bert "
            rb"T=(\d+) D=(\d+) time_us=(\S+)\s*$",
            log,
        )
        wanted_t = args[1].encode("ascii") if len(args) > 1 else b""
        if len(results) != 1:
            problems.append(
                f"{label}: expected one LayerNorm RESULT receipt, "
                f"found {len(results)}"
            )
        else:
            result_t, result_d, result_time = results[0]
            try:
                time_value = float(result_time)
            except ValueError:
                time_value = math.nan
            if (
                result_t != wanted_t
                or result_d != b"768"
                or not math.isfinite(time_value)
                or time_value <= 0
            ):
                problems.append(
                    f"{label}: LayerNorm RESULT receipt disagrees with the cell"
                )
        validation = single_finite_receipt(
            log,
            b"VALIDATION_MAX_ABS",
            label,
            problems,
            minimum=0.0,
        )
        if (
            validation is not None
            and validation > LAYERNORM_MAX_ABS_LIMIT
        ):
            problems.append(
                f"{label}: VALIDATION_MAX_ABS={validation!r}, "
                f"limit is {LAYERNORM_MAX_ABS_LIMIT}"
            )
        single_finite_receipt(
            log, b"CHECKSUM_LOGICAL", label, problems
        )
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )

    if (
        binary == "bench_kernel"
        and args[:1]
        and args[0] in {"rmsnorm", "residual", "rope", "swiglu"}
    ):
        single_finite_receipt(log, b"CHECKSUM_LOGICAL", label, problems)
        single_finite_receipt(
            log,
            b"CHECKSUM_LOGICAL_L1",
            label,
            problems,
            minimum=0.0,
            strict_minimum=True,
        )
        if args[0] in {"rmsnorm", "residual"}:
            validation = single_finite_receipt(
                log,
                b"VALIDATION_MAX_ABS",
                label,
                problems,
                minimum=0.0,
            )
            if (
                validation is not None
                and validation > LAYERNORM_MAX_ABS_LIMIT
            ):
                problems.append(
                    f"{label}: VALIDATION_MAX_ABS={validation!r}, "
                    f"limit is {LAYERNORM_MAX_ABS_LIMIT}"
                )
    return problems


def logical_output_pair(
    experiment: str, log: bytes
) -> tuple[float, float, float] | None:
    """Return the three logical-output invariants, if well formed."""
    signed_name = (
        b"CHECKSUM" if experiment == "software_alternatives"
        else b"CHECKSUM_LOGICAL"
    )
    signed = re.findall(
        rb"(?m)^\s*" + signed_name + rb":\s*(\S+)\s*$", log
    )
    l1 = re.findall(
        rb"(?m)^\s*CHECKSUM_LOGICAL_L1:\s*(\S+)\s*$", log
    )
    weighted = re.findall(
        rb"(?m)^\s*CHECKSUM_LOGICAL_WEIGHTED_L1:\s*(\S+)\s*$", log
    )
    if len(signed) != 1 or len(l1) != 1 or len(weighted) != 1:
        return None
    try:
        pair = (float(signed[0]), float(l1[0]), float(weighted[0]))
    except ValueError:
        return None
    if (
        not math.isfinite(pair[0])
        or not math.isfinite(pair[1])
        or not math.isfinite(pair[2])
        or pair[1] <= 0
        or pair[2] <= 0
    ):
        return None
    bound_tolerance = 1.0e-6 * max(pair[1], pair[2], 1.0)
    if (
        pair[2] < pair[1] - bound_tolerance
        or pair[2] > 2.0 * pair[1] + bound_tolerance
    ):
        return None
    return pair


def validate_numerical_group(
    label: str, values: list[tuple[str, tuple[float, float, float]]]
) -> list[str]:
    """Cross-check implementations that compute the same logical tensor."""
    if len(values) < 2:
        return [f"{label}: fewer than two numerical implementations"]
    problems: list[str] = []
    reference_name, (
        reference_sum,
        reference_l1,
        reference_weighted,
    ) = values[0]
    for name, (value_sum, value_l1, value_weighted) in values[1:]:
        sum_tolerance = (
            0.015 * max(abs(reference_sum), abs(value_sum)) + 1.0e-9
        )
        if abs(reference_sum - value_sum) > sum_tolerance:
            problems.append(
                f"{label}: logical sums differ: "
                f"{reference_name}={reference_sum!r}, {name}={value_sum!r}"
            )
        l1_tolerance = (
            0.015 * max(reference_l1, value_l1) + 1.0e-9
        )
        if abs(reference_l1 - value_l1) > l1_tolerance:
            problems.append(
                f"{label}: logical L1 sums differ: "
                f"{reference_name}={reference_l1!r}, {name}={value_l1!r}"
            )
        weighted_tolerance = (
            0.015
            * max(abs(reference_weighted), abs(value_weighted))
            + 1.0e-9
        )
        if abs(reference_weighted - value_weighted) > weighted_tolerance:
            problems.append(
                f"{label}: weighted logical sums differ: "
                f"{reference_name}={reference_weighted!r}, "
                f"{name}={value_weighted!r}"
            )
    return problems


def validate_recorded_config(
    config: configparser.ConfigParser,
    expected: dict[str, str | None],
    label: str,
) -> list[str]:
    problems: list[str] = []
    for key, wanted in expected.items():
        section, option = CONFIG_LOCATIONS[key]
        got = config.get(section, option, fallback=None)
        if got is not None:
            got = got.strip()
        if got != wanted:
            problems.append(
                f"{label}: {section}.{option}={got!r}, expected {wanted!r}"
            )
    return problems


def required_cell_files(
    experiment: str, pool: Path, name: str
) -> tuple[Path, ...]:
    run_dir = pool / name
    if experiment in {"capacity_motivation", "vl_length"}:
        log = pool / f"{name}.log"
    elif experiment == "software_alternatives":
        log = run_dir / "stdout.txt"
    else:
        log = run_dir / "stdout.log"
    required = (run_dir / "config.ini", run_dir / "stats.txt", log)
    if experiment == "software_alternatives":
        required += (
            run_dir / "RUN_ENV.txt",
            run_dir / "RUNNER_USED.txt",
            run_dir / "BINARY_USED.txt",
            run_dir / "BINARIES.sha256",
        )
    return required


def check_reference_semantics(root: Path) -> list[str]:
    """Match every canonical cell and recorded configuration to best.json."""
    problems: list[str] = []
    defaults = gem5_spm_defaults(root, problems)
    legacy_end_reference = uses_legacy_end_reference_profile(root)
    for experiment, pool_names in CANONICAL_POOLS.items():
        cells = load_best_cells(root, experiment, problems)
        if cells is None:
            continue
        numerical_groups: dict[
            str, list[tuple[str, tuple[float, float, float]]]
        ] = {}
        reference = root / "experiments" / experiment / "results" / "reference"
        expected_by_pool: dict[str, set[str]] = {
            pool_name: set() for pool_name in pool_names
        }
        for cell in cells:
            pool_name = cell_pool(experiment, cell)
            if pool_name not in expected_by_pool:
                problems.append(
                    f"{experiment}/{cell['name']}: maps to unknown pool {pool_name}"
                )
                continue
            expected_by_pool[pool_name].add(str(cell["name"]))

        for pool_name, expected_names in expected_by_pool.items():
            pool = reference / pool_name
            actual_names = (
                {path.name for path in pool.iterdir() if path.is_dir()}
                if pool.is_dir()
                else set()
            )
            if actual_names != expected_names:
                problems.append(
                    f"{experiment}/{pool_name}: canonical cell set differs "
                    f"from best.json: missing={sorted(expected_names - actual_names)}, "
                    f"extra={sorted(actual_names - expected_names)}"
                )

        for cell in cells:
            name = str(cell["name"])
            pool = reference / cell_pool(experiment, cell)
            required = required_cell_files(experiment, pool, name)
            missing = [
                path.name
                for path in required
                if not path.is_file() or path.stat().st_size == 0
            ]
            if missing:
                problems.append(
                    f"{experiment}/{name}: missing/empty "
                    f"{', '.join(missing)}"
                )
                continue

            config_path = pool / name / "config.ini"
            config = configparser.ConfigParser(
                interpolation=None, strict=True
            )
            try:
                with config_path.open(encoding="utf-8") as stream:
                    config.read_file(stream)
            except (OSError, configparser.Error) as error:
                problems.append(
                    f"{experiment}/{name}: unparseable config.ini: {error}"
                )
                continue
            memory_controllers = {
                section
                for section in config.sections()
                if re.fullmatch(r"system\.mem_ctrls\d+", section)
            }
            if memory_controllers != {
                "system.mem_ctrls0",
                "system.mem_ctrls1",
            }:
                problems.append(
                    f"{experiment}/{name}: memory-controller set "
                    f"{sorted(memory_controllers)!r}, expected exactly "
                    "['system.mem_ctrls0', 'system.mem_ctrls1']"
                )

            expected_config = expected_cell_config(
                experiment, cell, defaults, problems
            )
            if expected_config is not None:
                problems.extend(
                    validate_recorded_config(
                        config, expected_config, f"{experiment}/{name}"
                    )
                )

            command = config.get(
                "system.cpu.workload", "cmd", fallback=""
            )
            try:
                argv = shlex.split(command)
            except ValueError as error:
                problems.append(
                    f"{experiment}/{name}: malformed workload command: {error}"
                )
                argv = []
            expected_args = expected_workload_args(
                experiment, cell, problems
            )
            recorded_expected_args = expected_args
            if (
                expected_args is not None
                and experiment == "end2end"
                and legacy_end_reference
                and is_legacy_modeless_attention(cell)
            ):
                recorded_expected_args = legacy_recorded_workload_args(
                    cell, expected_args
                )
            actual_binary = Path(argv[0]).name if argv else None
            expected_binary = Path(str(cell["binary"])).name
            if actual_binary != expected_binary:
                problems.append(
                    f"{experiment}/{name}: workload binary "
                    f"{actual_binary!r}, expected {expected_binary!r}"
                )
            if (
                recorded_expected_args is not None
                and argv[1:] != recorded_expected_args
            ):
                problems.append(
                    f"{experiment}/{name}: workload args {argv[1:]!r}, "
                    f"expected {recorded_expected_args!r}"
                )

            if (
                expected_binary in ATTENTION_BINARIES
                and not (
                    experiment == "end2end"
                    and legacy_end_reference
                    and is_legacy_modeless_attention(cell)
                )
            ):
                expected_mode = expected_args[-1] if expected_args else None
                actual_mode = argv[-1] if len(argv) > 1 else None
                if actual_mode != expected_mode:
                    problems.append(
                        f"{experiment}/{name}: attention mode "
                        f"{actual_mode!r}, expected {expected_mode!r}"
                    )

            log = required[2].read_bytes()
            problems.extend(
                validate_result_receipts(
                    experiment,
                    cell,
                    log,
                    legacy_end_reference=(
                        experiment == "end2end" and legacy_end_reference
                    ),
                )
            )
            pair = logical_output_pair(experiment, log)
            if pair is not None:
                group: str | None = None
                if experiment == "capacity_motivation":
                    group = "Figure 2"
                elif experiment == "vl_length":
                    match = re.fullmatch(
                        r"(W\d+)_v[13]_vl(4|8|16)", name
                    )
                    if match is None:
                        problems.append(
                            f"{experiment}/{name}: cannot derive numerical "
                            "equivalence group"
                        )
                    else:
                        group = f"Figure 7/{match.group(1)}/VL{match.group(2)}"
                elif experiment == "software_alternatives":
                    workload = name.split("_", 1)[0]
                    if workload not in {"w3", "g2048"}:
                        problems.append(
                            f"{experiment}/{name}: cannot derive numerical "
                            "equivalence group"
                        )
                    else:
                        group = f"Figure 8/{workload}"
                if group is not None:
                    numerical_groups.setdefault(group, []).append((name, pair))

            if experiment == "software_alternatives":
                receipts = {
                    "RUN_ENV.txt": str(cell.get("env") or "none"),
                    "RUNNER_USED.txt": str(cell["runner"]),
                    "BINARY_USED.txt": str(cell["binary"]),
                }
                for filename, wanted in receipts.items():
                    path = pool / name / filename
                    got = path.read_text(encoding="utf-8").strip()
                    if got != wanted:
                        problems.append(
                            f"{experiment}/{name}: {filename}={got!r}, "
                            f"expected {wanted!r}"
                        )
                hash_lines = (
                    pool / name / "BINARIES.sha256"
                ).read_text(encoding="utf-8").splitlines()
                if len(hash_lines) != 2 or any(
                    re.fullmatch(r"[0-9a-f]{64}  [^/\s][^\s]*", line) is None
                    for line in hash_lines
                ):
                    problems.append(
                        f"{experiment}/{name}: malformed or non-relative "
                        "BINARIES.sha256"
                    )

        expected_group_shape: tuple[int, int] | None = {
            "capacity_motivation": (1, 21),
            "vl_length": (21, 2),
            "software_alternatives": (2, 6),
        }.get(experiment)
        if expected_group_shape is not None:
            expected_groups, expected_members = expected_group_shape
            if len(numerical_groups) != expected_groups:
                problems.append(
                    f"{experiment}: found {len(numerical_groups)} numerical "
                    f"equivalence groups, expected {expected_groups}"
                )
            for group, values in sorted(numerical_groups.items()):
                if len(values) != expected_members:
                    problems.append(
                        f"{group}: found {len(values)} implementations, "
                        f"expected {expected_members}"
                    )
                problems.extend(validate_numerical_group(group, values))
    return problems


def check_exact_reference_members(root: Path) -> list[str]:
    """Reject every file/directory not consumed by the canonical parsers."""
    problems: list[str] = []
    for experiment, pool_names in CANONICAL_POOLS.items():
        cells = load_best_cells(root, experiment, problems)
        if cells is None:
            continue
        reference = root / "experiments" / experiment / "results" / "reference"
        expected_files: set[Path] = set()
        expected_dirs: set[Path] = set()
        for pool_name in pool_names:
            relative = Path(pool_name)
            expected_dirs.add(relative)
            expected_dirs.update(
                parent for parent in relative.parents if parent != Path(".")
            )
        for cell in cells:
            pool_name = cell_pool(experiment, cell)
            pool = reference / pool_name
            name = str(cell["name"])
            cell_relative = Path(pool_name) / name
            expected_dirs.add(cell_relative)
            expected_dirs.update(
                parent
                for parent in cell_relative.parents
                if parent != Path(".")
            )
            expected_files.update(
                path.relative_to(reference)
                for path in required_cell_files(experiment, pool, name)
            )

        actual_files: set[Path] = set()
        actual_dirs: set[Path] = set()
        if reference.is_dir():
            for path in reference.rglob("*"):
                relative = path.relative_to(reference)
                if path.is_symlink():
                    problems.append(
                        f"{experiment}: canonical reference contains symlink "
                        f"{relative}"
                    )
                elif path.is_file():
                    if path != reference / "MANIFEST.sha256":
                        actual_files.add(relative)
                elif path.is_dir():
                    actual_dirs.add(relative)

        if actual_files != expected_files:
            problems.append(
                f"{experiment}: canonical file set is not exact: "
                f"missing={sorted(map(str, expected_files - actual_files))}, "
                f"extra={sorted(map(str, actual_files - expected_files))}"
            )
        if actual_dirs != expected_dirs:
            problems.append(
                f"{experiment}: canonical directory set is not exact: "
                f"missing={sorted(map(str, expected_dirs - actual_dirs))}, "
                f"extra={sorted(map(str, actual_dirs - expected_dirs))}"
            )
    return problems


def check_manifests(root: Path) -> list[str]:
    problems = check_pool_layout(root)
    for experiment in CANONICAL_POOLS:
        reference = root / "experiments" / experiment / "results" / "reference"
        if not reference.is_dir():
            continue
        manifest = reference / "MANIFEST.sha256"
        recorded, malformed = manifest_entries(manifest)
        problems.extend(malformed)
        actual = dict(reference_files(root, experiment))
        for relative in sorted(recorded.keys() - actual.keys()):
            problems.append(f"{manifest}: lists non-packaged/missing file {relative}")
        for relative in sorted(actual.keys() - recorded.keys()):
            problems.append(f"{manifest}: omits packaged file {relative}")
        for relative in sorted(recorded.keys() & actual.keys()):
            if digest(actual[relative]) != recorded[relative]:
                problems.append(f"{manifest}: checksum mismatch for {relative}")
    return problems


def validate_energy_counter_semantics(
    data: bytes, label: str
) -> list[str]:
    """Verify the command-counter invariant used by the paper energy model."""
    problems: list[str] = []
    begin = data.find(b"---------- Begin Simulation Statistics ----------")
    end = data.find(
        b"---------- End Simulation Statistics   ----------",
        begin + 1,
    )
    if begin < 0 or end < 0:
        return problems
    roi = data[begin:end]
    overall_classes = (
        "ReadReq",
        "WriteReq",
        "WriteLineReq",
        "ReadExReq",
        "ReadCleanReq",
        "ReadSharedReq",
        "SoftPFReq",
        "HardPFReq",
        "SoftPFExReq",
    )

    def value(name: str, *, required: bool = False) -> float | None:
        matches = re.findall(
            rb"(?m)^"
            + re.escape(name.encode("ascii"))
            + rb"\s+(\S+)",
            roi,
        )
        if len(matches) > 1:
            problems.append(f"{label}: duplicate ROI statistic {name}")
            return None
        if not matches:
            if required:
                problems.append(f"{label}: missing ROI statistic {name}")
            return 0.0
        try:
            result = float(matches[0])
        except ValueError:
            problems.append(f"{label}: non-numeric ROI statistic {name}")
            return None
        if not math.isfinite(result) or result < 0:
            problems.append(f"{label}: invalid ROI statistic {name}")
            return None
        return result

    for level in ("system.l2", "system.l3"):
        overall = value(f"{level}.overallAccesses::total", required=True)
        commands = {
            command: value(f"{level}.{command}.accesses::total")
            for command in overall_classes
        }
        if overall is None or any(item is None for item in commands.values()):
            continue
        command_sum = sum(item for item in commands.values() if item is not None)
        if not math.isclose(overall, command_sum, rel_tol=0.0, abs_tol=0.5):
            problems.append(
                f"{label}: {level}.overallAccesses={overall:g} does not "
                f"equal its command-class sum {command_sum:g}"
            )
        for command in ("WriteReq", "WriteLineReq"):
            if commands[command] != 0:
                problems.append(
                    f"{label}: {level}.{command}={commands[command]:g}; "
                    "the sealed energy accounting requires read-class "
                    "overallAccesses"
                )
        value(f"{level}.WritebackDirty.accesses::total", required=True)
    return problems


def check_cells(root: Path) -> list[str]:
    problems: list[str] = []
    for experiment in CANONICAL_POOLS:
        for relative, path in reference_files(root, experiment):
            if path.name == "stats.txt":
                if path.stat().st_size == 0:
                    problems.append(
                        f"empty canonical stats: {path.relative_to(root)}"
                    )
                else:
                    data = path.read_bytes()
                    if (
                        b"Begin Simulation Statistics" not in data
                        or b"End Simulation Statistics" not in data
                    ):
                        problems.append(
                            "incomplete simulation-statistics dump: "
                            f"{path.relative_to(root)}"
                        )
                    else:
                        problems.extend(
                            validate_energy_counter_semantics(
                                data, str(path.relative_to(root))
                            )
                        )
            if path.name == "config.ini":
                data = path.read_bytes()
                if b"type=BeladyRP" in data or b"type=OraclePrefetcher" in data:
                    problems.append(
                        f"oracle policy in canonical config: {path.relative_to(root)}"
                    )
            if (
                path.name in {"stdout.log", "stdout.txt", "stderr.log", "stderr.txt"}
                or path.suffix == ".log"
            ):
                data = path.read_bytes()
                if BAD_LOG.search(data):
                    problems.append(
                        f"fatal/crash marker in canonical log: {path.relative_to(root)}"
                    )
                is_result_log = path.name in {"stdout.log", "stdout.txt"} or (
                    path.suffix == ".log"
                    and path.name not in {"stderr.log", "stderr.txt"}
                )
                relative_parts = ()
                if experiment == "end2end":
                    relative_parts = path.relative_to(
                        root
                        / "experiments"
                        / "end2end"
                        / "results"
                        / "reference"
                        / "simulation_results"
                    ).parts
                accepted_incomplete_exit = (
                    len(relative_parts) == 3
                    and (relative_parts[0], relative_parts[1])
                    in LEGACY_INCOMPLETE_EXIT_CELLS
                    and relative_parts[2] == "stdout.log"
                )
                if (
                    is_result_log
                    and CLEAN_EXIT not in data
                    and not accepted_incomplete_exit
                ):
                    problems.append(
                        f"no clean gem5 exit in canonical log: {path.relative_to(root)}"
                    )
    return problems


def write_manifests(root: Path) -> int:
    blockers = (
        check_pool_layout(root)
        + check_cells(root)
        + check_reference_semantics(root)
        + check_exact_reference_members(root)
    )
    if blockers:
        for problem in blockers:
            print(f"FAIL: {problem}", file=sys.stderr)
        print(
            "FAIL: refusing to seal incomplete/crashed reference results.",
            file=sys.stderr,
        )
        return 1
    for experiment in CANONICAL_POOLS:
        reference = root / "experiments" / experiment / "results" / "reference"
        if not reference.is_dir():
            print(f"FAIL: missing reference directory: {reference}", file=sys.stderr)
            return 1
        manifest = reference / "MANIFEST.sha256"
        temporary = manifest.with_suffix(".sha256.tmp")
        lines = [
            f"{digest(path)}  {relative}\n"
            for relative, path in reference_files(root, experiment)
        ]
        temporary.write_text("".join(lines), encoding="utf-8")
        os.replace(temporary, manifest)
        print(f"wrote {manifest.relative_to(root)} ({len(lines)} files)")
    return 0


def privacy_scan(root: Path) -> int:
    """Scan exactly the archive members supplied on stdin by tar --list."""
    hits: list[tuple[str, str]] = []
    seen: set[Path] = set()

    # Generic /home and /data markers are useful for project-owned files, but
    # upstream gem5 legitimately contains example paths.  Independently reject
    # this machine's exact repository parent, home directory, login identity,
    # and hostname in *every* selected file and member name.  This catches a
    # generated gem5 source file or copied comment without flagging unrelated
    # upstream examples.
    local_path_markers = {
        str(candidate).encode()
        for candidate in (root, root.parent, Path.home())
        if candidate.is_absolute() and str(candidate) not in {"/", "/home", "/data"}
    }
    generic_identities = {
        "root",
        "home",
        "data",
        "tmp",
        "user",
        "runner",
        "workspace",
    }
    local_identity_tokens = {
        value.casefold().encode()
        for value in (
            os.environ.get("USER", ""),
            os.environ.get("LOGNAME", ""),
            root.parent.name,
            os.uname().nodename,
        )
        if len(value) >= 4 and value.casefold() not in generic_identities
    }

    for line in sys.stdin:
        member = line.rstrip("\n").split(" -> ", 1)[0]
        if member.startswith("./"):
            member = member[2:]
        if not member:
            continue
        member_bytes = member.casefold().encode()
        if any(token in member_bytes for token in local_identity_tokens):
            hits.append((member, "local identity in member name"))
        path = root / member
        try:
            if path in seen or path.is_symlink() or not path.is_file():
                continue
            seen.add(path)
            data = path.read_bytes()
        except OSError:
            continue
        local_path_hit = any(marker in data for marker in local_path_markers)
        local_identity_hit = any(
            token in data.lower() for token in local_identity_tokens
        )
        if local_path_hit:
            hits.append((member, "current-host absolute path"))
        if local_identity_hit:
            hits.append((member, "local identity token"))
        # Upstream gem5 contains example paths in tests and documentation-like
        # source comments.  Project-owned files, including every canonical
        # result receipt, must be free of absolute user/host paths.
        if not member.startswith("gem5/") and not local_path_hit:
            for marker in HOST_PATH_MARKERS:
                if marker in data:
                    hits.append((member, f"absolute host path ({marker.decode('ascii')})"))
                    break
            else:
                if WINDOWS_USER_PATH.search(data):
                    hits.append((member, "absolute Windows user path"))
        if (
            member.startswith("experiments/")
            and "/results/reference/" in member
            and NONANONYMOUS_GEM5_HOST.search(data)
        ):
            hits.append((member, "non-anonymous gem5 hostname receipt"))
    for member, marker in hits[:100]:
        print(f"FAIL: packaged identity/host data ({marker}): {member}", file=sys.stderr)
    if len(hits) > 100:
        print(f"FAIL: ... and {len(hits) - 100} more marker hits", file=sys.stderr)
    if hits:
        affected = len({member for member, _ in hits})
        print(
            f"FAIL: {len(hits)} local identity/host marker hits across "
            f"{affected} packaged files.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {len(seen)} packaged regular files contain no local host paths or result hostnames.")
    return 0


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.mode == "write-manifests":
        return write_manifests(root)
    if args.mode == "privacy":
        return privacy_scan(root)

    # Surface corrupt/incomplete cells before the more verbose manifest drift.
    problems = (
        check_shared_kernel_source(root)
        + check_cells(root)
        + check_reference_semantics(root)
        + check_exact_reference_members(root)
        + check_manifests(root)
    )
    for problem in problems[:100]:
        print(f"FAIL: {problem}", file=sys.stderr)
    if len(problems) > 100:
        print(f"FAIL: ... and {len(problems) - 100} more problem(s).", file=sys.stderr)
    if problems:
        print(f"FAIL: reference integrity found {len(problems)} problem(s).", file=sys.stderr)
        return 1
    print(
        "OK: canonical reference manifests, exact best-cell membership, "
        "recorded configurations, and clean exits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
