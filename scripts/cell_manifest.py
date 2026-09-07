#!/usr/bin/env python3
"""Validate a selected-cell manifest and emit the runner's TSV input.

Validation happens before a run creates an output directory.  The accepted
schema is intentionally narrow: these are the four paper experiments, not a
general command-execution format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Any


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
TOKEN = re.compile(r"[-+A-Za-z0-9_.]+\Z")
ASSIGNMENT = re.compile(r"CF_[A-Z0-9_]+=[-+A-Za-z0-9_.]+\Z")

CF_BINARIES = {
    "kernels/gemm/cacheflex/bin/v1_fused",
    "kernels/gemm/cacheflex/bin/v3_fused_vl4",
    "kernels/gemm/cacheflex/bin/v3_fused_vl8",
    "kernels/gemm/cacheflex/bin/v3_fused_vl16",
}
SOFTWARE_BINARIES = {
    "cacheopt": "kernels/gemm/software_alternatives/bin/v3_l1swpf_none",
    "pfoff": "kernels/gemm/software_alternatives/bin/v3_l1swpf_none",
    "pldl2keep": "kernels/gemm/software_alternatives/bin/v3_l2pf_only_b512",
    "nt": "kernels/gemm/software_alternatives/bin/v3_l2pin_nt_pf0",
    "shbw": "kernels/gemm/software_alternatives/bin/v3_spm_n0outer_VL_16",
    "cacheflex": "kernels/gemm/software_alternatives/bin/v3_spm_n0outer_VL_16",
}
SOFTWARE_ENV = {
    "cacheopt": "",
    "pfoff": "",
    "pldl2keep": "CF_NT_BYPASS=1",
    "nt": "CF_NT_BYPASS=1 CF_NT_CLEANRESP=1 CF_L2_PIN_KEEP=1",
    "shbw": "CF_SPM_SHARED_BW=1 CF_SPM_HDR_CHARGE=1",
    "cacheflex": "",
}
SOFTWARE_RUNNER = {
    "pfoff": (
        "experiments/software_alternatives/twosize_ladder/"
        "run_cell_hwpfoff_headh0.sh"
    )
}
DEFAULT_SOFTWARE_RUNNER = (
    "experiments/software_alternatives/runners/run_cell_headh0.sh"
)
END_BINARY_NAMES = {
    "bench_ffn_seq",
    "bench_ffn_seq_spm",
    "bench_kernel",
    "bench_kernel_spm",
    "bench_layernorm",
    "cache_flash_attn",
    "cache_unfused_attn",
    "spm_flash_attn",
    "spm_unfused_attn",
}


class ManifestError(ValueError):
    pass


def require_string(cell: dict[str, Any], field: str, label: str) -> str:
    value = cell.get(field)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label}: {field} must be a non-empty string")
    if any(character in value for character in "\t\r\n\0"):
        raise ManifestError(f"{label}: {field} contains a control separator")
    return value


def require_relative_path(value: str, field: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise ManifestError(f"{label}: unsafe {field} path {value!r}")
    return value


def simple_words(value: str, field: str, label: str) -> list[str]:
    try:
        words = shlex.split(value)
    except ValueError as error:
        raise ManifestError(f"{label}: malformed {field}: {error}") from error
    if not words or value != " ".join(words) or any(
        not TOKEN.fullmatch(word) for word in words
    ):
        raise ManifestError(f"{label}: {field} must contain simple tokens")
    return words


def environment(cell: dict[str, Any], label: str) -> str:
    value = cell.get("env", "")
    if not isinstance(value, str):
        raise ManifestError(f"{label}: env must be a string")
    if not value:
        return ""
    try:
        words = shlex.split(value)
    except ValueError as error:
        raise ManifestError(f"{label}: malformed env: {error}") from error
    if value != " ".join(words) or any(
        not ASSIGNMENT.fullmatch(word) for word in words
    ):
        raise ManifestError(
            f"{label}: env must be a space-separated CF_NAME=value list"
        )
    keys = [word.split("=", 1)[0] for word in words]
    if len(keys) != len(set(keys)):
        raise ManifestError(f"{label}: env contains duplicate variables")
    return value


def gem5_extra(cell: dict[str, Any], label: str) -> str:
    value = cell.get("gem5_extra", "")
    if not isinstance(value, str):
        raise ManifestError(f"{label}: gem5_extra must be a string")
    if not value:
        return ""
    try:
        words = shlex.split(value)
    except ValueError as error:
        raise ManifestError(
            f"{label}: malformed gem5_extra: {error}"
        ) from error
    if value != " ".join(words):
        raise ManifestError(f"{label}: gem5_extra must use canonical spacing")
    keys: set[str] = set()
    index = 0
    while index < len(words):
        word = words[index]
        if re.fullmatch(r"--l1d_(?:size|assoc)=[A-Za-z0-9]+", word):
            key = word.split("=", 1)[0]
            if key in keys:
                raise ManifestError(
                    f"{label}: gem5_extra contains duplicate parameter {key!r}"
                )
            keys.add(key)
            index += 1
            continue
        if word != "-P" or index + 1 >= len(words):
            raise ManifestError(
                f"{label}: unsupported gem5_extra token {word!r}"
            )
        assignment = words[index + 1]
        if not re.fullmatch(
            r"system\.cpu\[0\]\.(?:"
            r"cacheLoadPorts|cacheStorePorts|spmLoadPorts|spmStorePorts|"
            r"fuPool\.FUList\[(?:7|9)\]\.count|"
            r"isa\[0\]\.sve_vl_se)=[0-9]+",
            assignment,
        ):
            raise ManifestError(
                f"{label}: unsupported gem5 parameter {assignment!r}"
            )
        key = assignment.split("=", 1)[0]
        if key in keys:
            raise ManifestError(
                f"{label}: gem5_extra contains duplicate parameter {key!r}"
            )
        keys.add(key)
        index += 2
    return value


def gem5_parameter_map(extra: str) -> dict[str, str]:
    """Return the already-validated gem5_extra assignments by parameter."""
    words = shlex.split(extra)
    parameters: dict[str, str] = {}
    index = 0
    while index < len(words):
        word = words[index]
        if word.startswith("--l1d_"):
            key, value = word.split("=", 1)
            parameters[key] = value
            index += 1
        else:
            key, value = words[index + 1].split("=", 1)
            parameters[key] = value
            index += 2
    return parameters


def common(cell: Any, index: int) -> tuple[str, str, str, list[str], str]:
    label = f"cells[{index}]"
    if not isinstance(cell, dict):
        raise ManifestError(f"{label}: cell must be an object")
    name = require_string(cell, "name", label)
    if not NAME.fullmatch(name) or name in {".", ".."}:
        raise ManifestError(f"{label}: unsafe cell name {name!r}")
    binary = require_relative_path(
        require_string(cell, "binary", label), "binary", label
    )
    args_value = require_string(cell, "args", label)
    args = simple_words(args_value, "args", label)
    env = environment(cell, label)
    extra = gem5_extra(cell, label)
    return name, binary, env, args, extra


def vl_from_extra(extra: str, label: str) -> int:
    values = re.findall(r"system\.cpu\[0\]\.isa\[0\]\.sve_vl_se=(\d+)", extra)
    if len(values) != 1 or int(values[0]) not in {4, 8, 16}:
        raise ManifestError(f"{label}: gem5_extra must pin one supported VL")
    return int(values[0])


def emit_capacity(cell: dict[str, Any], index: int) -> list[str]:
    name, binary, env, args, extra = common(cell, index)
    label = f"cells[{index}]/{name}"
    if binary not in CF_BINARIES or env:
        raise ManifestError(f"{label}: unexpected binary or environment")
    if len(args) != 6 or not all(word.isdigit() for word in args):
        raise ManifestError(f"{label}: capacity args must be six integers")
    if args[:4] != ["256", "2048", "2048", "1"]:
        raise ManifestError(f"{label}: capacity problem size is not canonical")
    vl = vl_from_extra(extra, label)
    cache = re.fullmatch(
        r"fused_vl(\d+)_l1(\d+)k(\d+)w_ld(\d+)st(\d+)"
        r"_kc(\d+)_mc(\d+)",
        name,
    )
    cacheflex = re.fullmatch(
        r"v3fused_vl(\d+)_kc(\d+)_mc(\d+)",
        name,
    )
    match = cache if cache is not None else cacheflex
    if match is None:
        raise ManifestError(f"{label}: name, VL, KC, MC, and args disagree")
    if cacheflex is not None:
        name_vl, kc, mc = cacheflex.groups()
        wanted = f"kernels/gemm/cacheflex/bin/v3_fused_vl{vl}"
        expected_parameters = {
            "--l1d_size": "64kB",
            "--l1d_assoc": "4",
            "system.cpu[0].cacheLoadPorts": "2",
            "system.cpu[0].cacheStorePorts": "1",
            "system.cpu[0].spmLoadPorts": "1" if vl == 16 else "2",
            "system.cpu[0].spmStorePorts": "1",
            "system.cpu[0].fuPool.FUList[7].count": "2",
            "system.cpu[0].fuPool.FUList[9].count": "2",
            "system.cpu[0].isa[0].sve_vl_se": str(vl),
        }
    else:
        assert cache is not None
        name_vl, l1_size, l1_assoc, loads, stores, kc, mc = cache.groups()
        wanted = "kernels/gemm/cacheflex/bin/v1_fused"
        expected_parameters = {
            "--l1d_size": f"{l1_size}kB",
            "--l1d_assoc": l1_assoc,
            "system.cpu[0].cacheLoadPorts": loads,
            "system.cpu[0].cacheStorePorts": stores,
            "system.cpu[0].fuPool.FUList[7].count": loads,
            "system.cpu[0].fuPool.FUList[9].count": "2",
            "system.cpu[0].isa[0].sve_vl_se": str(vl),
        }
    if (
        int(name_vl) != vl
        or args[4:] != [kc, mc]
        or binary != wanted
        or gem5_parameter_map(extra) != expected_parameters
    ):
        raise ManifestError(
            f"{label}: name, binary, args, and gem5 parameters disagree"
        )
    return [name, env or "-", binary, " ".join(args), extra or "-"]


def emit_vl(cell: dict[str, Any], index: int) -> list[str]:
    name, binary, env, args, extra = common(cell, index)
    label = f"cells[{index}]/{name}"
    if binary not in CF_BINARIES or env:
        raise ManifestError(f"{label}: unexpected binary or environment")
    if len(args) != 6 or not all(word.isdigit() for word in args):
        raise ManifestError(f"{label}: VL args must be six integers")
    match = re.fullmatch(r"W[1-7]_(v1|v3)_vl(4|8|16)", name)
    vl = vl_from_extra(extra, label)
    workloads = {
        "W1": ["128", "4096", "11008"],
        "W2": ["256", "4096", "4096"],
        "W3": ["256", "2048", "2048"],
        "W4": ["512", "768", "768"],
        "W5": ["2048", "2048", "2048"],
        "W6": ["2048", "64", "2048"],
        "W7": ["784", "256", "1024"],
    }
    workload = name.split("_", 1)[0]
    if (
        match is None
        or int(match.group(2)) != vl
        or args[:3] != workloads.get(workload)
        or args[3] != "1"
        or int(args[4]) < 1
        or int(args[5]) < 1
    ):
        raise ManifestError(f"{label}: name, VL, and args disagree")
    wanted = (
        "kernels/gemm/cacheflex/bin/v1_fused"
        if match.group(1) == "v1"
        else f"kernels/gemm/cacheflex/bin/v3_fused_vl{vl}"
    )
    expected_parameters = {
        "system.cpu[0].isa[0].sve_vl_se": str(vl),
        "system.cpu[0].spmLoadPorts": "1" if vl == 16 else "2",
        "system.cpu[0].spmStorePorts": "1",
    }
    if binary != wanted or gem5_parameter_map(extra) != expected_parameters:
        raise ManifestError(
            f"{label}: implementation or gem5 parameters are inconsistent"
        )
    return [name, env or "-", binary, " ".join(args), extra or "-"]


def emit_software(cell: dict[str, Any], index: int) -> list[str]:
    name, binary, env, args, extra = common(cell, index)
    label = f"cells[{index}]/{name}"
    if extra:
        raise ManifestError(f"{label}: gem5_extra is not used")
    match = re.fullmatch(
        r"(w3|g2048)_(cacheopt|pfoff|pldl2keep|nt|shbw|cacheflex)"
        r"_kc(\d+)_mc(\d+)",
        name,
    )
    if match is None:
        raise ManifestError(f"{label}: malformed software-alternative name")
    tag, family, kc, mc = match.groups()
    if len(args) != 5 or not all(word.isdigit() for word in args):
        raise ManifestError(f"{label}: software args must be five integers")
    expected_m = "256" if tag == "w3" else "2048"
    if args != [expected_m, "2048", "2048", kc, mc]:
        raise ManifestError(f"{label}: name and problem-size args disagree")
    spm = cell.get("spm")
    expected_spm = 1 if family in {"shbw", "cacheflex"} else 0
    if type(spm) is not int or spm != expected_spm:
        raise ManifestError(f"{label}: spm must be {expected_spm}")
    runner = require_relative_path(
        require_string(cell, "runner", label), "runner", label
    )
    wanted_runner = SOFTWARE_RUNNER.get(family, DEFAULT_SOFTWARE_RUNNER)
    if (
        binary != SOFTWARE_BINARIES[family]
        or env != SOFTWARE_ENV[family]
        or runner != wanted_runner
    ):
        raise ManifestError(
            f"{label}: runner, binary, or environment does not match the row"
        )
    return [name, runner, binary, str(spm), env or "-", " ".join(args)]


def emit_end(cell: dict[str, Any], index: int) -> list[str]:
    name, binary, env, args, extra = common(cell, index)
    label = f"cells[{index}]/{name}"
    if env or extra:
        raise ManifestError(f"{label}: unexpected environment or gem5_extra")
    block = cell.get("block")
    if block not in {"vl4", "vl16"}:
        raise ManifestError(f"{label}: block must be vl4 or vl16")
    path = PurePosixPath(binary)
    if (
        len(path.parts) not in {4, 5}
        or path.parts[:3] != ("experiments", "end2end", "bin")
        or path.name not in END_BINARY_NAMES
        or (block == "vl16") != ("vl16" in path.parts)
    ):
        raise ManifestError(f"{label}: binary does not match its VL block")
    match = re.match(r"(llama|bert)_T(\d+)_", name)
    if match is None:
        raise ManifestError(f"{label}: malformed end-to-end cell name")
    model, sequence_length = match.groups()
    basename = path.name
    if "attn" in basename:
        wanted_mode = "noncausal"
        wanted_heads = ["32", "8"] if model == "llama" else ["12", "12"]
        if (
            args[0:1] != [sequence_length]
            or args[1:3] != wanted_heads
            or args[-1:] != [wanted_mode]
        ):
            raise ManifestError(
                f"{label}: attention model, size, heads, or mode disagree"
            )
        if args[3:5] != ["64", "1"]:
            raise ManifestError(f"{label}: attention width or iteration disagrees")
        flash = re.search(r"_(cflash|sflash)_br(\d+)_bc(\d+)\Z", name)
        if "flash_attn" in basename:
            if flash is None:
                raise ManifestError(f"{label}: malformed FlashAttention name")
            family, br, bc = flash.groups()
            expected_binary = (
                "cache_flash_attn" if family == "cflash" else "spm_flash_attn"
            )
            expected_length = 8 if family == "cflash" else 9
            if (
                basename != expected_binary
                or len(args) != expected_length
                or args[5:7] != [br, bc]
                or (family == "sflash" and args[7] != "0")
            ):
                raise ManifestError(
                    f"{label}: FlashAttention name, binary, or args disagree"
                )
        else:
            family = "spm" if "_spm_unfused" in name else "cache"
            expected_binary = (
                "spm_unfused_attn"
                if family == "spm"
                else "cache_unfused_attn"
            )
            if (
                basename != expected_binary
                or len(args) != 10
                or args[5:9] != ["128", "256", "128", "0"]
            ):
                raise ManifestError(
                    f"{label}: unfused-attention name, binary, or args disagree"
                )
    elif basename.startswith("bench_ffn_seq"):
        ffn = re.search(r"_ffn_seq(_spm)?_mc(\d+)_kc(\d+)\Z", name)
        if (
            ffn is None
            or len(args) != 5
            or args[0] != ("0" if model == "llama" else "1")
            or args[1] != sequence_length
            or args[2:4] != [ffn.group(2), ffn.group(3)]
            or args[4] != "1"
            or (ffn.group(1) is not None) != basename.endswith("_spm")
        ):
            raise ManifestError(f"{label}: FFN model or size disagrees")
    elif basename == "bench_layernorm":
        if (
            model != "bert"
            or not name.endswith("_layernorm")
            or args != ["1", sequence_length, "1"]
        ):
            raise ManifestError(f"{label}: LayerNorm model or size disagrees")
    elif basename.startswith("bench_kernel"):
        model_flag = "0" if model == "llama" else "1"
        gemm = re.search(
            r"_(cache|spm)_K(\d+)_N(\d+)_mc(\d+)_kc(\d+)\Z", name
        )
        simple = re.search(r"_(rmsnorm|residual|rope)\Z", name)
        if gemm is not None:
            family, k_size, n_size, mc, kc = gemm.groups()
            expected_binary = (
                "bench_kernel_spm" if family == "spm" else "bench_kernel"
            )
            if (
                basename != expected_binary
                or args
                != [
                    "gemm",
                    model_flag,
                    sequence_length,
                    k_size,
                    n_size,
                    mc,
                    kc,
                    "1",
                ]
            ):
                raise ManifestError(
                    f"{label}: GEMM name, binary, or args disagree"
                )
        elif simple is not None:
            if (
                basename != "bench_kernel"
                or args != [simple.group(1), model_flag, sequence_length, "1"]
            ):
                raise ManifestError(
                    f"{label}: elementwise name, binary, or args disagree"
                )
        else:
            raise ManifestError(f"{label}: unsupported end-to-end kernel name")
    figure = cell.get("figure")
    if figure not in {"Figure 9", "Table 4"}:
        raise ManifestError(f"{label}: unsupported paper result group")
    return [name, block, env or "-", binary, " ".join(args), extra or "-"]


EMITTERS = {
    "capacity_motivation": emit_capacity,
    "vl_length": emit_vl,
    "software_alternatives": emit_software,
    "end2end": emit_end,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EMITTERS), required=True)
    parser.add_argument("--best", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.best.read_text(encoding="utf-8"))
        cells = payload.get("cells") if isinstance(payload, dict) else None
        if not isinstance(cells, list):
            raise ManifestError("top-level cells must be a list")
        emitter = EMITTERS[args.experiment]
        rows = [emitter(cell, index) for index, cell in enumerate(cells)]
    except (OSError, json.JSONDecodeError, ManifestError, IndexError) as error:
        print(f"ERROR: cannot use {args.best}: {error}", file=sys.stderr)
        return 3
    for row in rows:
        print("\t".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
