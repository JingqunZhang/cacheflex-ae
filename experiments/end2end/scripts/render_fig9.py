#!/usr/bin/env python3
"""Render the canonical grouped Figure 9 from ``fig9_data.json``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


CONFIGS = ("BL", "CF", "BL+FA", "CF+FA")
MODELS = ("llama", "bert")
PAPER_T = (256, 4096)
PAPER_VL = (4, 16)
TIME_COMPONENTS = ("proj", "ffn", "attn", "other")
ENERGY_COMPONENTS = ("cache", "spm", "dram")

C_TIME = {
    "proj": "#4878A8",
    "ffn": "#E8853B",
    "attn": "#E6C05B",
    "other": "#8B8B8B",
}
C_ENERGY = {"cache": "#6BAF6B", "spm": "#8C6BB1", "dram": "#A0A0A0"}
C_SPEEDUP = "#B83E46"
C_REDUCTION = "#1769AA"

CANVAS_WIDTH_BP = 850.0
CANVAS_HEIGHT_BP = 440.0
MIN_SOURCE_FONT_PT = 12.2
BAR_X = np.array((0.0, 1.0, 2.25, 3.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_and_validate(path: Path) -> dict[tuple[str, int, int], dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Figure 9 data {path}: {error}") from error

    if payload.get("schema_version") != 1 or payload.get("figure") != "Figure 9":
        raise ValueError(f"unexpected Figure 9 schema in {path}")
    if payload.get("units") != {"time": "us", "energy": "mJ"}:
        raise ValueError(f"unexpected units in {path}")

    records: dict[tuple[str, int, int], dict] = {}
    for record in payload.get("records", []):
        try:
            key = (
                record["model"],
                record["sequence_length"],
                record["vector_length"],
            )
            configurations = record["configurations"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed panel in {path}: {error}") from error
        if key in records:
            raise ValueError(f"duplicate panel {key}")
        if tuple(configurations) != CONFIGS:
            raise ValueError(f"configuration order changed in panel {key}")

        baseline_time = configurations["BL"]["time_us"]["total"]
        baseline_energy = configurations["BL"]["energy_mJ"]["total"]
        if not all(
            math.isfinite(value) and value > 0
            for value in (baseline_time, baseline_energy)
        ):
            raise ValueError(f"non-positive baseline total in panel {key}")
        for config in CONFIGS:
            entry = configurations[config]
            if tuple(entry["time_us"]["components"]) != TIME_COMPONENTS:
                raise ValueError(f"time-component order changed in {key}/{config}")
            if tuple(entry["energy_mJ"]["components"]) != ENERGY_COMPONENTS:
                raise ValueError(f"energy-component order changed in {key}/{config}")
            time_components = tuple(entry["time_us"]["components"].values())
            energy_components = tuple(entry["energy_mJ"]["components"].values())
            if not all(
                math.isfinite(value) and value >= 0
                for value in time_components + energy_components
            ):
                raise ValueError(f"invalid component value in {key}/{config}")
            time_sum = sum(time_components)
            energy_sum = sum(energy_components)
            if time_sum <= 0 or energy_sum <= 0:
                raise ValueError(f"non-positive total in {key}/{config}")
            if not math.isclose(
                time_sum, entry["time_us"]["total"], rel_tol=1e-12
            ):
                raise ValueError(f"time total mismatch in {key}/{config}")
            if not math.isclose(
                energy_sum, entry["energy_mJ"]["total"], rel_tol=1e-12
            ):
                raise ValueError(f"energy total mismatch in {key}/{config}")
            if not math.isclose(
                baseline_time / time_sum,
                entry["speedup_vs_BL"],
                rel_tol=1e-12,
            ):
                raise ValueError(f"speedup mismatch in {key}/{config}")
            if not math.isclose(
                (1.0 - energy_sum / baseline_energy) * 100.0,
                entry["energy_reduction_vs_BL_pct"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"energy reduction mismatch in {key}/{config}")
        records[key] = record

    expected = {
        (model, sequence_length, vl)
        for model in MODELS
        for sequence_length in PAPER_T
        for vl in PAPER_VL
    }
    if set(records) != expected:
        raise ValueError(
            f"panel set changed: missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )
    return records


def style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#9B9B9B", linewidth=0.45, alpha=0.24)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(
        axis="y",
        labelsize=MIN_SOURCE_FONT_PT,
        colors="#383838",
        width=0.6,
        length=2.6,
        pad=2.0,
    )
    axis.tick_params(axis="x", width=0.0, length=0.0, pad=2.5)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"
            )
        )
    )
    axis.set_xlim(-0.55, 3.80)


def add_metric_label(axis, x, text, color) -> None:
    axis.text(
        x,
        0.895,
        text,
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=MIN_SOURCE_FONT_PT,
        fontfamily="DejaVu Sans",
        fontstretch="condensed",
        fontweight="normal",
        color=color,
        clip_on=False,
        zorder=4,
    )


def configure_x_axis(axis) -> None:
    axis.set_xticks(BAR_X)
    axis.set_xticklabels([])


def draw_time(axis, record: dict) -> None:
    configurations = record["configurations"]
    x = BAR_X
    bottoms = np.zeros(len(CONFIGS))
    for component in TIME_COMPONENTS:
        values = np.array(
            [
                configurations[config]["time_us"]["components"][component]
                for config in CONFIGS
            ]
        ) / 1000.0
        axis.bar(
            x,
            values,
            0.66,
            bottom=bottoms,
            color=C_TIME[component],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        bottoms += values

    axis.set_ylim(0.0, max(bottoms) * 1.34)
    band_floor = axis.get_ylim()[1] * 0.82
    axis.axhspan(band_floor, axis.get_ylim()[1], color="#FAFAFA", zorder=0)
    axis.axhline(band_floor, color="#D9D9D9", linewidth=0.45, zorder=1)
    axis.axvline(1.625, color="#D9D9D9", linewidth=0.5, zorder=1)
    for index, config in enumerate(CONFIGS[1:], start=1):
        add_metric_label(
            axis,
            x[index],
            f"{configurations[config]['speedup_vs_BL']:.2f}",
            C_SPEEDUP,
        )
    configure_x_axis(axis)
    style_axis(axis)


def energy_reduction_text(value: float) -> str:
    if abs(value) < 0.05:
        return "0.0"
    return f"{'−' if value < 0 else ''}{abs(value):.1f}"


def draw_energy(axis, record: dict) -> None:
    configurations = record["configurations"]
    x = BAR_X
    bottoms = np.zeros(len(CONFIGS))
    for component in ENERGY_COMPONENTS:
        values = np.array(
            [
                configurations[config]["energy_mJ"]["components"][component]
                for config in CONFIGS
            ]
        )
        axis.bar(
            x,
            values,
            0.66,
            bottom=bottoms,
            color=C_ENERGY[component],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        bottoms += values

    axis.set_ylim(0.0, max(bottoms) * 1.34)
    band_floor = axis.get_ylim()[1] * 0.82
    axis.axhspan(band_floor, axis.get_ylim()[1], color="#FAFAFA", zorder=0)
    axis.axhline(band_floor, color="#D9D9D9", linewidth=0.45, zorder=1)
    axis.axvline(1.625, color="#D9D9D9", linewidth=0.5, zorder=1)
    for index, config in enumerate(CONFIGS[1:], start=1):
        add_metric_label(
            axis,
            x[index],
            energy_reduction_text(
                configurations[config]["energy_reduction_vs_BL_pct"]
            ),
            C_REDUCTION,
        )
    configure_x_axis(axis)
    style_axis(axis)


def render(input_path: Path, output_dir: Path) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    records = load_and_validate(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": MIN_SOURCE_FONT_PT,
            "figure.dpi": 144,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": "#303030",
            "axes.labelcolor": "#202020",
        }
    )

    figure = plt.figure(
        figsize=(CANVAS_WIDTH_BP / 72.0, CANVAS_HEIGHT_BP / 72.0)
    )
    outer = figure.add_gridspec(
        2,
        4,
        left=0.086,
        right=0.992,
        bottom=0.064,
        top=0.859,
        hspace=0.20,
        wspace=0.35,
    )
    axes = np.empty((4, 4), dtype=object)
    for model_index in range(2):
        for column in range(4):
            pair = outer[model_index, column].subgridspec(2, 1, hspace=0.16)
            axes[2 * model_index, column] = figure.add_subplot(pair[0, 0])
            axes[2 * model_index + 1, column] = figure.add_subplot(pair[1, 0])

    columns = [
        (sequence_length, vl)
        for sequence_length in PAPER_T
        for vl in PAPER_VL
    ]
    for model_index, model in enumerate(MODELS):
        time_row = 2 * model_index
        energy_row = time_row + 1
        for column, (sequence_length, vl) in enumerate(columns):
            record = records[(model, sequence_length, vl)]
            draw_time(axes[time_row, column], record)
            draw_energy(axes[energy_row, column], record)
            if column == 0:
                axes[time_row, column].set_ylabel(
                    "Time (ms)", fontsize=12.4, fontweight="bold", labelpad=2.5
                )
                axes[energy_row, column].set_ylabel(
                    "Energy (mJ)", fontsize=12.4, fontweight="bold", labelpad=2.5
                )
            if time_row == 0:
                axes[time_row, column].set_title(
                    f"T={sequence_length}  ·  VL={vl}",
                    fontsize=13.8,
                    fontweight="bold",
                    pad=4.0,
                )

    for model_index, label in enumerate(("LLaMA", "BERT")):
        group_box = outer[model_index, :].get_position(figure)
        figure.text(
            0.008,
            group_box.y1 + 0.006,
            label,
            fontsize=13.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    figure.text(
        0.539,
        0.014,
        "Configuration order (left→right):  BL  ·  CF   |   BL+FA  ·  CF+FA",
        fontsize=11.8,
        ha="center",
        va="bottom",
        color="#303030",
    )

    time_legend_handles = [
        Patch(
            facecolor=C_TIME[key],
            edgecolor="white",
            linewidth=0.35,
            label={
                "proj": "QKVO proj.",
                "ffn": "FFN",
                "attn": "Attention",
                "other": "Other",
            }[key],
        )
        for key in TIME_COMPONENTS
    ]
    energy_legend_handles = [
        Patch(
            facecolor=C_ENERGY[key],
            edgecolor="white",
            linewidth=0.35,
            label={"cache": "Cache", "spm": "SPM", "dram": "DRAM"}[key],
        )
        for key in ENERGY_COMPONENTS
    ]
    section_handle = Patch(facecolor="none", edgecolor="none", linewidth=0.0)
    legend_handles = (
        [section_handle, *time_legend_handles, section_handle, *energy_legend_handles]
    )
    legend_labels = [
        "Time:",
        *[handle.get_label() for handle in time_legend_handles],
        "Energy:",
        *[handle.get_label() for handle in energy_legend_handles],
    ]
    legend = figure.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=9,
        frameon=False,
        fontsize=12.1,
        handlelength=1.10,
        handletextpad=0.30,
        columnspacing=0.68,
    )
    legend.get_texts()[0].set_fontweight("bold")
    legend.get_texts()[5].set_fontweight("bold")

    pdf_path = output_dir / "fig9.pdf"
    png_path = output_dir / "fig9.png"
    figure.savefig(pdf_path, bbox_inches=None, pad_inches=0)
    figure.savefig(png_path, bbox_inches=None, pad_inches=0, dpi=300)
    plt.close(figure)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> int:
    args = parse_args()
    try:
        render(args.input, args.output)
    except (ValueError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
