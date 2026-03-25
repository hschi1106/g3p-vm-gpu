#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "node_simple_x_plus_1_1024_speedup_20260322_220459"
DEFAULT_OUT_DIR = REPO_ROOT / "draw" / DEFAULT_LOG_DIR.name
EPSILON_MS = 1e-9


@dataclass(frozen=True)
class SegmentStyle:
    key: str
    label: str
    color: str
    hatch: str = ""
    one_time: bool = False


CPU_SEGMENTS = [
    SegmentStyle("compile", "Compile", "#4E79A7"),
    SegmentStyle("eval", "Eval", "#E15759"),
    SegmentStyle("repro_selection", "Selection", "#76B7B2"),
    SegmentStyle("repro_crossover", "Crossover", "#59A14F"),
    SegmentStyle("repro_mutation", "Mutation", "#EDC948"),
    SegmentStyle("repro_other", "Repro Other", "#B07AA1"),
    SegmentStyle("run_other", "Run Other", "#9C755F"),
]


ACCELERATED_SEGMENTS = [
    SegmentStyle("compile", "Compile", "#4E79A7"),
    SegmentStyle(
        "eval_init",
        "Eval Init (first gen only)",
        "#A0CBE8",
        hatch="//",
        one_time=True,
    ),
    SegmentStyle("eval_pack_upload", "Eval Pack/Launch/Upload", "#76B7B2"),
    SegmentStyle("eval_kernel", "Eval Kernel", "#E15759"),
    SegmentStyle("eval_finish", "Eval Finish", "#F28E2B"),
    SegmentStyle(
        "repro_setup",
        "Repro Setup (usually first gen only)",
        "#C7E9C0",
        hatch="\\\\",
        one_time=True,
    ),
    SegmentStyle("repro_prepare_pack", "Repro Prepare/Preprocess/Pack", "#59A14F"),
    SegmentStyle("repro_upload_kernel", "Repro Upload/Kernel", "#2D6A4F"),
    SegmentStyle("repro_finish", "Repro Finish", "#EDC948"),
    SegmentStyle("cpu_repro_selection", "CPU Repro Selection", "#59A14F"),
    SegmentStyle("cpu_repro_crossover", "CPU Repro Crossover", "#8CD17D"),
    SegmentStyle("cpu_repro_mutation", "CPU Repro Mutation", "#F1CE63"),
    SegmentStyle("cpu_repro_other", "CPU Repro Other", "#B07AA1"),
    SegmentStyle("run_other", "Run Other", "#9C755F"),
]


def positive_ms(value: float) -> float:
    if abs(value) <= EPSILON_MS:
        return 0.0
    return max(0.0, float(value))


def load_summary(log_dir: Path) -> dict:
    summary_path = log_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json under {log_dir}")
    return json.loads(summary_path.read_text())


def detect_sweep_axis(summary: Mapping[str, object]) -> tuple[str, str, str]:
    fixtures = summary.get("fixtures", [])
    if not fixtures:
        raise ValueError("summary.json contains no fixtures")

    sample = fixtures[0]
    if sample.get("target_node_count") is not None:
        return ("node_count", "Number of nodes", "nodes")
    population_json = sample.get("population_json")
    if population_json:
        try:
            payload = json.loads(Path(str(population_json)).read_text())
            if payload.get("target_depth") not in (None, "", 0):
                return ("max_expr_depth", "Exact Depth", "depth")
        except (OSError, json.JSONDecodeError):
            pass
    if sample.get("max_expr_depth") is not None:
        return ("max_expr_depth", "Max expression depth", "depth")
    raise ValueError("unable to detect sweep axis from summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw timing breakdown charts for a benchmark sweep."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("DRAW_LOG_DIR", DEFAULT_LOG_DIR)),
        help="benchmark log directory containing summary.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(os.environ.get("DRAW_OUT_DIR", DEFAULT_OUT_DIR)),
        help="output directory for generated charts",
    )
    parser.add_argument(
        "--gpu-mode",
        default=os.environ.get("DRAW_GPU_MODE", "gpu_repro"),
        choices=["gpu_eval", "gpu_repro"],
        help="accelerated mode used for GPU-side charts",
    )
    return parser.parse_args()


def extract_rows(summary: Mapping[str, object], gpu_mode: str) -> list[dict]:
    sweep_key, _, _ = detect_sweep_axis(summary)
    rows: list[dict] = []
    for fixture in summary.get("fixtures", []):
        fixture_modes = fixture["modes"]
        rows.append(
            {
                "x_value": int(fixture[sweep_key]),
                "label": fixture["population_label"],
                "cpu": fixture_modes["cpu"],
                "gpu": fixture_modes[gpu_mode],
            }
        )
    rows.sort(key=lambda row: row["x_value"])
    return rows


def build_cpu_segments(mode: Mapping[str, object]) -> dict[str, float]:
    compile_ms = positive_ms(mode["compile_ms"])
    eval_ms = positive_ms(mode["eval_ms"])
    selection_ms = positive_ms(mode["selection_ms"])
    crossover_ms = positive_ms(mode["crossover_ms"])
    mutation_ms = positive_ms(mode["mutation_ms"])
    repro_ms = positive_ms(mode["repro_ms"])
    total_ms = positive_ms(mode["total_ms"])

    repro_other = positive_ms(repro_ms - selection_ms - crossover_ms - mutation_ms)
    run_other = positive_ms(total_ms - compile_ms - eval_ms - repro_ms)

    return {
        "compile": compile_ms,
        "eval": eval_ms,
        "repro_selection": selection_ms,
        "repro_crossover": crossover_ms,
        "repro_mutation": mutation_ms,
        "repro_other": repro_other,
        "run_other": run_other,
    }


def build_accelerated_segments(mode: Mapping[str, object]) -> dict[str, float]:
    compile_ms = positive_ms(mode["compile_ms"])
    eval_ms = positive_ms(mode["eval_ms"])
    total_ms = positive_ms(mode["total_ms"])
    repro_ms = positive_ms(mode["repro_ms"])

    eval_init_ms = positive_ms(mode["gpu_eval_init_ms"])
    eval_pack_upload_ms = positive_ms(
        mode["gpu_eval_pack_ms"] + mode["gpu_eval_launch_prep_ms"] + mode["gpu_eval_upload_ms"]
    )
    eval_kernel_ms = positive_ms(mode["gpu_eval_kernel_ms"])
    eval_finish_ms = positive_ms(
        mode["gpu_eval_call_ms"]
        - mode["gpu_eval_pack_ms"]
        - mode["gpu_eval_launch_prep_ms"]
        - mode["gpu_eval_upload_ms"]
        - mode["gpu_eval_kernel_ms"]
    )

    segments = {
        "compile": compile_ms,
        "eval_init": eval_init_ms,
        "eval_pack_upload": eval_pack_upload_ms,
        "eval_kernel": eval_kernel_ms,
        "eval_finish": eval_finish_ms,
        "repro_setup": 0.0,
        "repro_prepare_pack": 0.0,
        "repro_upload_kernel": 0.0,
        "repro_finish": 0.0,
        "cpu_repro_selection": 0.0,
        "cpu_repro_crossover": 0.0,
        "cpu_repro_mutation": 0.0,
        "cpu_repro_other": 0.0,
        "run_other": 0.0,
    }

    if mode["repro_backend"] == "gpu":
        repro_setup_ms = positive_ms(mode["repro_setup_ms"])
        repro_prepare_pack_ms = positive_ms(
            mode["repro_prepare_inputs_ms"] + mode["repro_preprocess_ms"] + mode["repro_pack_ms"]
        )
        repro_upload_kernel_ms = positive_ms(mode["repro_upload_ms"] + mode["repro_kernel_ms"])
        repro_finish_ms = positive_ms(
            mode["repro_copyback_ms"]
            + mode["repro_decode_ms"]
            + mode["repro_teardown_ms"]
            + repro_ms
            - mode["repro_prepare_inputs_ms"]
            - mode["repro_setup_ms"]
            - mode["repro_preprocess_ms"]
            - mode["repro_pack_ms"]
            - mode["repro_upload_ms"]
            - mode["repro_kernel_ms"]
            - mode["repro_copyback_ms"]
            - mode["repro_decode_ms"]
            - mode["repro_teardown_ms"]
        )
        segments.update(
            {
                "repro_setup": repro_setup_ms,
                "repro_prepare_pack": repro_prepare_pack_ms,
                "repro_upload_kernel": repro_upload_kernel_ms,
                "repro_finish": repro_finish_ms,
            }
        )
    else:
        selection_ms = positive_ms(mode["selection_ms"])
        crossover_ms = positive_ms(mode["crossover_ms"])
        mutation_ms = positive_ms(mode["mutation_ms"])
        cpu_repro_other_ms = positive_ms(repro_ms - selection_ms - crossover_ms - mutation_ms)
        segments.update(
            {
                "cpu_repro_selection": selection_ms,
                "cpu_repro_crossover": crossover_ms,
                "cpu_repro_mutation": mutation_ms,
                "cpu_repro_other": cpu_repro_other_ms,
            }
        )

    run_other = positive_ms(total_ms - compile_ms - eval_ms - repro_ms)
    segments["run_other"] = run_other
    return segments


def build_segment_arrays(
    rows: list[dict],
    styles: Iterable[SegmentStyle],
    row_key: str,
    include_one_time: bool,
) -> tuple[list[int], list[SegmentStyle], dict[str, np.ndarray], np.ndarray]:
    x_values = [row["x_value"] for row in rows]
    style_list = list(styles)
    arrays: dict[str, np.ndarray] = {}

    for style in style_list:
        values = []
        for row in rows:
            row_segments = build_cpu_segments(row[row_key]) if row_key == "cpu" else build_accelerated_segments(row[row_key])
            value = row_segments[style.key]
            if style.one_time and not include_one_time:
                value = 0.0
            values.append(positive_ms(value))
        arrays[style.key] = np.array(values, dtype=float)

    visible_styles: list[SegmentStyle] = []
    for style in style_list:
        if np.all(arrays[style.key] <= EPSILON_MS):
            continue
        visible_styles.append(style)

    totals = np.zeros(len(x_values), dtype=float)
    for style in visible_styles:
        totals += arrays[style.key]

    return x_values, visible_styles, arrays, totals


def format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_stacked_bars(
    ax: plt.Axes,
    x_values: list[int],
    styles: list[SegmentStyle],
    arrays: Mapping[str, np.ndarray],
    totals: np.ndarray,
    title: str,
    xlabel: str,
    note: str | None = None,
) -> None:
    x = np.arange(len(x_values))
    bottoms = np.zeros(len(x_values), dtype=float)

    for style in styles:
        values = arrays[style.key]
        bar_kwargs = {
            "x": x,
            "height": values,
            "bottom": bottoms,
            "width": 0.78,
            "label": style.label,
            "color": style.color,
            "linewidth": 0.6,
        }
        if style.hatch:
            bar_kwargs["hatch"] = style.hatch
            bar_kwargs["edgecolor"] = "#222222"
        else:
            bar_kwargs["edgecolor"] = "#FFFFFF"
        ax.bar(**bar_kwargs)
        bottoms += values

    ymax = float(np.max(totals)) if len(totals) else 0.0
    y_offset = max(ymax * 0.015, 0.5)
    for xpos, total in zip(x, totals):
        ax.text(
            xpos,
            total + y_offset,
            format_ms(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, ymax * 1.12 + y_offset)

    if note:
        ax.text(
            0.01,
            0.99,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
        )


def plot_grouped_stacked_bars(
    ax: plt.Axes,
    node_counts: list[int],
    left_styles: list[SegmentStyle],
    left_arrays: Mapping[str, np.ndarray],
    left_totals: np.ndarray,
    left_label: str,
    right_styles: list[SegmentStyle],
    right_arrays: Mapping[str, np.ndarray],
    right_totals: np.ndarray,
    right_label: str,
    title: str,
    note: str | None = None,
) -> None:
    x = np.arange(len(node_counts), dtype=float)
    width = 0.34
    left_x = x - width / 2
    right_x = x + width / 2
    left_bottoms = np.zeros(len(node_counts), dtype=float)
    right_bottoms = np.zeros(len(node_counts), dtype=float)
    used_labels: set[str] = set()

    for style in left_styles:
        label = style.label if style.label not in used_labels else "_nolegend_"
        used_labels.add(style.label)
        bar_kwargs = {
            "x": left_x,
            "height": left_arrays[style.key],
            "bottom": left_bottoms,
            "width": width,
            "label": label,
            "color": style.color,
            "linewidth": 0.6,
        }
        if style.hatch:
            bar_kwargs["hatch"] = style.hatch
            bar_kwargs["edgecolor"] = "#222222"
        else:
            bar_kwargs["edgecolor"] = "#FFFFFF"
        ax.bar(**bar_kwargs)
        left_bottoms += left_arrays[style.key]

    for style in right_styles:
        label = style.label if style.label not in used_labels else "_nolegend_"
        used_labels.add(style.label)
        bar_kwargs = {
            "x": right_x,
            "height": right_arrays[style.key],
            "bottom": right_bottoms,
            "width": width,
            "label": label,
            "color": style.color,
            "linewidth": 0.6,
        }
        if style.hatch:
            bar_kwargs["hatch"] = style.hatch
            bar_kwargs["edgecolor"] = "#222222"
        else:
            bar_kwargs["edgecolor"] = "#FFFFFF"
        ax.bar(**bar_kwargs)
        right_bottoms += right_arrays[style.key]

    ymax = max(
        float(np.max(left_totals)) if len(left_totals) else 0.0,
        float(np.max(right_totals)) if len(right_totals) else 0.0,
    )
    y_offset = max(ymax * 0.015, 0.5)
    for xpos, total in zip(left_x, left_totals):
        ax.text(
            xpos,
            total + y_offset,
            format_ms(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for xpos, total in zip(right_x, right_totals):
        ax.text(
            xpos,
            total + y_offset,
            format_ms(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(node) for node in node_counts])
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, ymax * 1.12 + y_offset)

    group_note = f"Left bar in each node group: {left_label}. Right bar: {right_label}."
    if note:
        group_note = f"{group_note}\n{note}"
    ax.text(
        0.01,
        0.99,
        group_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )


def plot_total_comparison_bars(
    ax: plt.Axes,
    x_values: list[int],
    cpu_totals: np.ndarray,
    gpu_totals: np.ndarray,
    gpu_mode: str,
    title: str,
    xlabel: str,
    note: str | None = None,
) -> None:
    x = np.arange(len(x_values), dtype=float)
    width = 0.34
    cpu_x = x - width / 2
    gpu_x = x + width / 2

    ax.bar(
        cpu_x,
        cpu_totals,
        width=width,
        label="CPU",
        color="#E15759",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )
    ax.bar(
        gpu_x,
        gpu_totals,
        width=width,
        label="GPU",
        color="#4E79A7",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )

    ymax = max(
        float(np.max(cpu_totals)) if len(cpu_totals) else 0.0,
        float(np.max(gpu_totals)) if len(gpu_totals) else 0.0,
    )
    y_offset = max(ymax * 0.015, 0.5)
    speedup_offset = max(ymax * 0.045, 1.5)

    for xpos, total in zip(cpu_x, cpu_totals):
        ax.text(
            xpos,
            total + y_offset,
            format_ms(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for xpos, total in zip(gpu_x, gpu_totals):
        ax.text(
            xpos,
            total + y_offset,
            format_ms(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for idx, (cpu_total, gpu_total) in enumerate(zip(cpu_totals, gpu_totals)):
        if gpu_total <= EPSILON_MS:
            speedup_label = "inf x"
        else:
            speedup_label = f"{cpu_total / gpu_total:.1f}x"
        ax.text(
            x[idx],
            max(cpu_total, gpu_total) + speedup_offset,
            speedup_label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, ymax * 1.18 + speedup_offset)

    chart_note = f"Each group shows CPU total vs. GPU total. GPU mode: {gpu_mode}. Speedup label = CPU/GPU."
    if note:
        chart_note = f"{chart_note}\n{note}"
    ax.text(
        0.01,
        0.99,
        chart_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )


def save_gpu_chart(
    rows: list[dict],
    gpu_mode: str,
    outdir: Path,
    xlabel: str,
    axis_suffix: str,
) -> Path:
    x_values, styles, arrays, totals = build_segment_arrays(
        rows=rows,
        styles=ACCELERATED_SEGMENTS,
        row_key="gpu",
        include_one_time=True,
    )
    fig, ax = plt.subplots(figsize=(15, 8))
    note = (
        "Hatched segments are startup-biased costs.\n"
        "gpu_eval_init_ms is session init; repro_setup_ms is usually generation 1 only."
    )
    plot_stacked_bars(
        ax,
        x_values,
        styles,
        arrays,
        totals,
        title=f"{gpu_mode} timing breakdown vs. {xlabel.lower()}",
        xlabel=xlabel,
        note=note,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()

    outpath = outdir / f"01_gpu_timing_breakdown_vs_{axis_suffix}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def save_cpu_gpu_chart(
    rows: list[dict],
    gpu_mode: str,
    outdir: Path,
    include_one_time: bool,
    xlabel: str,
    axis_suffix: str,
) -> Path:
    cpu_x_values, _, _, cpu_totals = build_segment_arrays(
        rows=rows,
        styles=CPU_SEGMENTS,
        row_key="cpu",
        include_one_time=True,
    )
    _, _, _, gpu_totals = build_segment_arrays(
        rows=rows,
        styles=ACCELERATED_SEGMENTS,
        row_key="gpu",
        include_one_time=include_one_time,
    )
    fig, ax = plt.subplots(figsize=(15, 8))
    note = (
        "Hatched segments are startup-biased GPU costs:\n"
        "gpu_eval_init_ms is session init; repro_setup_ms is usually generation 1 only."
    )
    if not include_one_time:
        note = (
            "This view removes the explicitly measured startup-biased GPU costs:\n"
            "gpu_eval_init_ms and the usually-first-generation repro_setup_ms."
        )

    plot_total_comparison_bars(
        ax,
        cpu_x_values,
        cpu_totals,
        gpu_totals,
        gpu_mode,
        title=f"CPU vs GPU total time vs. {xlabel.lower()} ({gpu_mode})",
        xlabel=xlabel,
        note=note,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()

    outpath = outdir / f"02_cpu_gpu_timing_breakdown_vs_{axis_suffix}.png"
    if not include_one_time:
        outpath = outdir / f"03_cpu_gpu_timing_breakdown_vs_{axis_suffix}_no_first_gen.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_plot_manifest(
    rows: list[dict],
    gpu_mode: str,
    log_dir: Path,
    outdir: Path,
    sweep_key: str,
    xlabel: str,
) -> Path:
    cpu_nodes, cpu_styles, cpu_arrays, cpu_totals = build_segment_arrays(
        rows=rows,
        styles=CPU_SEGMENTS,
        row_key="cpu",
        include_one_time=True,
    )
    gpu_nodes, gpu_styles, gpu_arrays, gpu_totals = build_segment_arrays(
        rows=rows,
        styles=ACCELERATED_SEGMENTS,
        row_key="gpu",
        include_one_time=True,
    )
    _, gpu_steady_styles, gpu_steady_arrays, gpu_steady_totals = build_segment_arrays(
        rows=rows,
        styles=ACCELERATED_SEGMENTS,
        row_key="gpu",
        include_one_time=False,
    )
    gpu_speedup = np.divide(
        cpu_totals,
        gpu_totals,
        out=np.full_like(cpu_totals, np.inf),
        where=gpu_totals > EPSILON_MS,
    )
    gpu_steady_speedup = np.divide(
        cpu_totals,
        gpu_steady_totals,
        out=np.full_like(cpu_totals, np.inf),
        where=gpu_steady_totals > EPSILON_MS,
    )

    manifest = {
        "source_log_dir": str(log_dir),
        "gpu_mode": gpu_mode,
        "sweep_key": sweep_key,
        "sweep_label": xlabel,
        "x_values": cpu_nodes,
        "startup_biased_metrics": ["gpu_eval_init_ms", "repro_setup_ms"],
        "cpu_segments_ms": {style.label: cpu_arrays[style.key].tolist() for style in cpu_styles},
        "cpu_totals_ms": cpu_totals.tolist(),
        "gpu_segments_ms": {style.label: gpu_arrays[style.key].tolist() for style in gpu_styles},
        "gpu_totals_ms": gpu_totals.tolist(),
        "gpu_without_first_gen_segments_ms": {
            style.label: gpu_steady_arrays[style.key].tolist() for style in gpu_steady_styles
        },
        "gpu_without_first_gen_totals_ms": gpu_steady_totals.tolist(),
        "cpu_vs_gpu_speedup_x": gpu_speedup.tolist(),
        "cpu_vs_gpu_without_startup_speedup_x": gpu_steady_speedup.tolist(),
        "gpu_x_values": gpu_nodes,
    }
    outpath = outdir / "plot_manifest.json"
    outpath.write_text(json.dumps(manifest, indent=2))
    return outpath

def main() -> None:
    args = parse_args()
    summary = load_summary(args.log_dir)
    sweep_key, xlabel, axis_suffix = detect_sweep_axis(summary)
    rows = extract_rows(summary, args.gpu_mode)
    args.outdir.mkdir(parents=True, exist_ok=True)

    generated = [
        save_gpu_chart(rows, args.gpu_mode, args.outdir, xlabel, axis_suffix),
        save_cpu_gpu_chart(rows, args.gpu_mode, args.outdir, include_one_time=True, xlabel=xlabel, axis_suffix=axis_suffix),
        save_cpu_gpu_chart(rows, args.gpu_mode, args.outdir, include_one_time=False, xlabel=xlabel, axis_suffix=axis_suffix),
        write_plot_manifest(rows, args.gpu_mode, args.log_dir, args.outdir, sweep_key, xlabel),
    ]

    print(f"source: {args.log_dir}")
    print(f"outdir: {args.outdir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
