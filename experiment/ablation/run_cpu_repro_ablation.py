#!/usr/bin/env python3
"""Run the CPU reproduction ablation experiment and generate reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    extra_args: tuple[str, ...]


MODES: tuple[Mode, ...] = (
    Mode("baseline", "baseline", ("--engine", "gpu", "--repro-backend", "cpu")),
    Mode(
        "full",
        "full",
        ("--engine", "gpu", "--repro-backend", "gpu", "--repro-overlap", "on"),
    ),
    Mode(
        "cpu_gpu_selection",
        "cpu+3",
        (
            "--engine",
            "gpu",
            "--repro-backend",
            "cpu",
            "--cpu-repro-ablation",
            "gpu_selection",
        ),
    ),
    Mode(
        "cpu_gpu_candidates",
        "cpu+6",
        (
            "--engine",
            "gpu",
            "--repro-backend",
            "cpu",
            "--cpu-repro-ablation",
            "gpu_candidates",
        ),
    ),
    Mode(
        "cpu_gpu_coupled_donor",
        "cpu+78",
        (
            "--engine",
            "gpu",
            "--repro-backend",
            "cpu",
            "--cpu-repro-ablation",
            "gpu_coupled_donor",
        ),
    ),
)


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, stop_s = part.split("-", 1)
            start = int(start_s)
            stop = int(stop_s)
            if stop < start:
                raise argparse.ArgumentTypeError(f"invalid descending seed range: {part}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def select_modes(raw: str) -> list[Mode]:
    by_name = {mode.name: mode for mode in MODES}
    by_label = {mode.label: mode for mode in MODES}
    if raw == "all":
        return list(MODES)
    selected: list[Mode] = []
    for part in raw.split(","):
        key = part.strip()
        if not key:
            continue
        mode = by_name.get(key) or by_label.get(key)
        if mode is None:
            valid = ", ".join(mode.name for mode in MODES)
            raise argparse.ArgumentTypeError(f"unknown mode {key!r}; valid names: {valid}")
        selected.append(mode)
    if not selected:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return selected


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def mean(values: list[float]) -> float | None:
    return finite_or_none(statistics.fmean(values)) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return finite_or_none(statistics.stdev(values))


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values),
        "stdev": stdev(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def command_for(
    binary: Path,
    cases: Path,
    grammar_config: Path,
    mode: Mode,
    seed: int,
    population_size: int,
    generations: int,
    blocksize: int,
    out_json: Path,
    timing: str,
) -> list[str]:
    return [
        str(binary),
        "--cases",
        str(cases),
        "--grammar-config",
        str(grammar_config),
        *mode.extra_args,
        "--blocksize",
        str(blocksize),
        "--population-size",
        str(population_size),
        "--generations",
        str(generations),
        "--seed",
        str(seed),
        "--timing",
        timing,
        "--out-json",
        str(out_json),
    ]


def run_one(cmd: list[str], stdout_path: Path, dry_run: bool) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        stdout_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}\n{tail}")


def load_run_metrics(json_path: Path, mode: Mode, seed: int, generations: int) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    history = payload["history"]
    mean_curve = [float(row["mean_fitness"]) for row in history]
    best_curve = [float(row["best_fitness"]) for row in history]
    if len(mean_curve) < generations:
        raise RuntimeError(f"{json_path} has only {len(mean_curve)} generations, expected {generations}")

    window = mean_curve[:generations]
    timing = payload.get("timing", {})
    generation_total_ms = [float(x) for x in timing.get("generation_total_ms", [])]
    generation_repro_ms = [float(x) for x in timing.get("generation_repro_ms", [])]

    return {
        "mode": mode.name,
        "label": mode.label,
        "seed": seed,
        "auc_mean_fitness": sum(window),
        "gen_last_mean_fitness": window[generations - 1],
        "mean_fitness_gain": window[generations - 1] - window[0],
        "final_best_fitness": payload.get("final", {}).get("best_fitness"),
        "mean_curve": mean_curve,
        "best_curve": best_curve,
        "avg_generation_total_ms": mean(generation_total_ms),
        "avg_generation_repro_ms": mean(generation_repro_ms),
        "raw_json": str(json_path),
    }


def build_summary(runs: list[dict[str, Any]], modes: list[Mode], generations: int) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {mode.name: [] for mode in modes}
    for run in runs:
        by_mode[run["mode"]].append(run)

    baseline_by_seed = {run["seed"]: run for run in by_mode.get("baseline", [])}
    mode_summaries: dict[str, Any] = {}
    for mode in modes:
        mode_runs = sorted(by_mode[mode.name], key=lambda item: item["seed"])
        aucs = [float(run["auc_mean_fitness"]) for run in mode_runs]
        gen_last = [float(run["gen_last_mean_fitness"]) for run in mode_runs]
        gains = [float(run["mean_fitness_gain"]) for run in mode_runs]
        repro_ms = [
            float(run["avg_generation_repro_ms"])
            for run in mode_runs
            if run["avg_generation_repro_ms"] is not None
        ]
        total_ms = [
            float(run["avg_generation_total_ms"])
            for run in mode_runs
            if run["avg_generation_total_ms"] is not None
        ]
        auc_delta = [
            float(run["auc_mean_fitness"]) - float(baseline_by_seed[run["seed"]]["auc_mean_fitness"])
            for run in mode_runs
            if run["seed"] in baseline_by_seed
        ]
        gen_last_delta = [
            float(run["gen_last_mean_fitness"]) -
            float(baseline_by_seed[run["seed"]]["gen_last_mean_fitness"])
            for run in mode_runs
            if run["seed"] in baseline_by_seed
        ]
        mode_summaries[mode.name] = {
            "label": mode.label,
            "runs": mode_runs,
            "auc_mean_fitness": summarize_values(aucs),
            "auc_delta_vs_baseline": summarize_values(auc_delta),
            "gen_last_mean_fitness": summarize_values(gen_last),
            "gen_last_delta_vs_baseline": summarize_values(gen_last_delta),
            "mean_fitness_gain": summarize_values(gains),
            "avg_generation_repro_ms": summarize_values(repro_ms),
            "avg_generation_total_ms": summarize_values(total_ms),
        }

    return {
        "metric_note": f"AUC is the discrete sum of history.mean_fitness[0:{generations}].",
        "modes": mode_summaries,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_summary_md(summary: dict[str, Any], modes: list[Mode], path: Path) -> None:
    lines = [
        "# CPU Reproduction Ablation Summary",
        "",
        summary["metric_note"],
        "",
        "| mode | n | AUC mean | AUC delta vs baseline | gen-last mean | gen-last delta | gain | repro ms/gen | total ms/gen |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        item = summary["modes"][mode.name]
        lines.append(
            "| {label} | {n} | {auc} | {auc_delta} | {last} | {last_delta} | {gain} | {repro} | {total} |".format(
                label=item["label"],
                n=item["auc_mean_fitness"]["n"],
                auc=fmt(item["auc_mean_fitness"]["mean"]),
                auc_delta=fmt(item["auc_delta_vs_baseline"]["mean"]),
                last=fmt(item["gen_last_mean_fitness"]["mean"]),
                last_delta=fmt(item["gen_last_delta_vs_baseline"]["mean"]),
                gain=fmt(item["mean_fitness_gain"]["mean"]),
                repro=fmt(item["avg_generation_repro_ms"]["mean"]),
                total=fmt(item["avg_generation_total_ms"]["mean"]),
            )
        )
    lines.extend(["", "Higher fitness and higher deltas are better."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(summary: dict[str, Any], modes: list[Mode], out_dir: Path, generations: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#333333",
        "full": "#1f77b4",
        "cpu_gpu_selection": "#2ca02c",
        "cpu_gpu_candidates": "#ff7f0e",
        "cpu_gpu_coupled_donor": "#d62728",
    }

    x = list(range(generations))
    plt.figure(figsize=(10, 6))
    for mode in modes:
        runs = summary["modes"][mode.name]["runs"]
        curves = [run["mean_curve"][:generations] for run in runs]
        if not curves:
            continue
        y = [statistics.fmean(curve[gen] for curve in curves) for gen in x]
        if len(curves) >= 2:
            sem = [
                statistics.stdev(curve[gen] for curve in curves) / math.sqrt(len(curves))
                for gen in x
            ]
            lower = [a - b for a, b in zip(y, sem)]
            upper = [a + b for a, b in zip(y, sem)]
            plt.fill_between(x, lower, upper, color=colors.get(mode.name), alpha=0.12, linewidth=0)
        plt.plot(x, y, label=mode.label, color=colors.get(mode.name), linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Mean fitness")
    plt.title("Mean Fitness vs Generation")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_fitness_vs_generation.png", dpi=160)
    plt.close()

    labels: list[str] = []
    data: list[list[float]] = []
    for mode in modes:
        aucs = [float(run["auc_mean_fitness"]) for run in summary["modes"][mode.name]["runs"]]
        if aucs:
            labels.append(mode.label)
            data.append(aucs)
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel(f"AUC mean fitness, gen 0-{generations - 1}")
    plt.title("AUC Mean Fitness Distribution")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "auc_mean_fitness_boxplot.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "cpp/build/g3pvm_evolve_cli")
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "data/fixtures/psb1/median.train.json")
    parser.add_argument("--grammar-config", type=Path, default=REPO_ROOT / "configs/grammar/scalar.json")
    parser.add_argument("--population-size", type=int, default=8192)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0-9"))
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--timing", choices=("none", "summary", "per_gen", "all"), default="summary")
    parser.add_argument("--modes", type=select_modes, default=select_modes("all"))
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root
    if out_root is None:
        out_root = REPO_ROOT / f"logs/ablation_cpu_repro_median_pop8192_gen20_{timestamp}"
    out_root = out_root.resolve()
    raw_dir = out_root / "raw"
    reports_dir = out_root / "reports"
    plots_dir = out_root / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for mode in args.modes:
        for seed in args.seeds:
            mode_dir = raw_dir / mode.name
            out_json = mode_dir / f"seed_{seed}.json"
            stdout_path = mode_dir / f"seed_{seed}.stdout.txt"
            cmd = command_for(
                args.binary,
                args.cases,
                args.grammar_config,
                mode,
                seed,
                args.population_size,
                args.generations,
                args.blocksize,
                out_json,
                args.timing,
            )
            if args.resume and out_json.exists() and stdout_path.exists():
                print(f"[resume] {mode.label} seed={seed}")
            else:
                print(f"[run] {mode.label} seed={seed}")
                mode_dir.mkdir(parents=True, exist_ok=True)
                run_one(cmd, stdout_path, args.dry_run)
            if not args.dry_run:
                runs.append(load_run_metrics(out_json, mode, seed, args.generations))

    if args.dry_run:
        print(f"dry-run output written under {out_root}")
        return 0

    summary = build_summary(runs, args.modes, args.generations)
    summary["spec"] = {
        "cases": str(args.cases),
        "grammar_config": str(args.grammar_config),
        "population_size": args.population_size,
        "generations": args.generations,
        "seeds": args.seeds,
        "blocksize": args.blocksize,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary, args.modes, reports_dir / "summary.md")
    make_plots(summary, args.modes, plots_dir, args.generations)

    print(f"summary: {reports_dir / 'summary.md'}")
    print(f"plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
