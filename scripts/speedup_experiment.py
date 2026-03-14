#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = Path(__file__).with_suffix(".json")
EXAMPLE_CONFIG = Path(__file__).with_name("speedup_experiment.example.json")

BENCH_MODE_ALIASES = {
    "cpu": "cpu",
    "gpu_eval": "gpu_eval",
    "gpu_repro": "gpu_repro",
    "gpu_repro_overlap": "gpu_repro_overlap",
    "gpu": "gpu_repro",
    "gpu_overlap": "gpu_repro_overlap",
    "gpu_eval_repro_seq": "gpu_repro",
    "gpu_eval_repro_overlap": "gpu_repro_overlap",
}

BENCH_MODES = {
    "cpu": {"engine": "cpu", "repro_backend": "cpu", "repro_overlap": False},
    "gpu_eval": {"engine": "gpu", "repro_backend": "cpu", "repro_overlap": False},
    "gpu_repro": {"engine": "gpu", "repro_backend": "gpu", "repro_overlap": False},
    "gpu_repro_overlap": {"engine": "gpu", "repro_backend": "gpu", "repro_overlap": True},
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-population benchmark sweeps for CPU, GPU eval, and GPU reproduction modes."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to benchmark config JSON; default prefers scripts/speedup_experiment.json then falls back to scripts/speedup_experiment.example.json",
    )
    parser.add_argument(
        "--fixtures",
        default="",
        help="Comma-separated fixture stems or paths to run; default runs all configured fixtures",
    )
    parser.add_argument("--outdir", default="", help="Output directory; default uses config prefix + timestamp")
    parser.add_argument("--bench-cli", default="", help="Override benchmark CLI path")
    parser.add_argument(
        "--population-sizes",
        default="",
        help="Comma-separated population sizes to run; default uses configured population_sizes",
    )
    parser.add_argument(
        "--max-expr-depths",
        default="",
        help="Comma-separated max_expr_depth values to run; default uses configured max_expr_depths or max_expr_depth",
    )
    parser.add_argument(
        "--modes",
        default="",
        help="Comma-separated benchmark modes: cpu,gpu_eval,gpu_repro,gpu_repro_overlap",
    )
    parser.add_argument("--probe-cases", type=int, default=0, help="Override probe case count")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=-1.0,
        help="Override minimum accepted non-error probe success ratio",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_config_path() -> Path:
    if LOCAL_CONFIG.exists():
        return LOCAL_CONFIG
    return EXAMPLE_CONFIG


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def select_fixtures(config_fixtures: list[str], fixture_filter: str) -> list[Path]:
    configured = [resolve_path(item) for item in config_fixtures]
    if not fixture_filter:
        return configured
    requested = {item.strip() for item in fixture_filter.split(",") if item.strip()}
    selected: list[Path] = []
    for path in configured:
        if str(path) in requested or path.name in requested or path.stem in requested:
            selected.append(path)
    missing = requested - {str(p) for p in selected} - {p.name for p in selected} - {p.stem for p in selected}
    if missing:
        raise SystemExit(f"unknown fixtures requested: {sorted(missing)}")
    return selected


def parse_modes(config: dict[str, Any], raw_override: str) -> list[str]:
    raw_modes = [item.strip() for item in raw_override.split(",") if item.strip()] if raw_override else config.get(
        "modes",
        ["cpu", "gpu_eval", "gpu_repro", "gpu_repro_overlap"],
    )
    modes: list[str] = []
    for raw_mode in raw_modes:
        canonical = BENCH_MODE_ALIASES.get(raw_mode)
        if canonical is None:
            raise SystemExit(f"unknown benchmark mode: {raw_mode}")
        if canonical not in modes:
            modes.append(canonical)
    if not modes:
        raise SystemExit("no benchmark modes configured")
    if modes[0] != "cpu":
        raise SystemExit("benchmark modes must include cpu as the first baseline mode")
    return modes


def parse_value(raw: str) -> Any:
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_bench(stdout_text: str) -> dict[str, Any]:
    for line in stdout_text.splitlines():
        if not line.startswith("BENCH "):
            continue
        body: dict[str, Any] = {}
        for token in line[len("BENCH ") :].split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            body[key] = parse_value(value)
        return body
    raise RuntimeError("missing BENCH line")


def parse_population_sizes(config: dict[str, Any], raw_override: str) -> list[int]:
    if raw_override:
        values = [item.strip() for item in raw_override.split(",") if item.strip()]
        out = [int(item) for item in values]
    else:
        configured = config.get("population_sizes")
        if configured is None:
            configured = [config["population_size"]]
        out = [int(item) for item in configured]
    if not out:
        raise SystemExit("no population sizes configured")
    if any(value <= 0 for value in out):
        raise SystemExit("population sizes must be positive")
    return out


def parse_max_expr_depths(config: dict[str, Any], raw_override: str) -> list[int]:
    if raw_override:
        values = [item.strip() for item in raw_override.split(",") if item.strip()]
        out = [int(item) for item in values]
    else:
        configured = config.get("max_expr_depths")
        if configured is None:
            configured = [config["max_expr_depth"]]
        elif isinstance(configured, int):
            configured = [configured]
        out = [int(item) for item in configured]
    if not out:
        raise SystemExit("no max_expr_depth values configured")
    if any(value <= 0 for value in out):
        raise SystemExit("max_expr_depth values must be positive")
    deduped: list[int] = []
    for value in out:
        if value not in deduped:
            deduped.append(value)
    return deduped


def speedup(cpu_ms: Any, other_ms: Any) -> float | None:
    if cpu_ms is None or other_ms in (None, 0):
        return None
    return float(cpu_ms) / float(other_ms)


def fmt_speed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def run_one(
    fixture: Path,
    mode_name: str,
    outdir: Path,
    *,
    bench_cli: Path,
    blocksize: int,
    fuel: int,
    mutation_rate: float,
    mutation_subtree_prob: float,
    crossover_rate: float,
    penalty: float,
    selection_pressure: int,
    population_size: int,
    seed_start: int,
    probe_cases: int,
    min_success_rate: float,
    max_expr_depth: int,
    max_stmts_per_block: int,
    max_total_nodes: int,
    max_for_k: int,
    max_call_args: int,
) -> dict[str, Any]:
    mode = BENCH_MODES[mode_name]
    cmd = [
        str(bench_cli),
        "--cases",
        str(fixture),
        "--engine",
        mode["engine"],
        "--repro-backend",
        mode["repro_backend"],
        "--repro-overlap",
        "on" if mode["repro_overlap"] else "off",
        "--blocksize",
        str(blocksize),
        "--fuel",
        str(fuel),
        "--mutation-rate",
        str(mutation_rate),
        "--mutation-subtree-prob",
        str(mutation_subtree_prob),
        "--crossover-rate",
        str(crossover_rate),
        "--penalty",
        str(penalty),
        "--selection-pressure",
        str(selection_pressure),
        "--population-size",
        str(population_size),
        "--seed-start",
        str(seed_start),
        "--probe-cases",
        str(probe_cases),
        "--min-success-rate",
        str(min_success_rate),
        "--max-expr-depth",
        str(max_expr_depth),
        "--max-stmts-per-block",
        str(max_stmts_per_block),
        "--max-total-nodes",
        str(max_total_nodes),
        "--max-for-k",
        str(max_for_k),
        "--max-call-args",
        str(max_call_args),
        "--out-population-json",
        str(outdir / "fixed_population.seeds.json"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    console_path = outdir / f"one_gen_{mode_name}.console.log"
    console_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{mode_name} benchmark failed for {fixture.name}; see {console_path}")
    return parse_bench(proc.stdout)


def build_speedup_report(modes: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    cpu = modes["cpu"]
    out: dict[str, dict[str, float | None]] = {}
    for mode_name, result in modes.items():
        if mode_name == "cpu":
            continue
        out[mode_name] = {
            "compile_cpu_over_mode": speedup(cpu.get("compile_ms"), result.get("compile_ms")),
            "eval_cpu_over_mode": speedup(cpu.get("eval_ms"), result.get("eval_ms")),
            "repro_cpu_over_mode": speedup(cpu.get("repro_ms"), result.get("repro_ms")),
            "selection_cpu_over_mode": speedup(cpu.get("selection_ms"), result.get("selection_ms")),
            "crossover_cpu_over_mode": speedup(cpu.get("crossover_ms"), result.get("crossover_ms")),
            "mutation_cpu_over_mode": speedup(cpu.get("mutation_ms"), result.get("mutation_ms")),
            "total_cpu_over_mode": speedup(cpu.get("total_ms"), result.get("total_ms")),
        }
    return out


def write_fixture_report(outdir: Path, modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {
        "benchmark_type": "fixed_population_compare_v4",
        "population_json": str(outdir / "fixed_population.seeds.json"),
        "modes": modes,
        "speedup_vs_cpu": build_speedup_report(modes),
    }
    (outdir / "mode_compare.report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Fixed-Population Mode Benchmark",
        "",
        f"- population_json: `{report['population_json']}`",
        f"- benchmark_type: `{report['benchmark_type']}`",
        "",
        "## Speedup Vs CPU",
        "",
    ]
    for mode_name, speedup_map in report["speedup_vs_cpu"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in speedup_map.items():
            lines.append(f"- {key}: {fmt_speed(value)}")
        lines.append("")
    lines.append("## Raw Mode Timings")
    lines.append("")
    for mode_name, raw in report["modes"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in raw.items():
            lines.append(f"- {key}: {value if not isinstance(value, float) else fmt_ms(value)}")
        lines.append("")
    (outdir / "mode_compare.report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def average_speedup(
    aggregate: list[dict[str, Any]],
    mode_name: str,
    key: str,
    *,
    max_expr_depth: int | None = None,
) -> float | None:
    values = [
        item["speedup_vs_cpu"].get(mode_name, {}).get(key)
        for item in aggregate
        if item["speedup_vs_cpu"].get(mode_name, {}).get(key) is not None
        and (max_expr_depth is None or item["max_expr_depth"] == max_expr_depth)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config) if args.config else default_config_path()
    config = load_config(config_path)

    bench_cli = resolve_path(args.bench_cli or config["bench_cli"])
    if not bench_cli.exists():
        raise SystemExit(f"missing bench cli: {bench_cli}")

    fixtures = select_fixtures(config["fixtures"], args.fixtures)
    if not fixtures:
        raise SystemExit("no fixtures selected")

    modes = parse_modes(config, args.modes)
    population_sizes = parse_population_sizes(config, args.population_sizes)
    max_expr_depths = parse_max_expr_depths(config, args.max_expr_depths)
    probe_cases = args.probe_cases or int(config["probe_cases"])
    min_success_rate = args.min_success_rate if args.min_success_rate >= 0.0 else float(config["min_success_rate"])

    outdir = Path(args.outdir) if args.outdir else ROOT / f"{config['outdir_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    aggregate: list[dict[str, Any]] = []
    common = dict(
        bench_cli=bench_cli,
        blocksize=int(config["blocksize"]),
        fuel=int(config["fuel"]),
        mutation_rate=float(config["mutation_rate"]),
        mutation_subtree_prob=float(config["mutation_subtree_prob"]),
        crossover_rate=float(config["crossover_rate"]),
        penalty=float(config["penalty"]),
        selection_pressure=int(config["selection_pressure"]),
        seed_start=int(config["seed_start"]),
        probe_cases=probe_cases,
        min_success_rate=min_success_rate,
        max_stmts_per_block=int(config["max_stmts_per_block"]),
        max_total_nodes=int(config["max_total_nodes"]),
        max_for_k=int(config["max_for_k"]),
        max_call_args=int(config["max_call_args"]),
    )
    for max_expr_depth in max_expr_depths:
        depth_root = outdir if len(max_expr_depths) == 1 else outdir / f"depth{max_expr_depth}"
        for population_size in population_sizes:
            for fixture in fixtures:
                fixture_outdir = depth_root / f"{fixture.stem}_pop{population_size}"
                fixture_outdir.mkdir(parents=True, exist_ok=True)
                mode_results: dict[str, dict[str, Any]] = {}
                for mode_name in modes:
                    print(
                        f"[fixture-bench] {mode_name} fixture={fixture.stem} pop={population_size} depth={max_expr_depth}",
                        flush=True,
                    )
                    mode_results[mode_name] = run_one(
                        fixture,
                        mode_name,
                        fixture_outdir,
                        population_size=population_size,
                        max_expr_depth=max_expr_depth,
                        **common,
                    )
                report = write_fixture_report(fixture_outdir, mode_results)
                aggregate.append(
                    {
                        "fixture": fixture.stem,
                        "population_size": population_size,
                        "max_expr_depth": max_expr_depth,
                        "report_json": str(fixture_outdir / "mode_compare.report.json"),
                        "speedup_vs_cpu": report["speedup_vs_cpu"],
                    }
                )

    summary = {
        "benchmark_type": "fixture_speedup_modes_v1",
        "modes": modes,
        "max_expr_depths": max_expr_depths,
        "fixtures": aggregate,
        "average_speedup_vs_cpu": {
            mode_name: {
                key: average_speedup(aggregate, mode_name, key)
                for key in (
                    "compile_cpu_over_mode",
                    "eval_cpu_over_mode",
                    "repro_cpu_over_mode",
                    "selection_cpu_over_mode",
                    "crossover_cpu_over_mode",
                    "mutation_cpu_over_mode",
                    "total_cpu_over_mode",
                )
            }
            for mode_name in modes
            if mode_name != "cpu"
        },
        "average_speedup_vs_cpu_by_max_expr_depth": {
            str(max_expr_depth): {
                mode_name: {
                    key: average_speedup(aggregate, mode_name, key, max_expr_depth=max_expr_depth)
                    for key in (
                        "compile_cpu_over_mode",
                        "eval_cpu_over_mode",
                        "repro_cpu_over_mode",
                        "selection_cpu_over_mode",
                        "crossover_cpu_over_mode",
                        "mutation_cpu_over_mode",
                        "total_cpu_over_mode",
                    )
                }
                for mode_name in modes
                if mode_name != "cpu"
            }
            for max_expr_depth in max_expr_depths
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    lines = ["# Fixture Speedup Summary", "", f"- outdir: `{outdir}`", "", "## Average Speedup Vs CPU", ""]
    for mode_name, speedup_map in summary["average_speedup_vs_cpu"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in speedup_map.items():
            lines.append(f"- {key}: {fmt_speed(value)}")
        lines.append("")
    if len(max_expr_depths) > 1:
        lines.extend(["## Average Speedup Vs CPU By max_expr_depth", ""])
        for max_expr_depth in max_expr_depths:
            lines.append(f"### depth={max_expr_depth}")
            lines.append("")
            for mode_name, speedup_map in summary["average_speedup_vs_cpu_by_max_expr_depth"][str(max_expr_depth)].items():
                lines.append(f"#### `{mode_name}`")
                lines.append("")
                for key, value in speedup_map.items():
                    lines.append(f"- {key}: {fmt_speed(value)}")
                lines.append("")
    lines.extend(["## Fixtures", ""])
    for item in aggregate:
        parts = []
        for mode_name in modes:
            if mode_name == "cpu":
                continue
            speed_map = item["speedup_vs_cpu"].get(mode_name, {})
            parts.append(
                f"{mode_name}: total={fmt_speed(speed_map.get('total_cpu_over_mode'))}, eval={fmt_speed(speed_map.get('eval_cpu_over_mode'))}"
            )
        lines.append(
            f"- {item['fixture']} pop={item['population_size']} depth={item['max_expr_depth']}: " + "; ".join(parts)
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fixture-bench] summary json: {outdir / 'summary.json'}")
    print(f"[fixture-bench] summary md: {outdir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
