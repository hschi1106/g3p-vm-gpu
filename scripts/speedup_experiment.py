#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = Path(__file__).with_suffix(".json")
EXAMPLE_CONFIG = Path(__file__).with_name("speedup_experiment.example.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU/GPU one-generation benchmark for all configured fixtures."
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


def speedup(cpu_ms: Any, gpu_ms: Any) -> float | None:
    if cpu_ms is None or gpu_ms in (None, 0):
        return None
    return float(cpu_ms) / float(gpu_ms)


def fmt_speed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def run_one(
    fixture: Path,
    engine: str,
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
    cmd = [
        str(bench_cli),
        "--cases",
        str(fixture),
        "--engine",
        engine,
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
    console_path = outdir / f"one_gen_{engine}.console.log"
    console_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{engine} benchmark failed for {fixture.name}; see {console_path}")
    return parse_bench(proc.stdout)


def write_fixture_report(outdir: Path, cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    report = {
        "benchmark_type": "fixed_population_compare_v3",
        "population_json": str(outdir / "fixed_population.seeds.json"),
        "cpu": cpu,
        "gpu": gpu,
        "speedup": {
            "compile_cpu_over_gpu": speedup(cpu.get("compile_ms"), gpu.get("compile_ms")),
            "eval_cpu_over_gpu": speedup(cpu.get("eval_ms"), gpu.get("eval_ms")),
            "repro_cpu_over_gpu": speedup(cpu.get("repro_ms"), gpu.get("repro_ms")),
            "selection_cpu_over_gpu": speedup(cpu.get("selection_ms"), gpu.get("selection_ms")),
            "crossover_cpu_over_gpu": speedup(cpu.get("crossover_ms"), gpu.get("crossover_ms")),
            "mutation_cpu_over_gpu": speedup(cpu.get("mutation_ms"), gpu.get("mutation_ms")),
            "total_cpu_over_gpu": speedup(cpu.get("total_ms"), gpu.get("total_ms")),
        },
    }
    (outdir / "cpu_gpu_compare.report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    lines = [
        "# CPU vs GPU Fixed-Population Benchmark",
        "",
        f"- population_json: `{report['population_json']}`",
        f"- benchmark_type: `{report['benchmark_type']}`",
        "",
        "## Speedup",
        "",
    ]
    for key, value in report["speedup"].items():
        lines.append(f"- {key}: {fmt_speed(value)}")
    lines.extend(["", "## CPU", ""])
    for key, value in report["cpu"].items():
        lines.append(f"- {key}: {value if not isinstance(value, float) else fmt_ms(value)}")
    lines.extend(["", "## GPU", ""])
    for key, value in report["gpu"].items():
        lines.append(f"- {key}: {value if not isinstance(value, float) else fmt_ms(value)}")
    (outdir / "cpu_gpu_compare.report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


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

    population_sizes = parse_population_sizes(config, args.population_sizes)
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
        max_expr_depth=int(config["max_expr_depth"]),
        max_stmts_per_block=int(config["max_stmts_per_block"]),
        max_total_nodes=int(config["max_total_nodes"]),
        max_for_k=int(config["max_for_k"]),
        max_call_args=int(config["max_call_args"]),
    )
    for population_size in population_sizes:
        for fixture in fixtures:
            fixture_outdir = outdir / f"{fixture.stem}_pop{population_size}"
            fixture_outdir.mkdir(parents=True, exist_ok=True)
            print(f"[fixture-bench] cpu fixture={fixture.stem} pop={population_size}", flush=True)
            cpu = run_one(fixture, "cpu", fixture_outdir, population_size=population_size, **common)
            print(f"[fixture-bench] gpu fixture={fixture.stem} pop={population_size}", flush=True)
            gpu = run_one(fixture, "gpu", fixture_outdir, population_size=population_size, **common)
            report = write_fixture_report(fixture_outdir, cpu, gpu)
            aggregate.append(
                {
                    "fixture": fixture.stem,
                    "population_size": population_size,
                    "report_json": str(fixture_outdir / "cpu_gpu_compare.report.json"),
                    "speedup": report["speedup"],
                }
            )

    def average(key: str) -> float | None:
        values = [item["speedup"][key] for item in aggregate if item["speedup"][key] is not None]
        if not values:
            return None
        return sum(values) / len(values)

    summary = {
        "benchmark_type": "fixture_speedup_batch_v1",
        "fixtures": aggregate,
        "average_speedup": {
            "compile_cpu_over_gpu": average("compile_cpu_over_gpu"),
            "eval_cpu_over_gpu": average("eval_cpu_over_gpu"),
            "repro_cpu_over_gpu": average("repro_cpu_over_gpu"),
            "selection_cpu_over_gpu": average("selection_cpu_over_gpu"),
            "crossover_cpu_over_gpu": average("crossover_cpu_over_gpu"),
            "mutation_cpu_over_gpu": average("mutation_cpu_over_gpu"),
            "total_cpu_over_gpu": average("total_cpu_over_gpu"),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    lines = ["# Fixture Speedup Summary", "", f"- outdir: `{outdir}`", "", "## Average Speedup", ""]
    for key, value in summary["average_speedup"].items():
        lines.append(f"- {key}: {fmt_speed(value)}")
    lines.extend(["", "## Fixtures", ""])
    for item in aggregate:
        lines.append(
            f"- {item['fixture']} pop={item['population_size']}: total={fmt_speed(item['speedup']['total_cpu_over_gpu'])}, eval={fmt_speed(item['speedup']['eval_cpu_over_gpu'])}"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fixture-bench] summary json: {outdir / 'summary.json'}")
    print(f"[fixture-bench] summary md: {outdir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
