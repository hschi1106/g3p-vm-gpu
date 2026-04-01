#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
EVOLVE_CLI = ROOT / "cpp/build/g3pvm_evolve_cli"

GPU_ENV = {
    **os.environ,
    "G3PVM_CUDA_DEVICE": "1",
}

MODE_CONFIGS = {
    "cpu": {"engine": "cpu", "repro_backend": "cpu", "repro_overlap": "off"},
    "gpu_eval": {"engine": "gpu", "repro_backend": "cpu", "repro_overlap": "off"},
    "gpu_repro": {"engine": "gpu", "repro_backend": "gpu", "repro_overlap": "off"},
    "gpu_repro_overlap": {"engine": "gpu", "repro_backend": "gpu", "repro_overlap": "on"},
}

BENCH_COMMON = {
    "blocksize": 1024,
    "fuel": 20000,
    "mutation_rate": 0.5,
    "mutation_subtree_prob": 0.8,
    "crossover_rate": 0.9,
    "penalty": 1.0,
    "selection_pressure": 3,
    "probe_cases": 32,
    "min_success_rate": 0.10,
    "max_attempts": 2000000,
    "max_expr_depth": 5,
    "max_stmts_per_block": 6,
    "max_total_nodes": 80,
    "max_for_k": 16,
    "max_call_args": 3,
}

EVOLVE_COMMON = {
    "blocksize": 1024,
    "population_size": 1024,
    "generations": 40,
    "fuel": 20000,
    "mutation_rate": 0.5,
    "mutation_subtree_prob": 0.8,
    "crossover_rate": 0.9,
    "penalty": 1.0,
    "selection_pressure": 3,
    "max_expr_depth": 5,
    "max_stmts_per_block": 6,
    "max_total_nodes": 80,
    "max_for_k": 16,
    "max_call_args": 3,
}

EXP1_POP_SIZES = [1024, 2048, 4096, 8192, 16384]
EXP1_NUM_POPULATIONS = 5
EXP1_REPEATS = 3
EXP1_FIXTURE = ROOT / "data/fixtures/bouncing_balls_1024.json"

EXP2_TASKS = [
    ROOT / "data/fixtures/simple_exp_1024.json",
    ROOT / "data/fixtures/solve_boolean_1024.json",
    ROOT / "data/fixtures/middle_character_1024.json",
]
EXP2_SEEDS = list(range(10))

EXP3_FIXTURE = ROOT / "data/fixtures/simple_x_plus_1_1024.json"
EXP3_DEPTH_BUCKET_DIR = ROOT / "data/exp/depth_simple_x_plus_1_1024"
EXP3_NODE_BUCKET_DIR = ROOT / "data/exp/node_simple_x_plus_1_1024"
EXP3_DEPTHS = [5, 7, 9, 11, 13, 15]
EXP3_NODE_COUNTS = [20, 30, 40, 50, 60, 70]
EXP3_CELL_REPEATS = 3
EXP3_STRESS_POP_SIZES = [1024, 2048, 4096, 8192, 16384]
EXP3_EVAL_MODES = ("cpu", "gpu_eval")
EXP3_REPRO_MODES = ("gpu_eval", "gpu_repro", "gpu_repro_overlap")

GIB = 1024**3
CPU_FIXED_POP_VMEM_LIMIT_BYTES = 14 * GIB
FIXED_POP_TIMEOUT_SEC_CPU = 1800
FIXED_POP_TIMEOUT_SEC_GPU = 900


@dataclass
class ExperimentRoots:
    exp1: Path | None
    exp2: Path | None
    exp3: Path | None


@dataclass(frozen=True)
class FixedPopulationBucket:
    label: str
    axis: str
    axis_value: int
    population_json: Path
    metadata_json: Path | None


class CommandFailure(RuntimeError):
    def __init__(
        self,
        *,
        reason: str,
        cmd: list[str],
        console_path: Path,
        returncode: int | None = None,
    ) -> None:
        self.reason = reason
        self.cmd = cmd
        self.console_path = console_path
        self.returncode = returncode
        detail = f"{reason}: {' '.join(cmd)}\nsee {console_path}"
        if returncode is not None:
            detail = f"{detail}\nreturncode={returncode}"
        super().__init__(detail)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    # Native CLIs sometimes emit lowercase nan/inf tokens, which are not valid JSON.
    text = re.sub(r'(?<![A-Za-z0-9_"])-nan(?![A-Za-z0-9_"])', "NaN", text)
    text = re.sub(r'(?<![A-Za-z0-9_"])nan(?![A-Za-z0-9_"])', "NaN", text)
    text = re.sub(r'(?<![A-Za-z0-9_"])-inf(?![A-Za-z0-9_"])', "-Infinity", text)
    text = re.sub(r'(?<![A-Za-z0-9_"])inf(?![A-Za-z0-9_"])', "Infinity", text)
    return json.loads(text)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=float), q))


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = [float(v) for v in values if math.isfinite(float(v))]
    if not arr:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {
        "median": float(statistics.median(arr)),
        "q1": percentile(arr, 25),
        "q3": percentile(arr, 75),
    }


def mean_without_first(values: list[float]) -> float:
    if not values:
        return float("nan")
    trimmed = values[1:] if len(values) > 1 else values
    finite = [float(v) for v in trimmed if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def run_command(
    cmd: list[str],
    *,
    console_path: Path,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
    check: bool = True,
    timeout_sec: int | None = None,
    max_virtual_memory_bytes: int | None = None,
) -> subprocess.CompletedProcess[str]:
    def preexec() -> None:
        if max_virtual_memory_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_virtual_memory_bytes, max_virtual_memory_bytes),
            )

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
            preexec_fn=preexec if max_virtual_memory_bytes is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        console_path.parent.mkdir(parents=True, exist_ok=True)
        console_path.write_text((exc.stdout or "") + (exc.stderr or ""), encoding="utf-8")
        raise CommandFailure(reason="timeout", cmd=cmd, console_path=console_path) from exc

    console_path.parent.mkdir(parents=True, exist_ok=True)
    console_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if check and proc.returncode != 0:
        raise CommandFailure(
            reason="nonzero_exit",
            cmd=cmd,
            console_path=console_path,
            returncode=proc.returncode,
        )
    return proc


def fixed_pop_run_limits(mode_name: str) -> dict[str, int | None]:
    if mode_name == "cpu":
        return {
            "timeout_sec": FIXED_POP_TIMEOUT_SEC_CPU,
            "max_virtual_memory_bytes": CPU_FIXED_POP_VMEM_LIMIT_BYTES,
        }
    return {
        "timeout_sec": FIXED_POP_TIMEOUT_SEC_GPU,
        "max_virtual_memory_bytes": None,
    }


def first_timing_value(run: dict[str, Any], key: str) -> float:
    values = run.get("timing", {}).get(key, [])
    if not isinstance(values, list) or not values:
        return 0.0
    return float(values[0])


def run_to_fixed_pop_metrics(run: dict[str, Any]) -> dict[str, Any]:
    meta = run.get("meta", {})
    meta_timing = meta.get("timing", {})
    history = run.get("history", [])

    engine = str(meta.get("eval_engine", "cpu"))
    repro_backend = str(meta.get("reproduction_backend", "cpu"))
    repro_overlap = "on" if bool(meta.get("repro_overlap", False)) else "off"

    compile_ms = first_timing_value(run, "generation_gpu_compile_ms" if engine == "gpu" else "generation_cpu_compile_ms")
    generation_eval_ms = first_timing_value(run, "generation_eval_ms")
    gpu_eval_init_ms = float(meta_timing.get("gpu_eval_init_ms", 0.0))
    gpu_eval_call_ms = first_timing_value(run, "generation_gpu_eval_call_ms")
    total_ms = first_timing_value(run, "generation_total_ms") + (gpu_eval_init_ms if engine == "gpu" else 0.0)
    repro_ms = first_timing_value(run, "generation_repro_ms")
    repro_setup_ms = first_timing_value(run, "generation_repro_setup_ms")

    eval_ms = max(0.0, generation_eval_ms - compile_ms)
    if engine == "gpu":
        eval_ms = gpu_eval_init_ms + gpu_eval_call_ms

    if engine == "cpu":
        warm_total_proxy_ms = total_ms
    elif repro_backend == "cpu":
        warm_total_proxy_ms = total_ms - gpu_eval_init_ms
    else:
        warm_total_proxy_ms = total_ms - gpu_eval_init_ms - repro_setup_ms

    raw: dict[str, Any] = {
        "engine": engine,
        "repro_backend": repro_backend,
        "repro_overlap": repro_overlap,
        "population_size": int(meta.get("population_size", 0)),
        "population_source": meta.get("population_source", "generated"),
        "population_json": meta.get("population_json"),
        "compile_ms": compile_ms,
        "eval_ms": eval_ms,
        "repro_ms": repro_ms,
        "total_ms": total_ms,
        "steady_eval_ms": eval_ms if engine == "cpu" else gpu_eval_call_ms,
        "warm_total_proxy_ms": warm_total_proxy_ms,
        "selection_ms": first_timing_value(run, "generation_selection_ms"),
        "crossover_ms": first_timing_value(run, "generation_crossover_ms"),
        "mutation_ms": first_timing_value(run, "generation_mutation_ms"),
        "gpu_eval_init_ms": gpu_eval_init_ms,
        "gpu_eval_call_ms": gpu_eval_call_ms,
        "gpu_eval_pack_ms": first_timing_value(run, "generation_gpu_eval_pack_ms"),
        "gpu_eval_launch_prep_ms": first_timing_value(run, "generation_gpu_eval_launch_prep_ms"),
        "gpu_eval_upload_ms": first_timing_value(run, "generation_gpu_eval_upload_ms"),
        "gpu_eval_pack_upload_ms": first_timing_value(run, "generation_gpu_eval_pack_upload_ms"),
        "gpu_eval_kernel_ms": first_timing_value(run, "generation_gpu_eval_kernel_ms"),
        "gpu_eval_copyback_ms": first_timing_value(run, "generation_gpu_eval_copyback_ms"),
        "gpu_eval_teardown_ms": first_timing_value(run, "generation_gpu_eval_teardown_ms"),
        "repro_prepare_inputs_ms": first_timing_value(run, "generation_repro_prepare_inputs_ms"),
        "repro_setup_ms": repro_setup_ms,
        "repro_preprocess_ms": first_timing_value(run, "generation_repro_preprocess_ms"),
        "repro_pack_ms": first_timing_value(run, "generation_repro_pack_ms"),
        "repro_upload_ms": first_timing_value(run, "generation_repro_upload_ms"),
        "repro_kernel_ms": first_timing_value(run, "generation_repro_kernel_ms"),
        "repro_copyback_ms": first_timing_value(run, "generation_repro_copyback_ms"),
        "repro_decode_ms": first_timing_value(run, "generation_repro_decode_ms"),
        "repro_teardown_ms": first_timing_value(run, "generation_repro_teardown_ms"),
        "repro_selection_kernel_ms": first_timing_value(run, "generation_repro_selection_kernel_ms"),
        "repro_variation_kernel_ms": first_timing_value(run, "generation_repro_variation_kernel_ms"),
        "best_fitness": safe_float(history[0]["best_fitness"]) if history else float("nan"),
        "mean_fitness": safe_float(history[0]["mean_fitness"]) if history else float("nan"),
        "best_program_key": history[0]["program_key"] if history else "",
    }
    if repro_backend == "gpu" and repro_overlap == "on":
        raw["hidden_overlap_ms"] = max(
            0.0,
            raw["repro_prepare_inputs_ms"]
            + raw["repro_setup_ms"]
            + raw["repro_preprocess_ms"]
            + raw["repro_pack_ms"]
            + raw["repro_upload_ms"]
            + raw["repro_kernel_ms"]
            + raw["repro_copyback_ms"]
            + raw["repro_decode_ms"]
            + raw["repro_teardown_ms"]
            - raw["repro_ms"],
        )
    else:
        raw["hidden_overlap_ms"] = 0.0
    return raw


def evolve_fixed_pop_cmd(
    fixture: Path,
    mode_name: str,
    *,
    population_json: Path,
    out_json: Path,
    seed: int = 0,
) -> list[str]:
    mode = MODE_CONFIGS[mode_name]
    cmd = [
        str(EVOLVE_CLI),
        "--cases",
        str(fixture),
        "--engine",
        mode["engine"],
        "--repro-backend",
        mode["repro_backend"],
        "--repro-overlap",
        mode["repro_overlap"],
        "--blocksize",
        str(BENCH_COMMON["blocksize"]),
        "--generations",
        "1",
        "--fuel",
        str(BENCH_COMMON["fuel"]),
        "--mutation-rate",
        str(BENCH_COMMON["mutation_rate"]),
        "--mutation-subtree-prob",
        str(BENCH_COMMON["mutation_subtree_prob"]),
        "--penalty",
        str(BENCH_COMMON["penalty"]),
        "--selection-pressure",
        str(BENCH_COMMON["selection_pressure"]),
        "--seed",
        str(seed),
        "--population-json",
        str(population_json),
        "--skip-final-eval",
        "on",
        "--timing",
        "all",
        "--show-program",
        "none",
        "--out-json",
        str(out_json),
    ]
    return cmd


def evolve_cmd(fixture: Path, mode_name: str, *, seed: int, out_json: Path) -> list[str]:
    mode = MODE_CONFIGS[mode_name]
    return [
        str(EVOLVE_CLI),
        "--cases",
        str(fixture),
        "--engine",
        mode["engine"],
        "--repro-backend",
        mode["repro_backend"],
        "--repro-overlap",
        mode["repro_overlap"],
        "--blocksize",
        str(EVOLVE_COMMON["blocksize"]),
        "--population-size",
        str(EVOLVE_COMMON["population_size"]),
        "--generations",
        str(EVOLVE_COMMON["generations"]),
        "--fuel",
        str(EVOLVE_COMMON["fuel"]),
        "--mutation-rate",
        str(EVOLVE_COMMON["mutation_rate"]),
        "--mutation-subtree-prob",
        str(EVOLVE_COMMON["mutation_subtree_prob"]),
        "--penalty",
        str(EVOLVE_COMMON["penalty"]),
        "--selection-pressure",
        str(EVOLVE_COMMON["selection_pressure"]),
        "--max-expr-depth",
        str(EVOLVE_COMMON["max_expr_depth"]),
        "--max-stmts-per-block",
        str(EVOLVE_COMMON["max_stmts_per_block"]),
        "--max-total-nodes",
        str(EVOLVE_COMMON["max_total_nodes"]),
        "--max-for-k",
        str(EVOLVE_COMMON["max_for_k"]),
        "--max-call-args",
        str(EVOLVE_COMMON["max_call_args"]),
        "--seed",
        str(seed),
        "--timing",
        "all",
        "--show-program",
        "none",
        "--out-json",
        str(out_json),
    ]


def create_roots(selected: set[str]) -> ExperimentRoots:
    ts = now_timestamp()
    roots = ExperimentRoots(
        exp1=(LOGS / f"exp1_fixed_pop_adjusted_speedup_{ts}") if "1" in selected else None,
        exp2=(LOGS / f"exp2_closed_loop_effectiveness_{ts}") if "2" in selected else None,
        exp3=(LOGS / f"exp3_gpu_resource_boundary_{ts}") if "3" in selected else None,
    )
    for root in (roots.exp1, roots.exp2, roots.exp3):
        if root is None:
            continue
        mkdir(root / "raw")
        mkdir(root / "reports")
        mkdir(root / "plots")
        mkdir(root / "populations")
    return roots


def parse_selected_experiments(raw: str) -> set[str]:
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    if not selected:
        raise SystemExit("no experiments selected")
    invalid = selected - {"1", "2", "3"}
    if invalid:
        raise SystemExit(f"invalid experiment ids: {sorted(invalid)}")
    return selected


def latest_experiment_root(prefix: str) -> Path | None:
    candidates = sorted(path for path in LOGS.glob(f"{prefix}_*") if path.is_dir())
    return candidates[-1] if candidates else None


def load_population_seed_set(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("format_version") != "population-seeds-v1":
        raise RuntimeError(f"unexpected population seed set format: {path}")
    return payload


def load_metadata_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = load_json(path)
    if payload.get("format_version") != "population-metadata-v1":
        return payload
    return payload


def discover_exp3_buckets() -> tuple[list[FixedPopulationBucket], list[FixedPopulationBucket]]:
    depth_buckets: list[FixedPopulationBucket] = []
    for depth in EXP3_DEPTHS:
        population_json = EXP3_DEPTH_BUCKET_DIR / f"depth{depth}.population.json"
        metadata_json = EXP3_DEPTH_BUCKET_DIR / f"depth{depth}.metadata.json"
        if population_json.exists():
            depth_buckets.append(
                FixedPopulationBucket(
                    label=f"depth{depth}",
                    axis="depth",
                    axis_value=depth,
                    population_json=population_json,
                    metadata_json=metadata_json if metadata_json.exists() else None,
                )
            )

    node_buckets: list[FixedPopulationBucket] = []
    for node_count in EXP3_NODE_COUNTS:
        population_json = EXP3_NODE_BUCKET_DIR / f"node{node_count}.population.json"
        metadata_json = EXP3_NODE_BUCKET_DIR / f"node{node_count}.metadata.json"
        if population_json.exists():
            node_buckets.append(
                FixedPopulationBucket(
                    label=f"node{node_count}",
                    axis="node_count",
                    axis_value=node_count,
                    population_json=population_json,
                    metadata_json=metadata_json if metadata_json.exists() else None,
                )
            )
    if not depth_buckets and not node_buckets:
        raise RuntimeError("no Experiment 3 fixed-pop buckets found under data/exp/")
    return depth_buckets, node_buckets


def metadata_summary(metadata_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata_payload or "programs" not in metadata_payload:
        return {}
    programs = metadata_payload["programs"]
    if not programs:
        return {}
    payload_counts: dict[str, int] = {}
    depths = []
    node_counts = []
    code_lens = []
    for row in programs:
        payload = str(row.get("payload_flavor", "unknown"))
        payload_counts[payload] = payload_counts.get(payload, 0) + 1
        if "actual_depth" in row:
            depths.append(float(row["actual_depth"]))
        if "node_count" in row:
            node_counts.append(float(row["node_count"]))
        if "code_len" in row:
            code_lens.append(float(row["code_len"]))
    out: dict[str, Any] = {"payload_counts": payload_counts}
    if depths:
        out["actual_depth"] = summary_stats(depths)
    if node_counts:
        out["node_count"] = summary_stats(node_counts)
    if code_lens:
        out["code_len"] = summary_stats(code_lens)
    return out


def median_numeric_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    numeric_keys = {
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, (int, float))
    }
    for key in numeric_keys:
        out[key] = float(statistics.median([float(record.get(key, 0.0)) for record in records]))
    for key, value in records[0].items():
        if key not in out:
            out[key] = value
    return out


def run_fixed_pop_repeats(
    *,
    fixture: Path,
    mode_name: str,
    population_json: Path,
    out_dir: Path,
    repeats: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mkdir(out_dir)
    limits = fixed_pop_run_limits(mode_name)
    for repeat_index in range(repeats):
        run_json = out_dir / f"repeat_{repeat_index}.run.json"
        metrics_json = out_dir / f"repeat_{repeat_index}.metrics.json"
        console_log = out_dir / f"repeat_{repeat_index}.console.log"
        if metrics_json.exists():
            metrics = load_json(metrics_json)
        else:
            try:
                cmd = evolve_fixed_pop_cmd(
                    fixture,
                    mode_name,
                    population_json=population_json,
                    out_json=run_json,
                    seed=seed,
                )
                run_command(
                    cmd,
                    console_path=console_log,
                    env=GPU_ENV,
                    timeout_sec=limits["timeout_sec"],
                    max_virtual_memory_bytes=limits["max_virtual_memory_bytes"],
                )
                metrics = run_to_fixed_pop_metrics(load_json(run_json))
                metrics["success"] = True
            except Exception as exc:  # noqa: BLE001
                metrics = failed_fixed_pop_record(
                    mode_name=mode_name,
                    population_json=population_json,
                    error=exc,
                )
            write_json(metrics_json, metrics)
        results.append(metrics)
    return results


def write_tiled_population_seed_set(
    *,
    source_population_json: Path,
    out_population_json: Path,
    population_size: int,
    source_label: str,
) -> Path:
    payload = load_population_seed_set(source_population_json)
    seeds = payload["seeds"]
    if not seeds:
        raise RuntimeError(f"source population has no seeds: {source_population_json}")
    tiled = [dict(seeds[i % len(seeds)]) for i in range(population_size)]
    out_payload = dict(payload)
    out_payload["population_size"] = population_size
    out_payload["attempts"] = population_size
    out_payload["source_population_json"] = str(source_population_json.relative_to(ROOT))
    out_payload["source_label"] = source_label
    out_payload["seeds"] = tiled
    write_json(out_population_json, out_payload)
    return out_population_json


def ensure_fixed_population(exp_root: Path, population_size: int, population_index: int) -> Path:
    pop_dir = mkdir(exp_root / "populations" / f"pop{population_size}")
    pop_json = pop_dir / f"population_{population_index}.seeds.json"
    console = exp_root / "raw" / f"pop{population_size}" / f"population_{population_index}" / "generate.console.log"
    if pop_json.exists():
        return pop_json
    seed_start = population_size * 1_000_000 + population_index * 100_000
    payload = {
        "format_version": "population-seeds-v1",
        "cases_path": str(EXP1_FIXTURE),
        "population_size": population_size,
        "probe_cases": 0,
        "min_success_rate": 0.0,
        "fuel": BENCH_COMMON["fuel"],
        "attempts": population_size,
        "limits": {
            "max_expr_depth": BENCH_COMMON["max_expr_depth"],
            "max_stmts_per_block": BENCH_COMMON["max_stmts_per_block"],
            "max_total_nodes": BENCH_COMMON["max_total_nodes"],
            "max_for_k": BENCH_COMMON["max_for_k"],
            "max_call_args": BENCH_COMMON["max_call_args"],
        },
        "seeds": [{"seed": seed_start + i} for i in range(population_size)],
    }
    write_json(pop_json, payload)
    console.parent.mkdir(parents=True, exist_ok=True)
    console.write_text(
        f"synthesized population-seeds-v1 with seed_start={seed_start} population_size={population_size}\n",
        encoding="utf-8",
    )
    return pop_json


def plot_with_iqr(
    x_values: list[int],
    series: dict[str, dict[str, list[float]]],
    *,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    for mode_name, mode_values in series.items():
        y = [mode_values[str(x)]["median"] for x in x_values]
        y1 = [mode_values[str(x)]["q1"] for x in x_values]
        y3 = [mode_values[str(x)]["q3"] for x in x_values]
        plt.plot(x_values, y, marker="o", label=mode_name)
        plt.fill_between(x_values, y1, y3, alpha=0.2)
    plt.xlabel("population_size")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


FIXED_POP_METRIC_KEYS = (
    "compile_ms",
    "eval_ms",
    "steady_eval_ms",
    "total_ms",
    "warm_total_proxy_ms",
    "gpu_eval_init_ms",
    "gpu_eval_call_ms",
    "gpu_eval_pack_upload_ms",
    "gpu_eval_kernel_ms",
    "gpu_eval_teardown_ms",
    "repro_ms",
    "repro_prepare_inputs_ms",
    "repro_setup_ms",
    "repro_preprocess_ms",
    "repro_pack_ms",
    "repro_upload_ms",
    "repro_kernel_ms",
    "repro_copyback_ms",
    "repro_decode_ms",
    "repro_selection_kernel_ms",
    "repro_variation_kernel_ms",
    "hidden_overlap_ms",
)


def failed_fixed_pop_record(
    *,
    mode_name: str,
    population_json: Path,
    error: Exception,
) -> dict[str, Any]:
    error_type = type(error).__name__
    failure_phase = "process_exception"
    returncode: int | None = None
    if isinstance(error, CommandFailure):
        failure_phase = error.reason
        returncode = error.returncode
    return {
        "success": False,
        "mode": mode_name,
        "population_json": str(population_json.relative_to(ROOT)),
        "error_type": error_type,
        "failure_phase": failure_phase,
        "returncode": returncode,
        "error": str(error),
    }


def successful_fixed_pop_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if bool(record.get("success", True))]


def summarize_fixed_pop_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = successful_fixed_pop_records(records)
    summary: dict[str, Any] = {
        "all_success": len(successful) == len(records),
        "num_success": len(successful),
        "num_failures": len(records) - len(successful),
        "failures": [
            {
                "failure_phase": record.get("failure_phase"),
                "error_type": record.get("error_type"),
                "returncode": record.get("returncode"),
                "error": record.get("error"),
            }
            for record in records
            if not bool(record.get("success", True))
        ],
    }
    for key in FIXED_POP_METRIC_KEYS:
        summary[key] = summary_stats([float(record.get(key, 0.0)) for record in successful])
    return summary


def run_experiment_1(exp_root: Path) -> dict[str, Any]:
    raw_runs: dict[str, Any] = {}
    population_medians: dict[str, Any] = {}
    size_summary: dict[str, Any] = {}

    for population_size in EXP1_POP_SIZES:
        raw_runs[str(population_size)] = {}
        population_medians[str(population_size)] = {}
        for population_index in range(EXP1_NUM_POPULATIONS):
            pop_json = ensure_fixed_population(exp_root, population_size, population_index)
            pop_key = f"population_{population_index}"
            raw_runs[str(population_size)][pop_key] = {}
            population_medians[str(population_size)][pop_key] = {}
            for mode_name in MODE_CONFIGS:
                mode_dir = exp_root / "raw" / f"pop{population_size}" / pop_key / mode_name
                mode_runs = run_fixed_pop_repeats(
                    fixture=EXP1_FIXTURE,
                    mode_name=mode_name,
                    population_json=pop_json,
                    out_dir=mode_dir,
                    repeats=EXP1_REPEATS,
                )
                raw_runs[str(population_size)][pop_key][mode_name] = mode_runs

                median_bench: dict[str, Any] = {
                    "all_success": all(bool(run.get("success", True)) for run in mode_runs),
                    "num_success": len(successful_fixed_pop_records(mode_runs)),
                    "num_failures": len(mode_runs) - len(successful_fixed_pop_records(mode_runs)),
                }
                successful_runs = successful_fixed_pop_records(mode_runs)
                numeric_keys = {
                    key
                    for run in successful_runs
                    for key, value in run.items()
                    if isinstance(value, (int, float))
                }
                for key in numeric_keys:
                    median_bench[key] = float(statistics.median([float(run.get(key, 0.0)) for run in successful_runs]))
                exemplar = successful_runs[0] if successful_runs else mode_runs[0]
                for key, value in exemplar.items():
                    if key not in median_bench and key != "success":
                        median_bench[key] = value
                median_bench["success"] = bool(successful_runs)
                population_medians[str(population_size)][pop_key][mode_name] = median_bench

        size_summary[str(population_size)] = {}
        for mode_name in MODE_CONFIGS:
            medians = [
                population_medians[str(population_size)][f"population_{i}"][mode_name]
                for i in range(EXP1_NUM_POPULATIONS)
                if bool(population_medians[str(population_size)][f"population_{i}"][mode_name].get("success"))
            ]
            metric_summary: dict[str, Any] = {}
            for key in (
                "compile_ms",
                "total_ms",
                "eval_ms",
                "selection_ms",
                "crossover_ms",
                "mutation_ms",
                "gpu_eval_init_ms",
                "gpu_eval_call_ms",
                "gpu_eval_pack_upload_ms",
                "repro_setup_ms",
                "repro_ms",
                "repro_prepare_inputs_ms",
                "gpu_eval_kernel_ms",
                "repro_decode_ms",
                "repro_kernel_ms",
                "hidden_overlap_ms",
            ):
                values = [float(run.get(key, 0.0)) for run in medians]
                metric_summary[key] = summary_stats(values)
            metric_summary["num_successful_populations"] = len(medians)
            metric_summary["num_failed_populations"] = EXP1_NUM_POPULATIONS - len(medians)
            metric_summary["cold_total_ms"] = summary_stats([float(run.get("total_ms", 0.0)) for run in medians])
            metric_summary["cold_eval_ms"] = summary_stats([float(run.get("eval_ms", 0.0)) for run in medians])
            metric_summary["steady_eval_ms"] = summary_stats(
                [float(run["eval_ms"] if mode_name == "cpu" else run.get("gpu_eval_call_ms", 0.0)) for run in medians]
            )
            adjusted_values: list[float] = []
            for run in medians:
                total = float(run.get("total_ms", 0.0))
                if mode_name == "cpu":
                    adjusted = total
                elif mode_name == "gpu_eval":
                    adjusted = total - float(run.get("gpu_eval_init_ms", 0.0))
                else:
                    adjusted = total - float(run.get("gpu_eval_init_ms", 0.0)) - float(run.get("repro_setup_ms", 0.0))
                adjusted_values.append(adjusted)
            metric_summary["warm_total_proxy_ms"] = summary_stats(adjusted_values)
            size_summary[str(population_size)][mode_name] = metric_summary

        cpu_eval = size_summary[str(population_size)]["cpu"]["steady_eval_ms"]["median"]
        cpu_total = size_summary[str(population_size)]["cpu"]["warm_total_proxy_ms"]["median"]
        cpu_cold_total = size_summary[str(population_size)]["cpu"]["cold_total_ms"]["median"]
        cpu_cold_eval = size_summary[str(population_size)]["cpu"]["cold_eval_ms"]["median"]
        for mode_name, mode_summary in size_summary[str(population_size)].items():
            steady_eval = mode_summary["steady_eval_ms"]["median"]
            warm_total = mode_summary["warm_total_proxy_ms"]["median"]
            cold_total = mode_summary["cold_total_ms"]["median"]
            cold_eval = mode_summary["cold_eval_ms"]["median"]
            mode_summary["cold_eval_speedup_vs_cpu"] = cpu_cold_eval / cold_eval if cold_eval else float("nan")
            mode_summary["steady_eval_speedup_vs_cpu"] = cpu_eval / steady_eval if steady_eval else float("nan")
            mode_summary["cold_total_speedup_vs_cpu"] = cpu_cold_total / cold_total if cold_total else float("nan")
            mode_summary["warm_total_proxy_speedup_vs_cpu"] = cpu_total / warm_total if warm_total else float("nan")

        repro_plot = exp_root / "plots" / f"repro_breakdown_pop{population_size}.png"
        labels = ["gpu_repro", "gpu_repro_overlap"]
        decode_vals = [size_summary[str(population_size)][mode]["repro_decode_ms"]["median"] for mode in labels]
        kernel_vals = [size_summary[str(population_size)][mode]["repro_kernel_ms"]["median"] for mode in labels]
        hidden_vals = [size_summary[str(population_size)][mode]["hidden_overlap_ms"]["median"] for mode in labels]
        x = np.arange(len(labels))
        plt.figure(figsize=(8, 5))
        plt.bar(x, decode_vals, label="repro_decode_ms")
        plt.bar(x, kernel_vals, bottom=decode_vals, label="repro_kernel_ms")
        plt.bar(x, hidden_vals, bottom=np.array(decode_vals) + np.array(kernel_vals), label="hidden_overlap_ms")
        plt.xticks(x, labels)
        plt.ylabel("ms")
        plt.title(f"Reproduction breakdown pop{population_size}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(repro_plot)
        plt.close()

    plot_with_iqr(
        EXP1_POP_SIZES,
        {
            mode_name: {size: size_summary[size][mode_name]["steady_eval_ms"] for size in size_summary}
            for mode_name in MODE_CONFIGS
        },
        y_label="ms",
        title="Steady eval ms vs population size",
        out_path=exp_root / "plots" / "popsize_vs_steady_eval_ms.png",
    )
    plot_with_iqr(
        EXP1_POP_SIZES,
        {
            mode_name: {size: size_summary[size][mode_name]["warm_total_proxy_ms"] for size in size_summary}
            for mode_name in MODE_CONFIGS
        },
        y_label="ms",
        title="Warm total proxy ms vs population size",
        out_path=exp_root / "plots" / "popsize_vs_warm_total_proxy_ms.png",
    )
    plot_with_iqr(
        EXP1_POP_SIZES,
        {
            mode_name: {size: size_summary[size][mode_name]["cold_total_ms"] for size in size_summary}
            for mode_name in MODE_CONFIGS
        },
        y_label="ms",
        title="Cold total ms vs population size",
        out_path=exp_root / "plots" / "popsize_vs_cold_total_ms.png",
    )

    plt.figure(figsize=(9, 5))
    for mode_name in MODE_CONFIGS:
        y = [size_summary[str(x)][mode_name]["steady_eval_speedup_vs_cpu"] for x in EXP1_POP_SIZES]
        plt.plot(EXP1_POP_SIZES, y, marker="o", label=mode_name)
    plt.xlabel("population_size")
    plt.ylabel("speedup vs cpu")
    plt.title("Steady eval speedup vs cpu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_root / "plots" / "popsize_vs_steady_eval_speedup.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    for mode_name in MODE_CONFIGS:
        y = [size_summary[str(x)][mode_name]["warm_total_proxy_speedup_vs_cpu"] for x in EXP1_POP_SIZES]
        plt.plot(EXP1_POP_SIZES, y, marker="o", label=mode_name)
    plt.xlabel("population_size")
    plt.ylabel("speedup vs cpu")
    plt.title("Warm total proxy speedup vs cpu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_root / "plots" / "popsize_vs_warm_total_proxy_speedup.png")
    plt.close()

    def plot_mode_breakdown(mode_name: str, out_path: Path) -> None:
        labels = [str(size) for size in EXP1_POP_SIZES]
        positions = np.arange(len(EXP1_POP_SIZES), dtype=float)
        width = 0.72

        if mode_name == "cpu":
            compile_vals = []
            eval_vals = []
            selection_vals = []
            crossover_vals = []
            mutation_vals = []
            repro_vals = []
            host_other = []

            for population_size in EXP1_POP_SIZES:
                mode = size_summary[str(population_size)][mode_name]
                warm_total = float(mode["warm_total_proxy_ms"]["median"])
                compile_ms = float(mode["compile_ms"]["median"])
                eval_ms = float(mode["eval_ms"]["median"])
                selection_ms = float(mode["selection_ms"]["median"])
                crossover_ms = float(mode["crossover_ms"]["median"])
                mutation_ms = float(mode["mutation_ms"]["median"])
                repro_ms = float(mode["repro_ms"]["median"])

                accounted = compile_ms + eval_ms + selection_ms + crossover_ms + mutation_ms + repro_ms
                compile_vals.append(compile_ms)
                eval_vals.append(eval_ms)
                selection_vals.append(selection_ms)
                crossover_vals.append(crossover_ms)
                mutation_vals.append(mutation_ms)
                repro_vals.append(repro_ms)
                host_other.append(max(0.0, warm_total - accounted))

            plt.figure(figsize=(10, 5))
            bottom = np.zeros(len(EXP1_POP_SIZES), dtype=float)
            stacks = [
                ("compile_ms", compile_vals, "#4e79a7"),
                ("eval_ms", eval_vals, "#59a14f"),
                ("selection_ms", selection_vals, "#f28e2b"),
                ("crossover_ms", crossover_vals, "#e15759"),
                ("mutation_ms", mutation_vals, "#edc948"),
                ("repro_ms", repro_vals, "#b07aa1"),
                ("host_other_ms", host_other, "#bab0ab"),
            ]
            for name, values, color in stacks:
                values_arr = np.array(values, dtype=float)
                plt.bar(positions, values_arr, width=width, bottom=bottom, label=name, color=color)
                bottom = bottom + values_arr

            plt.xticks(positions, labels)
            plt.xlabel("population_size")
            plt.ylabel("warm_total_proxy_ms breakdown")
            plt.title(f"{mode_name} timing breakdown by population size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return

        eval_pack_upload = []
        eval_kernel = []
        eval_other = []
        repro_other = []
        repro_kernel = []
        repro_decode = []
        host_other = []
        hidden_overlap = []

        for population_size in EXP1_POP_SIZES:
            mode = size_summary[str(population_size)][mode_name]
            steady_eval = float(mode["steady_eval_ms"]["median"])
            warm_total = float(mode["warm_total_proxy_ms"]["median"])
            eval_pack = float(mode["gpu_eval_pack_upload_ms"]["median"])
            eval_k = float(mode["gpu_eval_kernel_ms"]["median"])
            repro_total = float(mode["repro_ms"]["median"])
            repro_k = float(mode["repro_kernel_ms"]["median"])
            repro_d = float(mode["repro_decode_ms"]["median"])
            hidden = float(mode["hidden_overlap_ms"]["median"])

            eval_other.append(max(0.0, steady_eval - eval_pack - eval_k))
            eval_pack_upload.append(eval_pack)
            eval_kernel.append(eval_k)
            repro_other.append(max(0.0, repro_total - repro_k - repro_d))
            repro_kernel.append(repro_k)
            repro_decode.append(repro_d)
            host_other.append(max(0.0, warm_total - steady_eval - repro_total))
            hidden_overlap.append(hidden)

        plt.figure(figsize=(10, 5))
        bottom = np.zeros(len(EXP1_POP_SIZES), dtype=float)
        stacks = [
            ("eval_other_ms", eval_other, "#4e79a7"),
            ("gpu_eval_pack_upload_ms", eval_pack_upload, "#59a14f"),
            ("gpu_eval_kernel_ms", eval_kernel, "#1f77b4"),
            ("repro_other_ms", repro_other, "#f28e2b"),
            ("repro_kernel_ms", repro_kernel, "#e15759"),
            ("repro_decode_ms", repro_decode, "#edc948"),
            ("host_other_ms", host_other, "#bab0ab"),
        ]
        for name, values, color in stacks:
            values_arr = np.array(values, dtype=float)
            plt.bar(positions, values_arr, width=width, bottom=bottom, label=name, color=color)
            bottom = bottom + values_arr

        if mode_name == "gpu_repro_overlap":
            plt.plot(positions, hidden_overlap, color="black", marker="o", label="hidden_overlap_ms")

        plt.xticks(positions, labels)
        plt.xlabel("population_size")
        plt.ylabel("warm_total_proxy_ms breakdown")
        plt.title(f"{mode_name} timing breakdown by population size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    for mode_name in MODE_CONFIGS:
        plot_mode_breakdown(mode_name, exp_root / "plots" / f"breakdown_{mode_name}_by_popsize.png")

    summary = {
        "fixture": str(EXP1_FIXTURE.relative_to(ROOT)),
        "population_sizes": EXP1_POP_SIZES,
        "num_fixed_populations": EXP1_NUM_POPULATIONS,
        "repeats": EXP1_REPEATS,
        "raw_runs": raw_runs,
        "population_medians": population_medians,
        "size_summary": size_summary,
    }
    write_json(exp_root / "reports" / "summary.json", summary)

    lines = [
        "# Experiment 1 Summary",
        "",
        f"- fixture: `{EXP1_FIXTURE.relative_to(ROOT)}`",
        f"- population_sizes: `{EXP1_POP_SIZES}`",
        f"- fixed_populations_per_size: `{EXP1_NUM_POPULATIONS}`",
        f"- repeats_per_mode: `{EXP1_REPEATS}`",
        "",
        "## Size Summary",
        "",
    ]
    for population_size in EXP1_POP_SIZES:
        lines.append(f"### pop{population_size}")
        lines.append("")
        for mode_name in MODE_CONFIGS:
            mode = size_summary[str(population_size)][mode_name]
            lines.append(
                f"- {mode_name}: successful_populations={mode['num_successful_populations']}/{EXP1_NUM_POPULATIONS}, "
                f"cold_total_ms={mode['cold_total_ms']['median']:.3f}, "
                f"cold_eval_ms={mode['cold_eval_ms']['median']:.3f}, "
                f"cold_total_speedup_vs_cpu={mode['cold_total_speedup_vs_cpu']:.3f}, "
                f"cold_eval_speedup_vs_cpu={mode['cold_eval_speedup_vs_cpu']:.3f}, "
                f"steady_eval_ms={mode['steady_eval_ms']['median']:.3f}, "
                f"warm_total_proxy_ms={mode['warm_total_proxy_ms']['median']:.3f}, "
                f"steady_eval_speedup_vs_cpu={mode['steady_eval_speedup_vs_cpu']:.3f}, "
                f"warm_total_proxy_speedup_vs_cpu={mode['warm_total_proxy_speedup_vs_cpu']:.3f}"
            )
        lines.append("")
    (exp_root / "reports" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def auc(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    if np.all(np.isfinite(arr)):
        return float(np.trapz(arr, dx=1.0))
    return float("nan")


def interpolate_quality(times_ms: list[float], values: list[float], grid: np.ndarray) -> np.ndarray:
    if not times_ms or not values:
        return np.full_like(grid, np.nan, dtype=float)
    out = np.empty_like(grid, dtype=float)
    idx = 0
    current = values[0]
    for i, t in enumerate(grid):
        while idx + 1 < len(times_ms) and times_ms[idx + 1] <= t:
            idx += 1
            current = values[idx]
        out[i] = current
    return out


def run_experiment_2(exp_root: Path) -> dict[str, Any]:
    raw_runs: dict[str, Any] = {}
    task_summary: dict[str, Any] = {}

    for fixture in EXP2_TASKS:
        task = fixture.stem
        raw_runs[task] = {}
        per_task_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {mode: [] for mode in MODE_CONFIGS}
        for mode_name in MODE_CONFIGS:
            raw_runs[task][mode_name] = {}
            for seed in EXP2_SEEDS:
                run_dir = exp_root / "raw" / task / mode_name
                mkdir(run_dir)
                console = run_dir / f"seed_{seed:03d}.console.log"
                out_json = run_dir / f"seed_{seed:03d}.run.json"
                if not out_json.exists():
                    cmd = evolve_cmd(fixture, mode_name, seed=seed, out_json=out_json)
                    run_command(cmd, console_path=console, env=GPU_ENV)
                payload = load_json(out_json)
                history = payload["history"]
                timing = payload["timing"]
                best_hist = [safe_float(row["best_fitness"]) for row in history]
                mean_hist = [safe_float(row["mean_fitness"]) for row in history]
                generation_total = [float(v) for v in timing["generation_total_ms"]]
                generation_eval = [float(v) for v in timing["generation_eval_ms"]]
                generation_repro = [float(v) for v in timing["generation_repro_ms"]]
                cumulative_time = list(np.cumsum(np.array(generation_total, dtype=float)))
                record = {
                    "seed": seed,
                    "final_best_fitness": float(payload["final"]["best_fitness"]),
                    "delta_best_fitness": float(best_hist[-1] - best_hist[0]),
                    "delta_mean_fitness": float(mean_hist[-1] - mean_hist[0]) if all(math.isfinite(v) for v in (mean_hist[0], mean_hist[-1])) else float("nan"),
                    "auc_best_fitness": auc(best_hist),
                    "auc_mean_fitness": auc(mean_hist),
                    "avg_generation_total_ms_no_first": mean_without_first(generation_total),
                    "avg_generation_eval_ms_no_first": mean_without_first(generation_eval),
                    "avg_generation_repro_ms_no_first": mean_without_first(generation_repro),
                    "best_history": best_hist,
                    "mean_history": mean_hist,
                    "cumulative_time_ms": cumulative_time,
                }
                raw_runs[task][mode_name][str(seed)] = record
                per_task_metrics[mode_name].append(record)

        task_summary[task] = {}
        for mode_name, records in per_task_metrics.items():
            task_summary[task][mode_name] = {
                "final_best_fitness": summary_stats([r["final_best_fitness"] for r in records]),
                "delta_best_fitness": summary_stats([r["delta_best_fitness"] for r in records]),
                "delta_mean_fitness": summary_stats([r["delta_mean_fitness"] for r in records]),
                "auc_best_fitness": summary_stats([r["auc_best_fitness"] for r in records]),
                "auc_mean_fitness": summary_stats([r["auc_mean_fitness"] for r in records]),
                "avg_generation_total_ms_no_first": summary_stats([r["avg_generation_total_ms_no_first"] for r in records]),
                "avg_generation_eval_ms_no_first": summary_stats([r["avg_generation_eval_ms_no_first"] for r in records]),
                "avg_generation_repro_ms_no_first": summary_stats([r["avg_generation_repro_ms_no_first"] for r in records]),
            }

        gens = np.arange(EVOLVE_COMMON["generations"])
        plt.figure(figsize=(9, 5))
        for mode_name, records in per_task_metrics.items():
            curves = np.array([r["best_history"] for r in records], dtype=float)
            med = np.median(curves, axis=0)
            q1 = np.percentile(curves, 25, axis=0)
            q3 = np.percentile(curves, 75, axis=0)
            plt.plot(gens, med, label=mode_name)
            plt.fill_between(gens, q1, q3, alpha=0.2)
        plt.xlabel("generation")
        plt.ylabel("best_fitness")
        plt.title(f"{task} best fitness vs generation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_root / "plots" / f"{task}_best_fitness_vs_generation.png")
        plt.close()

        plt.figure(figsize=(9, 5))
        for mode_name, records in per_task_metrics.items():
            curves = np.array([r["mean_history"] for r in records], dtype=float)
            med = np.nanmedian(curves, axis=0)
            q1 = np.nanpercentile(curves, 25, axis=0)
            q3 = np.nanpercentile(curves, 75, axis=0)
            plt.plot(gens, med, label=mode_name)
            plt.fill_between(gens, q1, q3, alpha=0.2)
        plt.xlabel("generation")
        plt.ylabel("mean_fitness")
        plt.title(f"{task} mean fitness vs generation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_root / "plots" / f"{task}_mean_fitness_vs_generation.png")
        plt.close()

        plt.figure(figsize=(9, 5))
        box_values = [np.array([r["final_best_fitness"] for r in per_task_metrics[mode]], dtype=float) for mode in MODE_CONFIGS]
        plt.boxplot(box_values, labels=list(MODE_CONFIGS.keys()))
        plt.ylabel("final_best_fitness")
        plt.title(f"{task} final best fitness")
        plt.tight_layout()
        plt.savefig(exp_root / "plots" / f"{task}_final_best_fitness_boxplot.png")
        plt.close()

        max_time = min(max(r["cumulative_time_ms"][-1] for r in per_task_metrics[mode]) for mode in MODE_CONFIGS)
        grid = np.linspace(0.0, max_time, num=100)
        plt.figure(figsize=(9, 5))
        for mode_name, records in per_task_metrics.items():
            curves = np.array([interpolate_quality(r["cumulative_time_ms"], r["best_history"], grid) for r in records], dtype=float)
            med = np.nanmedian(curves, axis=0)
            q1 = np.nanpercentile(curves, 25, axis=0)
            q3 = np.nanpercentile(curves, 75, axis=0)
            plt.plot(grid / 1000.0, med, label=mode_name)
            plt.fill_between(grid / 1000.0, q1, q3, alpha=0.2)
        plt.xlabel("cumulative time (s)")
        plt.ylabel("best_fitness")
        plt.title(f"{task} quality vs cumulative time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_root / "plots" / f"{task}_quality_vs_cumulative_time.png")
        plt.close()

    summary = {
        "tasks": [str(task.relative_to(ROOT)) for task in EXP2_TASKS],
        "seeds": EXP2_SEEDS,
        "raw_runs": raw_runs,
        "task_summary": task_summary,
    }
    write_json(exp_root / "reports" / "summary.json", summary)
    lines = [
        "# Experiment 2 Summary",
        "",
        f"- tasks: `{[task.relative_to(ROOT).as_posix() for task in EXP2_TASKS]}`",
        f"- seeds: `{EXP2_SEEDS}`",
        f"- generations: `{EVOLVE_COMMON['generations']}`",
        "",
    ]
    for task, modes in task_summary.items():
        lines.append(f"## {task}")
        lines.append("")
        for mode_name, metrics in modes.items():
            lines.append(
                f"- {mode_name}: final_best={metrics['final_best_fitness']['median']:.3f}, "
                f"delta_best={metrics['delta_best_fitness']['median']:.3f}, "
                f"delta_mean={metrics['delta_mean_fitness']['median']:.3f}, "
                f"auc_best={metrics['auc_best_fitness']['median']:.3f}, "
                f"avg_gen_total_no_first={metrics['avg_generation_total_ms_no_first']['median']:.3f}"
            )
        lines.append("")
    (exp_root / "reports" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_experiment_3(exp_root: Path) -> dict[str, Any]:
    depth_buckets, node_buckets = discover_exp3_buckets()
    all_buckets = depth_buckets + node_buckets
    bucket_results: dict[str, Any] = {}

    for bucket in all_buckets:
        metadata_payload = load_metadata_payload(bucket.metadata_json)
        bucket_entry = {
            "label": bucket.label,
            "axis": bucket.axis,
            "axis_value": bucket.axis_value,
            "population_json": str(bucket.population_json.relative_to(ROOT)),
            "metadata_json": str(bucket.metadata_json.relative_to(ROOT)) if bucket.metadata_json else None,
            "metadata_summary": metadata_summary(metadata_payload),
            "mode_summary": {},
        }
        for mode_name in MODE_CONFIGS:
            repeats = run_fixed_pop_repeats(
                fixture=EXP3_FIXTURE,
                mode_name=mode_name,
                population_json=bucket.population_json,
                out_dir=exp_root / "raw" / bucket.label / mode_name,
                repeats=EXP3_CELL_REPEATS,
            )
            bucket_entry["mode_summary"][mode_name] = summarize_fixed_pop_runs(repeats)

        cpu_eval = bucket_entry["mode_summary"]["cpu"]["steady_eval_ms"]["median"]
        gpu_eval = bucket_entry["mode_summary"]["gpu_eval"]["steady_eval_ms"]["median"]
        bucket_entry["eval_scaling"] = {
            "steady_eval_speedup_vs_cpu": cpu_eval / gpu_eval if gpu_eval else float("nan"),
            "cold_total_speedup_vs_cpu": (
                bucket_entry["mode_summary"]["cpu"]["total_ms"]["median"]
                / bucket_entry["mode_summary"]["gpu_eval"]["total_ms"]["median"]
                if bucket_entry["mode_summary"]["gpu_eval"]["total_ms"]["median"]
                else float("nan")
            ),
        }
        bucket_results[bucket.label] = bucket_entry

    def ordered_bucket_entries(axis: str) -> list[dict[str, Any]]:
        return sorted(
            [entry for entry in bucket_results.values() if entry["axis"] == axis],
            key=lambda item: item["axis_value"],
        )

    depth_entries = ordered_bucket_entries("depth")
    node_entries = ordered_bucket_entries("node_count")

    def plot_eval_scaling(entries: list[dict[str, Any]], *, x_label: str, title: str, out_path: Path) -> None:
        if not entries:
            return
        x = [entry["axis_value"] for entry in entries]
        cpu_eval = [entry["mode_summary"]["cpu"]["steady_eval_ms"]["median"] for entry in entries]
        gpu_modes = [
            ("gpu_eval", "gpu_eval"),
            ("gpu_repro", "gpu_repro"),
            ("gpu_repro_overlap", "gpu_repro_overlap"),
        ]
        positions = np.arange(len(entries), dtype=float)
        width = 0.24
        plt.figure(figsize=(10, 5))
        for offset, (mode_name, label) in zip((-width, 0.0, width), gpu_modes):
            speedups = []
            for entry, cpu_ms in zip(entries, cpu_eval):
                gpu_ms = entry["mode_summary"][mode_name]["steady_eval_ms"]["median"]
                if cpu_ms and math.isfinite(cpu_ms) and gpu_ms and math.isfinite(gpu_ms):
                    speedups.append(cpu_ms / gpu_ms)
                else:
                    speedups.append(float("nan"))
            plt.bar(positions + offset, speedups, width=width, label=f"{label} speedup")
        plt.xticks(positions, [str(value) for value in x])
        plt.xlabel(x_label)
        plt.ylabel("steady_eval_speedup_vs_cpu")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def plot_repro_breakdown(entries: list[dict[str, Any]], *, x_label: str, title: str, out_path: Path) -> None:
        if not entries:
            return
        x = [entry["axis_value"] for entry in entries]
        decode = [entry["mode_summary"]["gpu_repro"]["repro_decode_ms"]["median"] for entry in entries]
        kernel = [entry["mode_summary"]["gpu_repro"]["repro_kernel_ms"]["median"] for entry in entries]
        preprocess = [entry["mode_summary"]["gpu_repro"]["repro_preprocess_ms"]["median"] for entry in entries]
        pack = [entry["mode_summary"]["gpu_repro"]["repro_pack_ms"]["median"] for entry in entries]
        hidden = [entry["mode_summary"]["gpu_repro_overlap"]["hidden_overlap_ms"]["median"] for entry in entries]
        positions = np.arange(len(entries), dtype=float)
        width = 0.6
        plt.figure(figsize=(10, 5))
        plt.bar(positions, preprocess, width=width, label="gpu_repro preprocess")
        plt.bar(positions, pack, width=width, bottom=preprocess, label="gpu_repro pack")
        plt.bar(
            positions,
            kernel,
            width=width,
            bottom=np.array(preprocess) + np.array(pack),
            label="gpu_repro kernel",
        )
        plt.bar(
            positions,
            decode,
            width=width,
            bottom=np.array(preprocess) + np.array(pack) + np.array(kernel),
            label="gpu_repro decode",
        )
        plt.plot(positions, hidden, color="black", marker="o", label="gpu_repro_overlap hidden_overlap_ms")
        plt.xticks(positions, [str(value) for value in x])
        plt.xlabel(x_label)
        plt.ylabel("ms")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    plot_eval_scaling(
        depth_entries,
        x_label="depth",
        title="Steady eval speedup vs depth",
        out_path=exp_root / "plots" / "eval_scaling_vs_depth.png",
    )
    plot_eval_scaling(
        node_entries,
        x_label="node_count",
        title="Steady eval speedup vs node count",
        out_path=exp_root / "plots" / "eval_scaling_vs_nodes.png",
    )
    plot_repro_breakdown(
        depth_entries,
        x_label="depth",
        title="Reproduction breakdown vs depth",
        out_path=exp_root / "plots" / "repro_breakdown_vs_depth.png",
    )
    plot_repro_breakdown(
        node_entries,
        x_label="node_count",
        title="Reproduction breakdown vs node count",
        out_path=exp_root / "plots" / "repro_breakdown_vs_nodes.png",
    )

    representative_candidates = [
        entry
        for entry in bucket_results.values()
        if math.isfinite(entry["mode_summary"]["gpu_eval"]["gpu_eval_kernel_ms"]["median"])
    ]
    if not representative_candidates:
        raise RuntimeError("no successful gpu_eval bucket available for Experiment 3 stress selection")
    representative_bucket = max(
        representative_candidates,
        key=lambda entry: entry["mode_summary"]["gpu_eval"]["gpu_eval_kernel_ms"]["median"],
    )
    representative_population_json = ROOT / representative_bucket["population_json"]

    stress_results: dict[str, Any] = {}
    first_failing_point: int | None = None
    first_failing_phase: str | None = None
    last_stable_configuration: int | None = None

    for population_size in EXP3_STRESS_POP_SIZES:
        stress_label = f"{representative_bucket['label']}_pop{population_size}"
        pop_dir = mkdir(exp_root / "populations" / stress_label)
        pop_json = pop_dir / "population.seeds.json"
        meta_json = pop_dir / "metadata.json"
        if not pop_json.exists():
            write_tiled_population_seed_set(
                source_population_json=representative_population_json,
                out_population_json=pop_json,
                population_size=population_size,
                source_label=representative_bucket["label"],
            )
        if not meta_json.exists():
            write_json(
                meta_json,
                {
                    "format_version": "stress-population-metadata-v1",
                    "source_population_json": representative_bucket["population_json"],
                    "source_bucket_label": representative_bucket["label"],
                    "population_size": population_size,
                    "strategy": "seed-tiling",
                },
            )

        stress_entry = {
            "population_json": str(pop_json.relative_to(ROOT)),
            "metadata_json": str(meta_json.relative_to(ROOT)),
            "modes": {},
        }
        all_success = True
        for mode_name in EXP3_REPRO_MODES:
            mode_dir = exp_root / "raw" / stress_label / mode_name
            repeats = run_fixed_pop_repeats(
                fixture=EXP3_FIXTURE,
                mode_name=mode_name,
                population_json=pop_json,
                out_dir=mode_dir,
                repeats=EXP3_CELL_REPEATS,
            )
            summary = summarize_fixed_pop_runs(repeats)
            mode_success = summary["num_success"] > 0 and summary["num_failures"] == 0
            stress_entry["modes"][mode_name] = {
                "success": mode_success,
                "summary": summary,
            }
            if not mode_success:
                all_success = False
                first_failure = summary["failures"][0] if summary["failures"] else {"failure_phase": "unknown"}
                stress_entry["modes"][mode_name]["first_failing_phase"] = first_failure["failure_phase"]
                if first_failing_point is None:
                    first_failing_point = population_size
                    first_failing_phase = f"{mode_name}: {first_failure['failure_phase']}"
        stress_entry["all_success"] = all_success
        stress_results[str(population_size)] = stress_entry
        if all_success:
            last_stable_configuration = population_size

    def detect_performance_cliff(mode_name: str) -> int | None:
        previous_size: int | None = None
        previous_ms: float | None = None
        for population_size in EXP3_STRESS_POP_SIZES:
            mode_data = stress_results[str(population_size)]["modes"].get(mode_name, {})
            if not mode_data.get("success"):
                return None
            current_ms = float(mode_data["summary"]["warm_total_proxy_ms"]["median"])
            if previous_size is not None and previous_ms not in (None, 0.0):
                expected_ratio = population_size / previous_size
                actual_ratio = current_ms / previous_ms if previous_ms else float("inf")
                if actual_ratio > expected_ratio * 1.5:
                    return population_size
            previous_size = population_size
            previous_ms = current_ms
        return None

    stress_cliffs = {
        mode_name: detect_performance_cliff(mode_name)
        for mode_name in EXP3_REPRO_MODES
    }
    if first_failing_phase is None:
        for mode_name, cliff_at in stress_cliffs.items():
            if cliff_at is not None:
                first_failing_phase = f"{mode_name}: performance_cliff"
                if first_failing_point is None:
                    first_failing_point = cliff_at
                break

    stable_sizes = [
        population_size
        for population_size in EXP3_STRESS_POP_SIZES
        if stress_results[str(population_size)]["modes"]["gpu_eval"].get("success")
    ]
    if stable_sizes:
        curve = [
            stress_results[str(size)]["modes"]["gpu_eval"]["summary"]["gpu_eval_kernel_ms"]["median"]
            for size in stable_sizes
        ]
        plt.figure(figsize=(8, 5))
        plt.plot(stable_sizes, curve, marker="o")
        plt.xlabel("population_size")
        plt.ylabel("gpu_eval_kernel_ms")
        plt.title("Population size stress curve")
        plt.tight_layout()
        plt.savefig(exp_root / "plots" / "population_size_stress_curve.png")
        plt.close()

    heatmap_modes = list(EXP3_REPRO_MODES)
    heatmap = np.zeros((len(heatmap_modes), len(EXP3_STRESS_POP_SIZES)), dtype=float)
    for i, mode_name in enumerate(heatmap_modes):
        for j, population_size in enumerate(EXP3_STRESS_POP_SIZES):
            heatmap[i, j] = 1.0 if stress_results[str(population_size)]["modes"][mode_name].get("success") else 0.0
    plt.figure(figsize=(8, 4))
    plt.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(range(len(EXP3_STRESS_POP_SIZES)), [str(x) for x in EXP3_STRESS_POP_SIZES])
    plt.yticks(range(len(heatmap_modes)), heatmap_modes)
    plt.xlabel("population_size")
    plt.ylabel("mode")
    plt.title("Failure boundary heatmap")
    plt.tight_layout()
    plt.savefig(exp_root / "plots" / "failure_boundary_heatmap.png")
    plt.close()

    scaling_summary = {
        "fixture": str(EXP3_FIXTURE.relative_to(ROOT)),
        "depth_buckets": {entry["label"]: entry for entry in depth_entries},
        "node_buckets": {entry["label"]: entry for entry in node_entries},
    }
    boundary_summary = {
        "fixture": str(EXP3_FIXTURE.relative_to(ROOT)),
        "representative_bucket": representative_bucket["label"],
        "representative_bucket_population_json": representative_bucket["population_json"],
        "stress_results": stress_results,
        "performance_cliffs": stress_cliffs,
        "last_stable_configuration": last_stable_configuration,
        "first_failing_point": first_failing_point,
        "first_failing_phase": first_failing_phase,
    }
    write_json(exp_root / "reports" / "scaling_summary.json", scaling_summary)
    write_json(exp_root / "reports" / "boundary_summary.json", boundary_summary)

    scaling_lines = [
        "# Experiment 3 Scaling Summary",
        "",
        f"- fixture: `{EXP3_FIXTURE.relative_to(ROOT)}`",
        f"- depth_buckets: `{[entry['label'] for entry in depth_entries]}`",
        f"- node_buckets: `{[entry['label'] for entry in node_entries]}`",
        "",
        "## Depth Buckets",
        "",
    ]
    for entry in depth_entries:
        scaling_lines.append(
            f"- {entry['label']}: cpu_success={entry['mode_summary']['cpu']['num_success']}/{EXP3_CELL_REPEATS}, "
            f"gpu_success={entry['mode_summary']['gpu_eval']['num_success']}/{EXP3_CELL_REPEATS}, "
            f"cpu_eval_ms={entry['mode_summary']['cpu']['eval_ms']['median']:.3f}, "
            f"gpu_steady_eval_ms={entry['mode_summary']['gpu_eval']['steady_eval_ms']['median']:.3f}, "
            f"gpu_eval_kernel_ms={entry['mode_summary']['gpu_eval']['gpu_eval_kernel_ms']['median']:.3f}, "
            f"steady_eval_speedup_vs_cpu={entry['eval_scaling']['steady_eval_speedup_vs_cpu']:.3f}"
        )
    scaling_lines.extend(["", "## Node Buckets", ""])
    for entry in node_entries:
        scaling_lines.append(
            f"- {entry['label']}: cpu_success={entry['mode_summary']['cpu']['num_success']}/{EXP3_CELL_REPEATS}, "
            f"gpu_success={entry['mode_summary']['gpu_eval']['num_success']}/{EXP3_CELL_REPEATS}, "
            f"cpu_eval_ms={entry['mode_summary']['cpu']['eval_ms']['median']:.3f}, "
            f"gpu_steady_eval_ms={entry['mode_summary']['gpu_eval']['steady_eval_ms']['median']:.3f}, "
            f"gpu_eval_kernel_ms={entry['mode_summary']['gpu_eval']['gpu_eval_kernel_ms']['median']:.3f}, "
            f"steady_eval_speedup_vs_cpu={entry['eval_scaling']['steady_eval_speedup_vs_cpu']:.3f}"
        )
    (exp_root / "reports" / "scaling_summary.md").write_text("\n".join(scaling_lines) + "\n", encoding="utf-8")

    boundary_lines = [
        "# Experiment 3 Boundary Summary",
        "",
        f"- representative_bucket: `{representative_bucket['label']}`",
        f"- last_stable_configuration: `{last_stable_configuration}`",
        f"- first_failing_point: `{first_failing_point}`",
        f"- first_failing_phase: `{first_failing_phase}`",
        "",
        "## Stress Results",
        "",
    ]
    for population_size in EXP3_STRESS_POP_SIZES:
        row = stress_results[str(population_size)]
        parts = []
        for mode_name in EXP3_REPRO_MODES:
            mode_data = row["modes"][mode_name]
            if mode_data.get("success"):
                parts.append(
                    f"{mode_name}: warm_total_proxy_ms={mode_data['summary']['warm_total_proxy_ms']['median']:.3f}, "
                    f"gpu_eval_kernel_ms={mode_data['summary']['gpu_eval_kernel_ms']['median']:.3f}"
                )
            else:
                parts.append(
                    f"{mode_name}: FAIL ({mode_data.get('first_failing_phase', 'unknown')}, "
                    f"success_repeats={mode_data['summary']['num_success']}/{EXP3_CELL_REPEATS})"
                )
        boundary_lines.append(f"- pop{population_size}: " + "; ".join(parts))
    (exp_root / "reports" / "boundary_summary.md").write_text("\n".join(boundary_lines) + "\n", encoding="utf-8")

    return {
        "scaling_summary": scaling_summary,
        "boundary_summary": boundary_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="1,2,3", help="comma-separated subset of 1,2,3")
    parser.add_argument("--reuse-exp2", default="", help="existing exp2 root to reuse when experiment 2 is not selected")
    args = parser.parse_args()

    selected = parse_selected_experiments(args.experiments)
    roots = create_roots(selected)

    reused_exp2_root: Path | None = None
    if "2" not in selected:
        reused_exp2_root = Path(args.reuse_exp2) if args.reuse_exp2 else latest_experiment_root("exp2_closed_loop_effectiveness")
        if reused_exp2_root is None:
            raise SystemExit("experiment 2 not selected and no existing exp2 root available to reuse")

    manifest = {
        "exp1_root": roots.exp1.relative_to(ROOT).as_posix() if roots.exp1 else None,
        "exp2_root": (roots.exp2.relative_to(ROOT).as_posix() if roots.exp2 else reused_exp2_root.relative_to(ROOT).as_posix()),
        "exp3_root": roots.exp3.relative_to(ROOT).as_posix() if roots.exp3 else None,
    }
    print(json.dumps({"status": "starting", **manifest}, ensure_ascii=True), flush=True)

    exp1 = None
    exp2 = None
    exp3 = None

    if roots.exp1 is not None:
        exp1 = run_experiment_1(roots.exp1)
        print(
            json.dumps(
                {
                    "status": "experiment_1_complete",
                    "summary": roots.exp1.joinpath("reports/summary.json").relative_to(ROOT).as_posix(),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    if roots.exp2 is not None:
        exp2 = run_experiment_2(roots.exp2)
        print(
            json.dumps(
                {
                    "status": "experiment_2_complete",
                    "summary": roots.exp2.joinpath("reports/summary.json").relative_to(ROOT).as_posix(),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    if roots.exp3 is not None:
        exp3 = run_experiment_3(roots.exp3)
        print(
            json.dumps(
                {
                    "status": "experiment_3_complete",
                    "summary": roots.exp3.joinpath("reports/boundary_summary.json").relative_to(ROOT).as_posix(),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    final_manifest = {
        **manifest,
        "selected_experiments": sorted(selected),
        "exp1_summary": roots.exp1.joinpath("reports/summary.json").relative_to(ROOT).as_posix() if roots.exp1 else None,
        "exp2_summary": (
            roots.exp2.joinpath("reports/summary.json").relative_to(ROOT).as_posix()
            if roots.exp2
            else reused_exp2_root.joinpath("reports/summary.json").relative_to(ROOT).as_posix()
        ),
        "exp3_summary": roots.exp3.joinpath("reports/boundary_summary.json").relative_to(ROOT).as_posix() if roots.exp3 else None,
    }
    if exp1 is not None:
        final_manifest["exp1"] = exp1["size_summary"]
    if exp2 is not None:
        final_manifest["exp2"] = exp2["task_summary"]
    if exp3 is not None:
        final_manifest["exp3"] = exp3["boundary_summary"]

    write_json(LOGS / f"experiment_plan_manifest_{now_timestamp()}.json", final_manifest)
    print(json.dumps({"status": "done", **manifest}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
