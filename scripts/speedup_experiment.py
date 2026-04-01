#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
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

COARSE_TIMING_KEYS = (
    "compile_ms",
    "eval_ms",
    "repro_ms",
    "total_ms",
    "steady_eval_ms",
    "warm_total_proxy_ms",
)

CPU_REPRO_TIMING_KEYS = (
    "selection_ms",
    "crossover_ms",
    "mutation_ms",
)

GPU_EVAL_TIMING_KEYS = (
    "gpu_eval_init_ms",
    "gpu_eval_call_ms",
    "gpu_eval_pack_ms",
    "gpu_eval_launch_prep_ms",
    "gpu_eval_upload_ms",
    "gpu_eval_pack_upload_ms",
    "gpu_eval_kernel_ms",
    "gpu_eval_copyback_ms",
    "gpu_eval_teardown_ms",
)

GPU_EVAL_CALL_PHASE_KEYS = (
    "gpu_eval_pack_ms",
    "gpu_eval_launch_prep_ms",
    "gpu_eval_upload_ms",
    "gpu_eval_kernel_ms",
    "gpu_eval_copyback_ms",
    "gpu_eval_teardown_ms",
)

GPU_REPRO_PRIMARY_TIMING_KEYS = (
    "repro_prepare_inputs_ms",
    "repro_setup_ms",
    "repro_preprocess_ms",
    "repro_pack_ms",
    "repro_upload_ms",
    "repro_kernel_ms",
    "repro_copyback_ms",
    "repro_decode_ms",
    "repro_teardown_ms",
)

GPU_REPRO_KERNEL_SUBPHASE_KEYS = (
    "repro_selection_kernel_ms",
    "repro_variation_kernel_ms",
)

SUMMARY_TIMING_KEYS = (
    "compile_ms",
    "eval_ms",
    "repro_ms",
    "selection_ms",
    "crossover_ms",
    "mutation_ms",
    "gpu_eval_init_ms",
    "gpu_eval_call_ms",
    "gpu_eval_pack_ms",
    "gpu_eval_launch_prep_ms",
    "gpu_eval_upload_ms",
    "gpu_eval_pack_upload_ms",
    "gpu_eval_kernel_ms",
    "gpu_eval_copyback_ms",
    "gpu_eval_teardown_ms",
    "repro_prepare_inputs_ms",
    "repro_setup_ms",
    "repro_preprocess_ms",
    "repro_pack_ms",
    "repro_upload_ms",
    "repro_kernel_ms",
    "repro_copyback_ms",
    "repro_decode_ms",
    "repro_teardown_ms",
    "repro_selection_kernel_ms",
    "repro_variation_kernel_ms",
    "total_ms",
    "steady_eval_ms",
    "warm_total_proxy_ms",
)

SUMMARY_TIMING_ANALYSIS_PATHS = {
    "gpu_eval": {
        "gpu_eval_init_share_of_eval": ("mode_breakdown", "gpu_eval", "gpu_eval", "phase_share_of_eval", "gpu_eval_init_ms"),
        "gpu_eval_kernel_share_of_call": ("mode_breakdown", "gpu_eval", "gpu_eval", "call_phase_share_of_call", "gpu_eval_kernel_ms"),
        "gpu_eval_pack_upload_share_of_call": ("mode_breakdown", "gpu_eval", "gpu_eval", "derived_share_of_call", "gpu_eval_pack_upload_ms"),
        "gpu_eval_teardown_share_of_call": ("mode_breakdown", "gpu_eval", "gpu_eval", "call_phase_share_of_call", "gpu_eval_teardown_ms"),
        "steady_state_eval_speedup_vs_cpu": ("mode_breakdown", "gpu_eval", "gpu_eval", "steady_state_eval_speedup_vs_cpu"),
    },
    "gpu_repro": {
        "gpu_eval_init_share_of_eval": ("mode_breakdown", "gpu_repro", "gpu_eval", "phase_share_of_eval", "gpu_eval_init_ms"),
        "gpu_eval_kernel_share_of_call": ("mode_breakdown", "gpu_repro", "gpu_eval", "call_phase_share_of_call", "gpu_eval_kernel_ms"),
        "repro_decode_share_of_primary": ("mode_breakdown", "gpu_repro", "reproduction", "primary_share_of_repro_primary", "repro_decode_ms"),
        "repro_kernel_share_of_primary": ("mode_breakdown", "gpu_repro", "reproduction", "primary_share_of_repro_primary", "repro_kernel_ms"),
        "repro_wall_minus_primary_ms": ("mode_breakdown", "gpu_repro", "reproduction", "wall_minus_primary_sum_ms"),
    },
    "gpu_repro_overlap": {
        "gpu_eval_init_share_of_eval": ("mode_breakdown", "gpu_repro_overlap", "gpu_eval", "phase_share_of_eval", "gpu_eval_init_ms"),
        "gpu_eval_kernel_share_of_call": ("mode_breakdown", "gpu_repro_overlap", "gpu_eval", "call_phase_share_of_call", "gpu_eval_kernel_ms"),
        "repro_decode_share_of_primary": ("mode_breakdown", "gpu_repro_overlap", "reproduction", "primary_share_of_repro_primary", "repro_decode_ms"),
        "repro_kernel_share_of_primary": ("mode_breakdown", "gpu_repro_overlap", "reproduction", "primary_share_of_repro_primary", "repro_kernel_ms"),
        "repro_hidden_overlap_ms": ("mode_breakdown", "gpu_repro_overlap", "reproduction", "hidden_overlap_ms"),
        "repro_hidden_overlap_share": ("mode_breakdown", "gpu_repro_overlap", "reproduction", "hidden_overlap_share_of_primary"),
    },
}

SUMMARY_COMPARISON_PATHS = {
    "gpu_eval_vs_cpu": {
        "cold_eval_speedup": ("mode_comparisons", "gpu_eval_vs_cpu", "cold_eval_speedup"),
        "steady_state_eval_speedup": ("mode_comparisons", "gpu_eval_vs_cpu", "steady_state_eval_speedup"),
        "total_speedup": ("mode_comparisons", "gpu_eval_vs_cpu", "total_speedup"),
        "warm_total_proxy_speedup": ("mode_comparisons", "gpu_eval_vs_cpu", "warm_total_proxy_speedup"),
        "gpu_init_tax_ms": ("mode_comparisons", "gpu_eval_vs_cpu", "gpu_init_tax_ms"),
    },
    "gpu_repro_vs_gpu_eval": {
        "repro_speedup": ("mode_comparisons", "gpu_repro_vs_gpu_eval", "repro_speedup"),
        "repro_ms_saved": ("mode_comparisons", "gpu_repro_vs_gpu_eval", "repro_ms_saved"),
        "total_speedup": ("mode_comparisons", "gpu_repro_vs_gpu_eval", "total_speedup"),
        "total_ms_saved": ("mode_comparisons", "gpu_repro_vs_gpu_eval", "total_ms_saved"),
    },
    "gpu_repro_overlap_vs_gpu_repro": {
        "repro_speedup": ("mode_comparisons", "gpu_repro_overlap_vs_gpu_repro", "repro_speedup"),
        "repro_wall_ms_saved": ("mode_comparisons", "gpu_repro_overlap_vs_gpu_repro", "repro_wall_ms_saved"),
        "total_speedup": ("mode_comparisons", "gpu_repro_overlap_vs_gpu_repro", "total_speedup"),
        "total_ms_saved": ("mode_comparisons", "gpu_repro_overlap_vs_gpu_repro", "total_ms_saved"),
        "hidden_overlap_ms": ("mode_comparisons", "gpu_repro_overlap_vs_gpu_repro", "hidden_overlap_ms"),
    },
}


@dataclass(frozen=True)
class FixedPopulationSpec:
    label: str
    population_json: Path
    cases_arg: str
    cases_path: Path
    population_size: int | None
    target_depth: int | None
    target_node_count: int | None


FIXED_POP_EMPTY_LIST_KEYS = (
    "fixtures",
    "population_sizes",
    "max_expr_depths",
)

FIXED_POP_NULL_KEYS = (
    "seed_start",
    "probe_cases",
    "min_success_rate",
    "max_expr_depth",
    "max_stmts_per_block",
    "max_total_nodes",
    "max_for_k",
    "max_call_args",
)
def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'(?<![A-Za-z0-9_"])-nan(?![A-Za-z0-9_"])', "NaN", text)
    text = re.sub(r'(?<![A-Za-z0-9_"])nan(?![A-Za-z0-9_"])', "NaN", text)
    text = re.sub(r'(?<![A-Za-z0-9_"])-inf(?![A-Za-z0-9_"])', "-Infinity", text)
    text = re.sub(r'(?<![A-Za-z0-9_"])inf(?![A-Za-z0-9_"])', "Infinity", text)
    return json.loads(text)


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


def normalize_population_label(path: Path) -> str:
    label = path.stem
    if label.endswith(".population"):
        label = label[: -len(".population")]
    return label


def parse_fixed_populations(config: dict[str, Any], raw_override: str) -> list[FixedPopulationSpec]:
    if raw_override:
        raw_specs: list[Any] = [item.strip() for item in raw_override.split(",") if item.strip()]
    else:
        raw_specs = list(config.get("population_jsons", []))
    if not raw_specs:
        return []

    specs: list[FixedPopulationSpec] = []
    seen_labels: set[str] = set()
    for raw in raw_specs:
        if isinstance(raw, str):
            population_json = resolve_path(raw)
            label = normalize_population_label(population_json)
            cases_override_raw: str | None = None
        elif isinstance(raw, dict):
            if "population_json" not in raw:
                raise SystemExit("population_jsons items must include population_json")
            population_json = resolve_path(str(raw["population_json"]))
            label = str(raw.get("label") or normalize_population_label(population_json))
            cases_override_raw = str(raw["cases"]) if raw.get("cases") else None
        else:
            raise SystemExit("population_jsons must be a list of paths or objects")

        if not population_json.exists():
            raise SystemExit(f"missing population json: {population_json}")
        payload = json.loads(population_json.read_text(encoding="utf-8"))
        cases_raw = payload.get("cases_path")
        if cases_override_raw is not None:
            cases_arg = cases_override_raw
            cases_path = resolve_path(cases_override_raw)
        elif isinstance(cases_raw, str) and cases_raw:
            cases_arg = cases_raw
            cases_path = resolve_path(cases_raw)
        else:
            raise SystemExit(f"population json missing cases_path: {population_json}")
        if not cases_path.exists():
            raise SystemExit(f"missing cases path for population json {population_json}: {cases_path}")
        if label in seen_labels:
            raise SystemExit(f"duplicate fixed population label: {label}")
        seen_labels.add(label)
        specs.append(
            FixedPopulationSpec(
                label=label,
                population_json=population_json,
                cases_arg=cases_arg,
                cases_path=cases_path,
                population_size=int(payload["population_size"]) if payload.get("population_size") is not None else None,
                target_depth=int(payload["target_depth"]) if payload.get("target_depth") not in (None, "") else None,
                target_node_count=(
                    int(payload["target_node_count"]) if payload.get("target_node_count") not in (None, "", 0) else None
                ),
            )
        )
    return specs


def validate_fixed_population_config(config: dict[str, Any]) -> None:
    for key in FIXED_POP_EMPTY_LIST_KEYS:
        if config.get(key) != []:
            raise SystemExit(f"when population_jsons is non-empty, {key} must be []")
    for key in FIXED_POP_NULL_KEYS:
        if config.get(key) is not None:
            raise SystemExit(f"when population_jsons is non-empty, {key} must be null or omitted")


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


def write_population_seed_set(
    *,
    path: Path,
    cases_path: Path,
    population_size: int,
    seed_start: int,
    fuel: int,
    max_expr_depth: int,
    max_stmts_per_block: int,
    max_total_nodes: int,
    max_for_k: int,
    max_call_args: int,
) -> None:
    payload = {
        "format_version": "population-seeds-v1",
        "cases_path": str(cases_path),
        "population_size": population_size,
        "probe_cases": 0,
        "min_success_rate": 0.0,
        "fuel": fuel,
        "attempts": population_size,
        "limits": {
            "max_expr_depth": max_expr_depth,
            "max_stmts_per_block": max_stmts_per_block,
            "max_total_nodes": max_total_nodes,
            "max_for_k": max_for_k,
            "max_call_args": max_call_args,
        },
        "seeds": [{"seed": seed_start + i} for i in range(population_size)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def generate_population_seed_set(
    *,
    generator_cli: Path,
    path: Path,
    metadata_path: Path,
    cases_path: Path,
    population_size: int,
    seed_start: int,
    probe_cases: int,
    min_success_rate: float,
    fuel: int,
    max_expr_depth: int,
    max_stmts_per_block: int,
    max_total_nodes: int,
    max_for_k: int,
    max_call_args: int,
    max_attempts: int,
) -> None:
    cmd = [
        str(generator_cli),
        "--cases",
        str(cases_path),
        "--out-population-json",
        str(path),
        "--out-metadata-json",
        str(metadata_path),
        "--target-payload-flavor",
        "any",
        "--generator-root-type",
        "any",
        "--generator-mode",
        "native",
        "--target-depth",
        "0",
        "--target-node-count",
        "0",
        "--population-size",
        str(population_size),
        "--seed-start",
        str(seed_start),
        "--probe-cases",
        str(probe_cases),
        "--min-success-rate",
        str(min_success_rate),
        "--max-attempts",
        str(max_attempts),
        "--fuel",
        str(fuel),
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
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    console_path = metadata_path.with_suffix(".console.log")
    console_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"population generation failed for {cases_path.name}; see {console_path}")


def ensure_population_seed_set(
    *,
    generator_cli: Path | None,
    path: Path,
    cases_path: Path,
    population_size: int,
    seed_start: int,
    probe_cases: int,
    min_success_rate: float,
    fuel: int,
    max_expr_depth: int,
    max_stmts_per_block: int,
    max_total_nodes: int,
    max_for_k: int,
    max_call_args: int,
    max_attempts: int,
) -> Path:
    if not path.exists():
        if probe_cases > 0:
            if generator_cli is None:
                raise RuntimeError("population generation requires a configured generator cli")
            metadata_path = path.with_suffix(".metadata.json")
            generate_population_seed_set(
                generator_cli=generator_cli,
                path=path,
                metadata_path=metadata_path,
                cases_path=cases_path,
                population_size=population_size,
                seed_start=seed_start,
                probe_cases=probe_cases,
                min_success_rate=min_success_rate,
                fuel=fuel,
                max_expr_depth=max_expr_depth,
                max_stmts_per_block=max_stmts_per_block,
                max_total_nodes=max_total_nodes,
                max_for_k=max_for_k,
                max_call_args=max_call_args,
                max_attempts=max_attempts,
            )
        else:
            write_population_seed_set(
                path=path,
                cases_path=cases_path,
                population_size=population_size,
                seed_start=seed_start,
                fuel=fuel,
                max_expr_depth=max_expr_depth,
                max_stmts_per_block=max_stmts_per_block,
                max_total_nodes=max_total_nodes,
                max_for_k=max_for_k,
                max_call_args=max_call_args,
            )
    return path


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
        "mean_fitness": as_float(history[0]["mean_fitness"]) if history else None,
        "best_fitness": as_float(history[0]["best_fitness"]) if history else None,
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


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def fmt_delta_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.3f}"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def sum_timing(raw: dict[str, Any], keys: tuple[str, ...]) -> float:
    return sum(float(raw.get(key, 0.0)) for key in keys)


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def average_mode_metric(
    aggregate: list[dict[str, Any]],
    mode_name: str,
    key: str,
    *,
    max_expr_depth: int | None = None,
) -> float | None:
    values: list[float] = []
    for item in aggregate:
        if max_expr_depth is not None and item["max_expr_depth"] != max_expr_depth:
            continue
        raw = item["modes"].get(mode_name)
        if raw is None or key not in raw or not supports_metric(raw, key):
            continue
        values.append(float(raw[key]))
    if not values:
        return None
    return sum(values) / len(values)


def average_nested_metric(
    aggregate: list[dict[str, Any]],
    path: tuple[str, ...],
    *,
    max_expr_depth: int | None = None,
) -> float | None:
    values: list[float] = []
    for item in aggregate:
        if max_expr_depth is not None and item["max_expr_depth"] != max_expr_depth:
            continue
        value = nested_get(item["timing_analysis"], path)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def metric_map(raw: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(raw.get(key, 0.0)) for key in keys}


def share_map(values: dict[str, float], total: float | None) -> dict[str, float | None]:
    return {key: ratio(value, total) for key, value in values.items()}


def supports_metric(raw: dict[str, Any], key: str) -> bool:
    engine = str(raw.get("engine", "cpu"))
    repro_backend = str(raw.get("repro_backend", "cpu"))
    if key in GPU_EVAL_TIMING_KEYS:
        return engine == "gpu"
    if key in GPU_REPRO_PRIMARY_TIMING_KEYS or key in GPU_REPRO_KERNEL_SUBPHASE_KEYS:
        return repro_backend == "gpu"
    if key in CPU_REPRO_TIMING_KEYS:
        return repro_backend == "cpu"
    return True


def is_speed_metric(key: str) -> bool:
    return "speedup" in key


def is_share_metric(key: str) -> bool:
    return "_share_" in key or key.endswith("_share")


def is_delta_metric(key: str) -> bool:
    return key.endswith("_saved") or key.endswith("_saved_ms") or "residual" in key or "wall_minus" in key


def build_mode_timing_analysis(
    mode_name: str,
    raw: dict[str, Any],
    cpu_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    total_ms = as_float(raw.get("total_ms"))
    eval_ms = as_float(raw.get("eval_ms"))
    repro_ms = as_float(raw.get("repro_ms"))
    coarse_ms = metric_map(raw, COARSE_TIMING_KEYS)
    out: dict[str, Any] = {
        "coarse_ms": coarse_ms,
        "coarse_share_of_total": share_map(coarse_ms, total_ms),
    }

    if str(raw.get("engine", "cpu")) == "gpu":
        phase_ms = {
            "gpu_eval_init_ms": float(raw.get("gpu_eval_init_ms", 0.0)),
            "gpu_eval_call_ms": float(raw.get("gpu_eval_call_ms", 0.0)),
        }
        call_phase_ms = metric_map(raw, GPU_EVAL_CALL_PHASE_KEYS)
        call_ms = phase_ms["gpu_eval_call_ms"]
        derived_ms = {
            "gpu_eval_pack_upload_ms": float(raw.get("gpu_eval_pack_upload_ms", 0.0)),
        }
        out["gpu_eval"] = {
            "phase_ms": phase_ms,
            "phase_share_of_eval": share_map(phase_ms, eval_ms),
            "call_phase_ms": call_phase_ms,
            "call_phase_share_of_call": share_map(call_phase_ms, call_ms),
            "derived_ms": derived_ms,
            "derived_share_of_call": share_map(derived_ms, call_ms),
            "call_accounted_ms": sum(call_phase_ms.values()),
            "call_residual_ms": call_ms - sum(call_phase_ms.values()),
            "cold_start_tax_ms": phase_ms["gpu_eval_init_ms"],
            "cold_start_tax_share_of_eval": ratio(phase_ms["gpu_eval_init_ms"], eval_ms),
            "steady_state_eval_ms": call_ms,
            "cold_eval_speedup_vs_cpu": speedup(cpu_raw.get("eval_ms") if cpu_raw else None, eval_ms),
            "steady_state_eval_speedup_vs_cpu": speedup(cpu_raw.get("eval_ms") if cpu_raw else None, call_ms),
        }

    repro_backend = str(raw.get("repro_backend", "cpu"))
    if repro_backend == "gpu":
        primary_ms = metric_map(raw, GPU_REPRO_PRIMARY_TIMING_KEYS)
        primary_sum_ms = sum(primary_ms.values())
        kernel_subphase_ms = metric_map(raw, GPU_REPRO_KERNEL_SUBPHASE_KEYS)
        hidden_overlap_ms = 0.0
        if str(raw.get("repro_overlap", "off")) == "on":
            hidden_overlap_ms = max(0.0, primary_sum_ms - float(raw.get("repro_ms", 0.0)))
        out["reproduction"] = {
            "backend": "gpu",
            "reported_repro_ms": repro_ms,
            "primary_ms": primary_ms,
            "primary_share_of_repro_wall": share_map(primary_ms, repro_ms),
            "primary_share_of_repro_primary": share_map(primary_ms, primary_sum_ms),
            "primary_sum_ms": primary_sum_ms,
            "wall_minus_primary_sum_ms": (repro_ms - primary_sum_ms) if repro_ms is not None else None,
            "primary_minus_wall_ms": (primary_sum_ms - repro_ms) if repro_ms is not None else None,
            "hidden_overlap_ms": hidden_overlap_ms,
            "hidden_overlap_share_of_primary": ratio(hidden_overlap_ms, primary_sum_ms),
            "kernel_subphase_ms": kernel_subphase_ms,
            "kernel_subphase_share_of_kernel": share_map(kernel_subphase_ms, float(raw.get("repro_kernel_ms", 0.0))),
        }
    else:
        primary_ms = metric_map(raw, CPU_REPRO_TIMING_KEYS)
        primary_sum_ms = sum(primary_ms.values())
        out["reproduction"] = {
            "backend": "cpu",
            "reported_repro_ms": repro_ms,
            "primary_ms": primary_ms,
            "primary_share_of_repro_wall": share_map(primary_ms, repro_ms),
            "primary_share_of_repro_primary": share_map(primary_ms, primary_sum_ms),
            "primary_sum_ms": primary_sum_ms,
            "wall_minus_primary_sum_ms": (repro_ms - primary_sum_ms) if repro_ms is not None else None,
        }

    return out


def build_timing_analysis(modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cpu_raw = modes.get("cpu")
    mode_breakdown = {
        mode_name: build_mode_timing_analysis(mode_name, raw, cpu_raw)
        for mode_name, raw in modes.items()
    }
    comparisons: dict[str, dict[str, float | None]] = {}
    if "cpu" in modes and "gpu_eval" in modes:
        cpu = modes["cpu"]
        gpu_eval = modes["gpu_eval"]
        comparisons["gpu_eval_vs_cpu"] = {
            "cold_eval_speedup": speedup(cpu.get("eval_ms"), gpu_eval.get("eval_ms")),
            "steady_state_eval_speedup": speedup(cpu.get("eval_ms"), gpu_eval.get("gpu_eval_call_ms")),
            "total_speedup": speedup(cpu.get("total_ms"), gpu_eval.get("total_ms")),
            "warm_total_proxy_speedup": speedup(cpu.get("warm_total_proxy_ms"), gpu_eval.get("warm_total_proxy_ms")),
            "gpu_init_tax_ms": as_float(gpu_eval.get("gpu_eval_init_ms")),
            "gpu_init_tax_share_of_eval": ratio(gpu_eval.get("gpu_eval_init_ms"), gpu_eval.get("eval_ms")),
        }
    if "gpu_eval" in modes and "gpu_repro" in modes:
        gpu_eval = modes["gpu_eval"]
        gpu_repro = modes["gpu_repro"]
        comparisons["gpu_repro_vs_gpu_eval"] = {
            "repro_speedup": speedup(gpu_eval.get("repro_ms"), gpu_repro.get("repro_ms")),
            "repro_ms_saved": as_float(gpu_eval.get("repro_ms")) - as_float(gpu_repro.get("repro_ms")),
            "total_speedup": speedup(gpu_eval.get("total_ms"), gpu_repro.get("total_ms")),
            "total_ms_saved": as_float(gpu_eval.get("total_ms")) - as_float(gpu_repro.get("total_ms")),
        }
    if "gpu_repro" in modes and "gpu_repro_overlap" in modes:
        gpu_repro = modes["gpu_repro"]
        gpu_overlap = modes["gpu_repro_overlap"]
        overlap_breakdown = mode_breakdown["gpu_repro_overlap"]["reproduction"]
        comparisons["gpu_repro_overlap_vs_gpu_repro"] = {
            "repro_speedup": speedup(gpu_repro.get("repro_ms"), gpu_overlap.get("repro_ms")),
            "repro_wall_ms_saved": as_float(gpu_repro.get("repro_ms")) - as_float(gpu_overlap.get("repro_ms")),
            "total_speedup": speedup(gpu_repro.get("total_ms"), gpu_overlap.get("total_ms")),
            "total_ms_saved": as_float(gpu_repro.get("total_ms")) - as_float(gpu_overlap.get("total_ms")),
            "hidden_overlap_ms": as_float(overlap_breakdown.get("hidden_overlap_ms")),
        }
    return {
        "mode_breakdown": mode_breakdown,
        "mode_comparisons": comparisons,
    }


def ordered_raw_items(raw: dict[str, Any]) -> list[tuple[str, Any]]:
    ordered_keys = (
        "engine",
        "repro_backend",
        "repro_overlap",
        "population_size",
        *COARSE_TIMING_KEYS,
        *CPU_REPRO_TIMING_KEYS,
        *GPU_EVAL_TIMING_KEYS,
        *GPU_REPRO_PRIMARY_TIMING_KEYS,
        *GPU_REPRO_KERNEL_SUBPHASE_KEYS,
        "mean_fitness",
        "best_fitness",
        "best_program_key",
        "population_source",
        "generation_attempts",
        "probe_cases",
        "min_successes",
        "population_json",
        "out_population_json",
    )
    seen: set[str] = set()
    out: list[tuple[str, Any]] = []
    for key in ordered_keys:
        if key in raw:
            out.append((key, raw[key]))
            seen.add(key)
    for key, value in raw.items():
        if key not in seen:
            out.append((key, value))
    return out


def run_one(
    fixture: Path,
    mode_name: str,
    outdir: Path,
    *,
    evolve_cli: Path,
    generator_cli: Path | None,
    blocksize: int,
    fuel: int,
    mutation_rate: float,
    mutation_subtree_prob: float,
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
    max_attempts: int,
    population_json: Path | None = None,
) -> dict[str, Any]:
    mode = BENCH_MODES[mode_name]
    if population_json is None:
        population_json = ensure_population_seed_set(
            generator_cli=generator_cli,
            path=outdir / "fixed_population.seeds.json",
            cases_path=fixture,
            population_size=population_size,
            seed_start=seed_start,
            probe_cases=probe_cases,
            min_success_rate=min_success_rate,
            fuel=fuel,
            max_expr_depth=max_expr_depth,
            max_stmts_per_block=max_stmts_per_block,
            max_total_nodes=max_total_nodes,
            max_for_k=max_for_k,
            max_call_args=max_call_args,
            max_attempts=max_attempts,
        )
    run_json = outdir / f"one_gen_{mode_name}.run.json"
    cmd = [
        str(evolve_cli),
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
        "--generations",
        "1",
        "--fuel",
        str(fuel),
        "--mutation-rate",
        str(mutation_rate),
        "--mutation-subtree-prob",
        str(mutation_subtree_prob),
        "--penalty",
        str(penalty),
        "--selection-pressure",
        str(selection_pressure),
        "--seed",
        "0",
        "--population-json",
        str(population_json),
        "--skip-final-eval",
        "on",
        "--timing",
        "all",
        "--show-program",
        "none",
        "--out-json",
        str(run_json),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    console_path = outdir / f"one_gen_{mode_name}.console.log"
    console_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{mode_name} benchmark failed for {fixture.name}; see {console_path}")
    return run_to_fixed_pop_metrics(load_json_file(run_json))


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
            "steady_eval_cpu_over_mode": speedup(cpu.get("steady_eval_ms"), result.get("steady_eval_ms")),
            "warm_total_proxy_cpu_over_mode": speedup(cpu.get("warm_total_proxy_ms"), result.get("warm_total_proxy_ms")),
        }
    return out


def write_fixture_report(outdir: Path, modes: dict[str, dict[str, Any]], population_json_path: Path | None = None) -> dict[str, Any]:
    timing_analysis = build_timing_analysis(modes)
    if population_json_path is None:
        population_json_path = outdir / "fixed_population.seeds.json"
    report = {
        "benchmark_type": "fixed_population_evolve_cli_1gen_v1",
        "population_json": str(population_json_path),
        "modes": modes,
        "speedup_vs_cpu": build_speedup_report(modes),
        "timing_analysis": timing_analysis,
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
    lines.append("## Timing Analysis")
    lines.append("")
    for mode_name, analysis in report["timing_analysis"]["mode_breakdown"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        lines.append("#### Coarse Timing")
        lines.append("")
        for key in COARSE_TIMING_KEYS:
            lines.append(
                f"- {key}: {fmt_ms(analysis['coarse_ms'].get(key))} ({fmt_pct(analysis['coarse_share_of_total'].get(key))} of total)"
            )
        lines.append("")
        if "gpu_eval" in analysis:
            gpu_eval = analysis["gpu_eval"]
            lines.append("#### GPU Eval Breakdown")
            lines.append("")
            lines.append(
                f"- cold_eval_speedup_vs_cpu: {fmt_speed(gpu_eval.get('cold_eval_speedup_vs_cpu'))}"
            )
            lines.append(
                f"- steady_state_eval_speedup_vs_cpu: {fmt_speed(gpu_eval.get('steady_state_eval_speedup_vs_cpu'))}"
            )
            lines.append(
                f"- gpu_eval_init_ms: {fmt_ms(gpu_eval['phase_ms'].get('gpu_eval_init_ms'))} ({fmt_pct(gpu_eval['phase_share_of_eval'].get('gpu_eval_init_ms'))} of eval)"
            )
            lines.append(
                f"- gpu_eval_call_ms: {fmt_ms(gpu_eval['phase_ms'].get('gpu_eval_call_ms'))} ({fmt_pct(gpu_eval['phase_share_of_eval'].get('gpu_eval_call_ms'))} of eval)"
            )
            lines.append(
                f"- gpu_eval_pack_upload_ms: {fmt_ms(gpu_eval['derived_ms'].get('gpu_eval_pack_upload_ms'))} ({fmt_pct(gpu_eval['derived_share_of_call'].get('gpu_eval_pack_upload_ms'))} of call)"
            )
            for key in GPU_EVAL_CALL_PHASE_KEYS:
                lines.append(
                    f"- {key}: {fmt_ms(gpu_eval['call_phase_ms'].get(key))} ({fmt_pct(gpu_eval['call_phase_share_of_call'].get(key))} of call)"
                )
            lines.append(f"- gpu_eval_call_residual_ms: {fmt_delta_ms(gpu_eval.get('call_residual_ms'))}")
            lines.append("")
        repro = analysis["reproduction"]
        lines.append("#### Reproduction Breakdown")
        lines.append("")
        lines.append(f"- backend: `{repro['backend']}`")
        lines.append(f"- reported_repro_ms: {fmt_ms(repro.get('reported_repro_ms'))}")
        lines.append(f"- primary_sum_ms: {fmt_ms(repro.get('primary_sum_ms'))}")
        lines.append(f"- wall_minus_primary_sum_ms: {fmt_delta_ms(repro.get('wall_minus_primary_sum_ms'))}")
        if repro["backend"] == "gpu":
            lines.append(f"- hidden_overlap_ms: {fmt_ms(repro.get('hidden_overlap_ms'))}")
            lines.append(f"- hidden_overlap_share_of_primary: {fmt_pct(repro.get('hidden_overlap_share_of_primary'))}")
        for key, value in repro["primary_ms"].items():
            lines.append(
                f"- {key}: {fmt_ms(value)} ({fmt_pct(repro['primary_share_of_repro_wall'].get(key))} of repro wall, {fmt_pct(repro['primary_share_of_repro_primary'].get(key))} of repro primary)"
            )
        if repro["backend"] == "gpu":
            lines.append("")
            lines.append("#### GPU Repro Kernel Subphases")
            lines.append("")
            for key, value in repro["kernel_subphase_ms"].items():
                lines.append(
                    f"- {key}: {fmt_ms(value)} ({fmt_pct(repro['kernel_subphase_share_of_kernel'].get(key))} of repro_kernel_ms)"
                )
        lines.append("")
    if report["timing_analysis"]["mode_comparisons"]:
        lines.append("## Timing Comparisons")
        lines.append("")
        for comp_name, comp in report["timing_analysis"]["mode_comparisons"].items():
            lines.append(f"### `{comp_name}`")
            lines.append("")
            for key, value in comp.items():
                if is_speed_metric(key):
                    lines.append(f"- {key}: {fmt_speed(value)}")
                elif is_share_metric(key):
                    lines.append(f"- {key}: {fmt_pct(value)}")
                elif is_delta_metric(key):
                    lines.append(f"- {key}: {fmt_delta_ms(value)}")
                else:
                    lines.append(f"- {key}: {fmt_ms(value)}")
            lines.append("")
    lines.append("## Raw Mode Timings")
    lines.append("")
    for mode_name, raw in report["modes"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in ordered_raw_items(raw):
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


def build_average_mode_timings(
    aggregate: list[dict[str, Any]],
    modes: list[str],
    *,
    max_expr_depth: int | None = None,
) -> dict[str, dict[str, float | None]]:
    return {
        mode_name: {
            key: average_mode_metric(aggregate, mode_name, key, max_expr_depth=max_expr_depth)
            for key in SUMMARY_TIMING_KEYS
        }
        for mode_name in modes
    }


def build_average_timing_analysis(
    aggregate: list[dict[str, Any]],
    *,
    max_expr_depth: int | None = None,
) -> dict[str, dict[str, float | None]]:
    return {
        mode_name: {
            metric_name: average_nested_metric(aggregate, path, max_expr_depth=max_expr_depth)
            for metric_name, path in metric_paths.items()
        }
        for mode_name, metric_paths in SUMMARY_TIMING_ANALYSIS_PATHS.items()
    }


def build_average_timing_comparisons(
    aggregate: list[dict[str, Any]],
    *,
    max_expr_depth: int | None = None,
) -> dict[str, dict[str, float | None]]:
    return {
        comp_name: {
            metric_name: average_nested_metric(aggregate, path, max_expr_depth=max_expr_depth)
            for metric_name, path in metric_paths.items()
        }
        for comp_name, metric_paths in SUMMARY_COMPARISON_PATHS.items()
    }


def main() -> int:
    config = load_config(EXAMPLE_CONFIG)

    evolve_cli = resolve_path(config.get("evolve_cli", config.get("bench_cli", "cpp/build/g3pvm_evolve_cli")))
    if not evolve_cli.exists():
        raise SystemExit(f"missing evolve cli: {evolve_cli}")
    generator_cli_raw = config.get("population_bucket_cli", "cpp/build/g3pvm_population_bucket_cli")
    generator_cli = resolve_path(generator_cli_raw) if generator_cli_raw else None
    if generator_cli is not None and not generator_cli.exists():
        raise SystemExit(f"missing population bucket cli: {generator_cli}")

    fixtures = select_fixtures(config["fixtures"], "")
    fixed_populations = parse_fixed_populations(config, "")
    if fixed_populations:
        validate_fixed_population_config(config)
        fixtures = []
    if not fixtures and not fixed_populations:
        raise SystemExit("no fixtures selected")

    modes = parse_modes(config, "")
    population_sizes = [] if fixed_populations else parse_population_sizes(config, "")
    max_expr_depths = [] if fixed_populations else parse_max_expr_depths(config, "")
    probe_cases = 0 if fixed_populations else int(config["probe_cases"])
    min_success_rate = 0.0 if fixed_populations else float(config["min_success_rate"])

    outdir = ROOT / f"{config['outdir_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    aggregate: list[dict[str, Any]] = []
    common = dict(
        evolve_cli=evolve_cli,
        generator_cli=generator_cli,
        blocksize=int(config["blocksize"]),
        fuel=int(config["fuel"]),
        mutation_rate=float(config["mutation_rate"]),
        mutation_subtree_prob=float(config["mutation_subtree_prob"]),
        penalty=float(config["penalty"]),
        selection_pressure=int(config["selection_pressure"]),
        seed_start=0 if fixed_populations else int(config["seed_start"]),
        probe_cases=probe_cases,
        min_success_rate=min_success_rate,
        max_stmts_per_block=0 if fixed_populations else int(config["max_stmts_per_block"]),
        max_total_nodes=0 if fixed_populations else int(config["max_total_nodes"]),
        max_for_k=0 if fixed_populations else int(config["max_for_k"]),
        max_call_args=0 if fixed_populations else int(config["max_call_args"]),
        max_attempts=int(config.get("max_attempts", 500000)),
    )
    if fixed_populations:
        for spec in fixed_populations:
            fixture_outdir = outdir / spec.label
            fixture_outdir.mkdir(parents=True, exist_ok=True)
            mode_results: dict[str, dict[str, Any]] = {}
            for mode_name in modes:
                print(
                    f"[fixture-bench] {mode_name} population={spec.label} cases={spec.cases_path.stem}",
                    flush=True,
                )
                mode_results[mode_name] = run_one(
                    Path(spec.cases_arg),
                    mode_name,
                    fixture_outdir,
                    population_size=spec.population_size or 0,
                    max_expr_depth=spec.target_depth or 1,
                    population_json=spec.population_json,
                    **common,
                )
            report = write_fixture_report(fixture_outdir, mode_results, spec.population_json)
            aggregate.append(
                {
                    "fixture": spec.cases_path.stem,
                    "population_label": spec.label,
                    "population_json": str(spec.population_json),
                    "population_size": spec.population_size,
                    "max_expr_depth": spec.target_depth,
                    "target_node_count": spec.target_node_count,
                    "report_json": str(fixture_outdir / "mode_compare.report.json"),
                    "modes": report["modes"],
                    "speedup_vs_cpu": report["speedup_vs_cpu"],
                    "timing_analysis": report["timing_analysis"],
                }
            )
    else:
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
                            "population_label": None,
                            "population_json": report["population_json"],
                            "population_size": population_size,
                            "max_expr_depth": max_expr_depth,
                            "target_node_count": None,
                            "report_json": str(fixture_outdir / "mode_compare.report.json"),
                            "modes": report["modes"],
                            "speedup_vs_cpu": report["speedup_vs_cpu"],
                            "timing_analysis": report["timing_analysis"],
                        }
                    )

    summary = {
        "benchmark_type": "fixed_population_file_sweep_v1" if fixed_populations else "fixture_speedup_modes_v2",
        "modes": modes,
        "max_expr_depths": max_expr_depths,
        "population_jsons": [str(spec.population_json) for spec in fixed_populations] if fixed_populations else [],
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
                    "steady_eval_cpu_over_mode",
                    "warm_total_proxy_cpu_over_mode",
                )
            }
            for mode_name in modes
            if mode_name != "cpu"
        },
        "average_speedup_vs_cpu_by_max_expr_depth": ({
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
                        "steady_eval_cpu_over_mode",
                        "warm_total_proxy_cpu_over_mode",
                    )
                }
                for mode_name in modes
                if mode_name != "cpu"
            }
            for max_expr_depth in max_expr_depths
        } if max_expr_depths else {}),
        "average_mode_timings": build_average_mode_timings(aggregate, modes),
        "average_mode_timings_by_max_expr_depth": ({
            str(max_expr_depth): build_average_mode_timings(aggregate, modes, max_expr_depth=max_expr_depth)
            for max_expr_depth in max_expr_depths
        } if max_expr_depths else {}),
        "average_timing_analysis": {
            "mode_breakdown": build_average_timing_analysis(aggregate),
            "mode_comparisons": build_average_timing_comparisons(aggregate),
        },
        "average_timing_analysis_by_max_expr_depth": ({
            str(max_expr_depth): {
                "mode_breakdown": build_average_timing_analysis(aggregate, max_expr_depth=max_expr_depth),
                "mode_comparisons": build_average_timing_comparisons(aggregate, max_expr_depth=max_expr_depth),
            }
            for max_expr_depth in max_expr_depths
        } if max_expr_depths else {}),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    lines = ["# Fixture Speedup Summary", "", f"- outdir: `{outdir}`", "", "## Average Speedup Vs CPU", ""]
    for mode_name, speedup_map in summary["average_speedup_vs_cpu"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in speedup_map.items():
            lines.append(f"- {key}: {fmt_speed(value)}")
        lines.append("")
    lines.extend(["## Average Mode Timings", ""])
    for mode_name, timing_map in summary["average_mode_timings"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in timing_map.items():
            lines.append(f"- {key}: {fmt_ms(value)}")
        lines.append("")
    lines.extend(["## Average Timing Analysis", ""])
    for mode_name, analysis_map in summary["average_timing_analysis"]["mode_breakdown"].items():
        lines.append(f"### `{mode_name}`")
        lines.append("")
        for key, value in analysis_map.items():
            if is_speed_metric(key):
                lines.append(f"- {key}: {fmt_speed(value)}")
            elif is_share_metric(key):
                lines.append(f"- {key}: {fmt_pct(value)}")
            elif is_delta_metric(key):
                lines.append(f"- {key}: {fmt_delta_ms(value)}")
            else:
                lines.append(f"- {key}: {fmt_ms(value)}")
        lines.append("")
    for comp_name, comp_map in summary["average_timing_analysis"]["mode_comparisons"].items():
        lines.append(f"### comparison `{comp_name}`")
        lines.append("")
        for key, value in comp_map.items():
            if is_speed_metric(key):
                lines.append(f"- {key}: {fmt_speed(value)}")
            elif is_share_metric(key):
                lines.append(f"- {key}: {fmt_pct(value)}")
            elif is_delta_metric(key):
                lines.append(f"- {key}: {fmt_delta_ms(value)}")
            else:
                lines.append(f"- {key}: {fmt_ms(value)}")
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
        lines.extend(["## Average Mode Timings By max_expr_depth", ""])
        for max_expr_depth in max_expr_depths:
            lines.append(f"### depth={max_expr_depth}")
            lines.append("")
            for mode_name, timing_map in summary["average_mode_timings_by_max_expr_depth"][str(max_expr_depth)].items():
                lines.append(f"#### `{mode_name}`")
                lines.append("")
                for key, value in timing_map.items():
                    lines.append(f"- {key}: {fmt_ms(value)}")
                lines.append("")
        lines.extend(["## Average Timing Analysis By max_expr_depth", ""])
        for max_expr_depth in max_expr_depths:
            lines.append(f"### depth={max_expr_depth}")
            lines.append("")
            for mode_name, analysis_map in summary["average_timing_analysis_by_max_expr_depth"][str(max_expr_depth)]["mode_breakdown"].items():
                lines.append(f"#### `{mode_name}`")
                lines.append("")
                for key, value in analysis_map.items():
                    if is_speed_metric(key):
                        lines.append(f"- {key}: {fmt_speed(value)}")
                    elif is_share_metric(key):
                        lines.append(f"- {key}: {fmt_pct(value)}")
                    elif is_delta_metric(key):
                        lines.append(f"- {key}: {fmt_delta_ms(value)}")
                    else:
                        lines.append(f"- {key}: {fmt_ms(value)}")
                lines.append("")
    lines.extend(["## Fixtures", ""])
    for item in aggregate:
        parts = []
        for mode_name in modes:
            if mode_name == "cpu":
                continue
            speed_map = item["speedup_vs_cpu"].get(mode_name, {})
            parts.append(
                f"{mode_name}: cold_total={fmt_speed(speed_map.get('total_cpu_over_mode'))}, "
                f"steady_eval={fmt_speed(speed_map.get('steady_eval_cpu_over_mode'))}, "
                f"warm_total_proxy={fmt_speed(speed_map.get('warm_total_proxy_cpu_over_mode'))}"
            )
        prefix = f"{item['fixture']}"
        if item.get("population_label"):
            prefix += f" label={item['population_label']}"
        if item.get("population_size") is not None:
            prefix += f" pop={item['population_size']}"
        if item.get("max_expr_depth") is not None:
            prefix += f" depth={item['max_expr_depth']}"
        if item.get("target_node_count") is not None:
            prefix += f" node={item['target_node_count']}"
        lines.append(f"- {prefix}: " + "; ".join(parts))
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fixture-bench] summary json: {outdir / 'summary.json'}")
    print(f"[fixture-bench] summary md: {outdir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
