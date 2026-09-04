#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from ..shared.hashing import sha256_file
from ..shared.metrics import interpolated_percentile as percentile
from ..shared.schemas import PSB_REGRESSION_SUMMARY, require_format


PHASE_TIMING_KEYS: tuple[tuple[str, str], ...] = (
    ("eval", "generation_eval_ms"),
    ("repro", "generation_repro_ms"),
    ("gpu_eval_call", "generation_gpu_eval_call_ms"),
    ("gpu_kernel", "generation_gpu_eval_kernel_ms"),
    ("repro_decode", "generation_repro_decode_ms"),
    ("repro_kernel", "generation_repro_kernel_ms"),
)


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    require_format(payload, PSB_REGRESSION_SUMMARY, path)
    return payload


def load_problem_tolerances(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    if payload.get("format_version") != "psb-problem-tolerances":
        raise ValueError(f"unsupported PSB tolerance format in {path}")
    default = payload.get("default", {})
    problems = payload.get("problems", {})
    if not isinstance(default, dict):
        raise ValueError("problem tolerance default must be an object")
    if not isinstance(problems, dict):
        raise ValueError("problem tolerance problems must be an object")
    _validate_tolerance_row(default, "default")
    for problem, row in problems.items():
        if not isinstance(problem, str):
            raise ValueError("problem tolerance keys must be strings")
        if not isinstance(row, dict):
            raise ValueError(f"problem tolerance for {problem} must be an object")
        _validate_tolerance_row(row, problem)
    return payload


def _validate_tolerance_row(row: Mapping[str, Any], label: str) -> None:
    for key in ("median_fitness_tolerance", "test_median_fitness_tolerance"):
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or value < 0.0:
            raise ValueError(f"{label}.{key} must be a non-negative number")


def numeric_values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def group_ok_runs(summary: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary.get("runs", []):
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        problem = row.get("problem")
        if isinstance(problem, str):
            grouped.setdefault(problem, []).append(row)
    return grouped


def compare_problem(
    problem: str,
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    problem_tolerances: Dict[str, Any] | None,
) -> Dict[str, Any]:
    baseline_solved = sum(1 for row in baseline_rows if row.get("solved"))
    candidate_solved = sum(1 for row in candidate_rows if row.get("solved"))
    solved_regression = max(0, baseline_solved - candidate_solved)
    baseline_test_solved = sum(1 for row in baseline_rows if row.get("test_solved"))
    candidate_test_solved = sum(1 for row in candidate_rows if row.get("test_solved"))
    test_solved_regression = max(0, baseline_test_solved - candidate_test_solved)

    baseline_fitness = numeric_values(baseline_rows, "best_fitness")
    candidate_fitness = numeric_values(candidate_rows, "best_fitness")
    baseline_median_fitness = statistics.median(baseline_fitness) if baseline_fitness else None
    candidate_median_fitness = statistics.median(candidate_fitness) if candidate_fitness else None

    fitness_regression = None
    if baseline_median_fitness is not None and candidate_median_fitness is not None:
        fitness_regression = baseline_median_fitness - candidate_median_fitness

    baseline_test_fitness = numeric_values(baseline_rows, "test_best_fitness")
    candidate_test_fitness = numeric_values(candidate_rows, "test_best_fitness")
    baseline_median_test_fitness = statistics.median(baseline_test_fitness) if baseline_test_fitness else None
    candidate_median_test_fitness = statistics.median(candidate_test_fitness) if candidate_test_fitness else None

    test_fitness_regression = None
    if baseline_median_test_fitness is not None and candidate_median_test_fitness is not None:
        test_fitness_regression = baseline_median_test_fitness - candidate_median_test_fitness

    baseline_total = numeric_values(baseline_rows, "total_ms")
    candidate_total = numeric_values(candidate_rows, "total_ms")
    baseline_median_total = statistics.median(baseline_total) if baseline_total else None
    candidate_median_total = statistics.median(candidate_total) if candidate_total else None
    baseline_p90_total = percentile(baseline_total, 0.90)
    candidate_p90_total = percentile(candidate_total, 0.90)

    median_total_ratio = ratio(candidate_median_total, baseline_median_total)
    p90_total_ratio = ratio(candidate_p90_total, baseline_p90_total)

    baseline_kernel = timing_generation_values(baseline_rows, "generation_gpu_eval_kernel_ms")
    candidate_kernel = timing_generation_values(candidate_rows, "generation_gpu_eval_kernel_ms")
    baseline_median_kernel = statistics.median(baseline_kernel) if baseline_kernel else None
    candidate_median_kernel = statistics.median(candidate_kernel) if candidate_kernel else None
    kernel_ratio = ratio(candidate_median_kernel, baseline_median_kernel)
    phase_ratios = timing_phase_ratios(baseline_rows, candidate_rows)
    quality_thresholds = effective_quality_thresholds(problem, args, problem_tolerances)

    failures: List[str] = []
    if solved_regression > args.solved_regression_tolerance:
        failures.append("solved_count")
    if test_solved_regression > args.solved_regression_tolerance:
        failures.append("test_solved_count")
    if (
        args.stable_solved_threshold > 0
        and baseline_solved >= args.stable_solved_threshold
        and candidate_solved < args.stable_candidate_min
    ):
        failures.append("stable_solved")
    if fitness_regression is not None and fitness_regression > quality_thresholds["median_fitness_tolerance"]:
        failures.append("median_best_fitness")
    if (
        test_fitness_regression is not None
        and test_fitness_regression > quality_thresholds["test_median_fitness_tolerance"]
    ):
        failures.append("test_median_best_fitness")
    if median_total_ratio is not None and median_total_ratio > args.median_runtime_ratio:
        failures.append("median_total_ms")
    if p90_total_ratio is not None and p90_total_ratio > args.p90_runtime_ratio:
        failures.append("p90_total_ms")
    if kernel_ratio is not None and kernel_ratio > args.kernel_runtime_ratio:
        failures.append("gpu_kernel_ms")
    speed_attribution = speed_failure_attribution(failures, phase_ratios)
    quality_categories = quality_failure_categories(failures)

    return {
        "problem": problem,
        "baseline_runs": len(baseline_rows),
        "candidate_runs": len(candidate_rows),
        "baseline_solved": baseline_solved,
        "candidate_solved": candidate_solved,
        "solved_regression": solved_regression,
        "baseline_test_solved": baseline_test_solved if baseline_test_fitness else None,
        "candidate_test_solved": candidate_test_solved if candidate_test_fitness else None,
        "test_solved_regression": test_solved_regression if baseline_test_fitness and candidate_test_fitness else None,
        "baseline_median_best_fitness": baseline_median_fitness,
        "candidate_median_best_fitness": candidate_median_fitness,
        "fitness_regression": fitness_regression,
        "baseline_median_test_best_fitness": baseline_median_test_fitness,
        "candidate_median_test_best_fitness": candidate_median_test_fitness,
        "test_fitness_regression": test_fitness_regression,
        "baseline_median_total_ms": baseline_median_total,
        "candidate_median_total_ms": candidate_median_total,
        "median_total_ratio": median_total_ratio,
        "baseline_p90_total_ms": baseline_p90_total,
        "candidate_p90_total_ms": candidate_p90_total,
        "p90_total_ratio": p90_total_ratio,
        "baseline_median_gpu_eval_kernel_ms": baseline_median_kernel,
        "candidate_median_gpu_eval_kernel_ms": candidate_median_kernel,
        "gpu_kernel_ratio": kernel_ratio,
        "timing_phase_ratios": phase_ratios,
        "quality_thresholds": quality_thresholds,
        "failures": failures,
        "speed_failure_attribution": speed_attribution,
        "quality_failure_categories": quality_categories,
        "failure_context": failure_context(baseline_rows, candidate_rows) if failures else None,
        "passed": not failures,
    }


def effective_quality_thresholds(
    problem: str,
    args: argparse.Namespace,
    problem_tolerances: Dict[str, Any] | None,
) -> Dict[str, float]:
    train = float(args.median_fitness_tolerance)
    test = float(args.median_fitness_tolerance)
    if problem_tolerances is not None:
        default = problem_tolerances.get("default", {})
        if isinstance(default, dict):
            train = float(default.get("median_fitness_tolerance", train))
            test = float(default.get("test_median_fitness_tolerance", test))
        problems = problem_tolerances.get("problems", {})
        row = problems.get(problem) if isinstance(problems, dict) else None
        if isinstance(row, dict):
            train = float(row.get("median_fitness_tolerance", train))
            test = float(row.get("test_median_fitness_tolerance", test))
    return {
        "median_fitness_tolerance": train,
        "test_median_fitness_tolerance": test,
    }


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    return candidate / baseline


def timing_generation_values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        timing = row.get("timing")
        if not isinstance(timing, dict):
            continue
        values = timing.get(key)
        if isinstance(values, list):
            out.extend(float(v) for v in values if isinstance(v, (int, float)))
    return out


def timing_phase_ratios(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for phase, key in PHASE_TIMING_KEYS:
        baseline_values = timing_generation_values(baseline_rows, key)
        candidate_values = timing_generation_values(candidate_rows, key)
        baseline_median = statistics.median(baseline_values) if baseline_values else None
        candidate_median = statistics.median(candidate_values) if candidate_values else None
        out[phase] = {
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": candidate_median,
            "ratio": ratio(candidate_median, baseline_median),
        }
    return out


def speed_failure_attribution(
    failures: List[str],
    phase_ratios: Dict[str, Dict[str, float | None]],
) -> List[str]:
    out: List[str] = []
    if "gpu_kernel_ms" in failures:
        out.append("gpu_kernel")
    if "median_total_ms" in failures or "p90_total_ms" in failures:
        ranked: List[tuple[float, str]] = []
        for phase, row in phase_ratios.items():
            value = row.get("ratio")
            if isinstance(value, (int, float)) and value > 1.0:
                ranked.append((float(value), phase))
        ranked.sort(reverse=True)
        out.extend(phase for _, phase in ranked[:3])
        if not ranked:
            out.append("total_wall")
    deduped: List[str] = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped


def quality_failure_categories(failures: List[str]) -> List[str]:
    mapping = {
        "solved_count": "train_solved_count",
        "test_solved_count": "test_solved_count",
        "stable_solved": "stable_train_solved",
        "median_best_fitness": "train_median_best_fitness",
        "test_median_best_fitness": "test_median_best_fitness",
    }
    return [mapping[item] for item in failures if item in mapping]


def failure_context(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "seeds": sorted(
            {
                int(row["seed"])
                for row in baseline_rows + candidate_rows
                if isinstance(row.get("seed"), int)
            }
        ),
        "schema_hashes": sorted(
            {
                str(row["schema_hash"])
                for row in baseline_rows + candidate_rows
                if isinstance(row.get("schema_hash"), str)
            }
        ),
        "baseline_grammar_config": first_record(baseline_rows, "grammar_config"),
        "candidate_grammar_config": first_record(candidate_rows, "grammar_config"),
        "runs": run_context_rows(baseline_rows, candidate_rows),
    }


def first_record(rows: List[Dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def run_context_rows(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_seed = {row.get("seed"): row for row in baseline_rows}
    candidate_by_seed = {row.get("seed"): row for row in candidate_rows}
    seeds = sorted(seed for seed in set(baseline_by_seed) | set(candidate_by_seed) if isinstance(seed, int))
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        b_row = baseline_by_seed.get(seed, {})
        c_row = candidate_by_seed.get(seed, {})
        out.append(
            {
                "seed": seed,
                "baseline": compact_run_context(b_row),
                "candidate": compact_run_context(c_row),
            }
        )
    return out


def compact_run_context(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "best_fitness": row.get("best_fitness"),
        "test_best_fitness": row.get("test_best_fitness"),
        "solved": row.get("solved"),
        "test_solved": row.get("test_solved"),
        "total_ms": row.get("total_ms"),
        "program_key": row.get("program_key"),
        "test_program_key": row.get("test_program_key"),
    }


def compare(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    problem_tolerances = load_problem_tolerances(args.problem_tolerances)
    baseline_runs = group_ok_runs(baseline)
    candidate_runs = group_ok_runs(candidate)

    selected = sorted(set(baseline_runs) | set(candidate_runs))
    if args.problems:
        requested = {item.strip() for item in args.problems.split(",") if item.strip()}
        selected = [problem for problem in selected if problem in requested]

    problems: List[Dict[str, Any]] = []
    missing: List[str] = []
    for problem in selected:
        b_rows = baseline_runs.get(problem, [])
        c_rows = candidate_runs.get(problem, [])
        if not b_rows or not c_rows:
            missing.append(problem)
            problems.append(
                {
                    "problem": problem,
                    "baseline_runs": len(b_rows),
                    "candidate_runs": len(c_rows),
                    "failures": ["missing_runs"],
                    "passed": False,
                }
            )
            continue
        problems.append(compare_problem(problem, b_rows, c_rows, args, problem_tolerances))

    failed = [row for row in problems if not row.get("passed")]
    problem_tolerance_record = None
    if args.problem_tolerances is not None:
        problem_tolerance_record = {
            "path": str(args.problem_tolerances),
            "hash": sha256_file(args.problem_tolerances),
            "format_version": problem_tolerances.get("format_version") if problem_tolerances else None,
        }
    result = {
        "format_version": "psb-baseline-comparison",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "thresholds": {
            "solved_regression_tolerance": args.solved_regression_tolerance,
            "stable_solved_threshold": args.stable_solved_threshold,
            "stable_candidate_min": args.stable_candidate_min,
            "median_fitness_tolerance": args.median_fitness_tolerance,
            "median_runtime_ratio": args.median_runtime_ratio,
            "p90_runtime_ratio": args.p90_runtime_ratio,
            "kernel_runtime_ratio": args.kernel_runtime_ratio,
            "problem_tolerances": problem_tolerance_record,
        },
        "passed": not failed,
        "failed_problem_count": len(failed),
        "missing_problem_count": len(missing),
        "problems": problems,
    }
    return result, 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PSB regression summaries against a baseline.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--problems")
    parser.add_argument("--solved-regression-tolerance", type=int, default=1)
    parser.add_argument("--stable-solved-threshold", type=int, default=4)
    parser.add_argument("--stable-candidate-min", type=int, default=3)
    parser.add_argument("--median-fitness-tolerance", type=float, default=0.0)
    parser.add_argument("--problem-tolerances", type=Path)
    parser.add_argument("--median-runtime-ratio", type=float, default=1.15)
    parser.add_argument("--p90-runtime-ratio", type=float, default=1.25)
    parser.add_argument("--kernel-runtime-ratio", type=float, default=1.20)
    args = parser.parse_args()

    result, code = compare(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(f"PSB_COMPARE passed={str(result['passed']).lower()} failed={result['failed_problem_count']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
