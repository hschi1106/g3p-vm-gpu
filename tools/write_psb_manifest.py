#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def parse_csv(text: str | None) -> List[str] | None:
    if text is None or not text.strip():
        return None
    out = [item.strip() for item in text.split(",") if item.strip()]
    return out or None


def select_problem_rows(comparison: Dict[str, Any], problems: Iterable[str] | None) -> List[Dict[str, Any]]:
    rows = comparison.get("problems")
    if not isinstance(rows, list):
        raise ValueError("comparison JSON missing problems list")
    out = [row for row in rows if isinstance(row, dict) and isinstance(row.get("problem"), str)]
    if problems is not None:
        wanted = set(problems)
        out = [row for row in out if row["problem"] in wanted]
    out.sort(key=lambda row: str(row["problem"]))
    return out


def max_numeric(rows: List[Dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return max(values) if values else None


def min_numeric(rows: List[Dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return min(values) if values else None


def metadata_scope(metadata: Dict[str, Any], problems: List[str]) -> Dict[str, Any]:
    return {
        "problems": problems,
        "problem_count": len(problems),
        "seeds": metadata.get("seeds"),
        "population_size": metadata.get("population_size"),
        "generations": metadata.get("generations"),
        "engine": metadata.get("engine"),
        "repro_backend": metadata.get("repro_backend"),
        "repro_overlap": metadata.get("repro_overlap"),
        "eval_test": metadata.get("eval_test"),
        "cases_root": metadata.get("cases_root"),
    }


def load_excluded_problems(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = load_json(path)
    raw = payload.get("excluded_problems", {})
    if not isinstance(raw, dict):
        raise ValueError("excluded-problems manifest has non-object excluded_problems")
    return raw


def compact_problem(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "baseline_runs",
        "candidate_runs",
        "baseline_solved",
        "candidate_solved",
        "baseline_test_solved",
        "candidate_test_solved",
        "baseline_median_best_fitness",
        "candidate_median_best_fitness",
        "fitness_regression",
        "baseline_median_test_best_fitness",
        "candidate_median_test_best_fitness",
        "test_fitness_regression",
        "median_total_ratio",
        "p90_total_ratio",
        "gpu_kernel_ratio",
        "failures",
        "speed_failure_attribution",
        "quality_failure_categories",
        "passed",
    ]
    return {key: row.get(key) for key in keys if key in row}


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    comparison = load_json(args.comparison)
    problems = select_problem_rows(comparison, parse_csv(args.problems))
    problem_names = [str(row["problem"]) for row in problems]
    baseline_meta = baseline.get("metadata", {})
    candidate_meta = candidate.get("metadata", {})
    if not isinstance(baseline_meta, dict) or not isinstance(candidate_meta, dict):
        raise ValueError("summary metadata must be objects")

    baseline_aggregate = baseline.get("aggregate", {})
    candidate_aggregate = candidate.get("aggregate", {})
    if not isinstance(baseline_aggregate, dict) or not isinstance(candidate_aggregate, dict):
        raise ValueError("summary aggregate must be objects")

    failed_count = sum(1 for row in problems if not row.get("passed"))
    manifest = {
        "format_version": args.format_version,
        "generated_at": args.generated_at,
        "suite": candidate_meta.get("suite", baseline_meta.get("suite")),
        "profile": candidate_meta.get("profile"),
        "status": "passed" if failed_count == 0 else "failed",
        "scope": metadata_scope(candidate_meta, problem_names)
        | {
            "baseline_cases_root": baseline_meta.get("cases_root"),
            "excluded_problems": load_excluded_problems(args.excluded_manifest),
        },
        "fairness": {
            "baseline_grammar_config": baseline_meta.get("grammar_config"),
            "candidate_base_grammar_config": candidate_meta.get("base_grammar_config"),
            "candidate_generated_config_count": len(candidate_meta.get("generated_grammar_configs", {}))
            if isinstance(candidate_meta.get("generated_grammar_configs"), dict)
            else 0,
        },
        "raw_evidence": {
            "baseline_summary": str(args.baseline),
            "candidate_summary": str(args.candidate),
            "comparison": str(args.comparison),
        },
        "thresholds": comparison.get("thresholds", {}),
        "aggregate": {
            "baseline_runs": baseline_aggregate.get("runs"),
            "baseline_ok_runs": baseline_aggregate.get("ok_runs"),
            "candidate_runs": candidate_aggregate.get("runs"),
            "candidate_ok_runs": candidate_aggregate.get("ok_runs"),
            "failed_problem_count": failed_count,
            "max_median_total_ratio": max_numeric(problems, "median_total_ratio"),
            "min_median_total_ratio": min_numeric(problems, "median_total_ratio"),
            "max_p90_total_ratio": max_numeric(problems, "p90_total_ratio"),
            "min_p90_total_ratio": min_numeric(problems, "p90_total_ratio"),
            "max_gpu_kernel_ratio": max_numeric(problems, "gpu_kernel_ratio"),
            "min_gpu_kernel_ratio": min_numeric(problems, "gpu_kernel_ratio"),
        },
        "problems": {str(row["problem"]): compact_problem(row) for row in problems},
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a compact manifest from PSB comparison artifacts.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--format-version", default="psb-comparison-manifest")
    parser.add_argument("--generated-at", default="2026-07-10")
    parser.add_argument("--problems", help="optional comma-separated problem subset for the manifest")
    parser.add_argument("--excluded-manifest", type=Path, help="optional fixture materialization manifest with exclusions")
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PSB_MANIFEST {args.out}")
    print(f"PSB_MANIFEST_STATUS {manifest['status']}")
    print(f"PSB_MANIFEST_PROBLEMS {len(manifest['problems'])}")
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
