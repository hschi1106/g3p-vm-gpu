#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def number_list(payload: Dict[str, Any], key: str) -> List[float]:
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("run JSON missing timing object")
    values = timing.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"run JSON missing non-empty timing.{key}")
    out = []
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"timing.{key} contains non-finite value")
        out.append(float(value))
    return out


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def metric_summary(values: List[float]) -> Dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.9),
        "sum_ms": sum(values),
    }


def ratio(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return candidate / baseline


def best_fitness(payload: Dict[str, Any]) -> float | None:
    final = payload.get("final")
    if isinstance(final, dict) and isinstance(final.get("best_fitness"), (int, float)):
      return float(final["best_fitness"])
    history = payload.get("history")
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict) and isinstance(last.get("best_fitness"), (int, float)):
            return float(last["best_fitness"])
    return None


def comparable_meta(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    baseline_meta = baseline.get("meta", {})
    candidate_meta = candidate.get("meta", {})
    if not isinstance(baseline_meta, dict) or not isinstance(candidate_meta, dict):
        raise ValueError("run JSON meta must be objects")
    keys = [
        "cases_path",
        "population_size",
        "generations",
        "selection",
        "crossover_method",
        "eval_engine",
        "reproduction_backend",
        "repro_overlap",
        "skip_final_eval",
        "retain_final_population",
        "gpu_blocksize",
        "seed",
    ]
    fields = {}
    mismatches = {}
    default_false = {"skip_final_eval", "retain_final_population", "repro_overlap"}
    for key in keys:
        b = baseline_meta.get(key)
        c = candidate_meta.get(key)
        if key in default_false:
            b = bool(b) if b is not None else False
            c = bool(c) if c is not None else False
        fields[key] = {"baseline": b, "candidate": c}
        if b != c:
            mismatches[key] = {"baseline": b, "candidate": c}
    return {
        "fields": fields,
        "mismatches": mismatches,
        "baseline_grammar_config": baseline_meta.get("grammar_config"),
        "candidate_grammar_config": candidate_meta.get("grammar_config"),
    }


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    metric_keys = [
        "generation_total_ms",
        "generation_eval_ms",
        "generation_repro_ms",
        "generation_repro_decode_ms",
        "generation_repro_preprocess_ms",
        "generation_repro_pack_ms",
        "generation_gpu_eval_call_ms",
        "generation_gpu_eval_kernel_ms",
    ]

    metrics: Dict[str, Any] = {}
    for key in metric_keys:
        b = metric_summary(number_list(baseline, key))
        c = metric_summary(number_list(candidate, key))
        metrics[key] = {
            "baseline": b,
            "candidate": c,
            "ratio": {
                "median": ratio(c["median_ms"], b["median_ms"]),
                "p90": ratio(c["p90_ms"], b["p90_ms"]),
                "sum": ratio(c["sum_ms"], b["sum_ms"]),
            },
        }

    fairness = comparable_meta(baseline, candidate)
    total = metrics["generation_total_ms"]["ratio"]
    kernel = metrics["generation_gpu_eval_kernel_ms"]["ratio"]
    speed_failures = []
    if fairness["mismatches"]:
        speed_failures.append("metadata_mismatch")
    if total["median"] is None or total["median"] > args.max_median_total_ratio:
        speed_failures.append("median_total_ratio")
    if total["p90"] is None or total["p90"] > args.max_p90_total_ratio:
        speed_failures.append("p90_total_ratio")
    if kernel["median"] is None or kernel["median"] > args.max_gpu_kernel_ratio:
        speed_failures.append("gpu_kernel_ratio")

    baseline_best = best_fitness(baseline)
    candidate_best = best_fitness(candidate)
    quality_delta = None
    if baseline_best is not None and candidate_best is not None:
        quality_delta = candidate_best - baseline_best

    return {
        "format_version": args.format_version,
        "generated_at": args.generated_at,
        "status": "passed" if not speed_failures else "failed",
        "thresholds": {
            "max_median_total_ratio": args.max_median_total_ratio,
            "max_p90_total_ratio": args.max_p90_total_ratio,
            "max_gpu_kernel_ratio": args.max_gpu_kernel_ratio,
        },
        "raw_evidence": {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
        },
        "fairness": fairness,
        "speed_failures": speed_failures,
        "metrics": metrics,
        "quality_observation": {
            "gate": "observation_only_single_seed_simple_exp",
            "baseline_final_best_fitness": baseline_best,
            "candidate_final_best_fitness": candidate_best,
            "candidate_minus_baseline": quality_delta,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a compact simple_exp speed comparison manifest.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--format-version", default="simple-exp-speed-manifest")
    parser.add_argument("--generated-at", default="2026-07-11")
    parser.add_argument("--max-median-total-ratio", type=float, default=1.15)
    parser.add_argument("--max-p90-total-ratio", type=float, default=1.15)
    parser.add_argument("--max-gpu-kernel-ratio", type=float, default=1.20)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SIMPLE_EXP_MANIFEST {args.out}")
    print(f"SIMPLE_EXP_MANIFEST_STATUS {manifest['status']}")
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
