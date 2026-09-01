#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .convert_psb import (
    FORMAT_CURRENT,
    build_train_test_payloads,
    resolve_problem_files,
    suite_config,
    write_json,
    _load_schema_overrides,
)


def parse_csv(text: str | None) -> List[str] | None:
    if text is None or not text.strip():
        return None
    out = [item.strip() for item in text.split(",") if item.strip()]
    return out or None


def schema_override_path(schema_root: Path, suite: str, problem: str, format_version: str) -> Path | None:
    if format_version != FORMAT_CURRENT:
        return None
    candidate = schema_root / suite / f"{problem}.json"
    return candidate if candidate.exists() else None


def categorize_error(message: str) -> str:
    lowered = message.lower()
    if "multi-output" in lowered:
        return "multi_output"
    if "not enough random rows" in lowered:
        return "insufficient_random_rows"
    if "only empty lists" in lowered or "provide an explicit schema" in lowered:
        return "schema_required"
    if "mixes int and float list elements" in lowered:
        return "mixed_numeric_list"
    if "none" in lowered:
        return "none_value"
    if "missing" in lowered and "field" in lowered:
        return "schema_mismatch"
    return "conversion_error"


def conversion_attempts(args: argparse.Namespace) -> List[tuple[int, int, str]]:
    attempts = [(args.n_train, args.n_test, "primary")]
    if args.fallback_n_train is not None or args.fallback_n_test is not None:
        fallback_train = args.fallback_n_train if args.fallback_n_train is not None else args.n_train
        fallback_test = args.fallback_n_test if args.fallback_n_test is not None else args.n_test
        if (fallback_train, fallback_test) != (args.n_train, args.n_test):
            attempts.append((fallback_train, fallback_test, "fallback"))
    return attempts


def materialize_one(args: argparse.Namespace, problem: str) -> Dict[str, Any]:
    config = suite_config(args.suite)
    record: Dict[str, Any] = {
        "problem": problem,
        "status": "failed",
        "format_version": args.format_version,
    }
    try:
        resolved_problem, edge_file, random_file = resolve_problem_files(
            suite=config.label,
            problem=problem,
            datasets_root=args.datasets_root,
            edge_file=None,
            random_file=None,
        )
        schema_json = schema_override_path(args.schema_root, config.label, resolved_problem, args.format_version)
        schema_overrides = _load_schema_overrides(schema_json)
        source = {"suite": config.label, "problem": resolved_problem}
        last_error = ""
        last_category = "conversion_error"
        for n_train, n_test, attempt_kind in conversion_attempts(args):
            try:
                train_payload, test_payload, field_schemas, type_counts = build_train_test_payloads(
                    edge_file=edge_file,
                    random_file=random_file,
                    n_train=n_train,
                    n_test=n_test,
                    seed=args.seed,
                    source=source,
                    format_version=args.format_version,
                    schema_overrides=schema_overrides,
                )
            except ValueError as exc:
                last_error = str(exc)
                last_category = categorize_error(last_error)
                if last_category == "insufficient_random_rows" and attempt_kind == "primary":
                    continue
                break

            train_path = args.out_dir / f"{resolved_problem}.train.json"
            test_path = args.out_dir / f"{resolved_problem}.test.json"
            summary_path = args.out_dir / "_summaries" / f"{resolved_problem}.summary.json"
            write_json(train_path, train_payload)
            write_json(test_path, test_payload)
            schema_hash = train_payload.get("source", {}).get("schema_hash", "")
            summary = {
                "ok": True,
                "format_version": args.format_version,
                "suite": config.label,
                "problem": resolved_problem,
                "train_cases": len(train_payload["cases"]),
                "test_cases": len(test_payload["cases"]),
                "type_counts": type_counts,
                "out_train": str(train_path),
                "out_test": str(test_path),
                "field_schemas": field_schemas,
                "schema_hash": schema_hash,
                "schema_json": str(schema_json or ""),
                "attempt": attempt_kind,
            }
            write_json(summary_path, summary)
            record.update(
                {
                    "status": "ok",
                    "attempt": attempt_kind,
                    "train_cases": len(train_payload["cases"]),
                    "test_cases": len(test_payload["cases"]),
                    "out_train": str(train_path),
                    "out_test": str(test_path),
                    "summary_json": str(summary_path),
                    "schema_hash": schema_hash,
                    "schema_json": str(schema_json or ""),
                    "field_schemas": field_schemas,
                    "type_counts": type_counts,
                }
            )
            return record

        record.update({"error_category": last_category, "error": last_error})
        return record
    except Exception as exc:
        record.update({"error_category": categorize_error(str(exc)), "error": str(exc)})
        return record


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_problems(suite: str, explicit: Iterable[str] | None) -> List[str]:
    config = suite_config(suite)
    if explicit is None:
        return list(config.problems)
    valid = set(config.problems)
    out = []
    for problem in explicit:
        if problem not in valid:
            raise ValueError(f"unknown {config.label.upper()} problem: {problem}")
        out.append(problem)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize PSB train/test fixtures and a support manifest.")
    parser.add_argument("--suite", required=True, choices=("psb1", "psb2"))
    parser.add_argument("--format-version", default=FORMAT_CURRENT, choices=(FORMAT_CURRENT,))
    parser.add_argument("--datasets-root", type=Path, default=None)
    parser.add_argument("--schema-root", type=Path, default=Path("configs/psb_schemas"))
    parser.add_argument("--problems", help="comma-separated problem subset; omit for the whole suite")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--fallback-n-train", type=int)
    parser.add_argument("--fallback-n-test", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest-json", type=Path)
    args = parser.parse_args()

    config = suite_config(args.suite)
    if args.datasets_root is None:
        args.datasets_root = Path(config.default_datasets_root)

    problems = selected_problems(config.label, parse_csv(args.problems))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [materialize_one(args, problem) for problem in problems]
    ok_records = [row for row in records if row.get("status") == "ok"]
    failed_records = [row for row in records if row.get("status") != "ok"]
    exclusions = {
        str(row["problem"]): {
            "category": row.get("error_category", "conversion_error"),
            "error": row.get("error", ""),
        }
        for row in failed_records
    }
    manifest = {
        "format_version": "psb-fixtures-manifest",
        "suite": config.label,
        "fixture_format_version": args.format_version,
        "datasets_root": str(args.datasets_root),
        "schema_root": str(args.schema_root),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "requested_n_train": args.n_train,
        "requested_n_test": args.n_test,
        "fallback_n_train": args.fallback_n_train,
        "fallback_n_test": args.fallback_n_test,
        "problem_count": len(records),
        "ok_count": len(ok_records),
        "failed_count": len(failed_records),
        "ok_problems": [str(row["problem"]) for row in ok_records],
        "excluded_problems": exclusions,
        "problems": {str(row["problem"]): row for row in records},
    }
    manifest_path = args.manifest_json or (args.out_dir / "manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"PSB_FIXTURES_SUITE {config.label}")
    print(f"PSB_FIXTURES_FORMAT {args.format_version}")
    print(f"PSB_FIXTURES_OK {len(ok_records)}")
    print(f"PSB_FIXTURES_FAILED {len(failed_records)}")
    print(f"PSB_FIXTURES_MANIFEST {manifest_path}")
    return 0 if ok_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
