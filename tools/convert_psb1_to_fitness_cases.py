#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from convert_psb2_to_fitness_cases import (
    _build_cases_payload,
    _convert_row,
    _infer_dataset_schema,
    _load_jsonl,
    _sample_train_test,
)


def _resolve_input_files(args: argparse.Namespace) -> tuple[str, Path, Path]:
    problem = str(args.problem or "").strip()
    edge_file = Path(args.edge_file) if args.edge_file else None
    random_file = Path(args.random_file) if args.random_file else None

    if problem:
        if edge_file is not None or random_file is not None:
            raise ValueError("--problem cannot be combined with --edge-file/--random-file")
        datasets_root = Path(args.datasets_root)
        edge_file = datasets_root / problem / f"{problem}-edge.json"
        random_file = datasets_root / problem / f"{problem}-random.json"
    else:
        if edge_file is None or random_file is None:
            raise ValueError("provide either --problem or both --edge-file and --random-file")
        problem = edge_file.parent.name or edge_file.stem.replace("-edge", "")

    assert edge_file is not None and random_file is not None
    if not edge_file.exists():
        raise ValueError(f"edge file does not exist: {edge_file}")
    if not random_file.exists():
        raise ValueError(f"random file does not exist: {random_file}")
    return problem, edge_file, random_file


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PSB1 JSONL files to fitness-cases-v1 JSON.")
    parser.add_argument("--problem", default="", help="PSB1 problem name under --datasets-root.")
    parser.add_argument("--datasets-root", default="data/psb1_datasets", help="Root directory containing PSB1 datasets.")
    parser.add_argument("--edge-file", default="", help="Explicit PSB1 edge JSONL file.")
    parser.add_argument("--random-file", default="", help="Explicit PSB1 random JSONL file.")
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output path for training fitness-cases-v1 JSON.")
    parser.add_argument("--out-test", default="", help="Optional output path for test fitness-cases-v1 JSON.")
    parser.add_argument("--summary-json", default="", help="Optional output path for conversion summary JSON.")
    args = parser.parse_args()

    problem, edge_file, random_file = _resolve_input_files(args)
    out = Path(args.out)
    out_test = Path(args.out_test) if args.out_test else None
    summary_path = Path(args.summary_json) if args.summary_json else None

    edge_rows = _load_jsonl(edge_file)
    random_rows = _load_jsonl(random_file)
    input_keys, output_key, field_schemas = _infer_dataset_schema(edge_rows + random_rows)
    train_rows, test_rows = _sample_train_test(edge_rows, random_rows, args.n_train, args.n_test, args.seed)

    type_counts: Dict[str, int] = {}
    train_cases = [_convert_row(r, input_keys, output_key, field_schemas, type_counts) for r in train_rows]
    test_cases = [_convert_row(r, input_keys, output_key, field_schemas, type_counts) for r in test_rows]

    source = {
        "suite": "psb1",
        "problem": problem,
        "edge_file": str(edge_file),
        "random_file": str(random_file),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "field_schemas": field_schemas,
    }
    _write_json(out, _build_cases_payload(train_cases, source=source | {"split": "train"}))
    if out_test is not None:
        _write_json(out_test, _build_cases_payload(test_cases, source=source | {"split": "test"}))

    unsupported_types = sorted(
        t for t in type_counts.keys()
        if t not in {"none", "bool", "int", "float", "string", "num_list", "string_list"}
    )
    summary = {
        "ok": True,
        "suite": "psb1",
        "problem": problem,
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "type_counts": type_counts,
        "runtime_compatible": len(unsupported_types) == 0,
        "unsupported_types": unsupported_types,
        "out_train": str(out),
        "out_test": (str(out_test) if out_test is not None else ""),
    }
    if summary_path is not None:
        _write_json(summary_path, summary)

    print(f"CONVERT_PSB1_PROBLEM {problem}")
    print(f"CONVERT_OUT {out}")
    if out_test is not None:
        print(f"CONVERT_OUT_TEST {out_test}")
    print(f"CONVERT_RUNTIME_COMPATIBLE {1 if summary['runtime_compatible'] else 0}")
    print(f"CONVERT_UNSUPPORTED_TYPES {','.join(unsupported_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
