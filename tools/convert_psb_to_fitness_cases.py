#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PSB1_PROBLEMS: tuple[str, ...] = (
    "checksum",
    "collatz-numbers",
    "compare-string-lengths",
    "count-odds",
    "digits",
    "double-letters",
    "even-squares",
    "for-loop-index",
    "grade",
    "last-index-of-zero",
    "median",
    "mirror-image",
    "negative-to-zero",
    "number-io",
    "pig-latin",
    "replace-space-with-newline",
    "scrabble-score",
    "small-or-large",
    "smallest",
    "string-differences",
    "string-lengths-backwards",
    "sum-of-squares",
    "super-anagrams",
    "syllables",
    "vector-average",
    "vectors-summed",
    "wallis-pi",
    "word-stats",
    "x-word-lines",
)

PSB2_PROBLEMS: tuple[str, ...] = (
    "basement",
    "bouncing-balls",
    "bowling",
    "camel-case",
    "coin-sums",
    "cut-vector",
    "dice-game",
    "find-pair",
    "fizz-buzz",
    "fuel-cost",
    "gcd",
    "indices-of-substring",
    "leaders",
    "luhn",
    "mastermind",
    "middle-character",
    "paired-digits",
    "shopping-list",
    "snow-day",
    "solve-boolean",
    "spin-words",
    "square-digits",
    "substitution-cipher",
    "twitter",
    "vector-distance",
)

RUNTIME_VALUE_TYPES = {"none", "bool", "int", "float", "string", "num_list", "string_list"}


@dataclass(frozen=True)
class PsbSuiteConfig:
    label: str
    problems: tuple[str, ...]
    default_datasets_root: str


PSB1_CONFIG = PsbSuiteConfig(
    label="psb1",
    problems=PSB1_PROBLEMS,
    default_datasets_root="data/psb1_datasets",
)

PSB2_CONFIG = PsbSuiteConfig(
    label="psb2",
    problems=PSB2_PROBLEMS,
    default_datasets_root="data/psb2_datasets",
)


def suite_config(suite: str) -> PsbSuiteConfig:
    normalized = suite.strip().lower()
    if normalized == "psb1":
        return PSB1_CONFIG
    if normalized == "psb2":
        return PSB2_CONFIG
    raise ValueError(f"unsupported PSB suite: {suite}")


def resolve_problem_files(
    *,
    suite: str,
    problem: str,
    datasets_root: Path,
    edge_file: Path | None,
    random_file: Path | None,
) -> tuple[str, Path, Path]:
    config = suite_config(suite)
    problem = problem.strip()
    if problem:
        if edge_file is not None or random_file is not None:
            raise ValueError("--problem cannot be combined with --edge-file/--random-file")
        if problem not in config.problems:
            raise ValueError(
                f"unknown {config.label.upper()} problem: {problem}. "
                + "Use --problem with one of: "
                + ", ".join(config.problems)
            )
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{ln}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"each JSONL row must be object at {path}:{ln}")
            rows.append(obj)
    return rows


def sort_io_keys(keys: Iterable[str], prefix: str) -> List[str]:
    out: List[Tuple[int, str]] = []
    for k in keys:
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix) :]
        if not suffix.isdigit():
            continue
        out.append((int(suffix), k))
    out.sort(key=lambda x: x[0])
    return [k for _, k in out]


def _typed_scalar(v: Any, type_counts: Dict[str, int]) -> Dict[str, Any]:
    if v is None:
        type_counts["none"] = type_counts.get("none", 0) + 1
        return {"type": "none", "value": None}
    if isinstance(v, bool):
        type_counts["bool"] = type_counts.get("bool", 0) + 1
        return {"type": "bool", "value": v}
    if isinstance(v, int):
        type_counts["int"] = type_counts.get("int", 0) + 1
        return {"type": "int", "value": v}
    if isinstance(v, float):
        type_counts["float"] = type_counts.get("float", 0) + 1
        return {"type": "float", "value": v}
    if isinstance(v, str):
        type_counts["string"] = type_counts.get("string", 0) + 1
        return {"type": "string", "value": v}
    raise ValueError(f"unsupported JSON value type: {type(v).__name__}")


def _list_element_kind(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        raise ValueError("typed lists do not support bool/none elements")
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "string"
    raise ValueError(f"typed lists do not support {type(v).__name__} elements")


def _infer_field_schema(rows: List[Dict[str, Any]], key: str) -> str:
    values = []
    for row in rows:
        if key not in row:
            raise ValueError(f"row missing required field: {key}")
        values.append(row[key])

    has_list = any(isinstance(v, list) for v in values)
    if not has_list:
        for v in values:
            _typed_scalar(v, {})
        return "scalar"

    if not all(isinstance(v, list) for v in values):
        raise ValueError(f"field {key} mixes list and non-list values")

    elem_kind: str | None = None
    for v in values:
        for item in v:
            item_kind = _list_element_kind(item)
            if elem_kind is None:
                elem_kind = item_kind
            elif elem_kind != item_kind:
                raise ValueError(f"field {key} mixes numeric and string list elements")
    if elem_kind is None:
        raise ValueError(f"field {key} has only empty lists; cannot infer NumList vs StringList")
    return "num_list" if elem_kind == "num" else "string_list"


def _typed_with_schema(v: Any, schema: str, type_counts: Dict[str, int]) -> Dict[str, Any]:
    if schema == "scalar":
        return _typed_scalar(v, type_counts)
    if schema == "num_list":
        if not isinstance(v, list):
            raise ValueError("num_list schema requires list value")
        out: List[int | float] = []
        for item in v:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise ValueError("num_list elements must be int or float")
            out.append(item)
        type_counts["num_list"] = type_counts.get("num_list", 0) + 1
        return {"type": "num_list", "value": out}
    if schema == "string_list":
        if not isinstance(v, list):
            raise ValueError("string_list schema requires list value")
        out_s: List[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("string_list elements must be strings")
            out_s.append(item)
        type_counts["string_list"] = type_counts.get("string_list", 0) + 1
        return {"type": "string_list", "value": out_s}
    raise ValueError(f"unknown schema: {schema}")


def infer_dataset_schema(rows: List[Dict[str, Any]]) -> Tuple[List[str], str, Dict[str, str]]:
    if not rows:
        raise ValueError("cannot infer schema from empty rows")
    input_keys = sort_io_keys({k for row in rows for k in sort_io_keys(row.keys(), "input")}, "input")
    if not input_keys:
        raise ValueError("rows have no inputK fields")
    output_keys = sort_io_keys({k for row in rows for k in sort_io_keys(row.keys(), "output")}, "output")
    if not output_keys:
        raise ValueError("rows have no outputK fields")
    if len(output_keys) != 1:
        raise ValueError("multi-output rows are not runtime-compatible yet; do not encode them as list values")

    schemas: Dict[str, str] = {}
    for key in input_keys + output_keys:
        schemas[key] = _infer_field_schema(rows, key)
    return input_keys, output_keys[0], schemas


def convert_row(
    row: Dict[str, Any],
    input_keys: List[str],
    output_key: str,
    schemas: Dict[str, str],
    type_counts: Dict[str, int],
) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for k in input_keys:
        inputs[k] = _typed_with_schema(row[k], schemas[k], type_counts)
    expected = _typed_with_schema(row[output_key], schemas[output_key], type_counts)
    return {"inputs": inputs, "expected": expected}


def sample_train_test(
    edge_rows: List[Dict[str, Any]],
    random_rows: List[Dict[str, Any]],
    n_train: int,
    n_test: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be > 0")
    if not edge_rows and not random_rows:
        raise ValueError("both edge and random datasets are empty")

    rng = random.Random(seed)

    if n_train <= len(edge_rows):
        train = rng.sample(edge_rows, n_train)
    else:
        need_random = n_train - len(edge_rows)
        if need_random > len(random_rows):
            raise ValueError(f"not enough random rows for n_train={n_train} (need {need_random}, have {len(random_rows)})")
        train = list(edge_rows) + rng.sample(random_rows, need_random)

    if n_test > len(random_rows):
        raise ValueError(f"not enough random rows for n_test={n_test} (have {len(random_rows)})")
    test = rng.sample(random_rows, n_test)
    return train, test


def build_cases_payload(cases: List[Dict[str, Any]], source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format_version": "fitness-cases-v1",
        "source": source,
        "cases": cases,
    }


def build_train_test_payloads(
    edge_file: Path,
    random_file: Path,
    n_train: int,
    n_test: int,
    seed: int,
    source: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], Dict[str, int]]:
    edge_rows = load_jsonl(edge_file)
    random_rows = load_jsonl(random_file)
    input_keys, output_key, field_schemas = infer_dataset_schema(edge_rows + random_rows)
    train_rows, test_rows = sample_train_test(edge_rows, random_rows, n_train, n_test, seed)

    type_counts: Dict[str, int] = {}
    train_cases = [convert_row(r, input_keys, output_key, field_schemas, type_counts) for r in train_rows]
    test_cases = [convert_row(r, input_keys, output_key, field_schemas, type_counts) for r in test_rows]

    source_with_schema = source | {
        "edge_file": str(edge_file),
        "random_file": str(random_file),
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "field_schemas": field_schemas,
    }
    train_payload = build_cases_payload(train_cases, source=source_with_schema | {"split": "train"})
    test_payload = build_cases_payload(test_cases, source=source_with_schema | {"split": "test"})
    return train_payload, test_payload, field_schemas, type_counts


def unsupported_runtime_types(type_counts: Dict[str, int]) -> List[str]:
    return sorted(t for t in type_counts.keys() if t not in RUNTIME_VALUE_TYPES)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PSB1/PSB2 JSONL files to fitness-cases-v1 JSON.")
    parser.add_argument("--suite", required=True, choices=("psb1", "psb2"), help="PSB dataset suite.")
    parser.add_argument("--problem", default="", help="Problem name under --datasets-root.")
    parser.add_argument("--datasets-root", default="", help="Dataset root; defaults to the selected suite's data root.")
    parser.add_argument("--edge-file", default="", help="Explicit edge JSONL file.")
    parser.add_argument("--random-file", default="", help="Explicit random JSONL file.")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output path for training fitness-cases-v1 JSON.")
    parser.add_argument("--out-test", default="", help="Optional output path for test fitness-cases-v1 JSON.")
    parser.add_argument("--summary-json", default="", help="Optional output path for conversion summary JSON.")
    args = parser.parse_args()

    config = suite_config(args.suite)
    problem, edge_file, random_file = resolve_problem_files(
        suite=config.label,
        problem=str(args.problem or ""),
        datasets_root=Path(args.datasets_root or config.default_datasets_root),
        edge_file=Path(args.edge_file) if args.edge_file else None,
        random_file=Path(args.random_file) if args.random_file else None,
    )

    out = Path(args.out)
    out_test = Path(args.out_test) if args.out_test else None
    summary_path = Path(args.summary_json) if args.summary_json else None

    source = {
        "suite": config.label,
        "problem": problem,
    }
    train_payload, test_payload, field_schemas, type_counts = build_train_test_payloads(
        edge_file=edge_file,
        random_file=random_file,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        source=source,
    )

    write_json(out, train_payload)
    if out_test is not None:
        write_json(out_test, test_payload)

    unsupported_types = unsupported_runtime_types(type_counts)
    summary: Dict[str, Any] = {
        "ok": True,
        "suite": config.label,
        "problem": problem,
        "train_cases": len(train_payload["cases"]),
        "test_cases": len(test_payload["cases"]),
        "type_counts": type_counts,
        "runtime_compatible": len(unsupported_types) == 0,
        "unsupported_types": unsupported_types,
        "out_train": str(out),
        "out_test": (str(out_test) if out_test is not None else ""),
        "field_schemas": field_schemas,
    }
    if summary_path is not None:
        write_json(summary_path, summary)

    print(f"CONVERT_SUITE {config.label}")
    print(f"CONVERT_PROBLEM {problem}")
    print(f"CONVERT_OUT {out}")
    if out_test is not None:
        print(f"CONVERT_OUT_TEST {out_test}")
    print(f"CONVERT_RUNTIME_COMPATIBLE {1 if summary['runtime_compatible'] else 0}")
    print(f"CONVERT_UNSUPPORTED_TYPES {','.join(unsupported_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
