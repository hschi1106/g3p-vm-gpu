#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _sort_io_keys(keys: Iterable[str], prefix: str) -> List[str]:
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


def _infer_dataset_schema(rows: List[Dict[str, Any]]) -> Tuple[List[str], str, Dict[str, str]]:
    if not rows:
        raise ValueError("cannot infer schema from empty rows")
    input_keys = _sort_io_keys({k for row in rows for k in _sort_io_keys(row.keys(), "input")}, "input")
    if not input_keys:
        raise ValueError("rows have no inputK fields")
    output_keys = _sort_io_keys({k for row in rows for k in _sort_io_keys(row.keys(), "output")}, "output")
    if not output_keys:
        raise ValueError("rows have no outputK fields")
    if len(output_keys) != 1:
        raise ValueError("multi-output rows are not runtime-compatible yet; do not encode them as list values")

    schemas: Dict[str, str] = {}
    for key in input_keys + output_keys:
        schemas[key] = _infer_field_schema(rows, key)
    return input_keys, output_keys[0], schemas


def _convert_row(
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


def _sample_train_test(
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


def _build_cases_payload(cases: List[Dict[str, Any]], source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format_version": "fitness-cases-v1",
        "source": source,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PSB2 JSONL files to fitness-cases-v1 JSON.")
    parser.add_argument("--edge-file", required=True, help="Path to <task>-edge.json (JSONL).")
    parser.add_argument("--random-file", required=True, help="Path to <task>-random.json (JSONL).")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output path for training fitness-cases-v1 JSON.")
    parser.add_argument("--out-test", default="", help="Optional output path for test fitness-cases-v1 JSON.")
    parser.add_argument("--summary-json", default="", help="Optional output path for conversion summary JSON.")
    args = parser.parse_args()

    edge_file = Path(args.edge_file)
    random_file = Path(args.random_file)
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
        "edge_file": str(edge_file),
        "random_file": str(random_file),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "field_schemas": field_schemas,
    }
    train_payload = _build_cases_payload(train_cases, source=source | {"split": "train"})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(train_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if out_test is not None:
        test_payload = _build_cases_payload(test_cases, source=source | {"split": "test"})
        out_test.parent.mkdir(parents=True, exist_ok=True)
        out_test.write_text(json.dumps(test_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    unsupported_types = sorted(t for t in type_counts.keys() if t not in {"none", "bool", "int", "float", "string", "num_list", "string_list"})
    summary = {
        "ok": True,
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "type_counts": type_counts,
        "runtime_compatible": len(unsupported_types) == 0,
        "unsupported_types": unsupported_types,
        "out_train": str(out),
        "out_test": (str(out_test) if out_test is not None else ""),
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"CONVERT_OUT {out}")
    if out_test is not None:
        print(f"CONVERT_OUT_TEST {out_test}")
    print(f"CONVERT_RUNTIME_COMPATIBLE {1 if summary['runtime_compatible'] else 0}")
    print(f"CONVERT_UNSUPPORTED_TYPES {','.join(unsupported_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
