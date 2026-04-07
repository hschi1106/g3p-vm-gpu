#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{ln} row is not object")
            rows.append(obj)
    return rows


def _iter_values(v: Any) -> Iterable[Any]:
    if isinstance(v, dict):
        for x in v.values():
            yield from _iter_values(x)
        return
    if isinstance(v, list):
        for x in v:
            yield from _iter_values(x)
        return
    yield v


def _type_name(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        elem_types = {_type_name(x) for x in v}
        if not elem_types:
            return "empty_list"
        if elem_types.issubset({"int", "float"}):
            return "num_list"
        if elem_types == {"string"}:
            return "string_list"
        return "unsupported_list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _count_io_keys(keys: Iterable[str], prefix: str) -> int:
    n = 0
    for k in keys:
        if k.startswith(prefix) and k[len(prefix) :].isdigit():
            n += 1
    return n


def audit_task(task_dir: Path) -> Dict[str, Any]:
    task = task_dir.name
    edge = task_dir / f"{task}-edge.json"
    rnd = task_dir / f"{task}-random.json"
    if not edge.exists() or not rnd.exists():
        return {"task": task, "status": "missing_files"}

    edge_rows = _load_jsonl(edge)
    rnd_rows = _load_jsonl(rnd)
    sample_rows = edge_rows[: min(200, len(edge_rows))] + rnd_rows[: min(200, len(rnd_rows))]
    if not sample_rows:
        return {"task": task, "status": "empty"}

    max_inputs = 0
    max_outputs = 0
    value_types = Counter()
    has_multi_output = False
    for row in sample_rows:
        in_n = _count_io_keys(row.keys(), "input")
        out_n = _count_io_keys(row.keys(), "output")
        max_inputs = max(max_inputs, in_n)
        max_outputs = max(max_outputs, out_n)
        if out_n != 1:
            has_multi_output = True
        for v in _iter_values(row):
            value_types[_type_name(v)] += 1

    return {
        "task": task,
        "status": "ok",
        "edge_rows": len(edge_rows),
        "random_rows": len(rnd_rows),
        "max_inputs": max_inputs,
        "max_outputs": max_outputs,
        "has_multi_output": has_multi_output,
        "value_types": dict(value_types),
        "runtime_compatible_current": (not has_multi_output)
        and set(value_types.keys()).issubset({"none", "bool", "int", "float", "string", "num_list", "string_list", "empty_list"}),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Audit PSB2 dataset tasks for runtime compatibility planning.")
    p.add_argument("--datasets-root", default="data/psb2_datasets")
    p.add_argument("--out-json", default="logs/psb2_all_tasks/audit.json")
    args = p.parse_args()

    root = Path(args.datasets_root)
    tasks = sorted([x for x in root.iterdir() if x.is_dir()])
    results = [audit_task(t) for t in tasks]
    summary = {
        "tasks_total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "runtime_compatible_current": sum(1 for r in results if r.get("runtime_compatible_current")),
        "multi_output_tasks": sum(1 for r in results if r.get("has_multi_output")),
        "results": results,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"PSB2_AUDIT_JSON {out}")
    print(
        f"PSB2_AUDIT_COUNTS total={summary['tasks_total']} ok={summary['ok']} "
        f"runtime_compatible_current={summary['runtime_compatible_current']} multi_output={summary['multi_output_tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
