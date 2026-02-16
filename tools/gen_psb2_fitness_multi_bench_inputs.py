#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import psb2  # type: ignore
except ImportError:
    psb2 = None


def typed_int(v: int) -> Dict[str, Any]:
    return {"type": "int", "value": int(v)}


def typed_float(v: float) -> Dict[str, Any]:
    return {"type": "float", "value": float(v)}


def typed_bool(v: bool) -> Dict[str, Any]:
    return {"type": "bool", "value": bool(v)}


def typed_none() -> Dict[str, Any]:
    return {"type": "none"}


def make_pass_program() -> Dict[str, Any]:
    return {
        "n_locals": 2,
        "consts": [],
        "code": [
            {"op": "LOAD", "a": 0, "b": None},
            {"op": "RETURN", "a": None, "b": None},
        ],
    }


def make_fail_program() -> Dict[str, Any]:
    return {
        "n_locals": 0,
        "consts": [typed_bool(True)],
        "code": [
            {"op": "PUSH_CONST", "a": 0, "b": None},
            {"op": "NEG", "a": None, "b": None},
        ],
    }


def make_timeout_program() -> Dict[str, Any]:
    return {
        "n_locals": 0,
        "consts": [],
        "code": [
            {"op": "JMP", "a": 0, "b": None},
        ],
    }


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_task_json_files(psb2_root: Path, task_name: str) -> List[Path]:
    norm = normalize_name(task_name)
    candidates: List[Path] = []
    for path in psb2_root.rglob("*.json"):
        if norm in normalize_name(path.stem) or norm in normalize_name(path.as_posix()):
            candidates.append(path)
    return sorted(candidates)


def scalarize(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return typed_none()
    if isinstance(value, bool):
        return typed_bool(value)
    if isinstance(value, int):
        return typed_int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return typed_int(int(value))
        return typed_float(value)
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in ("none", "null"):
            return typed_none()
        if lower in ("true", "false"):
            return typed_bool(lower == "true")
        try:
            i = int(text)
            return typed_int(i)
        except ValueError:
            pass
        try:
            f = float(text)
            if math.isfinite(f):
                if f.is_integer():
                    return typed_int(int(f))
                return typed_float(f)
        except ValueError:
            return None
        return None
    if isinstance(value, dict):
        for k in ("output", "outputs", "label", "target", "result", "value", "y"):
            if k in value:
                out = scalarize(value[k])
                if out is not None:
                    return out
        for v in value.values():
            out = scalarize(v)
            if out is not None:
                return out
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            out = scalarize(item)
            if out is not None:
                return out
        return None
    return None


def iter_examples(node: Any) -> Iterator[Tuple[Any, Any]]:
    if isinstance(node, dict):
        io_key_pairs = (
            ("input", "output"),
            ("inputs", "output"),
            ("input", "outputs"),
            ("in", "out"),
            ("x", "y"),
            ("features", "label"),
            ("arguments", "output"),
        )
        for in_key, out_key in io_key_pairs:
            if in_key in node and out_key in node:
                yield (node[in_key], node[out_key])

        for key in ("train", "test", "examples", "cases", "data", "dataset"):
            if key in node:
                yield from iter_examples(node[key])

        for value in node.values():
            if isinstance(value, (dict, list, tuple)):
                yield from iter_examples(value)
        return

    if isinstance(node, list):
        for item in node:
            yield from iter_examples(item)


def extract_cases_from_file(path: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set[Tuple[str, str]] = set()
    for raw_in, raw_out in iter_examples(payload):
        ans = scalarize(raw_out)
        inp = scalarize(raw_in)
        if ans is None:
            continue
        if inp is None:
            inp = typed_int(0)
        key = (json.dumps(inp, sort_keys=True), json.dumps(ans, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append((inp, ans))
    return out


def collect_task_cases(psb2_root: Path, task_name: str) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Path]]:
    files = find_task_json_files(psb2_root, task_name)
    samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for path in files:
        samples.extend(extract_cases_from_file(path))
    return samples, files


def extract_from_psb2_examples(examples: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set[Tuple[str, str]] = set()
    for row in examples:
        if not isinstance(row, dict):
            continue

        answer: Optional[Dict[str, Any]] = None
        for key in sorted(row.keys()):
            if key.lower().startswith("output"):
                answer = scalarize(row[key])
                if answer is not None:
                    break
        if answer is None:
            answer = scalarize(row.get("output"))
        if answer is None:
            continue

        inp: Optional[Dict[str, Any]] = None
        for key in sorted(row.keys()):
            if key.lower().startswith("input"):
                inp = scalarize(row[key])
                if inp is not None:
                    break
        if inp is None:
            inp = typed_int(0)

        sig = (json.dumps(inp, sort_keys=True), json.dumps(answer, sort_keys=True))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((inp, answer))
    return out


def collect_task_cases_via_psb2(
    psb2_root: Path,
    task_name: str,
    train_count: int,
    test_count: int,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Optional[str]]:
    if psb2 is None:
        return [], "psb2 module is not installed"
    try:
        train_data, test_data = psb2.fetch_examples(str(psb2_root), task_name, train_count, test_count)
    except Exception as exc:
        return [], str(exc)
    merged: List[Dict[str, Any]] = []
    merged.extend(train_data)
    merged.extend(test_data)
    return extract_from_psb2_examples(merged), None


def build_shared_cases(
    task_to_samples: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]],
    cases_per_program: int,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, int]]:
    tasks = list(task_to_samples.keys())
    if not tasks:
        raise SystemExit("at least one task is required")

    for task_name, samples in task_to_samples.items():
        if not samples:
            raise SystemExit(f"no usable scalar examples found for task: {task_name}")

    shared_cases: List[List[Dict[str, Any]]] = []
    shared_answer: List[Dict[str, Any]] = []
    used_per_task = {task: 0 for task in tasks}
    pick_idx = {task: 0 for task in tasks}

    for i in range(cases_per_program):
        task = tasks[i % len(tasks)]
        samples = task_to_samples[task]
        sample = samples[pick_idx[task] % len(samples)]
        pick_idx[task] += 1
        used_per_task[task] += 1
        input_scalar, answer_scalar = sample

        shared_cases.append(
            [
                {"idx": 0, "value": answer_scalar},
                {"idx": 1, "value": input_scalar},
            ]
        )
        shared_answer.append(answer_scalar)

    return shared_cases, shared_answer, used_per_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fitness_multi_bench_inputs from PSB2 bouncing-balls dataset."
    )
    parser.add_argument("--psb2-root", required=True, help="Path to PSB2 dataset root.")
    parser.add_argument("--out", default="data/fixtures/fitness_multi_bench_inputs_psb2.json")
    parser.add_argument("--task-train-count", type=int, default=200)
    parser.add_argument("--task-test-count", type=int, default=2000)
    parser.add_argument(
        "--require-psb2-fetch",
        action="store_true",
        help="Fail unless task samples come from psb2.fetch_examples (disable JSON fallback).",
    )
    parser.add_argument("--program-count", type=int, default=4096)
    parser.add_argument("--cases-per-program", type=int, default=1024)
    parser.add_argument("--pass-programs", type=int, default=2048)
    parser.add_argument("--fail-programs", type=int, default=1024)
    parser.add_argument("--timeout-programs", type=int, default=1024)
    parser.add_argument("--fuel", type=int, default=64)
    args = parser.parse_args()

    if (
        args.program_count <= 0
        or args.cases_per_program <= 0
        or args.pass_programs < 0
        or args.fail_programs < 0
        or args.timeout_programs < 0
        or args.fuel <= 0
        or args.task_train_count <= 0
        or args.task_test_count <= 0
    ):
        raise SystemExit("invalid non-positive arguments")
    if args.pass_programs + args.fail_programs + args.timeout_programs != args.program_count:
        raise SystemExit("bucket counts must sum to program_count")

    psb2_root = Path(args.psb2_root)
    if psb2_root.exists() and not psb2_root.is_dir():
        raise SystemExit(f"invalid --psb2-root: {psb2_root}")
    psb2_root.mkdir(parents=True, exist_ok=True)

    task_names = ("bouncing-balls",)
    task_samples: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    task_files: Dict[str, List[Path]] = {}
    source_mode: Dict[str, str] = {}
    for task in task_names:
        samples, fetch_err = collect_task_cases_via_psb2(
            psb2_root=psb2_root,
            task_name=task,
            train_count=args.task_train_count,
            test_count=args.task_test_count,
        )
        if samples:
            task_samples[task] = samples
            task_files[task] = []
            source_mode[task] = "psb2.fetch_examples"
            continue

        if args.require_psb2_fetch:
            msg = fetch_err or "unknown fetch error"
            raise SystemExit(f"psb2.fetch_examples failed for task {task}: {msg}")

        samples, files = collect_task_cases(psb2_root, task)
        task_samples[task] = samples
        task_files[task] = files
        source_mode[task] = "json_scan_fallback"
        if not files:
            raise SystemExit(f"task not found under psb2 root: {task}")

    shared_cases, shared_answer, used_per_task = build_shared_cases(task_samples, args.cases_per_program)

    pass_p = make_pass_program()
    fail_p = make_fail_program()
    timeout_p = make_timeout_program()
    programs: List[Dict[str, Any]] = []
    programs.extend([pass_p] * args.pass_programs)
    programs.extend([fail_p] * args.fail_programs)
    programs.extend([timeout_p] * args.timeout_programs)

    expected_fitness = [args.cases_per_program] * args.pass_programs + [
        -10 * args.cases_per_program
    ] * (args.fail_programs + args.timeout_programs)

    payload: Dict[str, Any] = {
        "bytecode_program_inputs": {
            "format_version": "bytecode-json-v0.1",
            "fuel": args.fuel,
            "programs": programs,
            "shared_cases": shared_cases,
            "shared_answer": shared_answer,
        },
        "expected_fitness": expected_fitness,
        "meta": {
            "generator": "tools/gen_psb2_fitness_multi_bench_inputs.py",
            "program_count": args.program_count,
            "cases_per_program": args.cases_per_program,
            "pass_programs": args.pass_programs,
            "fail_programs": args.fail_programs,
            "timeout_programs": args.timeout_programs,
            "tasks": list(task_names),
            "cases_used_per_task": used_per_task,
            "task_train_count": args.task_train_count,
            "task_test_count": args.task_test_count,
            "source_mode": source_mode,
            "source_files": {k: [str(p) for p in v] for k, v in task_files.items()},
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
