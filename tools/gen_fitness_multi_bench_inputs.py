#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def typed_int(v: int) -> Dict[str, Any]:
    return {"type": "int", "value": int(v)}


def typed_bool(v: bool) -> Dict[str, Any]:
    return {"type": "bool", "value": bool(v)}


def make_pass_program() -> Dict[str, Any]:
    return {
        "n_locals": 1,
        "consts": [typed_int(1)],
        "code": [
            {"op": "LOAD", "a": 0, "b": None},
            {"op": "PUSH_CONST", "a": 0, "b": None},
            {"op": "ADD", "a": None, "b": None},
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate canonical bytecode_program_inputs JSON for fitness multi-bench style workloads."
    )
    parser.add_argument("--program-count", type=int, default=4096)
    parser.add_argument("--cases-per-program", type=int, default=1024)
    parser.add_argument("--pass-programs", type=int, default=2048)
    parser.add_argument("--fail-programs", type=int, default=1024)
    parser.add_argument("--timeout-programs", type=int, default=1024)
    parser.add_argument("--fuel", type=int, default=64)
    parser.add_argument("--out", default="data/fixtures/fitness_multi_bench_inputs.json")
    args = parser.parse_args()

    if (
        args.program_count <= 0
        or args.cases_per_program <= 0
        or args.pass_programs < 0
        or args.fail_programs < 0
        or args.timeout_programs < 0
        or args.fuel <= 0
    ):
        raise SystemExit("invalid non-positive arguments")
    if args.pass_programs + args.fail_programs + args.timeout_programs != args.program_count:
        raise SystemExit("bucket counts must sum to program_count")

    pass_p = make_pass_program()
    fail_p = make_fail_program()
    timeout_p = make_timeout_program()

    programs: List[Dict[str, Any]] = []
    programs.extend([pass_p] * args.pass_programs)
    programs.extend([fail_p] * args.fail_programs)
    programs.extend([timeout_p] * args.timeout_programs)

    shared_cases: List[List[Dict[str, Any]]] = []
    shared_answer: List[Dict[str, Any]] = []
    for i in range(args.cases_per_program):
        shared_cases.append([{"idx": 0, "value": typed_int(i)}])
        # Pass program oracle is LOAD(0)+1
        shared_answer.append(typed_int(i + 1))

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
            "generator": "tools/gen_fitness_multi_bench_inputs.py",
            "program_count": args.program_count,
            "cases_per_program": args.cases_per_program,
            "pass_programs": args.pass_programs,
            "fail_programs": args.fail_programs,
            "timeout_programs": args.timeout_programs,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

