#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def encode_value(v: Any) -> Dict[str, Any]:
    if v is None:
        return {"type": "none"}
    if isinstance(v, bool):
        return {"type": "bool", "value": v}
    if isinstance(v, int):
        return {"type": "int", "value": v}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    raise TypeError(f"unsupported value type: {type(v)}")


def make_program() -> Dict[str, Any]:
    # locals[0] = mode, locals[1] = x
    # mode==0: return x + 1                 -> pass
    # mode==1: NEG(True)                    -> TypeError
    # mode==2: infinite loop via JMP 14     -> Timeout
    return {
        "n_locals": 2,
        "consts": [
            encode_value(0),
            encode_value(1),
            encode_value(2),
            encode_value(True),
        ],
        "code": [
            {"op": "LOAD", "a": 0},
            {"op": "PUSH_CONST", "a": 0},
            {"op": "EQ"},
            {"op": "JMP_IF_FALSE", "a": 8},
            {"op": "LOAD", "a": 1},
            {"op": "PUSH_CONST", "a": 1},
            {"op": "ADD"},
            {"op": "RETURN"},
            {"op": "LOAD", "a": 0},
            {"op": "PUSH_CONST", "a": 1},
            {"op": "EQ"},
            {"op": "JMP_IF_FALSE", "a": 14},
            {"op": "PUSH_CONST", "a": 3},
            {"op": "NEG"},
            {"op": "JMP", "a": 14},
        ],
    }


def make_case(mode: int, x: int) -> List[Dict[str, Any]]:
    return [
        {"idx": 0, "value": encode_value(mode)},
        {"idx": 1, "value": encode_value(x)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate batch GPU fixture with deterministic buckets: pass/fail/timeout "
            "using a single program and many input cases."
        )
    )
    parser.add_argument("--out", default="data/fixtures/gpu_batch_cases.json")
    parser.add_argument("--pass-count", type=int, default=2048)
    parser.add_argument("--fail-count", type=int, default=1024)
    parser.add_argument("--timeout-count", type=int, default=1024)
    parser.add_argument("--min-cases", type=int, default=1024)
    parser.add_argument("--fuel", type=int, default=64)
    parser.add_argument("--blocksize", type=int, default=256)
    args = parser.parse_args()

    if args.pass_count < 0 or args.fail_count < 0 or args.timeout_count < 0:
        raise ValueError("counts must be non-negative")
    if args.fuel <= 0:
        raise ValueError("fuel must be > 0")
    if args.blocksize <= 0:
        raise ValueError("blocksize must be > 0")

    total_cases = args.pass_count + args.fail_count + args.timeout_count
    if total_cases < args.min_cases:
        raise ValueError(
            f"total cases {total_cases} is smaller than required min-cases {args.min_cases}"
        )

    program = make_program()
    cases: List[List[Dict[str, Any]]] = []
    expected: List[Dict[str, Any]] = []

    for i in range(args.pass_count):
        x = (i % 100000) - 50000
        cases.append(make_case(mode=0, x=x))
        expected.append(
            {
                "bucket": "pass",
                "kind": "return",
                "value": encode_value(x + 1),
            }
        )

    for i in range(args.fail_count):
        x = i
        cases.append(make_case(mode=1, x=x))
        expected.append(
            {
                "bucket": "failed",
                "kind": "error",
                "code": "TypeError",
            }
        )

    for i in range(args.timeout_count):
        x = i
        cases.append(make_case(mode=2, x=x))
        expected.append(
            {
                "bucket": "timeout",
                "kind": "error",
                "code": "Timeout",
            }
        )

    output = {
        "format_version": "bytecode-batch-fixture-v0.1",
        "meta": {
            "generator": "tools/gen_gpu_batch_cases.py",
            "bytecode_format_version": "bytecode-json-v0.1",
            "counts": {
                "pass": args.pass_count,
                "failed": args.fail_count,
                "timeout": args.timeout_count,
                "total": total_cases,
            },
            "min_cases": args.min_cases,
            "fuel": args.fuel,
            "blocksize": args.blocksize,
        },
        "batch_request": {
            "format_version": "bytecode-json-v0.1",
            "engine": "gpu",
            "fuel": args.fuel,
            "blocksize": args.blocksize,
            "bytecode": program,
            "cases": cases,
        },
        "expected": expected,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {total_cases} cases (pass={args.pass_count}, failed={args.fail_count}, "
        f"timeout={args.timeout_count}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
