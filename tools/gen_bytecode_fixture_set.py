#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.g3p_vm_gpu.ast import Block
from src.g3p_vm_gpu.compiler import BytecodeProgram, compile_program
from src.g3p_vm_gpu.errors import ErrCode
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


@dataclass(frozen=True)
class Quota:
    passed: int
    failed: int
    timeout: int


def encode_value(v: Any) -> Dict[str, Any]:
    if v is None:
        return {"type": "none"}
    if isinstance(v, bool):
        return {"type": "bool", "value": v}
    if isinstance(v, int):
        return {"type": "int", "value": v}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    raise TypeError(f"unsupported value for fixture: {type(v)}")


def encode_program(bc: BytecodeProgram) -> Dict[str, Any]:
    return {
        "consts": [encode_value(v) for v in bc.consts],
        "code": [{"op": ins.op, "a": ins.a, "b": ins.b} for ins in bc.code],
        "n_locals": bc.n_locals,
    }


def classify_case(prog: Block, fuel: int) -> tuple[str, Dict[str, Any]]:
    bc = compile_program(prog)
    out = run_bytecode(bc, {}, fuel=fuel)
    if isinstance(out, VMReturn):
        return "pass", {
            "kind": "return",
            "value": encode_value(out.value),
        }
    if isinstance(out, VMError):
        if out.err.code == ErrCode.TIMEOUT:
            return "timeout", {
                "kind": "error",
                "code": out.err.code.value,
            }
        return "failed", {
            "kind": "error",
            "code": out.err.code.value,
        }
    raise AssertionError("unexpected VM result type")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed bytecode fixtures from fuzz programs with target pass/failed/timeout counts."
    )
    parser.add_argument(
        "--out",
        default="data/fixtures/bytecode_cases.json",
        help="Output fixture JSON path.",
    )
    parser.add_argument("--pass-count", type=int, default=1000)
    parser.add_argument("--failed-count", type=int, default=1000)
    parser.add_argument("--timeout-count", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--fuel-main",
        type=int,
        default=20_000,
        help="Fuel for pass/failed buckets.",
    )
    parser.add_argument(
        "--fuel-timeout",
        type=int,
        default=1,
        help="Fuel used when searching timeout bucket.",
    )
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    quota = Quota(
        passed=args.pass_count,
        failed=args.failed_count,
        timeout=args.timeout_count,
    )
    counts = {"pass": 0, "failed": 0, "timeout": 0}
    targets = {"pass": quota.passed, "failed": quota.failed, "timeout": quota.timeout}
    cases: List[Dict[str, Any]] = []
    seed = args.start_seed

    while (
        counts["pass"] < quota.passed
        or counts["failed"] < quota.failed
        or counts["timeout"] < quota.timeout
    ):
        prog = make_random_program(seed=seed, depth=args.depth)

        if counts["pass"] < quota.passed or counts["failed"] < quota.failed:
            bucket, expected = classify_case(prog, args.fuel_main)
            if bucket in ("pass", "failed") and counts[bucket] < targets[bucket]:
                bc = compile_program(prog)
                cases.append(
                    {
                        "id": f"{bucket}_{counts[bucket]:04d}",
                        "bucket": bucket,
                        "seed": seed,
                        "depth": args.depth,
                        "fuel": args.fuel_main,
                        "bytecode": encode_program(bc),
                        "expected": expected,
                    }
                )
                counts[bucket] += 1

        if counts["timeout"] < quota.timeout:
            bucket_t, expected_t = classify_case(prog, args.fuel_timeout)
            if bucket_t == "timeout":
                bc = compile_program(prog)
                cases.append(
                    {
                        "id": f"timeout_{counts['timeout']:04d}",
                        "bucket": "timeout",
                        "seed": seed,
                        "depth": args.depth,
                        "fuel": args.fuel_timeout,
                        "bytecode": encode_program(bc),
                        "expected": expected_t,
                    }
                )
                counts["timeout"] += 1

        if seed % 2000 == 0 and seed != args.start_seed:
            print(
                f"seed={seed} pass={counts['pass']}/{quota.passed} "
                f"failed={counts['failed']}/{quota.failed} "
                f"timeout={counts['timeout']}/{quota.timeout}"
            )

        seed += 1

    output = {
        "format_version": "bytecode-fixture-v0.1",
        "meta": {
            "generator": "tools/gen_bytecode_fixture_set.py",
            "bytecode_format_version": "bytecode-json-v0.1",
            "depth": args.depth,
            "start_seed": args.start_seed,
            "final_seed_exclusive": seed,
            "counts": counts,
            "fuel_main": args.fuel_main,
            "fuel_timeout": args.fuel_timeout,
        },
        "cases": cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
