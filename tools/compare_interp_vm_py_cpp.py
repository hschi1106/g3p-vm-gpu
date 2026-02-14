#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from src.g3p_vm_gpu.compiler import BytecodeProgram, Instr, compile_program
from src.g3p_vm_gpu.errors import Failed, Returned
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.interp import run_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


ROOT = Path(__file__).resolve().parents[1]


def parse_cpp_output(stdout: str) -> Tuple[str, Any]:
    line = stdout.strip().splitlines()[0].strip()
    if line.startswith("ERR "):
        return ("ERR", line.split(" ", 1)[1])
    if not line.startswith("OK "):
        raise AssertionError(f"unexpected cpp output: {stdout!r}")
    payload = line[3:]
    if payload == "none":
        return ("OK", None)
    t, raw = payload.split(" ", 1)
    if t == "int":
        return ("OK", int(raw))
    if t == "float":
        return ("OK", float(raw))
    if t == "bool":
        return ("OK", raw == "1")
    raise AssertionError(f"unknown cpp value type: {t}")


def to_cli_input(case: Dict[str, Any]) -> str:
    req = {
        "format_version": "bytecode-json-v0.1",
        "fuel": int(case["fuel"]),
        "bytecode": case["bytecode"],
        "inputs": [],
    }
    return json.dumps(req, ensure_ascii=True)


def vm_py_result(program: BytecodeProgram, fuel: int) -> Tuple[str, Any]:
    out = run_bytecode(program, {}, fuel=fuel)
    if isinstance(out, VMReturn):
        return ("OK", out.value)
    if isinstance(out, VMError):
        return ("ERR", out.err.code.value)
    raise AssertionError("unexpected vm_py result type")


def interp_result(seed: int, depth: int, fuel: int) -> Tuple[str, Any]:
    prog = make_random_program(seed=seed, depth=depth)
    _env, out = run_program(prog, {}, fuel=fuel)
    if isinstance(out, Returned):
        return ("OK", out.value)
    if isinstance(out, Failed):
        return ("ERR", out.err.code.value)
    raise AssertionError("top-level interpreter returned Normal unexpectedly")


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare interp_py == vm_py == vm_cpp on fixture-derived programs.")
    parser.add_argument("--fixture", default="data/fixtures/bytecode_cases.json")
    parser.add_argument("--limit", type=int, default=0, help="Optional max case count (0 means all eligible cases)")
    parser.add_argument(
        "--include-timeout",
        action="store_true",
        help="Include timeout bucket. Default excludes it due to AST-vs-bytecode fuel granularity differences.",
    )
    args = parser.parse_args()

    gxx = shutil.which("g++")
    if gxx is None:
        raise SystemExit("g++ not found")

    payload = json.loads((ROOT / args.fixture).read_text(encoding="utf-8"))
    all_cases = payload["cases"]
    if args.include_timeout:
        cases = all_cases
    else:
        cases = [c for c in all_cases if c.get("bucket") != "timeout"]

    if args.limit > 0:
        cases = cases[: args.limit]

    with tempfile.TemporaryDirectory(prefix="g3p_triplet_") as tmpdir:
        bin_path = Path(tmpdir) / "g3p_vm_cpu_cli"
        cmd = [
            gxx,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "builtins.cpp"),
            str(ROOT / "cpp" / "src" / "vm_cpu.cpp"),
            str(ROOT / "cpp" / "src" / "vm_cpu_cli.cpp"),
            "-o",
            str(bin_path),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

        mismatches = 0
        for i, case in enumerate(cases):
            seed = int(case["seed"])
            depth = int(case["depth"])
            fuel = int(case["fuel"])

            prog = make_random_program(seed=seed, depth=depth)
            bc_py = compile_program(prog)
            bc_case = case["bytecode"]
            if (
                len(bc_py.consts) != len(bc_case["consts"])
                or len(bc_py.code) != len(bc_case["code"])
                or bc_py.n_locals != int(bc_case["n_locals"])
            ):
                mismatches += 1
                print(f"mismatch idx={i} id={case.get('id')} reason=bytecode-shape-drift")
                continue

            interp_k, interp_v = interp_result(seed=seed, depth=depth, fuel=fuel)
            vm_py_k, vm_py_v = vm_py_result(bc_py, fuel=fuel)

            proc = subprocess.run(
                [str(bin_path)],
                input=to_cli_input(case),
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
            )
            vm_cpp_k, vm_cpp_v = parse_cpp_output(proc.stdout)

            same_kind = interp_k == vm_py_k == vm_cpp_k
            same_vals = values_equal(interp_v, vm_py_v) and values_equal(vm_py_v, vm_cpp_v)
            if not same_kind or not same_vals:
                mismatches += 1
                print(
                    f"mismatch idx={i} id={case.get('id')} "
                    f"interp=({interp_k},{interp_v}) vm_py=({vm_py_k},{vm_py_v}) "
                    f"vm_cpp=({vm_cpp_k},{vm_cpp_v})"
                )

        print(f"cases={len(cases)} mismatches={mismatches}")
        if mismatches > 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
