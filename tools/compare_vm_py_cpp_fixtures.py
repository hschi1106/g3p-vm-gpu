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

from src.g3p_vm_gpu.compiler import BytecodeProgram, Instr
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


ROOT = Path(__file__).resolve().parents[1]


def decode_value(raw: Dict[str, Any]) -> Any:
    t = raw["type"]
    if t == "none":
        return None
    if t == "bool":
        return bool(raw["value"])
    if t == "int":
        return int(raw["value"])
    if t == "float":
        return float(raw["value"])
    raise ValueError(f"unknown value type: {t}")


def encode_cli_value(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return f"bool {1 if v else 0}"
    if isinstance(v, int):
        return f"int {v}"
    if isinstance(v, float):
        return f"float {repr(v)}"
    raise TypeError(f"unsupported value type: {type(v)}")


def to_py_program(case: Dict[str, Any]) -> BytecodeProgram:
    bc = case["bytecode"]
    consts = [decode_value(v) for v in bc["consts"]]
    code = [Instr(op=ins["op"], a=ins["a"], b=ins["b"]) for ins in bc["code"]]
    return BytecodeProgram(
        consts=consts,
        code=code,
        n_locals=int(bc["n_locals"]),
        var2idx={},
    )


def to_cli_input(case: Dict[str, Any]) -> str:
    bc = case["bytecode"]
    lines = [
        f"FUEL {case['fuel']}",
        f"N_LOCALS {bc['n_locals']}",
        f"N_CONSTS {len(bc['consts'])}",
    ]
    for c in bc["consts"]:
        lines.append(f"CONST {encode_cli_value(decode_value(c))}")
    lines.append(f"N_CODE {len(bc['code'])}")
    for ins in bc["code"]:
        a = "x" if ins["a"] is None else str(ins["a"])
        b = "x" if ins["b"] is None else str(ins["b"])
        lines.append(f"INS {ins['op']} {a} {b}")
    lines.append("N_INPUTS 0")
    return "\n".join(lines) + "\n"


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


def py_result(program: BytecodeProgram, fuel: int) -> Tuple[str, Any]:
    out = run_bytecode(program, {}, fuel=fuel)
    if isinstance(out, VMReturn):
        return ("OK", out.value)
    if isinstance(out, VMError):
        return ("ERR", out.err.code.value)
    raise AssertionError("unknown python vm result type")


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vm_py and vm_cpp on fixture JSON cases.")
    parser.add_argument("--fixture", default="data/fixtures/bytecode_cases.json")
    args = parser.parse_args()

    gxx = shutil.which("g++")
    if gxx is None:
        raise SystemExit("g++ not found")

    fixture_path = ROOT / args.fixture
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = payload["cases"]

    with tempfile.TemporaryDirectory(prefix="g3p_cpp_vm_cmp_") as tmpdir:
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
            prog = to_py_program(case)
            py_kind, py_val = py_result(prog, int(case["fuel"]))

            proc = subprocess.run(
                [str(bin_path)],
                input=to_cli_input(case),
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
            )
            cpp_kind, cpp_val = parse_cpp_output(proc.stdout)

            if py_kind != cpp_kind or not values_equal(py_val, cpp_val):
                mismatches += 1
                print(
                    f"mismatch idx={i} id={case.get('id')} "
                    f"py=({py_kind}, {py_val}) cpp=({cpp_kind}, {cpp_val})"
                )

        print(f"cases={len(cases)} mismatches={mismatches}")
        if mismatches > 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
