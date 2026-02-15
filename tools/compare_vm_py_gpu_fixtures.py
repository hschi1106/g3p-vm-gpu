#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
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


def to_py_program(case: Dict[str, Any]) -> BytecodeProgram:
    bc = case["bytecode"]
    consts = [decode_value(v) for v in bc["consts"]]
    code = [Instr(op=ins["op"], a=ins["a"], b=ins["b"]) for ins in bc["code"]]
    return BytecodeProgram(consts=consts, code=code, n_locals=int(bc["n_locals"]), var2idx={})


def to_cli_input(case: Dict[str, Any]) -> str:
    req = {
        "format_version": "bytecode-json-v0.1",
        "engine": "gpu",
        "fuel": int(case["fuel"]),
        "bytecode": case["bytecode"],
        "inputs": [],
    }
    return json.dumps(req, ensure_ascii=True)


def parse_cli_output(stdout: str) -> Tuple[str, Any, str]:
    lines = [x.strip() for x in stdout.strip().splitlines() if x.strip()]
    line = lines[0]
    msg = ""
    for x in lines[1:]:
        if x.startswith("MSG "):
            msg = x.split(" ", 1)[1]
            break
    if line.startswith("ERR "):
        return ("ERR", line.split(" ", 1)[1], msg)
    if not line.startswith("OK "):
        raise AssertionError(f"unexpected cli output: {stdout!r}")
    payload = line[3:]
    if payload == "none":
        return ("OK", None, msg)
    t, raw = payload.split(" ", 1)
    if t == "int":
        return ("OK", int(raw), msg)
    if t == "float":
        return ("OK", float(raw), msg)
    if t == "bool":
        return ("OK", raw == "1", msg)
    raise AssertionError(f"unknown gpu value type: {t}")


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
    parser = argparse.ArgumentParser(description="Compare vm_py and vm_gpu on fixture JSON cases.")
    parser.add_argument("--fixture", default="data/fixtures/bytecode_cases.json")
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    build_dir = ROOT / "cpp" / "build"
    exe = build_dir / "g3pvm_vm_cpu_cli"
    if not exe.exists():
        raise SystemExit(f"missing executable: {exe}; run cmake -S cpp -B cpp/build && cmake --build cpp/build")

    payload = json.loads((ROOT / args.fixture).read_text(encoding="utf-8"))
    cases = payload["cases"][: args.limit] if args.limit > 0 else payload["cases"]

    # Preflight: if GPU runtime is unavailable, CLI returns ValueError here.
    probe = {
        "format_version": "bytecode-json-v0.1",
        "engine": "gpu",
        "fuel": 10,
        "bytecode": {
            "n_locals": 0,
            "consts": [{"type": "int", "value": 1}],
            "code": [{"op": "PUSH_CONST", "a": 0, "b": None}, {"op": "RETURN", "a": None, "b": None}],
        },
        "inputs": [],
    }
    probe_proc = subprocess.run(
        [str(exe)],
        input=json.dumps(probe, ensure_ascii=True),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    probe_k, probe_v, probe_msg = parse_cli_output(probe_proc.stdout)
    if probe_k == "ERR":
        if probe_v == "ValueError" and "device unavailable" in probe_msg:
            print(f"gpu check skipped: {probe_v} ({probe_msg})")
            return
        raise SystemExit(f"gpu probe failed: {probe_v} ({probe_msg})")

    mismatches = 0
    for i, case in enumerate(cases):
        py_k, py_v = py_result(to_py_program(case), fuel=int(case["fuel"]))

        proc = subprocess.run(
            [str(exe)],
            input=to_cli_input(case),
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        gpu_k, gpu_v, _gpu_msg = parse_cli_output(proc.stdout)

        if py_k != gpu_k or not values_equal(py_v, gpu_v):
            mismatches += 1
            print(
                f"mismatch idx={i} id={case.get('id')} "
                f"py=({py_k},{py_v}) gpu=({gpu_k},{gpu_v})"
            )

    print(f"cases={len(cases)} mismatches={mismatches}")
    if mismatches > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
