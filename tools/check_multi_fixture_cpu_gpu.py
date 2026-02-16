#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Tuple


ROOT = Path(__file__).resolve().parents[1]


def parse_multi_summary(stdout: str) -> Tuple[bool, int, int, int, int, str]:
    lines = [x.strip() for x in stdout.strip().splitlines() if x.strip()]
    if not lines:
        return (False, 0, 0, 0, 0, "empty output")
    if lines[0].startswith("ERR "):
        msg = ""
        for line in lines[1:]:
            if line.startswith("MSG "):
                msg = line.split(" ", 1)[1]
                break
        return (False, 0, 0, 0, 0, msg)
    parts = lines[0].split()
    # Expected: OK programs <P> cases <C> return <R> error <E>
    if len(parts) != 9 or parts[0] != "OK" or parts[1] != "programs":
        return (False, 0, 0, 0, 0, lines[0])
    return (True, int(parts[2]), int(parts[4]), int(parts[6]), int(parts[8]), "")


def run_cli(exe: Path, req_text: str, engine: str, blocksize: int) -> Tuple[bool, int, int, int, int, str]:
    cmd = [str(exe), "--engine", engine]
    if engine == "gpu":
        cmd += ["--blocksize", str(blocksize)]
        last_msg = "cuda device unavailable"
        last = (False, 0, 0, 0, 0, last_msg)
        for dev in ("0", "1"):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = dev
            proc = subprocess.run(
                cmd,
                input=req_text,
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
                env=env,
            )
            ok, p, c, r, e, msg = parse_multi_summary(proc.stdout)
            if ok:
                return (ok, p, c, r, e, msg)
            last_msg = msg or last_msg
            last = (ok, p, c, r, e, last_msg)
        return last

    proc = subprocess.run(
        cmd,
        input=req_text,
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return parse_multi_summary(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi execution from generated PSB2 fixture JSON.")
    parser.add_argument("--fixture", default="data/fixtures/fitness_multi_bench_inputs_psb2.json")
    parser.add_argument("--cli", default="cpp/build_release/g3pvm_vm_cpu_cli")
    parser.add_argument("--blocksize", type=int, default=256)
    args = parser.parse_args()

    fixture_path = ROOT / args.fixture
    cli_path = ROOT / args.cli
    if not fixture_path.exists():
        raise SystemExit(f"missing fixture: {fixture_path}")
    if not cli_path.exists():
        raise SystemExit(f"missing cli executable: {cli_path}")

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    req_payload = {"bytecode_program_inputs": payload["bytecode_program_inputs"]}
    # Multi mode: no shared_answer field.
    req_payload["bytecode_program_inputs"].pop("shared_answer", None)
    req_text = json.dumps(req_payload, ensure_ascii=True)

    meta = payload.get("meta", {})
    prog_count = int(meta["program_count"])
    cases_per_program = int(meta["cases_per_program"])
    pass_programs = int(meta["pass_programs"])
    fail_programs = int(meta["fail_programs"])
    timeout_programs = int(meta["timeout_programs"])
    expected_cases = prog_count * cases_per_program
    expected_return = pass_programs * cases_per_program
    expected_error = (fail_programs + timeout_programs) * cases_per_program

    ok_cpu, p_cpu, c_cpu, r_cpu, e_cpu, msg_cpu = run_cli(cli_path, req_text, "cpu", args.blocksize)
    if not ok_cpu:
        raise SystemExit(f"cpu multi run failed: {msg_cpu}")
    if (p_cpu, c_cpu, r_cpu, e_cpu) != (prog_count, expected_cases, expected_return, expected_error):
        raise SystemExit("cpu multi summary mismatch")
    print(f"cpu multi check: OK programs={p_cpu} cases={c_cpu}")

    ok_gpu, p_gpu, c_gpu, r_gpu, e_gpu, msg_gpu = run_cli(cli_path, req_text, "gpu", args.blocksize)
    if not ok_gpu:
        if "device unavailable" in msg_gpu:
            print(f"gpu multi check: SKIP ({msg_gpu})")
            return
        raise SystemExit(f"gpu multi run failed: {msg_gpu}")
    if (p_gpu, c_gpu, r_gpu, e_gpu) != (prog_count, expected_cases, expected_return, expected_error):
        raise SystemExit("gpu multi summary mismatch")
    print(f"gpu multi check: OK programs={p_gpu} cases={c_gpu}")


if __name__ == "__main__":
    main()
