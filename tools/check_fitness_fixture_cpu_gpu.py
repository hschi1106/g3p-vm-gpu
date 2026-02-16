#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def parse_fitness_output(stdout: str) -> Tuple[bool, List[int], str]:
    lines = [x.strip() for x in stdout.strip().splitlines() if x.strip()]
    if not lines:
        return (False, [], "empty output")
    if lines[0].startswith("ERR "):
        msg = ""
        for line in lines[1:]:
            if line.startswith("MSG "):
                msg = line.split(" ", 1)[1]
                break
        return (False, [], msg)
    if not lines[0].startswith("OK fitness_count "):
        return (False, [], lines[0])
    vals: List[int] = []
    for line in lines[1:]:
        if line.startswith("FIT "):
            vals.append(int(line.split()[2]))
    return (True, vals, "")


def run_cli(exe: Path, payload_text: str, engine: str, blocksize: int) -> Tuple[bool, List[int], str]:
    cmd = [str(exe), "--engine", engine]
    if engine == "gpu":
        cmd += ["--blocksize", str(blocksize)]
        last_msg = "cuda device unavailable"
        last = (False, [], last_msg)
        for dev in ("0", "1"):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = dev
            proc = subprocess.run(
                cmd,
                input=payload_text,
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
                env=env,
            )
            ok, vals, msg = parse_fitness_output(proc.stdout)
            if ok:
                return (ok, vals, msg)
            last_msg = msg or last_msg
            last = (ok, vals, last_msg)
        return last

    proc = subprocess.run(
        cmd,
        input=payload_text,
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return parse_fitness_output(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fitness fixture JSON through CPU/GPU CLI paths.")
    parser.add_argument("--fixture", default="data/fixtures/fitness_multi_bench_inputs.json")
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
    expected = payload.get("expected_fitness")
    if not isinstance(expected, list):
        raise SystemExit("fixture missing expected_fitness")
    expected_vals = [int(x) for x in expected]

    request_text = json.dumps(
        {"bytecode_program_inputs": payload["bytecode_program_inputs"]}, ensure_ascii=True
    )

    ok_cpu, cpu_vals, cpu_msg = run_cli(cli_path, request_text, "cpu", args.blocksize)
    if not ok_cpu:
        raise SystemExit(f"cpu fitness run failed: {cpu_msg}")
    if cpu_vals != expected_vals:
        raise SystemExit("cpu fitness mismatch against expected_fitness")
    print(f"cpu fitness check: OK count={len(cpu_vals)}")

    ok_gpu, gpu_vals, gpu_msg = run_cli(cli_path, request_text, "gpu", args.blocksize)
    if not ok_gpu:
        # In CI/dev where GPU unavailable, allow skip.
        if "device unavailable" in gpu_msg:
            print(f"gpu fitness check: SKIP ({gpu_msg})")
            return
        raise SystemExit(f"gpu fitness run failed: {gpu_msg}")
    if gpu_vals != expected_vals:
        raise SystemExit("gpu fitness mismatch against expected_fitness")
    print(f"gpu fitness check: OK count={len(gpu_vals)}")


if __name__ == "__main__":
    main()
