#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
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


def one_run(exe: Path, request_text: str, engine: str, blocksize: int) -> Tuple[float, bool, str]:
    cmd = [str(exe), "--engine", engine]
    if engine == "gpu":
        cmd += ["--blocksize", str(blocksize)]
        last_msg = "cuda device unavailable"
        last = (0.0, False, last_msg)
        for dev in ("0", "1"):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = dev
            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                input=request_text,
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
                env=env,
            )
            t1 = time.perf_counter()
            ok, _fitness, msg = parse_fitness_output(proc.stdout)
            if ok:
                return ((t1 - t0) * 1000.0, ok, msg)
            last_msg = msg or last_msg
            last = ((t1 - t0) * 1000.0, ok, last_msg)
        return last

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=request_text,
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    t1 = time.perf_counter()
    ok, _fitness, msg = parse_fitness_output(proc.stdout)
    return ((t1 - t0) * 1000.0, ok, msg)


def one_parse_only_run(exe: Path, request_text: str, blocksize: int) -> float:
    # Use an invalid engine to force JSON parse/decode path without VM execution.
    cmd = [str(exe), "--engine", "invalid", "--blocksize", str(blocksize)]
    t0 = time.perf_counter()
    subprocess.run(
        cmd,
        input=request_text,
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU fitness from one generated fixture JSON.")
    parser.add_argument("--fixture", default="data/fixtures/fitness_multi_bench_inputs.json")
    parser.add_argument("--cli", default="cpp/build_release/g3pvm_vm_cpu_cli")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--blocksize", type=int, default=256)
    parser.add_argument(
        "--subtract-parse",
        action="store_true",
        help="Also report compute-only estimate by subtracting parse/decode baseline.",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs must be > 0")

    fixture_path = ROOT / args.fixture
    cli_path = ROOT / args.cli
    if not fixture_path.exists():
        raise SystemExit(f"missing fixture: {fixture_path}")
    if not cli_path.exists():
        raise SystemExit(f"missing cli executable: {cli_path}")

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    request_text = json.dumps({"bytecode_program_inputs": payload["bytecode_program_inputs"]}, ensure_ascii=True)

    cpu_ms: List[float] = []
    for _ in range(args.runs):
        ms, ok, msg = one_run(cli_path, request_text, "cpu", args.blocksize)
        if not ok:
            raise SystemExit(f"cpu run failed: {msg}")
        cpu_ms.append(ms)

    gpu_ms: List[float] = []
    gpu_skipped = False
    for _ in range(args.runs):
        ms, ok, msg = one_run(cli_path, request_text, "gpu", args.blocksize)
        if not ok:
            if "device unavailable" in msg:
                gpu_skipped = True
                break
            raise SystemExit(f"gpu run failed: {msg}")
        gpu_ms.append(ms)

    cpu_avg = statistics.fmean(cpu_ms)
    cpu_min = min(cpu_ms)
    cpu_max = max(cpu_ms)
    print(f"cpu_ms avg={cpu_avg:.3f} min={cpu_min:.3f} max={cpu_max:.3f} runs={len(cpu_ms)}")

    if gpu_skipped:
        print("gpu_ms SKIP (cuda device unavailable)")
        return

    gpu_avg = statistics.fmean(gpu_ms)
    gpu_min = min(gpu_ms)
    gpu_max = max(gpu_ms)
    speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float("inf")
    print(f"gpu_ms avg={gpu_avg:.3f} min={gpu_min:.3f} max={gpu_max:.3f} runs={len(gpu_ms)}")
    print(f"speedup_cpu_over_gpu={speedup:.2f}x")

    if args.subtract_parse:
        parse_ms: List[float] = []
        for _ in range(args.runs):
            parse_ms.append(one_parse_only_run(cli_path, request_text, args.blocksize))
        parse_avg = statistics.fmean(parse_ms)
        cpu_compute = max(cpu_avg - parse_avg, 0.0)
        gpu_compute = max(gpu_avg - parse_avg, 0.0)
        speedup_compute = (cpu_compute / gpu_compute) if gpu_compute > 0 else float("inf")
        print(
            f"parse_only_ms avg={parse_avg:.3f} min={min(parse_ms):.3f} max={max(parse_ms):.3f} runs={len(parse_ms)}"
        )
        print(f"cpu_compute_ms_est={cpu_compute:.3f}")
        print(f"gpu_compute_ms_est={gpu_compute:.3f}")
        print(f"speedup_cpu_over_gpu_compute_est={speedup_compute:.2f}x")


if __name__ == "__main__":
    main()
