#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]

_GEN_RE = re.compile(
    r"^GEN\s+(?P<gen>\d+)\s+best=(?P<best>[-+]?\d+(?:\.\d+)?)\s+mean=(?P<mean>[-+]?\d+(?:\.\d+)?)\s+hash=(?P<hash>[0-9a-fA-F]+)$"
)
_FINAL_RE = re.compile(
    r"^FINAL\s+best=(?P<best>[-+]?\d+(?:\.\d+)?)\s+hash=(?P<hash>[0-9a-fA-F]+)\s+selection=(?P<selection>[a-z_]+)\s+crossover=(?P<crossover>[a-z_]+)$"
)
_TIMING_PHASE_RE = re.compile(r"^TIMING\s+phase=(?P<phase>[a-z_]+)\s+ms=(?P<ms>[-+]?\d+(?:\.\d+)?)$")
_TIMING_GEN_LINE_RE = re.compile(r"^TIMING\s+gen=(?P<gen>\d+)\s+(?P<rest>.+)$")
_TIMING_GPU_GEN_RE = re.compile(
    r"^TIMING\s+gpu_gen=(?P<gen>\d+)\s+compile_ms=(?P<compile>[-+]?\d+(?:\.\d+)?)\s+pack_upload_ms=(?P<upload>[-+]?\d+(?:\.\d+)?)\s+kernel_ms=(?P<kernel>[-+]?\d+(?:\.\d+)?)\s+copyback_ms=(?P<copyback>[-+]?\d+(?:\.\d+)?)$"
)


@dataclass
class StageRecord:
    stage: str
    start_perf_s: float
    end_perf_s: float

    @property
    def elapsed_ms(self) -> float:
        return (self.end_perf_s - self.start_perf_s) * 1000.0


class StageTracker:
    def __init__(self) -> None:
        self._records: List[StageRecord] = []

    def run(self, stage: str, fn):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            t1 = time.perf_counter()
            self._records.append(StageRecord(stage=stage, start_perf_s=t0, end_perf_s=t1))

    def records(self) -> List[StageRecord]:
        return list(self._records)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run end-to-end C++ evolution CLI with configurable parameters and detailed timing logs."
        ),
    )
    p.add_argument("--cases", required=True, help="Path to fitness-cases-v1 JSON.")
    p.add_argument("--cpp-cli", default="cpp/build/g3pvm_evolve_cli", help="Path to C++ evolve CLI executable.")
    p.add_argument("--engine", choices=["cpu", "gpu"], default="cpu", help="Evaluation engine for evolve_cli.")
    p.add_argument("--blocksize", type=int, default=256, help="GPU blocksize when --engine gpu.")
    p.add_argument("--population-size", type=int, default=64)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--mutation-rate", type=float, default=0.5)
    p.add_argument("--crossover-rate", type=float, default=0.9)
    p.add_argument("--crossover-method", choices=["top_level_splice", "hybrid"], default="hybrid")
    p.add_argument("--selection", choices=["tournament", "roulette", "rank", "truncation", "random"], default="tournament")
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--truncation-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fuel", type=int, default=20_000)
    p.add_argument("--max-expr-depth", type=int, default=5)
    p.add_argument("--max-stmts-per-block", type=int, default=6)
    p.add_argument("--max-total-nodes", type=int, default=80)
    p.add_argument("--max-for-k", type=int, default=16)
    p.add_argument("--max-call-args", type=int, default=3)
    p.add_argument("--show-program", choices=["none", "ast", "bytecode", "both"], default="none")
    p.add_argument("--cpp-timing", choices=["none", "summary", "per_gen", "all"], default="all")
    p.add_argument("--timeout-sec", type=int, default=0, help="Subprocess timeout in seconds, 0 means no timeout.")
    p.add_argument("--log-dir", default="logs/cpp_evolution", help="Directory for run logs and JSON artifacts.")
    p.add_argument("--run-tag", default="", help="Optional run tag for artifact naming. Default uses UTC timestamp.")
    p.add_argument("--print-command", action="store_true", help="Print exact CLI command before execution.")
    return p


def _parse_kv_floats(rest: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in rest.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _parse_cli_output(stdout: str) -> Dict[str, Any]:
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    history: List[Dict[str, Any]] = []
    final: Optional[Dict[str, Any]] = None
    timing_summary: Dict[str, float] = {}
    timing_per_gen: List[Dict[str, Any]] = []
    timing_gpu_per_gen: List[Dict[str, Any]] = []

    for line in lines:
        m = _GEN_RE.match(line)
        if m is not None:
            history.append(
                {
                    "generation": int(m.group("gen")),
                    "best": float(m.group("best")),
                    "mean": float(m.group("mean")),
                    "hash": m.group("hash"),
                }
            )
            continue

        m = _FINAL_RE.match(line)
        if m is not None:
            final = {
                "best": float(m.group("best")),
                "hash": m.group("hash"),
                "selection": m.group("selection"),
                "crossover": m.group("crossover"),
            }
            continue

        m = _TIMING_PHASE_RE.match(line)
        if m is not None:
            timing_summary[m.group("phase")] = float(m.group("ms"))
            continue

        m = _TIMING_GEN_LINE_RE.match(line)
        if m is not None:
            row = {"generation": int(m.group("gen"))}
            row.update(_parse_kv_floats(m.group("rest")))
            timing_per_gen.append(row)
            continue

        m = _TIMING_GPU_GEN_RE.match(line)
        if m is not None:
            timing_gpu_per_gen.append(
                {
                    "generation": int(m.group("gen")),
                    "compile_ms": float(m.group("compile")),
                    "pack_upload_ms": float(m.group("upload")),
                    "kernel_ms": float(m.group("kernel")),
                    "copyback_ms": float(m.group("copyback")),
                }
            )

    if not history:
        raise ValueError("missing GEN lines in cpp cli stdout")
    if final is None:
        raise ValueError("missing FINAL line in cpp cli stdout")

    return {
        "history": history,
        "final": final,
        "timing_summary": timing_summary,
        "timing_per_gen": timing_per_gen,
        "timing_gpu_per_gen": timing_gpu_per_gen,
    }


def _dump_stage_table(records: List[StageRecord]) -> str:
    return "\n".join(f"{r.stage:24s} {r.elapsed_ms:12.3f} ms" for r in records)


def main() -> None:
    start_wall = datetime.now(tz=timezone.utc)
    overall_t0 = time.perf_counter()
    tracker = StageTracker()

    args = tracker.run("parse_args", lambda: _build_arg_parser().parse_args())

    def _resolve_paths() -> Dict[str, Path]:
        cases_path = (ROOT / args.cases).resolve()
        cpp_cli = (ROOT / args.cpp_cli).resolve()
        log_dir = (ROOT / args.log_dir).resolve()
        if not cases_path.exists():
            raise SystemExit(f"missing cases file: {cases_path}")
        if not cpp_cli.exists():
            raise SystemExit(f"missing cpp cli executable: {cpp_cli}")
        log_dir.mkdir(parents=True, exist_ok=True)
        return {"cases_path": cases_path, "cpp_cli": cpp_cli, "log_dir": log_dir}

    paths = tracker.run("resolve_paths", _resolve_paths)

    _ = tracker.run("load_cases_payload", lambda: json.loads(paths["cases_path"].read_text(encoding="utf-8")))

    run_tag = args.run_tag.strip() or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_base = f"cpp_evo_{run_tag}_pid{os.getpid()}"
    stdout_path = paths["log_dir"] / f"{run_base}.stdout.log"
    stderr_path = paths["log_dir"] / f"{run_base}.stderr.log"
    stage_path = paths["log_dir"] / f"{run_base}.timings.log"
    summary_path = paths["log_dir"] / f"{run_base}.summary.json"
    out_json_path = paths["log_dir"] / f"{run_base}.evolution.json"

    def _build_command() -> List[str]:
        return [
            str(paths["cpp_cli"]),
            "--cases",
            str(paths["cases_path"]),
            "--engine",
            args.engine,
            "--blocksize",
            str(args.blocksize),
            "--population-size",
            str(args.population_size),
            "--generations",
            str(args.generations),
            "--elitism",
            str(args.elitism),
            "--mutation-rate",
            str(args.mutation_rate),
            "--crossover-rate",
            str(args.crossover_rate),
            "--crossover-method",
            args.crossover_method,
            "--selection",
            args.selection,
            "--tournament-k",
            str(args.tournament_k),
            "--truncation-ratio",
            str(args.truncation_ratio),
            "--seed",
            str(args.seed),
            "--fuel",
            str(args.fuel),
            "--max-expr-depth",
            str(args.max_expr_depth),
            "--max-stmts-per-block",
            str(args.max_stmts_per_block),
            "--max-total-nodes",
            str(args.max_total_nodes),
            "--max-for-k",
            str(args.max_for_k),
            "--max-call-args",
            str(args.max_call_args),
            "--show-program",
            args.show_program,
            "--timing",
            args.cpp_timing,
            "--out-json",
            str(out_json_path),
        ]

    cmd = tracker.run("build_command", _build_command)

    if args.print_command:
        print("COMMAND", shlex.join(cmd))

    proc = tracker.run(
        "run_cpp_cli",
        lambda: subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=(None if args.timeout_sec <= 0 else args.timeout_sec),
        ),
    )

    def _parse_output() -> Dict[str, Any]:
        if proc.returncode != 0:
            raise RuntimeError(f"cpp evolve cli failed with returncode={proc.returncode}; see stderr log")
        return _parse_cli_output(proc.stdout)

    parsed = tracker.run("parse_cli_output", _parse_output)

    def _write_artifacts() -> None:
        write_t0 = time.perf_counter()
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        records = tracker.records()
        write_elapsed_ms = (time.perf_counter() - write_t0) * 1000.0
        all_rows = [{"stage": r.stage, "elapsed_ms": r.elapsed_ms} for r in records] + [
            {"stage": "write_artifacts", "elapsed_ms": write_elapsed_ms}
        ]

        timing_table = "[outer_python]\n" + _dump_stage_table(records)
        timing_table += f"\n{'write_artifacts':24s} {write_elapsed_ms:12.3f} ms"
        timing_table += f"\n{'total':24s} {sum(r.elapsed_ms for r in records) + write_elapsed_ms:12.3f} ms"

        if parsed["timing_summary"]:
            timing_table += "\n\n[inner_cpp_summary]"
            for k in sorted(parsed["timing_summary"].keys()):
                timing_table += f"\n{k:24s} {parsed['timing_summary'][k]:12.3f} ms"

        if parsed["timing_per_gen"]:
            timing_table += "\n\n[inner_cpp_per_gen]"
            for row in parsed["timing_per_gen"]:
                cols = [f"gen={row['generation']:03d}"]
                for k, v in sorted(row.items()):
                    if k == "generation":
                        continue
                    cols.append(f"{k}={v:.3f}")
                timing_table += "\n" + " ".join(cols)

        if parsed["timing_gpu_per_gen"]:
            timing_table += "\n\n[inner_cpp_gpu_per_gen]"
            for row in parsed["timing_gpu_per_gen"]:
                timing_table += (
                    "\n"
                    + f"gen={row['generation']:03d} compile_ms={row['compile_ms']:.3f} "
                    + f"pack_upload_ms={row['pack_upload_ms']:.3f} kernel_ms={row['kernel_ms']:.3f} "
                    + f"copyback_ms={row['copyback_ms']:.3f}"
                )

        stage_path.write_text(timing_table + "\n", encoding="utf-8")

        total_ms = (time.perf_counter() - overall_t0) * 1000.0
        summary = {
            "run": {
                "tag": run_tag,
                "utc_start": start_wall.isoformat(),
                "utc_end": datetime.now(tz=timezone.utc).isoformat(),
                "total_elapsed_ms": total_ms,
                "cwd": str(ROOT),
            },
            "inputs": {
                "cases": str(paths["cases_path"]),
                "cpp_cli": str(paths["cpp_cli"]),
                "args": vars(args),
                "command": cmd,
            },
            "subprocess": {
                "returncode": proc.returncode,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "timings_log": str(stage_path),
                "out_json": str(out_json_path),
            },
            "timings": all_rows,
            "parsed": parsed,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    tracker.run("write_artifacts", _write_artifacts)

    print(f"RUN_TAG {run_tag}")
    print(f"SUMMARY_JSON {summary_path}")
    print(f"STDOUT_LOG {stdout_path}")
    print(f"STDERR_LOG {stderr_path}")
    print(f"TIMINGS_LOG {stage_path}")
    print(f"EVOLUTION_JSON {out_json_path}")
    print(
        "FINAL_METRIC "
        f"best={parsed['final']['best']:.6f} "
        f"hash={parsed['final']['hash']} "
        f"selection={parsed['final']['selection']} "
        f"crossover={parsed['final']['crossover']}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        raise SystemExit(124)
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
