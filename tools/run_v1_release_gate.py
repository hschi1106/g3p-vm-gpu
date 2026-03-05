#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateStep:
    name: str
    ok: bool
    returncode: int
    detail: str
    command: List[str]
    stdout_log: str
    stderr_log: str


def _utc_tag() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_logged(name: str, cmd: List[str], out_dir: Path) -> GateStep:
    safe = name.replace("/", "_")
    stdout_log = out_dir / f"{safe}.stdout.log"
    stderr_log = out_dir / f"{safe}.stderr.log"
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    detail = ""
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip())[:500]
    return GateStep(
        name=name,
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        detail=detail,
        command=cmd,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
    )


def _parse_key_line(stdout: str, key: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(" ", 1)[1].strip()
    raise RuntimeError(f"missing line: {key}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# v1.0 Release Gate Report")
    lines.append("")
    lines.append(f"- utc_time: {summary['utc_time']}")
    lines.append(f"- overall_pass: {summary['overall_pass']}")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for g in summary["gates"]:
        lines.append(f"- {g['name']}: {'PASS' if g['pass'] else 'FAIL'} ({g['detail']})")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    for s in summary["steps"]:
        lines.append(f"- {s['name']}: {'ok' if s['ok'] else 'fail'} rc={s['returncode']}")
        lines.append(f"  cmd: `{shlex.join(s['command'])}`")
        lines.append(f"  stdout: `{s['stdout_log']}`")
        lines.append(f"  stderr: `{s['stderr_log']}`")
        if s["detail"]:
            lines.append(f"  detail: {s['detail']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Run v1.0 final gate: tests + speedup + evolution + PSB2 batch.")
    p.add_argument("--out-dir", default="", help="Output dir; default logs/v1_release_report/<utc_tag>")
    p.add_argument("--cpp-build-dir", default="cpp/build")
    p.add_argument("--cpp-cli", default="cpp/build/g3pvm_evolve_cli")
    p.add_argument("--baseline-speed-report", default="logs/v1_baseline/speed_bouncing_1024/cpu_gpu_compare.report.json")
    p.add_argument("--speedup-threshold-ratio", type=float, default=0.8)
    p.add_argument("--speed-population-size", type=int, default=1024)
    p.add_argument("--speed-generations", type=int, default=40)
    p.add_argument("--speed-blocksize", type=int, default=256)
    p.add_argument("--exp-cases", default="data/fixtures/simple_evo_exp_1024.json")
    p.add_argument("--exp-population-size", type=int, default=1024)
    p.add_argument("--exp-generations", type=int, default=40)
    p.add_argument("--exp-engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--psb2-datasets-root", default="data/psb2_datasets")
    p.add_argument("--psb2-n-train", type=int, default=64)
    p.add_argument("--psb2-n-test", type=int, default=64)
    p.add_argument("--psb2-population-size", type=int, default=32)
    p.add_argument("--psb2-generations", type=int, default=1)
    p.add_argument("--skip-python-tests", action="store_true")
    p.add_argument("--skip-cpp-tests", action="store_true")
    args = p.parse_args()

    out_dir = (ROOT / args.out_dir).resolve() if args.out_dir else (ROOT / "logs" / "v1_release_report" / _utc_tag()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps: List[GateStep] = []

    if not args.skip_python_tests:
        steps.append(
            _run_logged(
                "python_tests",
                ["python3", "-m", "unittest", "discover", "-s", "python/tests", "-p", "test_*.py", "-v"],
                out_dir,
            )
        )
    if not args.skip_cpp_tests:
        steps.append(
            _run_logged(
                "cpp_tests",
                ["ctest", "--test-dir", args.cpp_build_dir, "--output-on-failure"],
                out_dir,
            )
        )

    speed_out = out_dir / "speed"
    speed_out.mkdir(parents=True, exist_ok=True)
    steps.append(
        _run_logged(
            "speed_benchmark",
            [
                "bash",
                "tools/run_cpu_gpu_speedup_experiment.sh",
                "--popsize",
                str(args.speed_population_size),
                "--generations",
                str(args.speed_generations),
                "--blocksize",
                str(args.speed_blocksize),
                "--cases",
                "data/fixtures/speedup_cases_bouncing_balls_1024.json",
                "--cpp-cli",
                args.cpp_cli,
                "--outdir",
                str(speed_out),
            ],
            out_dir,
        )
    )

    exp_out = out_dir / "exp"
    exp_out.mkdir(parents=True, exist_ok=True)
    exp_cmd = [
        "python3",
        "tools/run_cpp_evolution.py",
        "--cases",
        args.exp_cases,
        "--cpp-cli",
        args.cpp_cli,
        "--engine",
        args.exp_engine,
        "--blocksize",
        str(args.speed_blocksize),
        "--population-size",
        str(args.exp_population_size),
        "--generations",
        str(args.exp_generations),
        "--cpp-timing",
        "all",
        "--log-dir",
        str(exp_out),
        "--run-tag",
        "v1_gate_exp",
    ]
    if args.exp_engine == "gpu":
        exp_cmd = ["scripts/run_gpu_command.sh", "--"] + exp_cmd
    steps.append(_run_logged("exp_evolution", exp_cmd, out_dir))

    psb2_out = out_dir / "psb2_all"
    steps.append(
        _run_logged(
            "psb2_all_tasks",
            [
                "python3",
                "tools/run_psb2_all_tasks.py",
                "--datasets-root",
                args.psb2_datasets_root,
                "--tasks",
                "all",
                "--engine",
                "cpu",
                "--n-train",
                str(args.psb2_n_train),
                "--n-test",
                str(args.psb2_n_test),
                "--population-size",
                str(args.psb2_population_size),
                "--generations",
                str(args.psb2_generations),
                "--log-dir",
                str(psb2_out),
            ],
            out_dir,
        )
    )

    gate_list: List[Dict[str, Any]] = []

    tests_pass = all(s.ok for s in steps if s.name in {"python_tests", "cpp_tests"})
    if args.skip_python_tests and args.skip_cpp_tests:
        tests_pass = True
    gate_list.append({"name": "tests_pass", "pass": tests_pass, "detail": "python/cpp tests"})

    speed_gate_pass = False
    speed_detail = "speed benchmark failed"
    speed_step = next(s for s in steps if s.name == "speed_benchmark")
    if speed_step.ok:
        baseline = _load_json((ROOT / args.baseline_speed_report).resolve())
        current = _load_json(speed_out / "cpu_gpu_compare.report.json")
        base_inner = float(baseline["speedup"]["inner_total_cpu_over_gpu"])
        base_eval = float(baseline["speedup"]["eval_only_cpu_over_gpu"])
        cur_inner = float(current["speedup"]["inner_total_cpu_over_gpu"])
        cur_eval = float(current["speedup"]["eval_only_cpu_over_gpu"])
        min_inner = base_inner * args.speedup_threshold_ratio
        min_eval = base_eval * args.speedup_threshold_ratio
        speed_gate_pass = (cur_inner >= min_inner) and (cur_eval >= min_eval)
        speed_detail = (
            f"inner={cur_inner:.3f} (min {min_inner:.3f}), "
            f"eval={cur_eval:.3f} (min {min_eval:.3f})"
        )
    gate_list.append({"name": "speed_ratio_gate", "pass": speed_gate_pass, "detail": speed_detail})

    evo_gate_pass = False
    evo_detail = "exp evolution failed"
    evo_step = next(s for s in steps if s.name == "exp_evolution")
    if evo_step.ok:
        evo_stdout = Path(evo_step.stdout_log).read_text(encoding="utf-8")
        evo_json = Path(_parse_key_line(evo_stdout, "EVOLUTION_JSON"))
        evo = _load_json(evo_json)
        hist = evo.get("history", [])
        if hist:
            best_start = float(hist[0]["best_fitness"])
            best_end = float(hist[-1]["best_fitness"])
            mean_start = float(hist[0]["mean_fitness"])
            mean_end = float(hist[-1]["mean_fitness"])
            evo_gate_pass = (best_end > best_start) and (mean_end > mean_start)
            evo_detail = (
                f"best {best_start:.6f}->{best_end:.6f}, "
                f"mean {mean_start:.6f}->{mean_end:.6f}"
            )
    gate_list.append({"name": "exp_evolution_progress", "pass": evo_gate_pass, "detail": evo_detail})

    psb2_gate_pass = False
    psb2_detail = "psb2 batch failed"
    psb2_step = next(s for s in steps if s.name == "psb2_all_tasks")
    if psb2_step.ok:
        psb2_stdout = Path(psb2_step.stdout_log).read_text(encoding="utf-8")
        psb2_json = Path(_parse_key_line(psb2_stdout, "PSB2_SUMMARY_JSON"))
        summary = _load_json(psb2_json)
        psb2_gate_pass = int(summary.get("failed", 0)) == 0
        psb2_detail = (
            f"total={summary.get('tasks_total')} ok={summary.get('ok')} "
            f"unsupported={summary.get('unsupported')} failed={summary.get('failed')}"
        )
    gate_list.append({"name": "psb2_all_tasks_nonstop", "pass": psb2_gate_pass, "detail": psb2_detail})

    overall_pass = all(bool(g["pass"]) for g in gate_list)
    result = {
        "utc_time": datetime.now(tz=timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "overall_pass": overall_pass,
        "gates": gate_list,
        "steps": [asdict(s) for s in steps],
    }

    out_json = out_dir / "release_gate.summary.json"
    out_md = out_dir / "release_gate.summary.md"
    out_json.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_make_md(result), encoding="utf-8")
    print(f"RELEASE_GATE_SUMMARY_JSON {out_json}")
    print(f"RELEASE_GATE_SUMMARY_MD {out_md}")
    print(f"RELEASE_GATE_PASS {1 if overall_pass else 0}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
