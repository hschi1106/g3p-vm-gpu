#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TaskResult:
    task: str
    status: str
    reason: str
    summary_json: str
    evolution_json: str
    best_start: float | None
    best_end: float | None
    mean_start: float | None
    mean_end: float | None


def _list_tasks(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _parse_key_line(stdout: str, key: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(" ", 1)[1].strip()
    raise RuntimeError(f"missing line: {key}")


def _run_one_task(
    task: str,
    datasets_root: Path,
    converted_dir: Path,
    args: argparse.Namespace,
) -> TaskResult:
    edge = datasets_root / task / f"{task}-edge.json"
    rnd = datasets_root / task / f"{task}-random.json"
    if not edge.exists() or not rnd.exists():
        return TaskResult(task, "skipped", "missing edge/random file", "", "", None, None, None, None)

    out_train = converted_dir / task / "train.fitness_cases.json"
    out_test = converted_dir / task / "test.fitness_cases.json"
    out_summary = converted_dir / task / "convert.summary.json"
    out_train.parent.mkdir(parents=True, exist_ok=True)

    convert_cmd = [
        "python3",
        str(ROOT / "tools" / "convert_psb2_to_fitness_cases.py"),
        "--edge-file",
        str(edge),
        "--random-file",
        str(rnd),
        "--n-train",
        str(args.n_train),
        "--n-test",
        str(args.n_test),
        "--seed",
        str(args.seed),
        "--out",
        str(out_train),
        "--out-test",
        str(out_test),
        "--summary-json",
        str(out_summary),
    ]
    cp = subprocess.run(convert_cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if cp.returncode != 0:
        return TaskResult(task, "failed", f"convert failed: {cp.stderr.strip()}", "", "", None, None, None, None)

    conv = json.loads(out_summary.read_text(encoding="utf-8"))
    if not conv.get("runtime_compatible", False):
        return TaskResult(
            task,
            "unsupported",
            "unsupported value types for current runtime: " + ",".join(conv.get("unsupported_types", [])),
            "",
            "",
            None,
            None,
            None,
            None,
        )

    run_cmd = [
        "python3",
        str(Path(args.run_cpp_tool)),
        "--cases",
        str(out_train),
        "--cpp-cli",
        str(Path(args.cpp_cli)),
        "--engine",
        args.engine,
        "--blocksize",
        str(args.blocksize),
        "--population-size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--numeric-type-penalty",
        str(args.numeric_type_penalty),
        "--selection-pressure",
        str(args.selection_pressure),
        "--seed",
        str(args.seed),
        "--log-dir",
        str(args.log_dir),
        "--run-tag",
        f"psb2_{task}",
    ]
    if args.engine == "gpu":
        cmd = [str(ROOT / "scripts" / "run_gpu_command.sh"), "--"] + run_cmd
    else:
        cmd = run_cmd

    rp = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if rp.returncode != 0:
        reason = rp.stderr.strip() or rp.stdout.strip()
        return TaskResult(task, "failed", f"evolve failed: {reason[:300]}", "", "", None, None, None, None)

    try:
        summary_json = _parse_key_line(rp.stdout, "SUMMARY_JSON")
        evolution_json = _parse_key_line(rp.stdout, "EVOLUTION_JSON")
    except Exception as exc:
        return TaskResult(task, "failed", f"missing output paths: {exc}", "", "", None, None, None, None)

    evo = json.loads(Path(evolution_json).read_text(encoding="utf-8"))
    hist = evo.get("history", [])
    if not hist:
        return TaskResult(task, "failed", "empty evolution history", summary_json, evolution_json, None, None, None, None)

    best_start = float(hist[0]["best_fitness"])
    best_end = float(hist[-1]["best_fitness"])
    mean_start = float(hist[0]["mean_fitness"])
    mean_end = float(hist[-1]["mean_fitness"])
    return TaskResult(task, "ok", "", summary_json, evolution_json, best_start, best_end, mean_start, mean_end)


def _render_md(results: List[TaskResult], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# PSB2 All Tasks Summary")
    lines.append("")
    lines.append(f"- tasks_total: {summary['tasks_total']}")
    lines.append(f"- ok: {summary['ok']}")
    lines.append(f"- skipped: {summary['skipped']}")
    lines.append(f"- unsupported: {summary['unsupported']}")
    lines.append(f"- failed: {summary['failed']}")
    lines.append("")
    lines.append("## Per Task")
    lines.append("")
    for r in results:
        line = f"- {r.task}: {r.status}"
        if r.status == "ok":
            line += (
                f", best {r.best_start:.6f}->{r.best_end:.6f}"
                f", mean {r.mean_start:.6f}->{r.mean_end:.6f}"
            )
        else:
            line += f", reason={r.reason}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert and run PSB2 tasks in batch.")
    parser.add_argument("--datasets-root", default="data/psb2_datasets")
    parser.add_argument("--tasks", default="all", help="Comma-separated task names or all.")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--blocksize", type=int, default=256)
    parser.add_argument("--population-size", type=int, default=1024)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--numeric-type-penalty", type=float, default=1.0)
    parser.add_argument("--selection-pressure", type=int, default=3)
    parser.add_argument("--cpp-cli", default="cpp/build/g3pvm_evolve_cli")
    parser.add_argument("--run-cpp-tool", default="tools/run_cpp_evolution.py")
    parser.add_argument("--log-dir", default="")
    args = parser.parse_args()

    datasets_root = (ROOT / args.datasets_root).resolve()
    if not datasets_root.exists():
        raise SystemExit(f"missing datasets root: {datasets_root}")

    if args.log_dir:
        log_dir = (ROOT / args.log_dir).resolve()
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = (ROOT / "logs" / "psb2_all_tasks" / ts).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    converted_dir = log_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(log_dir)

    if args.tasks == "all":
        tasks = _list_tasks(datasets_root)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("no tasks to run")

    results: List[TaskResult] = []
    for task in tasks:
        print(f"[psb2] task={task}")
        r = _run_one_task(task=task, datasets_root=datasets_root, converted_dir=converted_dir, args=args)
        results.append(r)
        print(f"[psb2] task={task} status={r.status} reason={r.reason}")

    summary: Dict[str, Any] = {
        "tasks_total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "unsupported": sum(1 for r in results if r.status == "unsupported"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "best_improved_tasks": sum(1 for r in results if r.status == "ok" and r.best_end is not None and r.best_start is not None and r.best_end > r.best_start),
        "mean_improved_tasks": sum(1 for r in results if r.status == "ok" and r.mean_end is not None and r.mean_start is not None and r.mean_end > r.mean_start),
        "results": [r.__dict__ for r in results],
    }
    unsupported_reason_counts: Dict[str, int] = {}
    for r in results:
        if r.status == "unsupported":
            unsupported_reason_counts[r.reason] = unsupported_reason_counts.get(r.reason, 0) + 1
    summary["unsupported_reason_counts"] = unsupported_reason_counts
    ok_count = max(1, summary["ok"])
    summary["best_improved_ratio_among_ok"] = summary["best_improved_tasks"] / ok_count
    summary["mean_improved_ratio_among_ok"] = summary["mean_improved_tasks"] / ok_count

    summary_json = log_dir / "summary.json"
    summary_md = log_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(_render_md(results, summary), encoding="utf-8")

    print(f"PSB2_SUMMARY_JSON {summary_json}")
    print(f"PSB2_SUMMARY_MD {summary_md}")
    print(
        f"PSB2_COUNTS total={summary['tasks_total']} ok={summary['ok']} skipped={summary['skipped']} "
        f"unsupported={summary['unsupported']} failed={summary['failed']}"
    )
    print(
        "PSB2_IMPROVEMENT "
        f"best_ratio={summary['best_improved_ratio_among_ok']:.3f} "
        f"mean_ratio={summary['mean_improved_ratio_among_ok']:.3f}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
