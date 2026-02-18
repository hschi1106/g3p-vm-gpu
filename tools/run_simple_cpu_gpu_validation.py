#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    fixture_path: str


FIXTURES: List[FixtureSpec] = [
    FixtureSpec(name="x_plus_1", fixture_path="data/fixtures/simple_evo_x_plus_1_1024.json"),
    FixtureSpec(name="affine_2x_plus_3", fixture_path="data/fixtures/simple_evo_affine_2x_plus_3_1024.json"),
    FixtureSpec(name="square_x2", fixture_path="data/fixtures/simple_evo_square_x2_1024.json"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run simple function evolution validation matrix across CPU/GPU and write summary.json + summary.md."
        )
    )
    parser.add_argument("--cpp-cli", default="cpp/build/g3pvm_evolve_cli")
    parser.add_argument("--population-size", type=int, default=1024)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated integer seeds.")
    parser.add_argument("--blocksize", type=int, default=256)
    parser.add_argument("--out-dir", default="")
    return parser


def _parse_seeds(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("--seeds must include at least one integer")
    return vals


def _extract_summary_path(stdout: str) -> Optional[Path]:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("SUMMARY_JSON "):
            return Path(line.split(" ", 1)[1].strip())
    return None


def _first_hit_generation(history: List[Dict[str, Any]], case_count: int) -> Optional[int]:
    target = float(case_count)
    for row in history:
        best = float(row.get("best", -1.0))
        if best == target:
            return int(row.get("generation", -1))
    return None


def _median_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _phase(parsed: Dict[str, Any], name: str) -> Optional[float]:
    timing = parsed.get("timing_summary", {})
    raw = timing.get(name)
    if raw is None:
        return None
    return float(raw)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _run_once(
    fixture: FixtureSpec,
    seed: int,
    engine: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    fixture_abs = (ROOT / fixture.fixture_path).resolve()
    payload = json.loads(fixture_abs.read_text(encoding="utf-8"))
    case_count = len(payload.get("cases", []))

    base_cmd = [
        "python3",
        "tools/run_cpp_evolution.py",
        "--cases",
        str(fixture_abs),
        "--cases-format",
        "simple",
        "--cpp-cli",
        str((ROOT / args.cpp_cli).resolve()),
        "--engine",
        engine,
        "--population-size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--selection",
        "tournament",
        "--crossover-method",
        "hybrid",
        "--cpp-timing",
        "all",
        "--seed",
        str(seed),
        "--fuel",
        "20000",
        "--log-dir",
        str(out_dir),
        "--run-tag",
        f"simple_{fixture.name}_{engine}_s{seed}",
    ]
    if engine == "gpu":
        base_cmd.extend(["--blocksize", str(args.blocksize)])
        cmd = ["scripts/run_gpu_command.sh", "--", *base_cmd]
    else:
        cmd = base_cmd

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    result: Dict[str, Any] = {
        "fixture": fixture.name,
        "fixture_path": fixture.fixture_path,
        "seed": seed,
        "engine": engine,
        "case_count": case_count,
        "status": "failed",
        "returncode": proc.returncode,
        "command": cmd,
        "console_stdout": proc.stdout,
        "console_stderr": proc.stderr,
        "summary_json": None,
        "success": False,
        "first_hit_generation": None,
        "final_best": None,
        "timing_ms": {},
        "error": "",
    }

    summary_path = _extract_summary_path(proc.stdout)
    if summary_path is None and proc.returncode == 0:
        result["error"] = "missing SUMMARY_JSON in tool output"
        return result

    if summary_path is None:
        result["error"] = (proc.stderr or proc.stdout or "run failed").strip()
        return result

    summary_abs = summary_path if summary_path.is_absolute() else (ROOT / summary_path)
    result["summary_json"] = _safe_rel(summary_abs)
    if not summary_abs.exists():
        result["error"] = f"summary json not found: {summary_abs}"
        return result

    try:
        summary = json.loads(summary_abs.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"failed to read summary json: {exc}"
        return result

    parsed = summary.get("parsed", {})
    history = parsed.get("history", [])
    final = parsed.get("final", {})
    first_hit = _first_hit_generation(history, case_count)
    final_best = float(final.get("best", -1.0)) if isinstance(final, dict) else None

    timing_ms: Dict[str, Optional[float]] = {
        "total": _phase(parsed, "total"),
        "eval": _phase(parsed, "generations_eval_total"),
        "repro": _phase(parsed, "generations_repro_total"),
    }
    if engine == "gpu":
        timing_ms.update(
            {
                "compile": _phase(parsed, "gpu_generations_program_compile_total"),
                "upload": _phase(parsed, "gpu_generations_pack_upload_total"),
                "kernel": _phase(parsed, "gpu_generations_kernel_total"),
                "copyback": _phase(parsed, "gpu_generations_copyback_total"),
            }
        )

    result["status"] = "ok" if proc.returncode == 0 else "failed"
    result["success"] = first_hit is not None
    result["first_hit_generation"] = first_hit
    result["final_best"] = final_best
    result["timing_ms"] = timing_ms

    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "run failed").strip()
    return result


def _aggregate_fixture(rows: List[Dict[str, Any]], engine: str) -> Dict[str, Any]:
    scoped = [r for r in rows if r["engine"] == engine]
    ok = [r for r in scoped if r["status"] == "ok"]
    success_cnt = sum(1 for r in ok if r["success"])

    hit_vals = [float(r["first_hit_generation"]) for r in ok if r["first_hit_generation"] is not None]
    total_vals = [float(r["timing_ms"].get("total")) for r in ok if r["timing_ms"].get("total") is not None]
    eval_vals = [float(r["timing_ms"].get("eval")) for r in ok if r["timing_ms"].get("eval") is not None]
    repro_vals = [float(r["timing_ms"].get("repro")) for r in ok if r["timing_ms"].get("repro") is not None]

    summary: Dict[str, Any] = {
        "runs_total": len(scoped),
        "runs_ok": len(ok),
        "success_count": success_cnt,
        "success_rate": (float(success_cnt) / float(len(scoped))) if scoped else 0.0,
        "median_first_hit_generation": _median_or_none(hit_vals),
        "median_total_ms": _median_or_none(total_vals),
        "median_eval_ms": _median_or_none(eval_vals),
        "median_repro_ms": _median_or_none(repro_vals),
    }

    if engine == "gpu":
        ratios: List[float] = []
        for r in ok:
            compile_ms = r["timing_ms"].get("compile")
            total_ms = r["timing_ms"].get("total")
            if compile_ms is None or total_ms is None:
                continue
            if float(total_ms) <= 0.0:
                continue
            ratios.append(float(compile_ms) / float(total_ms))
        summary["median_compile_ratio"] = _median_or_none(ratios)

    return summary


def _fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _write_markdown(path: Path, aggregates: Dict[str, Any], runs: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Simple CPU/GPU Evolution Validation")
    lines.append("")
    lines.append("## Per-function Summary")
    lines.append("")

    for fixture in FIXTURES:
        row = aggregates[fixture.name]
        cpu = row["cpu"]
        gpu = row["gpu"]
        lines.append(f"### {fixture.name}")
        lines.append("")
        lines.append(f"- CPU success rate: {cpu['success_count']}/{cpu['runs_total']} ({cpu['success_rate'] * 100.0:.1f}%)")
        lines.append(f"- GPU success rate: {gpu['success_count']}/{gpu['runs_total']} ({gpu['success_rate'] * 100.0:.1f}%)")
        lines.append(f"- CPU median first-hit generation: {_fmt_num(cpu['median_first_hit_generation'], 1)}")
        lines.append(f"- GPU median first-hit generation: {_fmt_num(gpu['median_first_hit_generation'], 1)}")
        lines.append(
            f"- CPU median timing (total/eval/repro ms): {_fmt_num(cpu['median_total_ms'])} / {_fmt_num(cpu['median_eval_ms'])} / {_fmt_num(cpu['median_repro_ms'])}"
        )
        lines.append(
            f"- GPU median timing (total/eval/repro ms): {_fmt_num(gpu['median_total_ms'])} / {_fmt_num(gpu['median_eval_ms'])} / {_fmt_num(gpu['median_repro_ms'])}"
        )
        lines.append(f"- GPU compile ratio (median compile/total): {_fmt_num(gpu.get('median_compile_ratio'))}")
        lines.append("")

    lines.append("## Failed Runs")
    lines.append("")
    failed = [r for r in runs if r["status"] != "ok"]
    if not failed:
        lines.append("- none")
    else:
        for r in failed:
            lines.append(
                f"- fixture={r['fixture']} engine={r['engine']} seed={r['seed']} rc={r['returncode']} error={r['error'][:180]}"
            )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)

    cpp_cli = (ROOT / args.cpp_cli).resolve()
    if not cpp_cli.exists():
        raise SystemExit(f"missing cpp cli executable: {cpp_cli}")

    missing_fixtures = [f.fixture_path for f in FIXTURES if not (ROOT / f.fixture_path).exists()]
    if missing_fixtures:
        raise SystemExit(f"missing fixture(s): {', '.join(missing_fixtures)}")

    if args.out_dir:
        out_dir = (ROOT / args.out_dir).resolve()
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = (ROOT / f"logs/simple_cpu_gpu_validation_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for fixture in FIXTURES:
        for seed in seeds:
            runs.append(_run_once(fixture=fixture, seed=seed, engine="cpu", args=args, out_dir=out_dir))
            runs.append(_run_once(fixture=fixture, seed=seed, engine="gpu", args=args, out_dir=out_dir))

    aggregate: Dict[str, Any] = {}
    for fixture in FIXTURES:
        subset = [r for r in runs if r["fixture"] == fixture.name]
        aggregate[fixture.name] = {
            "cpu": _aggregate_fixture(subset, "cpu"),
            "gpu": _aggregate_fixture(subset, "gpu"),
        }

    summary = {
        "meta": {
            "utc_time": datetime.now(tz=timezone.utc).isoformat(),
            "cpp_cli": str(cpp_cli),
            "population_size": args.population_size,
            "generations": args.generations,
            "seeds": seeds,
            "blocksize": args.blocksize,
            "fixtures": [f.fixture_path for f in FIXTURES],
        },
        "runs": runs,
        "aggregate": aggregate,
    }

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_markdown(summary_md, aggregate, runs)

    print(f"OUT_DIR {out_dir}")
    print(f"SUMMARY_JSON {summary_json}")
    print(f"SUMMARY_MD {summary_md}")


if __name__ == "__main__":
    main()
