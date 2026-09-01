#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .grammar_profiles import (
    build_compat_config,
    build_full_config,
    load_fixture_schema,
    schema_hash,
    write_profile,
)
from ..shared.hashing import sha256_file
from ..shared.schemas import PSB_REGRESSION_SUMMARY


ROOT = Path(__file__).resolve().parents[3]
NUMERIC_TYPES = {"int", "float"}
COMPAT_PROFILES = {"compat", "compact"}


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for raw in parse_csv(text):
        try:
            out.append(int(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed: {raw}") from exc
    if not out:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return out


def on_off(value: bool) -> str:
    return "on" if value else "off"


def parse_on_off(text: str | None) -> bool:
    if text is None:
        return True
    normalized = text.strip().lower()
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected on/off")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def discover_problem_files(cases_root: Path, problems: Iterable[str] | None) -> Dict[str, Path]:
    if problems is None:
        found = sorted(cases_root.glob("*.train.json"))
        return {path.name[: -len(".train.json")]: path for path in found}
    out: Dict[str, Path] = {}
    for problem in problems:
        path = cases_root / f"{problem}.train.json"
        if not path.exists():
            raise FileNotFoundError(f"missing train fixture for problem {problem}: {path}")
        out[problem] = path
    return out


def solved_target_for_cases(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"fixture has no cases: {path}")

    binary_cases = 0
    numeric_cases = 0
    for idx, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError(f"case {idx} is not an object in {path}")
        expected = row.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"case {idx} missing typed expected value in {path}")
        expected_type = expected.get("type")
        if expected_type in NUMERIC_TYPES:
            numeric_cases += 1
        else:
            binary_cases += 1

    schema_hash = None
    meta = payload.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("schema_hash"), str):
        schema_hash = meta["schema_hash"]
    source = payload.get("source")
    if schema_hash is None and isinstance(source, dict) and isinstance(source.get("schema_hash"), str):
        schema_hash = source["schema_hash"]

    return {
        "format_version": payload.get("format_version"),
        "case_count": len(cases),
        "numeric_cases": numeric_cases,
        "binary_cases": binary_cases,
        "target_fitness": float(binary_cases),
        "schema_hash": schema_hash,
    }


def extract_run_metrics(run_json: Path, target: Dict[str, Any]) -> Dict[str, Any]:
    payload = load_json(run_json)
    final = payload.get("final", {})
    history = payload.get("history", [])
    meta = payload.get("meta", {})
    timing = payload.get("timing", {})

    best_fitness: float | None = None
    program_key: str | None = None
    if isinstance(final, dict) and not final.get("skipped", False) and "best_fitness" in final:
        best_fitness = float(final["best_fitness"])
        program_key = final.get("program_key")
    elif isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict) and "best_fitness" in last:
            best_fitness = float(last["best_fitness"])
            program_key = last.get("program_key")

    solved = False
    if best_fitness is not None:
        solved = math.isclose(best_fitness, float(target["target_fitness"]), rel_tol=0.0, abs_tol=1e-9)

    meta_timing = meta.get("timing", {}) if isinstance(meta, dict) else {}
    total_ms = None
    if isinstance(meta_timing, dict) and "total_ms" in meta_timing:
        total_ms = float(meta_timing["total_ms"])

    return {
        "best_fitness": best_fitness,
        "target_fitness": target["target_fitness"],
        "solved": solved,
        "program_key": program_key,
        "timing_totals": meta_timing if isinstance(meta_timing, dict) else {},
        "timing": timing if isinstance(timing, dict) else {},
        "total_ms": total_ms,
    }


def extract_ast_eval_metrics(eval_json: Path, target: Dict[str, Any]) -> Dict[str, Any]:
    payload = load_json(eval_json)
    result = payload.get("result")
    if not isinstance(result, dict) or "fitness" not in result:
        raise ValueError(f"AST eval output missing result.fitness: {eval_json}")
    fitness = float(result["fitness"])
    solved = math.isclose(fitness, float(target["target_fitness"]), rel_tol=0.0, abs_tol=1e-9)
    return {
        "test_best_fitness": fitness,
        "test_target_fitness": target["target_fitness"],
        "test_solved": solved,
        "test_program_key": result.get("program_key"),
    }


def test_cases_path_for_train(cases_path: Path) -> Path:
    name = cases_path.name
    if name.endswith(".train.json"):
        return cases_path.with_name(name[: -len(".train.json")] + ".test.json")
    return cases_path.with_name(cases_path.stem + ".test.json")


def build_command(
    args: argparse.Namespace,
    cases_path: Path,
    seed: int,
    out_json: Path,
    grammar_config_path: Path | None,
) -> List[str]:
    cmd = [
        str(args.binary),
        "--cases",
        str(cases_path),
        "--engine",
        args.engine,
        "--repro-backend",
        args.repro_backend,
        "--repro-overlap",
        on_off(args.repro_overlap),
        "--blocksize",
        str(args.blocksize),
        "--population-size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--selection-pressure",
        str(args.selection_pressure),
        "--mutation-rate",
        str(args.mutation_rate),
        "--mutation-subtree-prob",
        str(args.mutation_subtree_prob),
        "--fuel",
        str(args.fuel),
        "--seed",
        str(seed),
        "--timing",
        "all",
        "--out-json",
        str(out_json),
    ]
    if grammar_config_path is not None:
        cmd.extend(["--grammar-config", str(grammar_config_path)])
    if args.skip_final_eval:
        cmd.extend(["--skip-final-eval", "on"])
    if args.retain_final_population:
        cmd.extend(["--retain-final-population", "on"])
    return cmd


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_problem_grammar_config(
    args: argparse.Namespace,
    problem: str,
    cases_path: Path,
    target: Dict[str, Any],
) -> tuple[Path | None, Dict[str, Any] | None]:
    if args.grammar_config is not None:
        return args.grammar_config, {
            "kind": "explicit",
            "path": str(args.grammar_config),
            "hash": sha256_file(args.grammar_config),
        }

    if args.profile in COMPAT_PROFILES:
        if args.base_grammar_config is None:
            raise ValueError(f"--profile {args.profile} requires --base-grammar-config or --grammar-config")
        field_schemas, fixture_schema_hash = load_fixture_schema(cases_path)
        fixture_schema_hash = target.get("schema_hash") or fixture_schema_hash or schema_hash(field_schemas)
        num_list_mode = "both" if args.profile == "compact" else "schema"
        payload = build_compat_config(
            base_config_path=args.base_grammar_config,
            field_schemas=field_schemas,
            fixture_schema_hash=str(fixture_schema_hash),
            profile=args.profile,
            num_list_mode=num_list_mode,
        )
        out_path = args.out_dir / "_grammar_configs" / f"{problem}.{args.profile}.json"
        generated_hash = write_profile(out_path, payload)
        return out_path, {
            "kind": f"generated_{args.profile}",
            "path": str(out_path),
            "hash": generated_hash,
            "base_config": {
                "path": str(args.base_grammar_config),
                "hash": sha256_file(args.base_grammar_config),
            },
            "fixture_schema_hash": fixture_schema_hash,
            "num_list_mode": num_list_mode,
        }

    if args.profile == "full":
        payload = build_full_config()
        out_path = args.out_dir / "_grammar_configs" / "full.json"
        generated_hash = write_profile(out_path, payload)
        return out_path, {
            "kind": "generated_full",
            "path": str(out_path),
            "hash": generated_hash,
        }

    return None, None


def run_one(
    args: argparse.Namespace,
    problem: str,
    cases_path: Path,
    seed: int,
    target: Dict[str, Any],
    grammar_config_path: Path | None,
    grammar_config_record: Dict[str, Any] | None,
) -> Dict[str, Any]:
    run_dir = args.out_dir / problem / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "run.json"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    cmd = build_command(args, cases_path, seed, out_json, grammar_config_path)

    record: Dict[str, Any] = {
        "problem": problem,
        "seed": seed,
        "cases_path": str(cases_path),
        "cases_hash": sha256_file(cases_path),
        "schema_hash": target.get("schema_hash"),
        "target": target,
        "command": cmd,
        "out_json": str(out_json),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if grammar_config_record is not None:
        record["grammar_config"] = grammar_config_record

    if args.dry_run:
        record.update(
            {
                "status": "dry_run",
                "returncode": 0,
                "best_fitness": None,
                "target_fitness": target["target_fitness"],
                "solved": False,
                "total_ms": None,
            }
        )
        return record

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    record["returncode"] = proc.returncode
    record["wall_ms"] = elapsed_ms

    if proc.returncode != 0:
        record.update({"status": "failed", "best_fitness": None, "solved": False, "total_ms": None})
        return record
    if not out_json.exists():
        record.update({"status": "missing_output", "best_fitness": None, "solved": False, "total_ms": None})
        return record

    metrics = extract_run_metrics(out_json, target)
    record.update({"status": "ok", **metrics})

    if args.eval_test:
        test_cases_path = test_cases_path_for_train(cases_path)
        record["test_cases_path"] = str(test_cases_path)
        if not test_cases_path.exists():
            record["test_status"] = "missing_test_cases"
            return record
        run_payload = load_json(out_json)
        final = run_payload.get("final")
        if not isinstance(final, dict) or not isinstance(final.get("ast"), dict):
            record["test_status"] = "missing_final_ast"
            return record
        best_ast_path = run_dir / "best_ast.json"
        write_json(best_ast_path, final["ast"])
        test_out_json = run_dir / "test_eval.json"
        test_stdout_path = run_dir / "test_stdout.txt"
        test_stderr_path = run_dir / "test_stderr.txt"
        test_cmd = [
            str(args.binary),
            "--cases",
            str(test_cases_path),
            "--eval-ast-json",
            str(best_ast_path),
            "--engine",
            "cpu",
            "--fuel",
            str(args.fuel),
            "--penalty",
            str(args.penalty),
            "--out-json",
            str(test_out_json),
        ]
        test_proc = subprocess.run(test_cmd, cwd=ROOT, text=True, capture_output=True)
        test_stdout_path.write_text(test_proc.stdout, encoding="utf-8")
        test_stderr_path.write_text(test_proc.stderr, encoding="utf-8")
        record["test_command"] = test_cmd
        record["test_out_json"] = str(test_out_json)
        record["test_stdout"] = str(test_stdout_path)
        record["test_stderr"] = str(test_stderr_path)
        record["test_returncode"] = test_proc.returncode
        if test_proc.returncode != 0:
            record["test_status"] = "failed"
            return record
        test_target = solved_target_for_cases(test_cases_path)
        record["test_target"] = test_target
        record.update(extract_ast_eval_metrics(test_out_json, test_target))
        record["test_status"] = "ok"
    return record


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_problem: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        by_problem.setdefault(str(run["problem"]), []).append(run)

    problems: Dict[str, Any] = {}
    for problem, rows in sorted(by_problem.items()):
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        solved_count = sum(1 for r in ok_rows if r.get("solved"))
        test_rows = [r for r in ok_rows if r.get("test_status") == "ok"]
        test_solved_count = sum(1 for r in test_rows if r.get("test_solved"))
        totals = [float(r["total_ms"]) for r in ok_rows if r.get("total_ms") is not None]
        test_fitness = [
            float(r["test_best_fitness"])
            for r in test_rows
            if isinstance(r.get("test_best_fitness"), (int, float))
        ]
        problems[problem] = {
            "runs": len(rows),
            "ok_runs": len(ok_rows),
            "solved_count": solved_count,
            "test_ok_runs": len(test_rows),
            "test_solved_count": test_solved_count if test_rows else None,
            "best_fitness": max((r.get("best_fitness") for r in ok_rows if r.get("best_fitness") is not None), default=None),
            "best_test_fitness": max(test_fitness, default=None),
            "median_test_fitness": median(test_fitness) if test_fitness else None,
            "median_total_ms": median(totals) if totals else None,
        }

    return {
        "runs": len(runs),
        "ok_runs": sum(1 for r in runs if r.get("status") == "ok"),
        "dry_runs": sum(1 for r in runs if r.get("status") == "dry_run"),
        "failed_runs": sum(1 for r in runs if r.get("status") not in {"ok", "dry_run"}),
        "problems": problems,
    }


def median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PSB evolution regression sweeps.")
    parser.add_argument("--suite", required=True, choices=["psb1", "psb2"])
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--cases-root", required=True, type=Path)
    parser.add_argument("--problems", help="comma-separated problem subset")
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--binary", type=Path, default=Path("cpp/build/g3pvm_evolve_cli"))
    parser.add_argument("--grammar-config", type=Path)
    parser.add_argument("--base-grammar-config", type=Path)
    parser.add_argument("--engine", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--repro-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--repro-overlap", nargs="?", const="on", default="off", type=parse_on_off)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--population-size", type=int, default=8192)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--selection-pressure", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.5)
    parser.add_argument("--mutation-subtree-prob", type=float, default=0.8)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--fuel", type=int, default=20000)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--retain-final-population", action="store_true")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate the final best AST on matching *.test.json cases.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    problems = parse_csv(args.problems) if args.problems else None
    problem_files = discover_problem_files(args.cases_root, problems)
    if not problem_files:
        raise SystemExit(f"no *.train.json fixtures found under {args.cases_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    generated_grammar_configs: Dict[str, Any] = {}
    for problem, cases_path in sorted(problem_files.items()):
        target = solved_target_for_cases(cases_path)
        grammar_config_path, grammar_config_record = materialize_problem_grammar_config(args, problem, cases_path, target)
        if grammar_config_record is not None and grammar_config_record.get("kind", "").startswith("generated_"):
            generated_grammar_configs[problem] = grammar_config_record
        for seed in args.seeds:
            run = run_one(args, problem, cases_path, seed, target, grammar_config_path, grammar_config_record)
            runs.append(run)
            print(
                f"PSB_RUN suite={args.suite} profile={args.profile} problem={problem} "
                f"seed={seed} status={run['status']}"
            )

    metadata: Dict[str, Any] = {
        "suite": args.suite,
        "profile": args.profile,
        "cases_root": str(args.cases_root),
        "problems": sorted(problem_files.keys()),
        "seeds": args.seeds,
        "binary": str(args.binary),
        "engine": args.engine,
        "repro_backend": args.repro_backend,
        "repro_overlap": args.repro_overlap,
        "blocksize": args.blocksize,
        "population_size": args.population_size,
        "generations": args.generations,
        "selection_pressure": args.selection_pressure,
        "mutation_rate": args.mutation_rate,
        "mutation_subtree_prob": args.mutation_subtree_prob,
        "penalty": args.penalty,
        "fuel": args.fuel,
        "eval_test": args.eval_test,
        "dry_run": args.dry_run,
    }
    if args.grammar_config is not None:
        metadata["grammar_config"] = {"path": str(args.grammar_config), "hash": sha256_file(args.grammar_config)}
    if args.base_grammar_config is not None:
        metadata["base_grammar_config"] = {
            "path": str(args.base_grammar_config),
            "hash": sha256_file(args.base_grammar_config),
        }
    if generated_grammar_configs:
        metadata["generated_grammar_configs"] = generated_grammar_configs

    summary = {
        "format_version": PSB_REGRESSION_SUMMARY,
        "metadata": metadata,
        "aggregate": aggregate_runs(runs),
        "runs": runs,
    }
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "manifest.json", {"format_version": "psb-regression-manifest", "metadata": metadata})
    print(f"PSB_SUMMARY {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
