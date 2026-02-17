#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.g3p_vm_gpu.evo_encoding import Limits, compile_for_eval
from src.g3p_vm_gpu.evolve import EvolutionConfig, FitnessCase, SelectionMethod, evolve_population


ROOT = Path(__file__).resolve().parents[1]


def _decode_typed_or_raw_value(v: Any) -> Any:
    if isinstance(v, dict) and "type" in v:
        t = v.get("type")
        if t == "none":
            return None
        if t in ("bool", "int", "float"):
            return v.get("value")
        raise ValueError(f"unsupported typed value type: {t}")
    return v


def _decode_inputs(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("case.inputs must be an object")
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        out[str(k)] = _decode_typed_or_raw_value(v)
    return out


def _decode_expected(raw: Any) -> Any:
    return _decode_typed_or_raw_value(raw)


def _parse_cases(payload: Dict[str, Any]) -> List[FitnessCase]:
    if "cases" not in payload or not isinstance(payload["cases"], list):
        raise ValueError("input JSON must include list field: cases")
    out: List[FitnessCase] = []
    for i, row in enumerate(payload["cases"]):
        if not isinstance(row, dict):
            raise ValueError(f"cases[{i}] must be an object")
        raw_inputs = row.get("inputs")
        if raw_inputs is None:
            raise ValueError(f"cases[{i}] missing inputs")
        raw_expected = row.get("expected")
        if raw_expected is None:
            raise ValueError(f"cases[{i}] missing expected")
        out.append(FitnessCase(inputs=_decode_inputs(raw_inputs), expected=_decode_expected(raw_expected)))
    return out


def _parse_idx_case_entries(raw_case_entries: Any, case_i: int) -> Dict[int, Any]:
    if not isinstance(raw_case_entries, list):
        raise ValueError(f"shared_cases[{case_i}] must be a list")
    out: Dict[int, Any] = {}
    for j, row in enumerate(raw_case_entries):
        if not isinstance(row, dict):
            raise ValueError(f"shared_cases[{case_i}][{j}] must be an object")
        if "idx" not in row or "value" not in row:
            raise ValueError(f"shared_cases[{case_i}][{j}] must include idx/value")
        idx = row["idx"]
        if not isinstance(idx, int):
            raise ValueError(f"shared_cases[{case_i}][{j}].idx must be int")
        out[idx] = _decode_typed_or_raw_value(row["value"])
    return out


def _parse_input_names(raw: str, n: int) -> List[str]:
    if not raw:
        if n == 1:
            return ["x"]
        return [f"x{i}" for i in range(n)]
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if len(names) != n:
        raise ValueError(f"--input-names expects {n} names, got {len(names)}")
    return names


def _parse_indices(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("input indices must not be empty")
    return out


def _auto_pick_indices(case_maps: List[Dict[int, Any]], expected_vals: List[Any]) -> List[int]:
    idxs = sorted({k for m in case_maps for k in m.keys()})
    if not idxs:
        raise ValueError("shared_cases does not contain any idx values")
    if 0 in idxs and len(idxs) > 1:
        equal_all = True
        for m, exp in zip(case_maps, expected_vals):
            if 0 not in m or m[0] != exp:
                equal_all = False
                break
        if equal_all:
            return [i for i in idxs if i != 0]
    return idxs


def _parse_psb2_fixture_cases(payload: Dict[str, Any], input_indices_raw: str, input_names_raw: str) -> List[FitnessCase]:
    root = payload.get("bytecode_program_inputs", payload)
    if not isinstance(root, dict):
        raise ValueError("fixture root must be an object")
    shared_cases = root.get("shared_cases")
    shared_answer = root.get("shared_answer")
    if not isinstance(shared_cases, list) or not isinstance(shared_answer, list):
        raise ValueError("fixture must include shared_cases(list) and shared_answer(list)")
    if len(shared_cases) != len(shared_answer):
        raise ValueError("shared_cases and shared_answer length mismatch")

    expected_vals = [_decode_typed_or_raw_value(v) for v in shared_answer]
    case_maps = [_parse_idx_case_entries(raw_case, i) for i, raw_case in enumerate(shared_cases)]

    if input_indices_raw == "auto":
        input_indices = _auto_pick_indices(case_maps, expected_vals)
    else:
        input_indices = _parse_indices(input_indices_raw)
    input_names = _parse_input_names(input_names_raw, len(input_indices))

    out: List[FitnessCase] = []
    for i, (case_map, exp) in enumerate(zip(case_maps, expected_vals)):
        inputs: Dict[str, Any] = {}
        for idx, name in zip(input_indices, input_names):
            if idx not in case_map:
                raise ValueError(f"shared_cases[{i}] missing idx={idx}")
            inputs[name] = case_map[idx]
        out.append(FitnessCase(inputs=inputs, expected=exp))
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AST evolution from JSON cases and print per-generation best fitness/program.")
    p.add_argument("--cases", required=True, help="Path to JSON file with `cases` list.")
    p.add_argument(
        "--cases-format",
        choices=["auto", "simple", "psb2_fixture"],
        default="auto",
        help="`simple`: {cases:[...]}; `psb2_fixture`: bytecode_program_inputs.shared_cases/shared_answer",
    )
    p.add_argument(
        "--input-indices",
        default="auto",
        help="For psb2_fixture: comma-separated shared_cases idx list. Use auto to infer (drops answer idx=0 when duplicated in expected).",
    )
    p.add_argument(
        "--input-names",
        default="x",
        help="For psb2_fixture: variable names mapped to --input-indices, comma-separated (e.g. x,y).",
    )
    p.add_argument("--population-size", type=int, default=64)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--mutation-rate", type=float, default=0.5)
    p.add_argument("--crossover-rate", type=float, default=0.9)
    p.add_argument("--crossover-method", default="hybrid", choices=["top_level_splice", "hybrid"])
    p.add_argument("--selection", default=SelectionMethod.TOURNAMENT.value, choices=[m.value for m in SelectionMethod])
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
    p.add_argument("--out-json", default="", help="Optional output summary JSON path.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    case_path = ROOT / args.cases
    if not case_path.exists():
        raise SystemExit(f"missing cases file: {case_path}")

    try:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        cases: List[FitnessCase]
        if args.cases_format == "simple":
            cases = _parse_cases(payload)
        elif args.cases_format == "psb2_fixture":
            cases = _parse_psb2_fixture_cases(payload, args.input_indices, args.input_names)
        else:
            if isinstance(payload, dict) and "cases" in payload:
                cases = _parse_cases(payload)
            else:
                cases = _parse_psb2_fixture_cases(payload, args.input_indices, args.input_names)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid cases payload: {exc}")

    cfg = EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        elitism=args.elitism,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        crossover_method=args.crossover_method,
        selection_method=SelectionMethod(args.selection),
        tournament_k=args.tournament_k,
        truncation_ratio=args.truncation_ratio,
        seed=args.seed,
        fuel=args.fuel,
        limits=Limits(
            max_expr_depth=args.max_expr_depth,
            max_stmts_per_block=args.max_stmts_per_block,
            max_total_nodes=args.max_total_nodes,
            max_for_k=args.max_for_k,
            max_call_args=args.max_call_args,
        ),
    )

    result = evolve_population(cases, cfg)

    history_rows: List[Dict[str, Any]] = []
    for i, (best_s, best_fit, mean_fit) in enumerate(
        zip(result.history_best, result.history_best_fitness, result.history_mean_fitness)
    ):
        row = {
            "generation": i,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "hash_key": best_s.genome.meta.hash_key,
        }
        history_rows.append(row)
        print(
            f"GEN {i:03d} best={best_fit:.6f} mean={mean_fit:.6f} hash={best_s.genome.meta.hash_key}"
        )
        if args.show_program in ("ast", "both"):
            print(f"AST {i:03d}: {best_s.genome.ast!r}")
        if args.show_program in ("bytecode", "both"):
            bc = compile_for_eval(best_s.genome)
            print(
                f"BYTECODE {i:03d}: n_locals={bc.n_locals} consts={len(bc.consts)} code={len(bc.code)}"
            )
            print(
                "BYTECODE_HEAD "
                + " ".join(
                    f"{idx}:{ins.op}"
                    for idx, ins in enumerate(bc.code[: min(12, len(bc.code))])
                )
            )

    print(
        f"FINAL best={result.best.fitness:.6f} hash={result.best.genome.meta.hash_key} "
        f"selection={cfg.selection_method.value} crossover={cfg.crossover_method}"
    )

    if args.out_json:
        out_path = ROOT / args.out_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "meta": {
                "cases_path": str(args.cases),
                "population_size": cfg.population_size,
                "generations": cfg.generations,
                "selection": cfg.selection_method.value,
                "crossover_method": cfg.crossover_method,
                "seed": cfg.seed,
            },
            "history": history_rows,
            "final": {
                "best_fitness": result.best.fitness,
                "hash_key": result.best.genome.meta.hash_key,
                "ast_repr": repr(result.best.genome.ast),
            },
        }
        out_path.write_text(json.dumps(out_payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
