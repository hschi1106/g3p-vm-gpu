#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ..shared.schemas import POPULATION_SEEDS


def fnv1a64_hex(text: str) -> str:
    h = 1469598103934665603
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{h:016x}"


def build_seed_rows(count: int, start_seed: int, stride: int) -> List[Dict[str, int]]:
    if count <= 0:
        raise ValueError("--count must be > 0")
    if stride <= 0:
        raise ValueError("--stride must be > 0")
    if start_seed < 0:
        raise ValueError("--start-seed must be >= 0")
    return [{"seed": start_seed + i * stride} for i in range(count)]


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format_version": POPULATION_SEEDS,
        "cases_path": str(args.cases),
        "limits": {
            "max_expr_depth": args.max_expr_depth,
            "max_stmts_per_block": args.max_stmts_per_block,
            "max_total_nodes": args.max_total_nodes,
            "max_for_k": args.max_for_k,
            "max_call_args": args.max_call_args,
        },
        "seeds": build_seed_rows(args.count, args.start_seed, args.stride),
    }
    if args.grammar_config is not None:
        grammar_text = args.grammar_config.read_text(encoding="utf-8")
        payload["grammar_config"] = {
            "path": str(args.grammar_config),
            "hash": fnv1a64_hex(grammar_text),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a population-seeds file for native fixed-pop benchmarks.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grammar-config", type=Path)
    parser.add_argument("--max-expr-depth", type=int, default=7)
    parser.add_argument("--max-stmts-per-block", type=int, default=6)
    parser.add_argument("--max-total-nodes", type=int, default=80)
    parser.add_argument("--max-for-k", type=int, default=16)
    parser.add_argument("--max-call-args", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = build_payload(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"POPULATION_SEEDS_OUT {args.out}")
    print(f"POPULATION_SEEDS_COUNT {args.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
