#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact-depth/payload bucket populations and benchmark GPU eval kernel cost."
    )
    parser.add_argument("--cases", default="data/fixtures/simple_x_plus_1_1024.json")
    parser.add_argument("--bucket-cli", default="cpp/build/g3pvm_population_bucket_cli")
    parser.add_argument("--bench-cli", default="cpp/build/g3pvm_population_bench_cli")
    parser.add_argument("--population-size", type=int, default=1024)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument("--depths", default="5,7,9,11,13")
    parser.add_argument("--payload-flavors", default="none,string,list,mixed")
    parser.add_argument("--probe-cases", type=int, default=32)
    parser.add_argument("--min-success-rate", type=float, default=0.5)
    parser.add_argument("--fuel", type=int, default=20000)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--max-stmts-per-block", type=int, default=6)
    parser.add_argument("--max-total-nodes", type=int, default=80)
    parser.add_argument("--max-for-k", type=int, default=16)
    parser.add_argument("--max-call-args", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--population-outdir", default="")
    parser.add_argument("--outdir", default="")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_bench(stdout_text: str) -> dict[str, Any]:
    for line in stdout_text.splitlines():
        if not line.startswith("BENCH "):
            continue
        out: dict[str, Any] = {}
        for token in line[len("BENCH ") :].split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                if any(ch in value for ch in ".eE"):
                    out[key] = float(value)
                else:
                    out[key] = int(value)
            except ValueError:
                out[key] = value
        return out
    raise RuntimeError("missing BENCH line")


def run(cmd: list[str], *, cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}; see {log_path}")
    return proc


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def quantile_bucket(value: float, cutpoints: list[float]) -> int:
    for idx, cut in enumerate(cutpoints):
        if value <= cut:
            return idx
    return len(cutpoints)


def group_r2(rows: list[dict[str, Any]], key_fn) -> float:
    ys = [row["kernel_ms_median"] for row in rows]
    y_mean = mean(ys)
    sst = sum((y - y_mean) ** 2 for y in ys)
    if sst <= 0:
        return 1.0
    groups: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row["kernel_ms_median"])
    sse = 0.0
    for row in rows:
        bucket = groups[key_fn(row)]
        pred = mean(bucket)
        sse += (row["kernel_ms_median"] - pred) ** 2
    return 1.0 - (sse / sst)


def metadata_summary(meta: dict[str, Any]) -> dict[str, float]:
    programs = meta["programs"]
    node_counts = [float(item["node_count"]) for item in programs]
    code_lens = [float(item["code_len"]) for item in programs]
    depths = [float(item["actual_depth"]) for item in programs]
    return {
        "node_count_mean": mean(node_counts),
        "node_count_median": statistics.median(node_counts),
        "node_count_p95": percentile(node_counts, 0.95),
        "code_len_mean": mean(code_lens),
        "code_len_median": statistics.median(code_lens),
        "code_len_p95": percentile(code_lens, 0.95),
        "actual_depth_mean": mean(depths),
    }


def main() -> int:
    args = parse_args()
    cases = resolve(args.cases)
    bucket_cli = resolve(args.bucket_cli)
    bench_cli = resolve(args.bench_cli)
    depths = parse_int_list(args.depths)
    payload_flavors = parse_str_list(args.payload_flavors)

    population_outdir = (
        resolve(args.population_outdir)
        if args.population_outdir
        else ROOT / "data" / "exp" / f"kernel_bucket_grid_{cases.stem}"
    )
    outdir = (
        resolve(args.outdir)
        if args.outdir
        else ROOT / "logs" / f"kernel_bucket_grid_{cases.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    population_outdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cases": str(cases),
        "population_size": args.population_size,
        "replicates": args.replicates,
        "bench_repeats": args.bench_repeats,
        "depths": depths,
        "payload_flavors": payload_flavors,
        "probe_cases": args.probe_cases,
        "min_success_rate": args.min_success_rate,
        "fuel": args.fuel,
        "blocksize": args.blocksize,
        "max_stmts_per_block": args.max_stmts_per_block,
        "max_total_nodes": args.max_total_nodes,
        "max_for_k": args.max_for_k,
        "max_call_args": args.max_call_args,
        "generator_mode": "synthetic",
    }
    (population_outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    population_rows: list[dict[str, Any]] = []
    for depth in depths:
        for payload_index, payload_flavor in enumerate(payload_flavors):
            for rep in range(args.replicates):
                stem = f"depth{depth:02d}_{payload_flavor}_rep{rep}"
                population_json = population_outdir / f"{stem}.population.json"
                metadata_json = population_outdir / f"{stem}.metadata.json"
                bucket_log = outdir / f"{stem}.bucket.console.log"

                seed_start = args.seed_base + depth * 10_000_000 + payload_index * 1_000_000 + rep * 100_000
                print(f"[bucket] depth={depth} flavor={payload_flavor} rep={rep}", flush=True)
                run(
                    [
                        str(bucket_cli),
                        "--cases",
                        str(cases),
                        "--target-depth",
                        str(depth),
                        "--max-expr-depth",
                        str(depth),
                        "--target-payload-flavor",
                        payload_flavor,
                        "--generator-mode",
                        "synthetic",
                        "--population-size",
                        str(args.population_size),
                        "--seed-start",
                        str(seed_start),
                        "--probe-cases",
                        str(args.probe_cases),
                        "--min-success-rate",
                        str(args.min_success_rate),
                        "--max-attempts",
                        str(args.max_attempts),
                        "--fuel",
                        str(args.fuel),
                        "--max-stmts-per-block",
                        str(args.max_stmts_per_block),
                        "--max-total-nodes",
                        str(args.max_total_nodes),
                        "--max-for-k",
                        str(args.max_for_k),
                        "--max-call-args",
                        str(args.max_call_args),
                        "--out-population-json",
                        str(population_json),
                        "--out-metadata-json",
                        str(metadata_json),
                    ],
                    cwd=ROOT,
                    log_path=bucket_log,
                )

                meta = json.loads(metadata_json.read_text(encoding="utf-8"))
                meta_summary = metadata_summary(meta)
                bench_runs: list[dict[str, Any]] = []
                for bench_rep in range(args.bench_repeats):
                    bench_log = outdir / f"{stem}.bench{bench_rep}.console.log"
                    print(f"[bench] depth={depth} flavor={payload_flavor} rep={rep} run={bench_rep}", flush=True)
                    proc = run(
                        [
                            str(bench_cli),
                            "--cases",
                            str(cases),
                            "--population-json",
                            str(population_json),
                            "--engine",
                            "gpu",
                            "--repro-backend",
                            "cpu",
                            "--repro-overlap",
                            "off",
                            "--blocksize",
                            str(args.blocksize),
                            "--fuel",
                            str(args.fuel),
                        ],
                        cwd=ROOT,
                        log_path=bench_log,
                    )
                    bench_runs.append(parse_bench(proc.stdout))

                kernel_samples = [float(run["kernel_ms"]) for run in bench_runs]
                eval_samples = [float(run["eval_ms"]) for run in bench_runs]
                pack_samples = [float(run["pack_upload_ms"]) for run in bench_runs]
                population_rows.append(
                    {
                        "depth": depth,
                        "payload_flavor": payload_flavor,
                        "replicate": rep,
                        "population_json": str(population_json),
                        "metadata_json": str(metadata_json),
                        "attempts": int(meta["attempts"]),
                        "kernel_ms_median": statistics.median(kernel_samples),
                        "eval_ms_median": statistics.median(eval_samples),
                        "pack_upload_ms_median": statistics.median(pack_samples),
                        **meta_summary,
                    }
                )

    # Aggregate by depth/payload cell.
    cell_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in population_rows:
        grouped[(row["depth"], row["payload_flavor"])].append(row)
    for (depth, payload_flavor), rows in sorted(grouped.items()):
        cell_rows.append(
            {
                "depth": depth,
                "payload_flavor": payload_flavor,
                "kernel_ms_median_of_reps": statistics.median([r["kernel_ms_median"] for r in rows]),
                "eval_ms_median_of_reps": statistics.median([r["eval_ms_median"] for r in rows]),
                "pack_upload_ms_median_of_reps": statistics.median([r["pack_upload_ms_median"] for r in rows]),
                "attempts_mean": mean([float(r["attempts"]) for r in rows]),
                "code_len_mean": mean([r["code_len_mean"] for r in rows]),
                "node_count_mean": mean([r["node_count_mean"] for r in rows]),
            }
        )

    code_len_cutpoints = [
        percentile([row["code_len_mean"] for row in population_rows], 0.25),
        percentile([row["code_len_mean"] for row in population_rows], 0.50),
        percentile([row["code_len_mean"] for row in population_rows], 0.75),
    ]
    node_count_cutpoints = [
        percentile([row["node_count_mean"] for row in population_rows], 0.25),
        percentile([row["node_count_mean"] for row in population_rows], 0.50),
        percentile([row["node_count_mean"] for row in population_rows], 0.75),
    ]
    for row in population_rows:
        row["code_len_bucket"] = quantile_bucket(row["code_len_mean"], code_len_cutpoints)
        row["node_count_bucket"] = quantile_bucket(row["node_count_mean"], node_count_cutpoints)

    analysis = {
        "pearson_corr": {
            "depth_vs_kernel_ms": pearson(
                [float(row["depth"]) for row in population_rows],
                [row["kernel_ms_median"] for row in population_rows],
            ),
            "code_len_mean_vs_kernel_ms": pearson(
                [row["code_len_mean"] for row in population_rows],
                [row["kernel_ms_median"] for row in population_rows],
            ),
            "node_count_mean_vs_kernel_ms": pearson(
                [row["node_count_mean"] for row in population_rows],
                [row["kernel_ms_median"] for row in population_rows],
            ),
        },
        "group_r2": {
            "payload": group_r2(population_rows, lambda row: row["payload_flavor"]),
            "depth": group_r2(population_rows, lambda row: row["depth"]),
            "payload_plus_depth": group_r2(population_rows, lambda row: (row["payload_flavor"], row["depth"])),
            "payload_plus_code_len_bucket": group_r2(
                population_rows, lambda row: (row["payload_flavor"], row["code_len_bucket"])
            ),
            "payload_plus_node_count_bucket": group_r2(
                population_rows, lambda row: (row["payload_flavor"], row["node_count_bucket"])
            ),
        },
    }

    summary = {
        "experiment_type": "gpu_eval_kernel_bucket_grid_v1",
        "manifest": manifest,
        "population_rows": population_rows,
        "cell_rows": cell_rows,
        "analysis": analysis,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        "# GPU Eval Kernel Bucket Experiment",
        "",
        f"- cases: `{cases}`",
        f"- population_outdir: `{population_outdir}`",
        f"- outdir: `{outdir}`",
        f"- generator_mode: `{manifest['generator_mode']}`",
        f"- population_size: `{args.population_size}`",
        f"- replicates: `{args.replicates}`",
        f"- bench_repeats: `{args.bench_repeats}`",
        "",
        "## Cell Summary",
        "",
        "| depth | payload | kernel_ms median | eval_ms median | pack_upload_ms median | mean code_len | mean node_count | mean attempts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cell_rows:
        lines.append(
            f"| `{row['depth']}` | `{row['payload_flavor']}` | `{row['kernel_ms_median_of_reps']:.3f}` | `{row['eval_ms_median_of_reps']:.3f}` | `{row['pack_upload_ms_median_of_reps']:.3f}` | `{row['code_len_mean']:.3f}` | `{row['node_count_mean']:.3f}` | `{row['attempts_mean']:.1f}` |"
        )

    lines.extend(
        [
            "",
            "## Feature Analysis",
            "",
            "Pearson correlation with `kernel_ms` across the 60 populations:",
            "",
            f"- `depth`: `{analysis['pearson_corr']['depth_vs_kernel_ms']:.3f}`" if analysis["pearson_corr"]["depth_vs_kernel_ms"] is not None else "- `depth`: n/a",
            f"- `mean_code_len`: `{analysis['pearson_corr']['code_len_mean_vs_kernel_ms']:.3f}`" if analysis["pearson_corr"]["code_len_mean_vs_kernel_ms"] is not None else "- `mean_code_len`: n/a",
            f"- `mean_node_count`: `{analysis['pearson_corr']['node_count_mean_vs_kernel_ms']:.3f}`" if analysis["pearson_corr"]["node_count_mean_vs_kernel_ms"] is not None else "- `mean_node_count`: n/a",
            "",
            "Grouped `R^2` using population-level median `kernel_ms` as the target:",
            "",
            f"- `payload`: `{analysis['group_r2']['payload']:.3f}`",
            f"- `depth`: `{analysis['group_r2']['depth']:.3f}`",
            f"- `payload + depth`: `{analysis['group_r2']['payload_plus_depth']:.3f}`",
            f"- `payload + code_len_bucket`: `{analysis['group_r2']['payload_plus_code_len_bucket']:.3f}`",
            f"- `payload + node_count_bucket`: `{analysis['group_r2']['payload_plus_node_count_bucket']:.3f}`",
        ]
    )

    best_group = max(analysis["group_r2"].items(), key=lambda item: item[1])
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Best grouping by this experiment's `R^2`: `{best_group[0]}` = `{best_group[1]:.3f}`",
            "- Use this result as a dispatch-hint study, not as a final proof of production kernel design. These populations were generated in synthetic exact-depth buckets to make the grid feasible.",
        ]
    )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[kernel-bucket] summary json: {outdir / 'summary.json'}")
    print(f"[kernel-bucket] summary md: {outdir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
