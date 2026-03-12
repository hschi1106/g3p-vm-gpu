#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CASES="data/fixtures/bouncing_balls_1024.json"
GENERATOR_CLI="cpp/build/g3pvm_generate_population_cli"
BENCH_CLI="cpp/build/g3pvm_population_bench_cli"
POPSIZE=1024
GENERATIONS=1
BLOCKSIZE=256
OUTDIR=""
SEED_START=0
PROBE_CASES=32
MIN_SUCCESS_RATE=0.10
FUEL=20000
MAX_EXPR_DEPTH=5
MAX_STMTS_PER_BLOCK=6
MAX_TOTAL_NODES=80
MAX_FOR_K=16
MAX_CALL_ARGS=3
MUTATION_RATE=0.5
MUTATION_SUBTREE_PROB=0.8
CROSSOVER_RATE=0.9
PENALTY=1.0
SELECTION_PRESSURE=3

usage() {
  cat <<USAGE
Usage: scripts/run_cpu_gpu_speedup_experiment.sh [options]

Run fixed-population CPU/GPU speed benchmark and write logs + compare report.

Primary metrics:
  - eval-only
  - one-gen-e2e

Options:
  --cases PATH                 Benchmark fixture (default: data/fixtures/bouncing_balls_1024.json)
  --popsize N                  Population size for generated fixed population (default: 1024)
  --generations N              Accepted for compatibility; ignored by this benchmark (default: 1)
  --blocksize N                GPU blocksize (default: 256)
  --generator-cli PATH         Population generator CLI (default: cpp/build/g3pvm_generate_population_cli)
  --bench-cli PATH             Population benchmark CLI (default: cpp/build/g3pvm_population_bench_cli)
  --cpp-cli PATH               Alias for --bench-cli for compatibility
  --outdir PATH                Output directory (default: logs/cpu_gpu_compare_pop<pop>_<timestamp>)
  --seed-start N               First RNG seed probed for population generation (default: 0)
  --probe-cases N              Cases sampled when filtering generated programs (default: 32)
  --min-success-rate F         Minimum non-error probe success ratio per program (default: 0.10)
  --fuel N                     Per-program execution budget (default: 20000)
  --max-expr-depth N           Generator max expression depth (default: 5)
  --max-stmts-per-block N      Generator max statements per block (default: 6)
  --max-total-nodes N          Generator max AST nodes (default: 80)
  --max-for-k N                Generator max for-range constant (default: 16)
  --max-call-args N            Generator max builtin call arity (default: 3)
  --mutation-rate F            One-gen benchmark mutation rate (default: 0.5)
  --mutation-subtree-prob F    One-gen benchmark mutation subtree probability (default: 0.8)
  --crossover-rate F           One-gen benchmark crossover rate (default: 0.9)
  --penalty F                  Fitness penalty (default: 1.0)
  --selection-pressure N       Tournament size (default: 3)
  --help                       Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases)
      CASES="$2"; shift 2 ;;
    --popsize)
      POPSIZE="$2"; shift 2 ;;
    --generations)
      GENERATIONS="$2"; shift 2 ;;
    --blocksize)
      BLOCKSIZE="$2"; shift 2 ;;
    --generator-cli)
      GENERATOR_CLI="$2"; shift 2 ;;
    --bench-cli)
      BENCH_CLI="$2"; shift 2 ;;
    --cpp-cli)
      BENCH_CLI="$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    --seed-start)
      SEED_START="$2"; shift 2 ;;
    --probe-cases)
      PROBE_CASES="$2"; shift 2 ;;
    --min-success-rate)
      MIN_SUCCESS_RATE="$2"; shift 2 ;;
    --fuel)
      FUEL="$2"; shift 2 ;;
    --max-expr-depth)
      MAX_EXPR_DEPTH="$2"; shift 2 ;;
    --max-stmts-per-block)
      MAX_STMTS_PER_BLOCK="$2"; shift 2 ;;
    --max-total-nodes)
      MAX_TOTAL_NODES="$2"; shift 2 ;;
    --max-for-k)
      MAX_FOR_K="$2"; shift 2 ;;
    --max-call-args)
      MAX_CALL_ARGS="$2"; shift 2 ;;
    --mutation-rate)
      MUTATION_RATE="$2"; shift 2 ;;
    --mutation-subtree-prob)
      MUTATION_SUBTREE_PROB="$2"; shift 2 ;;
    --crossover-rate)
      CROSSOVER_RATE="$2"; shift 2 ;;
    --penalty)
      PENALTY="$2"; shift 2 ;;
    --selection-pressure)
      SELECTION_PRESSURE="$2"; shift 2 ;;
    --help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="logs/cpu_gpu_compare_pop${POPSIZE}_${TS}"
fi
mkdir -p "$OUTDIR"

if [[ ! -f "$CASES" ]]; then
  echo "Missing cases file: $CASES" >&2
  exit 2
fi
if [[ ! -x "$GENERATOR_CLI" ]]; then
  echo "Missing or non-executable generator cli: $GENERATOR_CLI" >&2
  exit 2
fi
if [[ ! -x "$BENCH_CLI" ]]; then
  echo "Missing or non-executable bench cli: $BENCH_CLI" >&2
  exit 2
fi

POP_JSON="$OUTDIR/fixed_population.seeds.json"

echo "[exp] outdir: $OUTDIR"
echo "[exp] popsize=$POPSIZE blocksize=$BLOCKSIZE generations_ignored=$GENERATIONS"
echo "[exp] primary_metrics=eval-only,one-gen-e2e"

echo "[exp] generating fixed population..."
"$GENERATOR_CLI" \
  --cases "$CASES" \
  --out-json "$POP_JSON" \
  --population-size "$POPSIZE" \
  --seed-start "$SEED_START" \
  --probe-cases "$PROBE_CASES" \
  --min-success-rate "$MIN_SUCCESS_RATE" \
  --fuel "$FUEL" \
  --max-expr-depth "$MAX_EXPR_DEPTH" \
  --max-stmts-per-block "$MAX_STMTS_PER_BLOCK" \
  --max-total-nodes "$MAX_TOTAL_NODES" \
  --max-for-k "$MAX_FOR_K" \
  --max-call-args "$MAX_CALL_ARGS" | tee "$OUTDIR/generate_population.console.log"

run_bench() {
  local engine="$1"
  local mode="$2"
  local outfile="$3"
  local -a cmd=(
    "$BENCH_CLI"
    --cases "$CASES"
    --population-json "$POP_JSON"
    --mode "$mode"
    --engine "$engine"
    --blocksize "$BLOCKSIZE"
    --fuel "$FUEL"
    --mutation-rate "$MUTATION_RATE"
    --mutation-subtree-prob "$MUTATION_SUBTREE_PROB"
    --crossover-rate "$CROSSOVER_RATE"
    --penalty "$PENALTY"
    --selection-pressure "$SELECTION_PRESSURE"
  )
  if [[ "$engine" == "gpu" ]]; then
    scripts/run_gpu_command.sh -- "${cmd[@]}" | tee "$outfile"
  else
    "${cmd[@]}" | tee "$outfile"
  fi
}

echo "[exp] running eval-only CPU..."
run_bench cpu eval-only "$OUTDIR/eval_only_cpu.console.log"

echo "[exp] running eval-only GPU..."
run_bench gpu eval-only "$OUTDIR/eval_only_gpu.console.log"

echo "[exp] running one-gen-e2e CPU..."
run_bench cpu one-gen-e2e "$OUTDIR/one_gen_cpu.console.log"

echo "[exp] running one-gen-e2e GPU..."
run_bench gpu one-gen-e2e "$OUTDIR/one_gen_gpu.console.log"

echo "[exp] generating compare report..."
python3 - "$OUTDIR" <<'PY'
import json
import os
import re
import sys

outdir = sys.argv[1]

BENCH_RE = re.compile(r"^BENCH\s+(?P<body>.+)$")

def parse_value(raw: str):
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw

def parse_bench(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = BENCH_RE.match(line.strip())
            if not m:
                continue
            body = {}
            for token in m.group("body").split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                body[k] = parse_value(v)
            return body
    raise SystemExit(f"missing BENCH line in {path}")

def speedup(cpu_ms, gpu_ms):
    if cpu_ms is None or gpu_ms in (None, 0):
        return None
    return cpu_ms / gpu_ms

eval_cpu = parse_bench(os.path.join(outdir, "eval_only_cpu.console.log"))
eval_gpu = parse_bench(os.path.join(outdir, "eval_only_gpu.console.log"))
one_cpu = parse_bench(os.path.join(outdir, "one_gen_cpu.console.log"))
one_gpu = parse_bench(os.path.join(outdir, "one_gen_gpu.console.log"))

report = {
    "benchmark_type": "fixed_population_compare_v1",
    "population_json": os.path.join(outdir, "fixed_population.seeds.json"),
    "cpu": {
        "compile_ms": eval_cpu.get("compile_ms"),
        "eval_ms": eval_cpu.get("eval_ms"),
        "eval_only_ms": eval_cpu.get("total_ms"),
        "eval_only_checksum": eval_cpu.get("checksum"),
        "eval_pack_upload_ms": eval_cpu.get("pack_upload_ms"),
        "eval_kernel_ms": eval_cpu.get("kernel_ms"),
        "eval_copyback_ms": eval_cpu.get("copyback_ms"),
        "eval_session_init_ms": eval_cpu.get("session_init_ms"),
        "one_gen_e2e_total_ms": one_cpu.get("total_ms"),
        "one_gen_e2e_compile_ms": one_cpu.get("compile_ms"),
        "one_gen_e2e_eval_ms": one_cpu.get("eval_ms"),
        "one_gen_e2e_repro_ms": one_cpu.get("repro_ms"),
        "one_gen_e2e_selection_ms": one_cpu.get("selection_ms"),
        "one_gen_e2e_crossover_ms": one_cpu.get("crossover_ms"),
        "one_gen_e2e_mutation_ms": one_cpu.get("mutation_ms"),
        "one_gen_e2e_pack_upload_ms": one_cpu.get("pack_upload_ms"),
        "one_gen_e2e_kernel_ms": one_cpu.get("kernel_ms"),
        "one_gen_e2e_copyback_ms": one_cpu.get("copyback_ms"),
        "one_gen_e2e_session_init_ms": one_cpu.get("session_init_ms"),
        "one_gen_e2e_mean_fitness": one_cpu.get("mean_fitness"),
        "one_gen_e2e_best_fitness": one_cpu.get("best_fitness"),
        "one_gen_e2e_best_program_key": one_cpu.get("best_program_key"),
    },
    "gpu": {
        "compile_ms": eval_gpu.get("compile_ms"),
        "eval_ms": eval_gpu.get("eval_ms"),
        "eval_only_ms": eval_gpu.get("total_ms"),
        "eval_only_checksum": eval_gpu.get("checksum"),
        "eval_pack_upload_ms": eval_gpu.get("pack_upload_ms"),
        "eval_kernel_ms": eval_gpu.get("kernel_ms"),
        "eval_copyback_ms": eval_gpu.get("copyback_ms"),
        "eval_session_init_ms": eval_gpu.get("session_init_ms"),
        "one_gen_e2e_total_ms": one_gpu.get("total_ms"),
        "one_gen_e2e_compile_ms": one_gpu.get("compile_ms"),
        "one_gen_e2e_eval_ms": one_gpu.get("eval_ms"),
        "one_gen_e2e_repro_ms": one_gpu.get("repro_ms"),
        "one_gen_e2e_selection_ms": one_gpu.get("selection_ms"),
        "one_gen_e2e_crossover_ms": one_gpu.get("crossover_ms"),
        "one_gen_e2e_mutation_ms": one_gpu.get("mutation_ms"),
        "one_gen_e2e_pack_upload_ms": one_gpu.get("pack_upload_ms"),
        "one_gen_e2e_kernel_ms": one_gpu.get("kernel_ms"),
        "one_gen_e2e_copyback_ms": one_gpu.get("copyback_ms"),
        "one_gen_e2e_session_init_ms": one_gpu.get("session_init_ms"),
        "one_gen_e2e_mean_fitness": one_gpu.get("mean_fitness"),
        "one_gen_e2e_best_fitness": one_gpu.get("best_fitness"),
        "one_gen_e2e_best_program_key": one_gpu.get("best_program_key"),
    },
}
report["speedup"] = {
    "compile_cpu_over_gpu": speedup(report["cpu"]["compile_ms"], report["gpu"]["compile_ms"]),
    "eval_cpu_over_gpu": speedup(report["cpu"]["eval_ms"], report["gpu"]["eval_ms"]),
    "repro_cpu_over_gpu": speedup(report["cpu"]["one_gen_e2e_repro_ms"], report["gpu"]["one_gen_e2e_repro_ms"]),
    "eval_only_cpu_over_gpu": speedup(report["cpu"]["eval_only_ms"], report["gpu"]["eval_only_ms"]),
    "one_gen_e2e_cpu_over_gpu": speedup(report["cpu"]["one_gen_e2e_total_ms"], report["gpu"]["one_gen_e2e_total_ms"]),
    "inner_total_cpu_over_gpu": speedup(report["cpu"]["one_gen_e2e_total_ms"], report["gpu"]["one_gen_e2e_total_ms"]),
}

json_path = os.path.join(outdir, "cpu_gpu_compare.report.json")
md_path = os.path.join(outdir, "cpu_gpu_compare.report.md")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=True, indent=2)

def fmt_ms(v):
    return "n/a" if v is None else f"{v:.3f}"

def fmt_speed(v):
    return "n/a" if v is None else f"{v:.3f}x"

lines = []
lines.append("# CPU vs GPU Fixed-Population Benchmark")
lines.append("")
lines.append(f"- population_json: `{report['population_json']}`")
lines.append(f"- benchmark_type: `{report['benchmark_type']}`")
lines.append("")
lines.append("## Speedup")
lines.append("")
lines.append(f"- compile_cpu_over_gpu: {fmt_speed(report['speedup']['compile_cpu_over_gpu'])}")
lines.append(f"- eval_cpu_over_gpu: {fmt_speed(report['speedup']['eval_cpu_over_gpu'])}")
lines.append(f"- repro_cpu_over_gpu: {fmt_speed(report['speedup']['repro_cpu_over_gpu'])}")
lines.append(f"- eval_only_cpu_over_gpu: {fmt_speed(report['speedup']['eval_only_cpu_over_gpu'])}")
lines.append(f"- one_gen_e2e_cpu_over_gpu: {fmt_speed(report['speedup']['one_gen_e2e_cpu_over_gpu'])}")
lines.append("")
lines.append("## CPU")
lines.append("")
for key, value in report["cpu"].items():
    lines.append(f"- {key}: {value if not isinstance(value, float) else fmt_ms(value)}")
lines.append("")
lines.append("## GPU")
lines.append("")
for key, value in report["gpu"].items():
    lines.append(f"- {key}: {value if not isinstance(value, float) else fmt_ms(value)}")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"[exp] report json: {json_path}")
print(f"[exp] report md: {md_path}")
PY

echo "[exp] done"
echo "[exp] report: $OUTDIR/cpu_gpu_compare.report.md"
