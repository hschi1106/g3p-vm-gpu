#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CASES="data/fixtures/bouncing_balls_1024.json"
BENCH_CLI="cpp/build/g3pvm_population_bench_cli"
POPSIZE=1024
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

Run one-generation CPU/GPU benchmark on one fixed population and write logs + compare report.

Options:
  --cases PATH                 Benchmark fixture (default: data/fixtures/bouncing_balls_1024.json)
  --popsize N                  Population size for generated fixed population (default: 1024)
  --blocksize N                GPU blocksize (default: 256)
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
    --blocksize)
      BLOCKSIZE="$2"; shift 2 ;;
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
if [[ ! -x "$BENCH_CLI" ]]; then
  echo "Missing or non-executable bench cli: $BENCH_CLI" >&2
  exit 2
fi

POP_JSON="$OUTDIR/fixed_population.seeds.json"

echo "[exp] outdir: $OUTDIR"
echo "[exp] popsize=$POPSIZE blocksize=$BLOCKSIZE"
echo "[exp] primary_metric=one-generation-e2e"

run_bench() {
  local engine="$1"
  local outfile="$2"
  local -a cmd=(
    "$BENCH_CLI"
    --cases "$CASES"
    --engine "$engine"
    --blocksize "$BLOCKSIZE"
    --fuel "$FUEL"
    --mutation-rate "$MUTATION_RATE"
    --mutation-subtree-prob "$MUTATION_SUBTREE_PROB"
    --crossover-rate "$CROSSOVER_RATE"
    --penalty "$PENALTY"
    --selection-pressure "$SELECTION_PRESSURE"
  )
  if [[ -f "$POP_JSON" ]]; then
    cmd+=(--population-json "$POP_JSON")
  else
    cmd+=(
      --population-size "$POPSIZE"
      --seed-start "$SEED_START"
      --probe-cases "$PROBE_CASES"
      --min-success-rate "$MIN_SUCCESS_RATE"
      --max-expr-depth "$MAX_EXPR_DEPTH"
      --max-stmts-per-block "$MAX_STMTS_PER_BLOCK"
      --max-total-nodes "$MAX_TOTAL_NODES"
      --max-for-k "$MAX_FOR_K"
      --max-call-args "$MAX_CALL_ARGS"
      --out-population-json "$POP_JSON"
    )
  fi
  if [[ "$engine" == "gpu" ]]; then
    scripts/run_gpu_command.sh -- "${cmd[@]}" | tee "$outfile"
  else
    "${cmd[@]}" | tee "$outfile"
  fi
}

echo "[exp] running one-gen-e2e CPU..."
run_bench cpu "$OUTDIR/one_gen_cpu.console.log"

echo "[exp] running one-gen-e2e GPU..."
run_bench gpu "$OUTDIR/one_gen_gpu.console.log"

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

cpu = parse_bench(os.path.join(outdir, "one_gen_cpu.console.log"))
gpu = parse_bench(os.path.join(outdir, "one_gen_gpu.console.log"))

report = {
    "benchmark_type": "fixed_population_compare_v2",
    "population_json": os.path.join(outdir, "fixed_population.seeds.json"),
    "cpu": cpu,
    "gpu": gpu,
    "speedup": {
        "compile_cpu_over_gpu": speedup(cpu.get("compile_ms"), gpu.get("compile_ms")),
        "eval_cpu_over_gpu": speedup(cpu.get("eval_ms"), gpu.get("eval_ms")),
        "repro_cpu_over_gpu": speedup(cpu.get("repro_ms"), gpu.get("repro_ms")),
        "selection_cpu_over_gpu": speedup(cpu.get("selection_ms"), gpu.get("selection_ms")),
        "crossover_cpu_over_gpu": speedup(cpu.get("crossover_ms"), gpu.get("crossover_ms")),
        "mutation_cpu_over_gpu": speedup(cpu.get("mutation_ms"), gpu.get("mutation_ms")),
        "total_cpu_over_gpu": speedup(cpu.get("total_ms"), gpu.get("total_ms")),
    },
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
for key, value in report["speedup"].items():
    lines.append(f"- {key}: {fmt_speed(value)}")
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
