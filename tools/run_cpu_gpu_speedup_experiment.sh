#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CASES="data/fixtures/fitness_multi_bench_inputs_psb2.json"
CASES_FORMAT="psb2_fixture"
INPUT_INDICES="1"
INPUT_NAMES="x"
CPP_CLI="cpp/build/g3pvm_evolve_cli"
SELECTION="tournament"
CROSSOVER_METHOD="hybrid"
POPSIZE=4096
GENERATIONS=40
BLOCKSIZE=256
CPP_TIMING="all"
OUTDIR=""

usage() {
  cat <<USAGE
Usage: tools/run_cpu_gpu_speedup_experiment.sh [options]

Run CPU/GPU speedup experiment for C++ evolution and write logs + compare report.

Options:
  --popsize N            Population size (default: 4096)
  --generations N        Generations (default: 40)
  --blocksize N          GPU blocksize (default: 256)
  --cases PATH           Cases JSON path (default: data/fixtures/fitness_multi_bench_inputs_psb2.json)
  --input-indices STR    Input indices for psb2 fixture (default: 1)
  --input-names STR      Input names for psb2 fixture (default: x)
  --selection STR        Selection method (default: tournament)
  --crossover-method STR Crossover method (default: hybrid)
  --cpp-cli PATH         Evolve CLI path (default: cpp/build/g3pvm_evolve_cli)
  --outdir PATH          Output directory (default: logs/cpu_gpu_compare_pop<pop>_<timestamp>)
  --help                 Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --popsize)
      POPSIZE="$2"; shift 2 ;;
    --generations)
      GENERATIONS="$2"; shift 2 ;;
    --blocksize)
      BLOCKSIZE="$2"; shift 2 ;;
    --cases)
      CASES="$2"; shift 2 ;;
    --input-indices)
      INPUT_INDICES="$2"; shift 2 ;;
    --input-names)
      INPUT_NAMES="$2"; shift 2 ;;
    --selection)
      SELECTION="$2"; shift 2 ;;
    --crossover-method)
      CROSSOVER_METHOD="$2"; shift 2 ;;
    --cpp-cli)
      CPP_CLI="$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
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
if [[ ! -x "$CPP_CLI" ]]; then
  echo "Missing or non-executable cpp cli: $CPP_CLI" >&2
  exit 2
fi

echo "[exp] outdir: $OUTDIR"
echo "[exp] popsize=$POPSIZE generations=$GENERATIONS blocksize=$BLOCKSIZE"

echo "[exp] running CPU baseline..."
python3 tools/run_cpp_evolution.py \
  --cases "$CASES" \
  --cases-format "$CASES_FORMAT" \
  --input-indices "$INPUT_INDICES" \
  --input-names "$INPUT_NAMES" \
  --cpp-cli "$CPP_CLI" \
  --engine cpu \
  --selection "$SELECTION" \
  --crossover-method "$CROSSOVER_METHOD" \
  --population-size "$POPSIZE" \
  --generations "$GENERATIONS" \
  --cpp-timing "$CPP_TIMING" \
  --log-dir "$OUTDIR" \
  --run-tag cpu | tee "$OUTDIR/cpu_run.console.log"

echo "[exp] running GPU run..."
set +e
scripts/run_gpu_command.sh -- python3 tools/run_cpp_evolution.py \
  --cases "$CASES" \
  --cases-format "$CASES_FORMAT" \
  --input-indices "$INPUT_INDICES" \
  --input-names "$INPUT_NAMES" \
  --cpp-cli "$CPP_CLI" \
  --engine gpu \
  --blocksize "$BLOCKSIZE" \
  --selection "$SELECTION" \
  --crossover-method "$CROSSOVER_METHOD" \
  --population-size "$POPSIZE" \
  --generations "$GENERATIONS" \
  --cpp-timing "$CPP_TIMING" \
  --log-dir "$OUTDIR" \
  --run-tag gpu | tee "$OUTDIR/gpu_run.console.log"
GPU_RC=$?
set -e

if [[ $GPU_RC -ne 0 ]]; then
  echo "[exp] GPU run failed (rc=$GPU_RC). CPU logs are still available in $OUTDIR" >&2
  exit $GPU_RC
fi

echo "[exp] generating compare report..."
python3 - "$OUTDIR" <<'PY'
import glob
import json
import os
import sys

outdir = sys.argv[1]
cpu_candidates = glob.glob(os.path.join(outdir, 'cpp_evo_cpu_pid*.summary.json'))
gpu_candidates = glob.glob(os.path.join(outdir, 'cpp_evo_gpu_pid*.summary.json'))
if not cpu_candidates:
    raise SystemExit('missing CPU summary json')
if not gpu_candidates:
    raise SystemExit('missing GPU summary json')

cpu_sum = cpu_candidates[0]
gpu_sum = gpu_candidates[0]

with open(cpu_sum, 'r', encoding='utf-8') as f:
    cpu = json.load(f)
with open(gpu_sum, 'r', encoding='utf-8') as f:
    gpu = json.load(f)

def phase(d, name):
    return d['parsed']['timing_summary'].get(name)

def find_outer(d):
    for x in d['timings']:
        if x['stage'] == 'run_cpp_cli':
            return x['elapsed_ms']
    return None

def speedup(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b

cpu_total = phase(cpu, 'total')
gpu_total = phase(gpu, 'total')
cpu_eval = phase(cpu, 'generations_eval_total')
gpu_eval = phase(gpu, 'generations_eval_total')
cpu_repro = phase(cpu, 'generations_repro_total')
gpu_repro = phase(gpu, 'generations_repro_total')

report = {
    'cpu_summary': cpu_sum,
    'gpu_summary': gpu_sum,
    'cpu': {
        'inner_total_ms': cpu_total,
        'inner_eval_ms': cpu_eval,
        'inner_repro_ms': cpu_repro,
        'outer_run_cpp_cli_ms': find_outer(cpu),
    },
    'gpu': {
        'inner_total_ms': gpu_total,
        'inner_eval_ms': gpu_eval,
        'inner_repro_ms': gpu_repro,
        'gpu_session_init_ms': phase(gpu, 'gpu_session_init'),
        'gpu_program_compile_total_ms': phase(gpu, 'gpu_generations_program_compile_total'),
        'gpu_pack_upload_total_ms': phase(gpu, 'gpu_generations_pack_upload_total'),
        'gpu_kernel_total_ms': phase(gpu, 'gpu_generations_kernel_total'),
        'gpu_copyback_total_ms': phase(gpu, 'gpu_generations_copyback_total'),
        'outer_run_cpp_cli_ms': find_outer(gpu),
    },
    'speedup': {
        'inner_total_cpu_over_gpu': speedup(cpu_total, gpu_total),
        'inner_eval_cpu_over_gpu': speedup(cpu_eval, gpu_eval),
        'outer_run_cpp_cli_cpu_over_gpu': speedup(find_outer(cpu), find_outer(gpu)),
    },
}

json_path = os.path.join(outdir, 'cpu_gpu_compare.report.json')
md_path = os.path.join(outdir, 'cpu_gpu_compare.report.md')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=True, indent=2)

lines = []
lines.append('# CPU vs GPU Evolution Timing Comparison')
lines.append('')
lines.append(f'- CPU summary: `{cpu_sum}`')
lines.append(f'- GPU summary: `{gpu_sum}`')
lines.append('')
lines.append('## Key Metrics (ms)')
lines.append('')
lines.append(f'- CPU inner total: {cpu_total:.3f}')
lines.append(f'- GPU inner total: {gpu_total:.3f}')
lines.append(f'- CPU generations eval total: {cpu_eval:.3f}')
lines.append(f'- GPU generations eval total: {gpu_eval:.3f}')
lines.append(f'- CPU generations repro total: {cpu_repro:.3f}')
lines.append(f'- GPU generations repro total: {gpu_repro:.3f}')
lines.append('')
lines.append('## Speedup')
lines.append('')
for k, v in report['speedup'].items():
    if v is None:
      lines.append(f'- {k}: n/a')
    else:
      lines.append(f'- {k}: {v:.3f}x')
lines.append('')
lines.append('## GPU Breakdown (ms)')
lines.append('')
for k in ['gpu_session_init_ms', 'gpu_program_compile_total_ms', 'gpu_pack_upload_total_ms', 'gpu_kernel_total_ms', 'gpu_copyback_total_ms']:
    val = report['gpu'][k]
    if val is None:
      lines.append(f'- {k}: n/a')
    else:
      lines.append(f'- {k}: {val:.3f}')

with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(md_path)
PY

echo "[exp] done. compare report: $OUTDIR/cpu_gpu_compare.report.md"
