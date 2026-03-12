#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/exp"
RESULTS_DIR="${EXP_DIR}/results"
BENCH_BIN="${EXP_DIR}/bin/g3pvm_repro_proto_bench"

bash "${EXP_DIR}/build_repro_proto_bench.sh"

mkdir -p "${RESULTS_DIR}"

G3PVM_CUDA_DEVICE="${G3PVM_CUDA_DEVICE:-0}" \
  "${BENCH_BIN}" \
  --population-size 1024 \
  --child-count 1024 \
  --out-json "${RESULTS_DIR}/current_pop1024.json"

G3PVM_CUDA_DEVICE="${G3PVM_CUDA_DEVICE:-0}" \
  "${BENCH_BIN}" \
  --population-size 4096 \
  --child-count 4096 \
  --out-json "${RESULTS_DIR}/current_pop4096.json"

python3 "${EXP_DIR}/write_current_summary.py"

echo "updated ${RESULTS_DIR}/current_pop1024.json"
echo "updated ${RESULTS_DIR}/current_pop4096.json"
echo "updated ${RESULTS_DIR}/complete_summary.md"
