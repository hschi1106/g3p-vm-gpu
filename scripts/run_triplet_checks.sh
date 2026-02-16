#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Python unit tests"
G3PVM_RUN_PSB2_LIVE=1 PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v

echo "[2/5] Build and run native C++ VM tests"
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure

echo "[3/5] Generate PSB2 fitness fixture JSON (bouncing-balls)"
PYTHONPATH=python python3 tools/gen_psb2_fitness_multi_bench_inputs.py \
  --psb2-root data/psb2_datasets \
  --out data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --require-psb2-fetch

echo "[4/5] Validate CPU/GPU multi + fitness from generated fixture JSON"
PYTHONPATH=python python3 tools/check_multi_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cli cpp/build/g3pvm_vm_cpu_cli

PYTHONPATH=python python3 tools/check_fitness_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cli cpp/build/g3pvm_vm_cpu_cli

echo "[5/5] Benchmark CPU/GPU fitness from generated fixture JSON"
mkdir -p logs
BENCH_LOG="logs/fitness_bench_subtract_parse_$(date +%Y%m%d_%H%M%S).log"
PYTHONPATH=python python3 tools/bench_fitness_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cli cpp/build/g3pvm_vm_cpu_cli \
  --runs 3 \
  --blocksize 256 \
  --subtract-parse | tee "$BENCH_LOG"
echo "benchmark log: $BENCH_LOG"

echo "triplet checks: OK"
