#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Python unit tests"
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v

echo "[2/5] Build and run native C++ VM tests"
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure

echo "[3/5] Verify speedup fixture schema"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("data/fixtures/speedup_cases_bouncing_balls_1024.json")
payload = json.loads(p.read_text(encoding="utf-8"))
assert payload["format_version"] == "fitness-cases-v1"
assert isinstance(payload["cases"], list) and payload["cases"]
print("fixture schema: OK")
PY

echo "[4/5] Run CPU/GPU speedup experiment smoke"
bash tools/run_cpu_gpu_speedup_experiment.sh \
  --cases data/fixtures/speedup_cases_bouncing_balls_1024.json \
  --popsize 64 \
  --generations 2 \
  --outdir logs/triplet_speedup_smoke

echo "[5/5] Validate speedup report exists"
test -f logs/triplet_speedup_smoke/cpu_gpu_compare.report.json
test -f logs/triplet_speedup_smoke/cpu_gpu_compare.report.md

echo "triplet checks: OK"
