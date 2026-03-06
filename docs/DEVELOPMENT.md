# Development Runbook

## 1. Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compiler
- CUDA toolkit + NVIDIA driver (for GPU paths)

## 2. Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

## 3. Tests

### Python

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
```

### C++

```bash
ctest --test-dir cpp/build --output-on-failure
```

### Recommended full check

```bash
bash scripts/run_triplet_checks.sh
```

## 4. GPU Command Policy

Use wrapper for all GPU commands:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

This retries device 0 then 1 when GPU is unavailable/contended.

## 5. CPU/GPU Speedup Benchmark

```bash
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
```

Validation behavior:

- Public runners use fast evolution operators only.
- `validate_genome` remains available as internal test/debug scaffolding, not as a benchmark/runtime flag.

Current benchmark contract:

- Benchmark fixture: `data/fixtures/speedup_cases_bouncing_balls_1024.json`
- Evolution-progress fixture: `data/fixtures/simple_evo_exp_1024.json`
- Public crossover method: `typed_subtree`
- Public fitness rule: binary exact match per case

Inputs must use `fitness-cases-v1` schema (example: `data/fixtures/speedup_cases_bouncing_balls_1024.json`).

Outputs:

- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.md`
- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.json`
- per-run CPU/GPU summary JSON logs

## 6. Contribution Rules

1. Keep APIs prefix-native (`AstProgram`).
2. Preserve CPU/GPU fitness parity and timing report compatibility.
3. Run Python and C++ tests after behavior changes.
4. For speedup changes, attach generated comparison report.

## 7. v1.0 Release Gate

```bash
python3 tools/run_v1_release_gate.py
```

What it runs:
- Python tests (`unittest discover`)
- C++ tests (`ctest`)
- CPU/GPU speed benchmark (`bouncing-balls-1024`)
- evolution progress benchmark (`exp-1024`)
- PSB2 all-task batch run

Output:
- `logs/v1_release_report/<timestamp>/release_gate.summary.json`
- `logs/v1_release_report/<timestamp>/release_gate.summary.md`

Smoke record (2026-03-05):
- `logs/v1_release_report/20260305T154704Z/release_gate.summary.json`

## 8. PSB2 Pipeline

### 8.1 Convert PSB2 JSONL -> fitness-cases-v1

```bash
python3 tools/convert_psb2_to_fitness_cases.py \
  --edge-file data/psb2_datasets/bouncing-balls/bouncing-balls-edge.json \
  --random-file data/psb2_datasets/bouncing-balls/bouncing-balls-random.json \
  --n-train 1024 \
  --n-test 1024 \
  --seed 0 \
  --out logs/psb2/converted/bouncing-balls.train.json \
  --out-test logs/psb2/converted/bouncing-balls.test.json \
  --summary-json logs/psb2/converted/bouncing-balls.summary.json
```

Notes:
- Converter supports single-output and multi-output (`output1..M`) tasks.
- For multi-output rows, converter packs outputs as list typed value in `expected`.
- Runtime extension adds builtin `len(x)` for `String/List` (`CALL_BUILTIN` id `4`).
- Runtime extension adds builtin `concat(a,b)` for `String/String` and `List/List` (`CALL_BUILTIN` id `5`).
- Runtime extension adds builtin `slice(x,lo,hi)` for `String/List` (`CALL_BUILTIN` id `6`).
- Runtime extension adds builtin `index(x,i)` for `String/List` (`CALL_BUILTIN` id `7`).
- CPU runtime has payload registry support for typed `String/List` values. If payload is present, `concat/slice/index` execute exact payload semantics.
- GPU runtime uploads payload snapshots and enables exact payload execution for `concat/slice/index` via thread-local scratch backed by device global memory; overflow falls back to compact deterministic transport.
- Evolution AST generation now includes `CALL_LEN` / `CALL_CONCAT` / `CALL_SLICE` / `CALL_INDEX` (Python/C++ paths).
- Fitness scoring is binary per case: exact match `+1`, mismatch/error `+0` (CPU/GPU parity).

### 8.2 Run all PSB2 tasks in batch

```bash
python3 tools/run_psb2_all_tasks.py \
  --datasets-root data/psb2_datasets \
  --tasks all \
  --engine gpu \
  --population-size 1024 \
  --generations 20
```

Outputs:
- `summary.json`: machine-readable per-task status (`ok/skipped/unsupported/failed`)
- `summary.md`: human-readable report
- Current smoke baseline (2026-03-05): `logs/psb2_all_tasks/current_all_v5/summary.json` (`total=25, ok=25`)
