# Development Runbook

## Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compiler
- CUDA toolkit + NVIDIA driver for GPU paths

## Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

## Tests

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

## GPU Command Policy

Use the wrapper for all GPU commands:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

This retries device `0` then `1` when GPU is unavailable or contended.

## Benchmark Contract

Canonical speed benchmark:

```bash
scripts/run_gpu_command.sh -- bash scripts/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
```

Canonical evolution-progress run:

```bash
scripts/run_gpu_command.sh -- python3 tools/run_cpp_evolution.py --cases data/fixtures/simple_evo_exp_1024.json --cpp-cli cpp/build/g3pvm_evolve_cli --engine gpu --blocksize 256 --population-size 1024 --generations 20
```

Public behavior contract:
- fixture schema: `fitness-cases-v1`
- crossover: `typed_subtree`
- selection: tournament only with `selection_pressure`
- numeric fitness: negative MAE
- `Bool/None/String/List` fitness: binary exact match
- no heavyweight validate flag in public runners

Latest confirmed speed benchmark (`bouncing-balls-1024`, `2026-03-06`, `pop=1024`, `gen=40`):
- CPU inner total: `298884.479 ms`
- GPU inner total: `1767.595 ms`
- GPU kernel total: `375.385 ms`
- inner total speedup: `169.09x`
- eval-only speedup: `335.90x`
- outer speedup: `164.44x`

Interpretation note:
- This result is after the numeric fitness path switched to negative MAE.
- Do not compare it directly with older binary-fitness-era reports without calling out the semantics change.

Outputs:
- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.md`
- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.json`
- per-run CPU/GPU summary JSON logs

## PSB2 Pipeline

### Convert PSB2 JSONL -> fitness-cases-v1

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
- converter supports single-output and multi-output (`output1..M`) tasks
- multi-output rows are packed into list typed `expected`
- CPU runtime uses exact payload semantics for typed `String/List`
- GPU runtime supports exact payload execution with bounded fallback
- Python and C++ evolution paths both expose `len` / `concat` / `slice` / `index`
- fitness rule is mixed, not all-binary:
  - numeric => `-abs(actual - expected)`
  - `Bool/None/String/List` => exact match `1`, mismatch `0`
  - runtime error => `0`

### Run all PSB2 tasks

```bash
python3 tools/run_psb2_all_tasks.py \
  --datasets-root data/psb2_datasets \
  --tasks all \
  --engine gpu \
  --population-size 1024 \
  --generations 20
```

Outputs:
- `summary.json`: machine-readable per-task status
- `summary.md`: human-readable report

## Contribution Rules

1. Keep APIs prefix-native (`AstProgram`).
2. Preserve CPU/GPU fitness parity.
3. Run Python and C++ tests after behavior changes.
4. For benchmark-impacting changes, attach the generated comparison report.
5. Update `README.md`, `docs/`, `spec/`, and the repo skill if public behavior or structure changed.
