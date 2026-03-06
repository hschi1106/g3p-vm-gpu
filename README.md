# g3p-vm-gpu

Prefix-AST genetic programming VM with a Python reference path and C++ CPU/GPU evolution backends.

## Current Version

- One public program representation: linear prefix `AstProgram`.
- One public crossover path: `typed_subtree`.
- One public selection path: tournament selection with `selection_pressure`.
- One public fitness rule: numeric cases use negative MAE; `Bool/None/String/List` use binary exact match.
- One fixture schema: `fitness-cases-v1`.
- CPU and GPU both support typed `String/List` runtime values; CPU is exact, GPU is exact when payload scratch fits and falls back to deterministic compact transport otherwise.

## Repository Layout

- `python/src/g3p_vm_gpu/core/`: AST, errors, value semantics.
- `python/src/g3p_vm_gpu/runtime/`: builtins, compiler, interpreter, Python bytecode VM.
- `python/src/g3p_vm_gpu/evolution/`: genome metadata, random generation, mutation, crossover, evolution loop.
- `cpp/`: CPU runtime, GPU fitness runtime, evolution core, CLIs, benches, tests.
- `tools/`: runners and experiment scripts.
- `scripts/`: operational wrappers, including GPU device retry.
- `data/fixtures/`: `fitness-cases-v1` fixtures.

## Quick Start

### Build C++

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

### Run tests

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
ctest --test-dir cpp/build --output-on-failure
```

### Run Python demo

```bash
PYTHONPATH=python/src python3 -m g3p_vm_gpu.demo
```

## Entrypoints

### C++ evolution runner

```bash
python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/speedup_cases_bouncing_balls_1024.json \
  --cpp-cli cpp/build/g3pvm_evolve_cli \
  --engine gpu \
  --blocksize 256 \
  --cpp-timing all
```

### CPU vs GPU speedup benchmark

```bash
scripts/run_gpu_command.sh -- bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
```

Latest confirmed `bouncing-balls-1024` report (`2026-03-06`, `pop=1024`, `gen=40`):
- inner total speedup: `169.09x`
- eval-only speedup: `335.90x`
- outer end-to-end speedup: `164.44x`

Important note:
- This is not directly comparable to older binary-fitness-era reports.
- The current numeric fitness path uses negative MAE, which changed CPU and GPU evaluation cost.
- For performance analysis, track absolute `cpu inner total`, `gpu inner total`, and `gpu kernel total`, not only the speedup ratio.

Output reports are written to `logs/cpu_gpu_compare_pop*_*/cpu_gpu_compare.report.md` and `.json`.

### v1.0 release gate

```bash
python3 tools/run_v1_release_gate.py
```

Artifacts:
- `logs/v1_release_report/<timestamp>/release_gate.summary.json`
- `logs/v1_release_report/<timestamp>/release_gate.summary.md`

## Runtime Summary

- Builtins:
  - `abs`, `min`, `max`, `clip`
  - `len`, `concat`, `slice`, `index`
- Builtin ids:
  - `abs=0`, `min=1`, `max=2`, `clip=3`, `len=4`, `concat=5`, `slice=6`, `index=7`
- CPU runtime:
  - payload registry for typed `String/List`
  - exact payload semantics for `concat` / `slice` / `index`
- GPU runtime:
  - payload snapshots in device memory
  - exact payload-backed `concat` / `slice` / `index` when scratch fits
  - deterministic compact fallback when exact payload materialization cannot fit
- Fitness:
  - numeric => `-abs(actual - expected)`
  - `Bool/None/String/List` => exact match `1`, mismatch `0`
  - runtime errors => `0`

## PSB2 Pipeline

### Convert one PSB2 task

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

### Run all PSB2 tasks

```bash
python3 tools/run_psb2_all_tasks.py \
  --datasets-root data/psb2_datasets \
  --tasks all \
  --engine gpu \
  --population-size 1024 \
  --generations 20
```

## Specs

- Base language: [spec/subset_v1_0.md](spec/subset_v1_0.md)
- Base builtins: [spec/builtins_base_v1_0.md](spec/builtins_base_v1_0.md)
- Runtime extensions: [spec/builtins_runtime_v1_0.md](spec/builtins_runtime_v1_0.md)
- Bytecode ISA: [spec/bytecode_isa_v1_0.md](spec/bytecode_isa_v1_0.md)
- Bytecode JSON format: [spec/bytecode_format_v1_0.md](spec/bytecode_format_v1_0.md)

## GPU Runbook

Always run GPU commands through:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper retries `CUDA_VISIBLE_DEVICES=0` then `1` on device-unavailable failures.
