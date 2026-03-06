# g3p-vm-gpu

Prefix-AST genetic programming VM with Python reference implementation and C++ CPU/GPU evolution backends.

## Current Version

- One AST representation only: linear prefix `AstProgram`.
- One public crossover path only: `typed_subtree`.
- One public fitness rule only: binary per-case exact match (`+1` / `+0`).
- One fixture schema only: `fitness-cases-v1`.
- CPU and GPU both support typed `String/List` runtime values with payload-backed exact execution for `concat` / `slice` / `index`.

## Repository Layout

- `python/src/g3p_vm_gpu/`: Python AST/compiler/interpreter/VM/evolution reference.
- `cpp/`: CPU VM, GPU VM, evolution core, CLIs, and tests.
- `tools/`: speedup-focused runners.
- `scripts/`: operational wrappers (`run_gpu_command.sh`).
- `data/fixtures/`: unified input fixtures (`fitness-cases-v1`).

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

## Evolution / Benchmark Entrypoints

### C++ evolution runner (CPU/GPU)

```bash
python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/speedup_cases_bouncing_balls_1024.json \
  --cpp-cli cpp/build/g3pvm_evolve_cli \
  --engine gpu \
  --blocksize 256 \
  --cpp-timing all
```

### CPU vs GPU speedup experiment

```bash
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 4096 --generations 40
```

Output reports are written to `logs/cpu_gpu_compare_pop*_*/cpu_gpu_compare.report.md` and `.json`.

### v1.0 release gate (tests + speed + evolution + PSB2 all-task)

```bash
python3 tools/run_v1_release_gate.py
```

Artifacts:
- `logs/v1_release_report/<timestamp>/release_gate.summary.json`
- `logs/v1_release_report/<timestamp>/release_gate.summary.md`

Validation mode notes:

- Public runners always use the throughput path.
- Internal `validate_genome` hooks remain test/debug scaffolding only and are not exposed through the CLI.

Runtime summary:
- Builtin `len(x)` is available for `String/List` via `CALL_BUILTIN` id `4`.
- Builtin `concat(a,b)` is available for `String/String` and `List/List` via `CALL_BUILTIN` id `5`.
- Builtin `slice(x,lo,hi)` is available for `String/List` via `CALL_BUILTIN` id `6`.
- Builtin `index(x,i)` is available for `String/List` via `CALL_BUILTIN` id `7`.
- CPU runtime uses payload registry for typed `String/List` decoded from fixtures/CLI and executes exact payload semantics.
- GPU runtime uploads payload snapshots to device memory and executes payload-backed `concat` / `slice` / `index` on the exact path; bounded scratch overflow falls back to deterministic compact transport.
- Container comparison supports `EQ/NE` for same-tag `String`/`List`.
- `CALL_LEN` / `CALL_CONCAT` / `CALL_SLICE` / `CALL_INDEX` are wired into AST/compiler/interpreter/VM/evolution on Python and C++ paths.

Fitness note:
- Evolution fitness is binary per case: exact match = `+1`, otherwise `+0` (including runtime errors).

### PSB2 conversion and batch run

```bash
# convert one PSB2 task (JSONL) to fitness-cases-v1
python3 tools/convert_psb2_to_fitness_cases.py \
  --edge-file data/psb2_datasets/bouncing-balls/bouncing-balls-edge.json \
  --random-file data/psb2_datasets/bouncing-balls/bouncing-balls-random.json \
  --n-train 1024 \
  --n-test 1024 \
  --seed 0 \
  --out logs/psb2/converted/bouncing-balls.train.json \
  --out-test logs/psb2/converted/bouncing-balls.test.json \
  --summary-json logs/psb2/converted/bouncing-balls.summary.json

# run all discovered PSB2 tasks
python3 tools/run_psb2_all_tasks.py \
  --datasets-root data/psb2_datasets \
  --tasks all \
  --engine gpu \
  --population-size 1024 \
  --generations 20
```

PSB2 batch outputs:
- `logs/psb2_all_tasks/<timestamp>/summary.json`
- `logs/psb2_all_tasks/<timestamp>/summary.md`
- Current smoke baseline (2026-03-05): `logs/psb2_all_tasks/current_all_v5/summary.json` (`total=25, ok=25`)

## Language / Runtime Scope

- Base subset spec remains in [spec/subset_v1_0.md](/home/hschi1106/g3p-vm-gpu/spec/subset_v1_0.md).
- Base builtin whitelist remains in [spec/builtins_base_v1_0.md](/home/hschi1106/g3p-vm-gpu/spec/builtins_base_v1_0.md).
- v1.0 runtime extensions for `String/List`, new builtins, payload execution, and binary fitness are defined in [spec/builtins_runtime_v1_0.md](/home/hschi1106/g3p-vm-gpu/spec/builtins_runtime_v1_0.md).
- Bytecode behavior is defined in [spec/bytecode_isa_v1_0.md](/home/hschi1106/g3p-vm-gpu/spec/bytecode_isa_v1_0.md).

## GPU Runbook

Always run GPU commands via:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper retries `CUDA_VISIBLE_DEVICES=0` then `1` on device-unavailable failures.
