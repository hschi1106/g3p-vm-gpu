# g3p-vm-gpu

Prefix-AST genetic programming system with:
- a Python reference implementation,
- a C++ CPU execution and evolution backend,
- a C++ CUDA GPU fitness backend.

## What Is Stable

The current public contract is:
- program representation: prefix `AstProgram`
- fixture schema: `fitness-cases-v1`
- crossover: `typed_subtree`
- selection: tournament only, controlled by `selection_pressure`
- mutation: one public mutation path, internal mix controlled by `mutation_subtree_prob`
- fitness:
  - numeric expected + numeric actual => `-abs(actual - expected)`
  - numeric expected + non-numeric actual => `-penalty`
  - `Bool` / `None` / `String` / `List` => exact match `1`, same-type mismatch `0`, type mismatch `-penalty`
  - runtime error => `-penalty`

## Document Map

Use the documents below as the source of truth.

### Specs
- [grammar_v1_0.md](spec/grammar_v1_0.md): language grammar, typing rules, control flow, evaluation order
- [bytecode_isa_v1_0.md](spec/bytecode_isa_v1_0.md): bytecode execution contract
- [bytecode_format_v1_0.md](spec/bytecode_format_v1_0.md): internal JSON harness format used by bytecode parity tooling
- [builtins_base_v1_0.md](spec/builtins_base_v1_0.md): scalar builtins
- [builtins_runtime_v1_0.md](spec/builtins_runtime_v1_0.md): container builtins and payload behavior
- [fitness_v1_0.md](spec/fitness_v1_0.md): scoring rules and solved criteria

### Docs
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): system structure, module map, invariants
- [DEVELOPMENT.md](docs/DEVELOPMENT.md): build, test, benchmarks, public CLIs, adjustable arguments
- [structure.md](structure.md): terse repository directory map

## Repository Layout

- `python/src/g3p_vm_gpu/core/`: AST, shared error/value semantics
- `python/src/g3p_vm_gpu/runtime/`: builtins, compiler, interpreter, Python VM
- `python/src/g3p_vm_gpu/evolution/`: genome, random generation, mutation, crossover, evolution loop
- `cpp/include/g3pvm/`: public C++ headers
- `cpp/src/runtime/`: CPU runtime, GPU fitness runtime, payload support
- `cpp/src/evolution/`: genome analysis, compiler, mutation, crossover, evolution loop
- `cpp/src/cli/`: `evolve_cli`, benchmark/population utilities
- `cpp/src/bench/`: benchmark binaries
- `tools/`: conversion, orchestration, release-gate tools
- `scripts/`: execution wrappers and convenience scripts

## Quick Start

### Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

### Test

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
ctest --test-dir cpp/build --output-on-failure
```

### Full local check

```bash
bash scripts/run_triplet_checks.sh
```

## Main Entrypoints

### Run one evolution job

```bash
python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/simple_exp_1024.json \
  --cpp-cli cpp/build/g3pvm_evolve_cli \
  --engine gpu \
  --blocksize 256 \
  --population-size 1024 \
  --generations 20
```

### Run CPU vs GPU benchmark

```bash
bash scripts/run_cpu_gpu_speedup_experiment.sh \
  --cases data/fixtures/bouncing_balls_1024.json \
  --popsize 1024
```

This benchmark now generates one fixed population per run instead of using a multi-generation evolution run.
It always executes one complete generation and reports a phase breakdown:
- `compile`: genome-to-bytecode preparation and compile-cache lookup
- `eval`: fitness execution only; `compile` is intentionally excluded
- `repro`: one-generation host-side selection, crossover, and mutation work
- `selection`, `crossover`, `mutation`: the internal reproduction phases
- `total`: the full one-generation benchmark wall time

### Generate and benchmark one fixed population in one step

```bash
cpp/build/g3pvm_population_bench_cli \
  --cases data/fixtures/bouncing_balls_1024.json \
  --out-population-json logs/fixed_population.json \
  --population-size 1024 \
  --probe-cases 32 \
  --min-success-rate 0.10 \
  --engine cpu
```

### Convert PSB2 task into `fitness-cases-v1`

```bash
python3 tools/convert_psb2_to_fitness_cases.py \
  --edge-file data/psb2_datasets/bouncing-balls/bouncing-balls-edge.json \
  --random-file data/psb2_datasets/bouncing-balls/bouncing-balls-random.json \
  --out logs/psb2/bouncing-balls.train.json
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

## GPU Commands

Always run GPU commands through:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper retries `CUDA_VISIBLE_DEVICES=0` then `CUDA_VISIBLE_DEVICES=1` when the device is unavailable.

## Change Discipline

If you change code, update the matching documents in the same change:
- language or AST semantics => `spec/grammar_v1_0.md`, `spec/bytecode_isa_v1_0.md`, `docs/ARCHITECTURE.md`
- builtin or payload semantics => `spec/builtins_base_v1_0.md` or `spec/builtins_runtime_v1_0.md`, plus `docs/ARCHITECTURE.md`
- fitness semantics or adjustable scoring args => `spec/fitness_v1_0.md`, `docs/DEVELOPMENT.md`, `README.md`
- public CLI/tool args => `docs/DEVELOPMENT.md`, `README.md` if it changes the main workflow
- repo structure or module ownership => `docs/ARCHITECTURE.md`, `structure.md`, repo skill references
