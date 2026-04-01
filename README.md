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
- reproduction order: selected parent pairs always attempt `typed_subtree` crossover before child-level mutation
- default reproduction backend: `cpu`
- default selection: round-based tournament only, controlled by `selection_pressure`
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
- [CPP_RUNTIME_PAYLOAD.md](docs/CPP_RUNTIME_PAYLOAD.md): host/device container payload model and fallback behavior
- [GPU_REPRODUCTION.md](docs/GPU_REPRODUCTION.md): GPU reproduction backend pipeline, overlap model, and bottlenecks
- [DEVELOPMENT.md](docs/DEVELOPMENT.md): build, test, benchmarks, public CLIs, adjustable arguments
- [AGENTS.md](AGENTS.md): repo-local contributor guidance for coding agents
- [FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md): terse repository directory map

## Repository Layout

- `python/src/g3p_vm_gpu/core/`: AST, shared error/value semantics
- `python/src/g3p_vm_gpu/runtime/`: builtins, compiler, interpreter, Python VM
- `python/src/g3p_vm_gpu/evolution/`: genome, random generation, mutation, crossover, evolution loop
- `cpp/include/g3pvm/`: public C++ headers
- `cpp/src/runtime/`: CPU runtime, GPU fitness runtime, payload support
- `cpp/src/evolution/`: genome analysis, compiler, mutation, crossover, evolution loop
- `cpp/src/evolution/repro/`: reproduction backends, preprocess/pack, GPU reproduction backend
- `cpp/src/cli/`: `evolve_cli`, bucket/population utilities
- `cpp/src/bench/`: benchmark binaries
- `cpp/tests/`: native runtime, GPU smoke, parity, and evolution tests
- `data/fixtures/`: canonical benchmark and evolution fixtures
- `data/psb2_datasets/`: mirrored PSB2 datasets
- `tools/`: dataset conversion and audit utilities
- `scripts/`: execution wrappers and convenience scripts
- `meeting/`: meeting notes and discussion artifacts

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
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
cmake --build cpp/build -j4
ctest --test-dir cpp/build --output-on-failure
python3 scripts/speedup_experiment.py
```

## Main Entrypoints

### Run one evolution job

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --population-size 1024 \
  --generations 20 \
  --out-json logs/simple_exp_1024.run.json
```

### Run speed benchmark sweep

```bash
python3 scripts/speedup_experiment.py
```

This script reads all benchmark settings from
[speedup_experiment.example.json](scripts/speedup_experiment.example.json),
runs the configured sweep, and writes one report directory containing:
- per-fixture multi-mode reports
- an aggregate summary JSON/Markdown

It accepts either a single `max_expr_depth` or a list of `max_expr_depths` through the config file.
When multiple depths are configured, it runs one full sweep per depth and groups the reports under depth-specific subdirectories.
It can also benchmark existing fixed populations directly via the config key `population_jsons`, which is useful for exact-depth or exact-node experiment datasets. In that mode, keep `fixtures`, `population_sizes`, and `max_expr_depths` as empty arrays, and set generation-only controls such as `seed_start`, `probe_cases`, `min_success_rate`, and the AST-shape limits to `null`, so the inactive inputs are explicit.

Each fixture benchmark now runs `g3pvm_evolve_cli --generations 1 --skip-final-eval on` against a fixed `population-seeds-v1` input. When `population_jsons` is omitted, the script materializes a deterministic seed set from `seed_start` and the configured AST limits.
It can compare four formal benchmark modes:
- `cpu`: CPU eval + CPU reproduction
- `gpu_eval`: GPU eval + CPU reproduction
- `gpu_repro`: GPU eval + GPU reproduction
- `gpu_repro_overlap`: GPU eval + GPU reproduction with overlapped preparation

Each mode executes one complete generation and reports derived fixed-pop metrics:
- `compile`: genome-to-bytecode preparation and compile-cache lookup from generation 0
- `eval`: CPU fitness call for `cpu`, and `gpu_eval_init_ms + gpu_eval_call_ms` for GPU modes
- `steady_eval_ms`: CPU fitness call for `cpu`, and `gpu_eval_call_ms` for GPU modes
- `repro`: one-generation host-side or GPU reproduction work
- `warm_total_proxy_ms`: generation-0 total with cold GPU setup removed as documented in `experiment.md`
- `selection`, `crossover`, `mutation`: the internal reproduction phases
- `repro_prepare_inputs`, `repro_setup`, `repro_preprocess`, `repro_pack`, `repro_upload`, `repro_kernel`, `repro_copyback`, `repro_decode`: GPU reproduction phases

When `gpu_repro_overlap` is enabled, `repro_prepare_inputs` / `repro_preprocess` / `repro_pack`
may be partially hidden behind GPU fitness evaluation. Compare `total` first, then inspect the
reproduction subphases to see whether overlap reduced critical-path work or merely shifted it.

### Run one fixed-population sweep

Edit `scripts/speedup_experiment.example.json` and set `population_jsons`, then run:

```bash
python3 scripts/speedup_experiment.py
```

### Convert PSB2 task into `fitness-cases-v1`

```bash
python3 tools/convert_psb2_to_fitness_cases.py \
  --edge-file data/psb2_datasets/bouncing-balls/bouncing-balls-edge.json \
  --random-file data/psb2_datasets/bouncing-balls/bouncing-balls-random.json \
  --out logs/psb2/bouncing-balls.train.json
```

## GPU Commands

GPU-capable C++ paths select the least-used visible CUDA device internally.
To force a specific visible-device index, set:

```bash
G3PVM_CUDA_DEVICE=0
```

## Change Discipline

If you change code, update the matching documents in the same change:
- language or AST semantics => `spec/grammar_v1_0.md`, `spec/bytecode_isa_v1_0.md`, `docs/ARCHITECTURE.md`
- builtin or payload semantics => `spec/builtins_base_v1_0.md` or `spec/builtins_runtime_v1_0.md`, plus `docs/ARCHITECTURE.md`
- fitness semantics or adjustable scoring args => `spec/fitness_v1_0.md`, `docs/DEVELOPMENT.md`, `README.md`
- public CLI/tool args => `docs/DEVELOPMENT.md`, `README.md` if it changes the main workflow
- repo structure or module ownership => `docs/ARCHITECTURE.md`, `docs/FILE_STRUCTURE.md`, repo skill references
