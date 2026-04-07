# g3p-vm-gpu

Prefix-AST genetic programming system with:
- a Python reference implementation,
- a C++ CPU execution and evolution backend,
- a C++ CUDA GPU fitness backend.

## What Is Stable

The current public contract is:
- program representation: prefix `AstProgram`
- control flow grammar: `ForRange(x, e, body)` with evaluate-once non-negative integer bounds
- fixture schema: `fitness-cases-v1`
- crossover: `typed_subtree`
- reproduction order: selected parent pairs always attempt `typed_subtree` crossover before child-level mutation
- default reproduction backend: `cpu`
- default selection: round-based tournament only, controlled by `selection_pressure`
- mutation: one public mutation path, internal mix controlled by `mutation_subtree_prob`
- fitness:
  - numeric expected + numeric actual => `-abs(actual - expected)`
  - numeric expected + non-numeric actual => `-penalty`
  - `Bool` / `None` / `String` / `NumList` / `StringList` => exact match `1`, same-type mismatch `0`, type mismatch `-penalty`
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
- [GRAMMAR_CONFIG.md](docs/GRAMMAR_CONFIG.md): external config format for evolution grammar search-space controls
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
- `cpp/src/cli/`: `evolve_cli` and shared CLI helpers
- `cpp/src/bench/`: benchmark binaries
- `cpp/tests/`: native runtime, GPU smoke, parity, and evolution tests
- `configs/grammar/`: checked-in evolution grammar config presets
- `data/fixtures/`: canonical benchmark and evolution fixtures
- `data/psb1_datasets/`: mirrored PSB1 datasets
- `data/psb2_datasets/`: mirrored PSB2 datasets
- `tools/`: PSB dataset fetch/conversion/audit utilities
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

To restrict the evolution search space, pass a checked-in grammar config. The config affects generation and reproduction donor synthesis, not runtime execution of already-materialized programs:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --grammar-config configs/grammar/scalar.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --population-size 64 \
  --generations 5 \
  --out-json logs/simple_exp_1024.scalar.json
```

CPU and GPU reproduction both respect non-default grammar configs. GPU reproduction applies the config during host-side preprocess by filtering typed candidates and building config-aware donor buckets.

### Run one fixed-population benchmark

Use one prepared `population-seeds-v1` file and run one generation per mode:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/bouncing_balls_1024.json \
  --population-json logs/fixed_population.seeds.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap off \
  --blocksize 1024 \
  --generations 1 \
  --skip-final-eval on \
  --timing all \
  --out-json logs/fixed_population.run.json
```

For fair comparisons, reuse the same `population-seeds-v1` input across `cpu`, `gpu_eval`,
`gpu_repro`, and `gpu_repro_overlap` runs, then compare generation-0 timing fields.

### Convert PSB2 task into `fitness-cases-v1`

```bash
python3 tools/convert_psb2_to_fitness_cases.py \
  --edge-file data/psb2_datasets/bouncing-balls/bouncing-balls-edge.json \
  --random-file data/psb2_datasets/bouncing-balls/bouncing-balls-random.json \
  --out logs/psb2/bouncing-balls.train.json
```

The converter emits typed sequence values as `num_list` or `string_list` so empty list fields remain unambiguous.
Multi-output PSB rows are rejected by this converter path until runtime-level multi-output support is added; they are not encoded as fake list outputs.

### Fetch PSB1 datasets

```bash
python3 tools/fetch_psb1_datasets.py --out-dir data/psb1_datasets
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
