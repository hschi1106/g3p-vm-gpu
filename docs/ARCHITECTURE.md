# Architecture

## System Model

The system evolves prefix `AstProgram` programs, compiles them to bytecode, executes them on CPU or GPU-backed runtimes, scores them against `fitness-cases-v1`, and repeats reproduction over generations.

The execution stack has three layers:
- Python reference path: semantics reference for AST, interpreter, compiler, and VM behavior
- C++ CPU path: native execution and native evolution
- C++ GPU path: CUDA fitness evaluation over batched cases

## Core Invariants

These are the current 1.0 invariants.
- Public program representation is prefix `AstProgram`
- Public crossover is `typed_subtree`
- Public selection is tournament only, controlled by `selection_pressure`
- Public mutation API is single-path, with internal operator mix controlled by `mutation_subtree_prob`
- Public fixture schema is `fitness-cases-v1`
- Public runners do not expose heavyweight validate modes
- CPU and GPU must preserve fitness parity for the same inputs and configuration

## Runtime Model

### Value domain
- `Int`
- `Float`
- `Bool`
- `None`
- `String`
- `List`

`Bool` is not numeric.

### Builtins
Scalar builtins:
- `abs`
- `min`
- `max`
- `clip`

Container builtins:
- `len`
- `concat`
- `slice`
- `index`

### Payload execution
Container values use payload-backed execution.
- CPU runtime keeps decoded `String` and `List` payloads in a registry.
- GPU runtime uploads payload snapshots into global memory before kernel evaluation.
- GPU exact payload operations use bounded per-thread scratch.
- When exact payload materialization does not fit, GPU falls back to deterministic compact transport instead of aborting the full evaluation.

See also:
- [CPP_RUNTIME_PAYLOAD.md](CPP_RUNTIME_PAYLOAD.md) for the C++ container token, payload registry, exact/fallback split, and collision tradeoffs.

## Fitness Model

The scoring model is defined in [fitness_v1_0.md](../spec/fitness_v1_0.md).

Operational summary:
- numeric expected + numeric actual => negative absolute error
- numeric expected + non-numeric actual => `-penalty`
- `Bool` / `None` / `String` / `List` => exact match `1`, same-type mismatch `0`, type mismatch `-penalty`
- runtime error => `-penalty`

This keeps numeric tasks dense while keeping container semantics exact and simple.

## Python Module Map

### `python/src/g3p_vm_gpu/core/`
- `ast.py`: prefix AST definitions and traversal helpers
- `errors.py`: runtime outcome and error types
- `value_semantics.py`: shared scalar comparison and numeric promotion rules

### `python/src/g3p_vm_gpu/runtime/`
- `builtins.py`: reference builtin semantics
- `compiler.py`: AST to bytecode compiler
- `interp.py`: direct AST interpreter
- `vm.py`: Python bytecode VM

### `python/src/g3p_vm_gpu/evolution/`
- `genome.py`: genome container and compile-for-eval helpers
- `stmt_codec.py`: AST and statement codec helpers
- `random_tree.py`: typed random expression and statement generation
- `random_genome.py`: random genome generation
- `random_program.py`: generic fuzz/reference random programs
- `mutation.py`: Python mutation operator
- `crossover.py`: Python crossover operator
- `evolve.py`: Python evolution loop and reference fitness logic

## C++ Module Map

### `cpp/include/g3pvm/core/`
Public value, error, bytecode, and shared fitness/value semantics headers.

### `cpp/include/g3pvm/runtime/cpu/`
Public CPU execution, fitness, and builtin interfaces.

### `cpp/include/g3pvm/runtime/gpu/`
Public GPU host-side contracts for packed device types, constants, and host packing.

### `cpp/include/g3pvm/runtime/payload/`
Public payload registry interface for host-side string/list snapshots and lookup.

### `cpp/include/g3pvm/evolution/`
Public genome, mutation, crossover, and evolution interfaces.

### `cpp/src/runtime/cpu/`
- builtin implementation
- bytecode execution in `execute_bytecode_cpu.*`
- CPU fitness accumulation in `fitness_cpu.*`

### `cpp/src/runtime/gpu/`
- GPU fitness orchestration
- host-side packing and kernel launch
- device execution and builtin helpers

### `cpp/src/runtime/payload/`
- payload registry
- payload snapshot generation for GPU

### `cpp/src/evolution/`
- genome metadata and serialization
- subtree traversal and rewrite
- typed expression analysis
- AST to bytecode compiler
- random genome generation
- mutation
- crossover
- evolution loop

### `cpp/src/cli/`
- `runtime_cli.cpp`: runtime execution CLI
- `evolve_cli.cpp`: evolution CLI
- `generate_population_cli.cpp`: fixed-population generator for benchmark inputs
- `population_bench_cli.cpp`: fixed-population benchmark runner
- codec / json / options helpers

### `cpp/src/bench/`
Benchmark binaries for runtime-focused measurement.

## Data and Tooling Layout

- `data/fixtures/`: canonical benchmark fixtures
- `data/fixtures/programs/`: checked-in fixed-population benchmark inputs
- `data/psb2_datasets/`: PSB2 dataset mirror used by converters and batch runs
- `tools/`: dataset conversion, orchestration, reporting, release gates
- `scripts/`: operational wrappers that should be used directly by humans and agents
- `logs/`: generated run artifacts, benchmark reports, gate outputs

## What To Update When Code Changes

### AST, grammar, or bytecode changes
Update:
- `spec/grammar_v1_0.md`
- `spec/bytecode_isa_v1_0.md`
- `spec/bytecode_format_v1_0.md` if the wire format changed
- this file

### Builtin, type, or payload changes
Update:
- `spec/builtins_base_v1_0.md` or `spec/builtins_runtime_v1_0.md`
- `spec/bytecode_isa_v1_0.md` if opcode behavior changed
- this file

### Fitness or evolution-arg changes
Update:
- `spec/fitness_v1_0.md`
- `docs/DEVELOPMENT.md`
- `README.md` if the main workflow or key defaults changed

### Repo structure or entrypoint changes
Update:
- this file
- `structure.md`
- repo skill references under `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/references/`
