# Architecture

## 1. High-Level Design

The project uses one core representation: **linear prefix AST**.

Primary flow:

1. Build/obtain prefix `AstProgram`
2. Compile to bytecode
3. Execute in Python VM, C++ CPU VM, or C++ GPU VM
4. Run evolution loops with CPU/GPU fitness parity and profiling

Current invariants:

- Prefix AST is the only program representation exposed publicly.
- `typed_subtree` is the only public crossover method.
- Public evolution runners use mixed fitness: numeric cases use negative MAE; `Bool/None/String/List` use binary exact match.
- Public runners do not expose heavyweight validation flags.

## 2. Core Data Models

### Prefix AST (`AstProgram`)

Defined in `python/src/g3p_vm_gpu/ast.py` and mirrored in C++ evo AST structures.

- `nodes`: linear prefix token stream
- `names`: symbol table
- `consts`: constant pool
- `version`: `ast-prefix-v1`

### Unified Fixture Schema (`fitness-cases-v1`)

All evolution/benchmark inputs use one JSON schema:

- `format_version`: `fitness-cases-v1`
- `meta`: metadata map
- `cases`: list of `{inputs, expected}`
- value encoding supports typed values:
  - scalar: `int | float | bool | none`
  - container: `string | list`

## 3. C++ Evolution Pipeline

- `cpp/src/evolution/` contains genome generation, mutation, crossover, selection, and evolution loop code.
- `cpp/src/runtime/cpu/`, `cpp/src/runtime/gpu/`, and `cpp/src/runtime/payload/` split the runtime by execution responsibility.
- `cpp/src/cli/` contains user-facing C++ entrypoints and fixture/bytecode decode helpers.
- `cpp/src/bench/` contains benchmark binaries only.
- `evolve.cpp` evaluates CPU/GPU fitness with aligned scoring logic.
- Fitness is mixed per case:
- numeric expected/actual => `-abs(actual - expected)`
- `Bool/None/String/List` => exact match `1`, otherwise `0`
- runtime errors => `0`
- CPU and GPU both use shared-cases/shared-answer style evaluation internally.
- GPU path uses session reuse and reports compile/upload/kernel/copyback timings.
- Reproduction phase reports selection/crossover/mutation/elite timings.

Container execution:

- CPU path maintains a payload registry for typed `String/List` values decoded from fixtures and CLI inputs.
- GPU path uploads payload snapshots into device global memory and runs exact payload-backed `concat` / `slice` / `index` when scratch capacity allows.
- Compact container transport remains in the `Value` path for throughput and bounded fallback behavior.

## 4. CPU/GPU Performance Pipeline

Entry: `tools/run_cpu_gpu_speedup_experiment.sh`

It runs CPU and GPU evolution on the same `fitness-cases-v1` fixture, then reports:

- end-to-end speedup (`run_cpp_cli` wall)
- inner total speedup
- eval-only speedup
- reproduction breakdown
- GPU execution breakdown

## 5. Design Constraints

- Prefix-only architecture.
- CPU/GPU fitness parity is mandatory.
- GPU runtime selection should go through `scripts/run_gpu_command.sh`.
- Documentation and public CLI should reflect the current single-path API surface rather than historical alternatives.
