# Architecture

## 1. High-Level Design

The project uses one core representation: **linear prefix AST**.

Primary flow:

1. Build/obtain prefix `AstProgram`
2. Compile to bytecode
3. Execute in Python VM, C++ CPU VM, or C++ GPU VM
4. Run evolution loops with CPU/GPU fitness parity and profiling

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
- value encoding supports typed values (`int|float|bool|none`)

## 3. C++ Evolution Pipeline

- `evolve.cpp` evaluates CPU/GPU fitness with aligned scoring logic.
- CPU and GPU both use shared-cases/shared-answer style evaluation internally.
- GPU path uses session reuse and reports compile/upload/kernel/copyback timings.
- Reproduction phase reports selection/crossover/mutation/elite timings.

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
