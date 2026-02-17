# Architecture

## 1. High-Level Design

The project has one core program representation: **linear prefix AST**.

Primary flow:

1. Build/obtain prefix `AstProgram`
2. Compile to bytecode (optional for interpreter-only checks)
3. Execute in Python VM, C++ CPU VM, or C++ GPU VM
4. In evolution loops, mutate/crossover genomes at prefix level

No tree AST compatibility layer is part of the supported architecture.

## 2. Core Data Models

### Prefix AST (`AstProgram`)

Defined in `python/src/g3p_vm_gpu/ast.py` and mirrored in C++ evo AST structures.

- `nodes`: linear prefix token stream (`NodeKind` + payload slots)
- `names`: symbol table for vars/assign targets
- `consts`: constant pool
- `version`: `ast-prefix-v1`

Structural invariants:

- root node is `PROGRAM`
- arity-consistent prefix structure
- no trailing tokens
- const/name indices in range

### Bytecode Program

- instruction list + const pool + local variable map
- consumed by Python VM and C++ VM CLI tools

### Genome

- wraps an `AstProgram` + computed metadata (`node_count`, `max_depth`, hash, etc.)
- mutation/crossover operate on prefix-native forms

## 3. Python Layer Responsibilities

- `ast.py`: representation, validation, utility traversals, builder
- `compiler.py`: deterministic prefix -> bytecode lowering
- `interp.py`: direct prefix interpreter with fuel/error semantics
- `vm.py`: bytecode execution engine
- `evo_encoding.py`: random generation, mutate, crossover, constraints validation
- `evolve.py`: population evaluation, selection, and generation loop

## 4. C++ Layer Responsibilities

- `evo_ast.*`: prefix-native evolutionary operators and compilation integration
- `vm_cpu.*`: CPU bytecode runtime
- `vm_gpu.*`: GPU runtime and fitness/multi-batch execution
- `evolve_cli.*`: end-to-end evolution executable with timing outputs

## 5. CPU/GPU Performance Pipeline

Benchmark command:

`tools/run_cpu_gpu_speedup_experiment.sh`

It runs:

1. CPU evolution run (`engine=cpu`)
2. GPU evolution run (`engine=gpu` via wrapper)
3. report synthesis:
   - inner totals (`total`, `generations_eval_total`, `generations_repro_total`)
   - outer wall (`run_cpp_cli`)
   - CPU/GPU speedups

## 6. Design Constraints

- Prefix-only architecture: new feature work must not introduce tree AST paths.
- Interpreter/VM semantic parity is mandatory.
- Evolution operators must preserve AST structural validity and limits.
- GPU runtime selection should go through `scripts/run_gpu_command.sh` in shared environments.
