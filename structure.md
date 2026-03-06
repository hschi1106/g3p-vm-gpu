# g3p-vm-gpu Structure

## Overview

This repo is a prefix-AST genetic programming system with:
- Python reference implementation
- C++ CPU runtime and evolution backend
- C++ CUDA GPU fitness backend

Core flow:
1. Build or mutate prefix `AstProgram`
2. Compile to `BytecodeProgram`
3. Execute on Python runtime, C++ CPU runtime, or C++ GPU fitness runtime
4. Score against `fitness-cases-v1`
5. Reproduce and iterate generations

## Top-Level Layout

```text
g3p-vm-gpu/
├── python/
│   ├── src/g3p_vm_gpu/
│   │   ├── core/
│   │   ├── runtime/
│   │   ├── evolution/
│   │   ├── __init__.py
│   │   └── demo.py
│   └── tests/
├── cpp/
│   ├── include/g3pvm/
│   │   ├── core/
│   │   ├── runtime/
│   │   ├── evolution/
│   │   └── cli/
│   ├── src/
│   │   ├── evolution/
│   │   ├── runtime/
│   │   ├── cli/
│   │   └── bench/
│   └── tests/
├── spec/
├── docs/
├── tools/
├── scripts/
├── data/
└── logs/
```

## Python Package

### `python/src/g3p_vm_gpu/core/`
- `ast.py`: `AstNode`, `AstProgram`, prefix traversal helpers, `build_program`
- `errors.py`: `ErrCode`, `Err`, `Normal`, `Returned`, `Failed`
- `value_semantics.py`: numeric promotion and comparison rules

### `python/src/g3p_vm_gpu/runtime/`
- `builtins.py`: builtin semantics
- `compiler.py`: AST -> bytecode compiler
- `interp.py`: direct prefix interpreter
- `vm.py`: bytecode VM with `exec_bytecode`, `ExecReturn`, `ExecError`

### `python/src/g3p_vm_gpu/evolution/`
- `genome.py`: `Limits`, `GenomeMeta`, `ProgramGenome`, `compile_for_eval`
- `stmt_codec.py`: prefix AST <-> statement-list conversion helpers
- `random_tree.py`: typed random expr/stmt generation
- `random_genome.py`: random genome generation
- `random_program.py`: generic fuzz program generation
- `mutation.py`: mutation operator with `mutation_subtree_prob`
- `crossover.py`: single public crossover path
- `evolve.py`: tournament-selection evolution loop and mixed fitness scoring

## C++ Package

### `cpp/include/g3pvm/core/`
- value model, bytecode model, errors, shared value semantics

### `cpp/include/g3pvm/runtime/`
- CPU execution interface
- GPU fitness interface
- builtin and payload interfaces

### `cpp/include/g3pvm/evolution/`
- public genome, mutation, crossover, and evolution APIs

### `cpp/src/evolution/`
- `genome_meta.cpp`: genome metadata and AST serialization
- `subtree_utils.cpp`: subtree traversal and rewrite
- `typed_expr_analysis.cpp`: typed expression root discovery
- `compiler.cpp`: AST -> bytecode compiler
- `random_genome.cpp`: random genome generation
- `mutation.cpp`: mutation sub-operators
- `crossover.cpp`: crossover operator
- `evolve.cpp`: evolution loop and profiled run

### `cpp/src/runtime/`
- `cpu/`: builtin semantics, CPU execution, CPU fitness
- `gpu/`: GPU fitness orchestration, host packing, device helpers
- `payload/`: typed `String/List` payload registry and snapshot support

### `cpp/src/cli/`
- `runtime_cli.cpp`: runtime execution CLI
- `evolve_cli.cpp`: evolution CLI
- `codec.cpp`, `json.cpp`, `options.cpp`: CLI support

### `cpp/src/bench/`
- `runtime_multi_bench.cpp`: runtime benchmark binary

## Public Behavioral Invariants

- Program representation: prefix `AstProgram`
- Crossover: `typed_subtree`
- Selection: tournament only with `selection_pressure`
- Mutation: one public `mutate(...)` entry, internal mix controlled by `mutation_subtree_prob`
- Fitness:
  - numeric => `-abs(actual - expected)`
  - `Bool/None/String/List` exact match => `1`
  - mismatch => `0`
  - runtime error => `0`
- Fixture schema: `fitness-cases-v1`
- Public runners do not expose heavyweight validate flags

## Runtime Notes

- CPU path executes exact payload semantics for typed `String/List`
- GPU path uploads payload snapshots to device memory
- GPU exact payload execution uses bounded per-thread scratch
- GPU falls back to deterministic compact transport when exact payload materialization cannot fit
- Python remains the reference semantics path for AST/interpreter/VM behavior

## Benchmark Contract

- Speed benchmark fixture: `data/fixtures/speedup_cases_bouncing_balls_1024.json`
- Evolution-progress fixture: `data/fixtures/simple_evo_exp_1024.json`
- GPU commands must go through `scripts/run_gpu_command.sh -- ...`

Latest confirmed `bouncing-balls-1024` benchmark (`2026-03-06`, `pop=1024`, `gen=40`):
- CPU inner total: `298884.479 ms`
- GPU inner total: `1767.595 ms`
- GPU kernel total: `375.385 ms`
- inner total speedup: `169.09x`
- eval-only speedup: `335.90x`
- outer speedup: `164.44x`

Interpret benchmark speedups carefully if fitness semantics changed.
