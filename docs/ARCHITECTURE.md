# Architecture

## High-Level Design

The project uses one canonical program representation: linear prefix `AstProgram`.

Primary flow:
1. Build or mutate prefix `AstProgram`
2. Compile to bytecode
3. Execute in Python runtime, C++ CPU runtime, or C++ GPU fitness runtime
4. Evaluate fitness over `fitness-cases-v1`
5. Reproduce and iterate generations

## Current Invariants

- Public AST representation: prefix `AstProgram`
- Public crossover method: `typed_subtree`
- Public selection method: tournament only, with `selection_pressure`
- Public fitness rule:
  - numeric expected/actual => `-abs(actual - expected)`
  - `Bool/None/String/List` exact match => `1`
  - `Bool/None/String/List` mismatch => `0`
  - runtime error => `0`
- Public fixture schema: `fitness-cases-v1`
- Public runners do not expose heavyweight validate flags

## Python Layout

- `python/src/g3p_vm_gpu/core/ast.py`: prefix AST definitions and helpers
- `python/src/g3p_vm_gpu/core/errors.py`: error/outcome types
- `python/src/g3p_vm_gpu/core/value_semantics.py`: numeric promotion and comparisons
- `python/src/g3p_vm_gpu/runtime/builtins.py`: builtin semantics
- `python/src/g3p_vm_gpu/runtime/compiler.py`: AST -> bytecode compiler
- `python/src/g3p_vm_gpu/runtime/interp.py`: reference interpreter
- `python/src/g3p_vm_gpu/runtime/vm.py`: Python bytecode VM
- `python/src/g3p_vm_gpu/evolution/genome.py`: `Limits`, `GenomeMeta`, `ProgramGenome`
- `python/src/g3p_vm_gpu/evolution/stmt_codec.py`: AST <-> statement codec helpers
- `python/src/g3p_vm_gpu/evolution/random_tree.py`: typed random expression/statement generation
- `python/src/g3p_vm_gpu/evolution/random_genome.py`: random genome generation
- `python/src/g3p_vm_gpu/evolution/random_program.py`: generic random program fuzzing
- `python/src/g3p_vm_gpu/evolution/mutation.py`: mutation operator
- `python/src/g3p_vm_gpu/evolution/crossover.py`: crossover operator
- `python/src/g3p_vm_gpu/evolution/evolve.py`: Python evolution loop

## C++ Layout

- `cpp/src/evolution/`: genome metadata, subtree helpers, typed expr analysis, compiler, random generation, mutation, crossover, evolution loop
- `cpp/src/runtime/cpu/`: CPU runtime and CPU fitness helpers
- `cpp/src/runtime/gpu/`: GPU fitness orchestration and device code
- `cpp/src/runtime/payload/`: payload registry and snapshot support
- `cpp/src/cli/`: `runtime_cli`, `evolve_cli`, JSON/codec/options helpers
- `cpp/src/bench/`: benchmark binaries

## Runtime Model

Base scalar values:
- `Int`
- `Float`
- `Bool`
- `None`

Extended values:
- `String`
- `List`

Builtins:
- `abs`, `min`, `max`, `clip`
- `len`, `concat`, `slice`, `index`

Container execution:
- CPU keeps decoded payloads for typed `String/List`
- GPU uploads payload snapshots to device global memory
- GPU exact path uses bounded per-thread scratch
- GPU falls back to deterministic compact transport when exact payload materialization cannot fit

## Evolution Design

The project intentionally constrains the search space:
- typed generation reduces invalid AST production
- typed subtree crossover is the only public crossover path
- tournament selection is the only public selection path
- mutation keeps a single public entry with `mutation_subtree_prob` controlling internal mix
- mixed fitness keeps numeric tasks dense without adding soft container scoring yet

The core engineering tension remains:
- semantic correctness
- CPU/GPU parity
- speedup preservation
