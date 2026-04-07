# Architecture

## System Model

The system evolves prefix `AstProgram` programs, compiles them to bytecode, executes them on CPU or GPU-backed runtimes, scores them against `fitness-cases-v1`, and repeats reproduction over generations.

The execution stack has three layers:
- Python reference path: semantics reference for AST, interpreter, compiler, and VM behavior
- C++ CPU path: native execution and native evolution
- C++ GPU path: CUDA fitness evaluation plus an optional GPU reproduction backend

## Documentation Layers

Use the repo documents in this order:
- `spec/`: normative semantics and wire formats
- `README.md`: entrypoint and common workflows
- `docs/DEVELOPMENT.md`: commands, CLIs, and benchmark procedure
- `docs/TIMING.md`: canonical timing names, scope boundaries, and output mapping
- `docs/GRAMMAR_CONFIG.md`: external evolution grammar config contract
- `docs/CPP_RUNTIME_PAYLOAD.md`: host/device container transport details
- `docs/GPU_REPRODUCTION.md`: GPU reproduction backend design, overlap model, and current bottlenecks
- `docs/FILE_STRUCTURE.md`: terse directory map
- `AGENTS.md`: repo-local contributor guidance for coding agents

## Core Invariants

These are the current 1.0 invariants.
- Public program representation is prefix `AstProgram`
- Public crossover is `typed_subtree`
- Public reproduction attempts `typed_subtree` crossover on every selected parent pair before child-level mutation
- Default public reproduction backend is `cpu`
- Default public selection is round-based tournament only, controlled by `selection_pressure`
- Public mutation API is single-path, with internal operator mix controlled by `mutation_subtree_prob`
- Evolution grammar configs are search-space controls only; runtime/VM execution remains the all-enabled public grammar superset
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
- `NumList`
- `StringList`

`Bool` is not numeric.
`NumList` is a homogeneous list of `Int`/`Float` values.
`StringList` is a homogeneous list of `String` values.
Nested and heterogeneous lists are not part of the public value contract.

### Control flow
- The public grammar has one loop form: `ForRange(x, e, body)`.
- `ForRange` evaluates `e` once, stores the bound in a temporary local during bytecode lowering, and then runs integer loop indices while `i < bound`.
- The bound must be a non-negative integer. `0` is valid and executes zero iterations; `Bool` and `Float` are rejected.
- Current random genome generation still seeds loop bounds with integer constants, so existing benchmark population generation remains bounded by `max_for_k`.

### Evolution grammar configs
- `grammar-config-v1` files live under `configs/grammar/` and can be passed to the native CLI with `--grammar-config`.
- The config restricts random genome generation, CPU mutation donor synthesis, GPU reproduction preprocess candidate/donor generation, and seed replay regeneration.
- The config does not reject execution of existing ASTs or bytecode that use disabled constructs.
- CPU and GPU reproduction both respect non-default grammar configs.

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
- `append`
- `reverse`
- `find`
- `contains`

### Payload execution
Container values use payload-backed execution.
- CPU runtime keeps decoded `String`, `NumList`, and `StringList` payloads in a registry.
- GPU runtime keeps a session-local host payload cache, lazily fills it by packed token from the process-global registry, and then builds compact per-eval payload packs for only the tokens needed by the current accepted population plus shared cases.
- GPU payload evaluation always launches one production `Mixed` eval kernel across the full accepted population.
- the finer `StringOnly` / `ListOnly` / `Mixed` flavor classifier is still kept for experiment tooling and offline bucketing studies
- GPU exact payload operations use bounded per-thread scratch.
- When exact payload materialization does not fit, GPU falls back to deterministic compact transport instead of aborting the full evaluation.

See also:
- [CPP_RUNTIME_PAYLOAD.md](CPP_RUNTIME_PAYLOAD.md) for the C++ container token, payload registry, exact/fallback split, and collision tradeoffs.

## Fitness Model

The scoring model is defined in [fitness_v1_0.md](../spec/fitness_v1_0.md).

Operational summary:
- numeric expected + numeric actual => negative absolute error
- numeric expected + non-numeric actual => `-penalty`
- `Bool` / `None` / `String` / `NumList` / `StringList` => exact match `1`, same-type mismatch `0`, type mismatch `-penalty`
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
- `grammar_config.py`: Python-side `grammar-config-v1` parser and config object
- `stmt_codec.py`: AST and statement codec helpers
- `random_tree.py`: typed random expression and statement generation
- `random_genome.py`: random genome generation
- `random_program.py`: generic fuzz/reference random programs
- `mutation.py`: Python mutation operator
- `crossover.py`: Python crossover operator
- `evolve.py`: Python evolution loop and reference fitness logic

## C++ Module Map

### `cpp/include/g3pvm/core/`
Public value, error, builtin id, opcode, bytecode, and shared fitness/value semantics headers.

### `cpp/include/g3pvm/runtime/cpu/`
Public CPU execution, fitness, and builtin interfaces:
- `execute_bytecode_cpu.hpp`
- `fitness_cpu.hpp`
- `builtins_cpu.hpp`

### `cpp/include/g3pvm/runtime/gpu/`
Public GPU host-side contracts for fitness orchestration and packed device data:
- `fitness_gpu.hpp`
- `host_pack_gpu.hpp`
- `device_types_gpu.hpp`
- `constants_gpu.hpp`

### `cpp/include/g3pvm/runtime/payload/`
Public payload registry interface for host-side string/list snapshots and lookup.

### `cpp/include/g3pvm/evolution/`
Public evolution interfaces split by responsibility:
- `ast_program.hpp`: prefix AST program representation, shape limits, and canonical AST serialization helpers
- `genome.hpp`: genome metadata and `ProgramGenome` wrapper
- `grammar_config.hpp`: evolution grammar search-space config
- `genome_generation.hpp`: random genome generation
- `compiler.hpp`: AST-to-bytecode lowering
- `selection.hpp`: ranking and parent selection
- `mutation.hpp`, `crossover.hpp`, `evolve.hpp`: operators and orchestration
- `repro/`: reproduction backend contracts, preprocess/pack schema, and GPU reproduction backend entrypoints

### `cpp/src/runtime/cpu/`
- `builtins_cpu.cpp`: builtin implementation
- `execute_bytecode_cpu.cpp`: bytecode execution
- `fitness_cpu.cpp`: CPU fitness accumulation

### `cpp/src/runtime/gpu/`
- `fitness_gpu.cu`: GPU fitness orchestration
- `host_pack_gpu.cu`: host-side program and case packing
- `opcode_map_gpu.*`: host opcode-to-device opcode mapping
- `device/`: CUDA device-side execution, builtin, arithmetic, and kernel entry helpers

### `cpp/src/runtime/payload/`
- payload registry
- payload snapshot generation for GPU

### `cpp/src/evolution/`
- `ast_program.cpp`: canonical AST serialization and cache-key generation
- `genome.cpp`: genome metadata construction
- `grammar_config.cpp`: native grammar config validation and helper predicates
- `subtree_utils.*`: subtree traversal and rewrite
- `typed_expr_analysis.*`: typed expression root analysis
- `compiler.cpp`: AST-to-bytecode compiler
- `genome_generation.cpp`: random genome generation
- `selection.cpp`: ranking and parent selection
- `mutation.cpp`: mutation operators
- `crossover.cpp`: typed subtree exchange
- `repro/`: reproduction backend dispatch, preprocess/pack extraction, `gpu` arena/copyback logic, and sequential/overlap orchestration
- `evolve.cpp`: evolution loop orchestration

### `cpp/src/cli/`
- `evolve_cli.cpp`: evolution CLI; also supports fixed-population one-generation benchmark runs via `--population-json` and `--skip-final-eval`
- codec / json / options helpers

### `cpp/src/bench/`
Benchmark binaries for runtime-focused measurement.

### `cpp/tests/`
- `runtime/`: CPU VM smoke, edge, and CLI-harness tests
- `gpu/`: direct GPU smoke coverage
- `parity/`: CPU/GPU fitness and evolution parity regression tests
- `evolution/`: native evolution and genome tests

## Tooling And Script Map

### `tools/`
- `fetch_psb_datasets.py`: download PSB1/PSB2 JSON Lines datasets into `data/psb1_datasets/` or `data/psb2_datasets/`
- `convert_psb_to_fitness_cases.py`: convert PSB1/PSB2 JSON Lines into `fitness-cases-v1` fixtures with schema-aware `num_list` / `string_list` emission

## Data and Tooling Layout

- `data/fixtures/`: canonical benchmark fixtures
- `data/psb1_datasets/`: PSB1 dataset mirror used by fetch tooling
- `data/psb2_datasets/`: PSB2 dataset mirror used by fetch/convert utilities
- `tools/`: dataset fetch and conversion utilities
- `logs/`: generated run artifacts, benchmark reports, gate outputs
- `meeting/`: meeting notes and non-normative discussion artifacts

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
- `docs/FILE_STRUCTURE.md`
- repo skill references under `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/references/`
