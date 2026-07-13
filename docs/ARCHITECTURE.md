# Architecture

## System Model

The system evolves prefix `AstProgram` programs, compiles them to bytecode,
executes them on CPU or GPU-backed runtimes, scores them against
`fitness-cases` fixtures for current validation or `fitness-cases` fixtures
for compatibility baselines, and repeats reproduction over generations.

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

These are the current invariants.
- Public program representation is prefix `AstProgram`
- Public crossover is `typed_subtree`
- Public reproduction attempts `typed_subtree` crossover on every selected parent pair before child-level mutation
- Default public reproduction backend is `cpu`
- Default public selection is round-based tournament only, controlled by `selection_pressure`
- Public mutation API is single-path, with internal operator mix controlled by `mutation_subtree_prob`
- Evolution grammar configs are search-space controls only; runtime/VM execution remains the all-enabled public grammar superset
- Public fixture schema is `fitness-cases`.
- Public runners do not expose heavyweight validate modes
- CPU and GPU must preserve fitness parity for the same inputs and configuration while exact GPU payload materialization stays within bounded device limits; payload overflow is backend-specific and may diverge.
- Generated initial populations are expected-output-aware for payload return types: when all fixture cases have the same `String`, `IntList`, `FloatList`, or `StringList` expected output type and the active grammar allows that type, generation should force the top-level return type for generation 0. Other expected output types, mixed expected output types, unsupported expected values, disabled grammar types, and fixed `population-seeds` replay use the generic generation/replay path.
- The CPU reproduction backend selects parent indices and streams children into `next_population`; it does not materialize extra full-population `selected_parents` or `offspring` copies
- The evolution loop ranks the current population with lightweight scored references during each generation; it only materializes owned `ScoredGenome` values for public outputs such as history snapshots and final evaluated populations

## Runtime Model

### Value domain
- `Int`
- `Float`
- `Bool`
- `Char`
- `String`
- `IntList`
- `FloatList`
- `StringList`

`Bool` is not numeric.
`Char` is distinct from one-character `String`.
`IntList` and `FloatList` are distinct direct list tags.
`StringList` is a homogeneous list of `String` values.
Nested and heterogeneous lists are not part of the public value contract.

### Control flow
- The public grammar has one loop form: `ForRange(x, e, body)`.
- `ForRange` evaluates `e` once, stores the bound in a temporary local during bytecode lowering, and then runs integer loop indices while `i < bound`.
- The bound must be a non-negative integer. `0` is valid and executes zero iterations; `Bool` and `Float` are rejected.
- Current random genome generation still seeds loop bounds with integer constants, so existing benchmark population generation remains bounded by `max_for_k`.

### Structured expressions
- The Python reference path implements `BoundVar`, `MapList`, `FilterList`, and `LinearRec`.
- `BoundVar` uses a hidden binder namespace separate from ordinary mutable locals, so same-name `Var(x)` and `BoundVar(x)` are capture-safe and assignments cannot write binder slots.
- Python bytecode lowering expands `MapList`, `FilterList`, and `LinearRec` into ordinary loop/jump code plus private type/list helper opcodes; these opcodes are not part of the public current bytecode wire contract.
- `LinearRec` binder metadata uses an explicit `AstProgram` side table because the form has three binders and cannot be represented clearly with the two generic `AstNode` integer payload slots.
- Native C++ AST metadata, subtree traversal, grammar-config gating, cache keys, and typed-expression analysis understand the structured node set.
- Native CPU compiler/runtime lowering executes hand-authored `MapList`, `FilterList`, and `LinearRec` ASTs using hidden locals and private helper opcodes.
- Native GPU fitness execution implements the private structured-expression helper opcode slice needed by compiler-lowered `MapList`, `FilterList`, and `LinearRec` programs, including exact empty typed-list payload creation within bounded device payload state.
- Python and native random generation can emit conservative structured forms when enabled. Native typed-subtree mutation can synthesize `MapList` / `FilterList` donors, and typed root collection excludes binder-body fragments containing `BoundVar` so lexical binders cannot escape their scope during variation. CPU mutation, CPU crossover, CPU reproduction, and GPU reproduction preprocessing also filter ASGP phase-body roots before choosing typed-subtree replacement sites. Host-side CPU typed-subtree crossover/reproduction and packed GPU candidate selection use a current typed key that includes result type, visible scope, binder/scheme identity, ASGP phase identity, and ASGP-DP dependency arity.
- Native AST JSON includes `LinearRec` and ASGP binder/spec side-table metadata.
  GPU reproduction preserves structured children by packing parent/donor
  side tables, copying back the selected parent/candidate context, and
  rebuilding child metadata during host decode with the same keep/shift/insert
  rules as typed-subtree replacement.
- ASGP/DC and ASGP/DP node kinds are declared for the current source grammar and
  grammar-config shape. The Python interpreter, Python VM, and native CPU
  runtime have an ASGP-DC slice using internal phase bytecode segments for
  direct semantic testing of hand-authored ASTs. The Python interpreter/VM and
  native CPU runtime also have an ASGP-DP1D slice with side-table
  bounds/dependency metadata and memoized phase execution. The Python
  interpreter/VM and native CPU runtime now have an ASGP-DP2D semantic slice
  with required side-table bounds/dependency metadata and internal phase
  bytecode segments. Native GPU fitness execution supports ASGP-DC,
  ASGP-DP1D, and ASGP-DP2D bytecode semantic slices with explicit device
  frames and bounded device memo storage. Native random generation and CPU
  subtree mutation can emit conservative ASGP-DC, ASGP-DP1D, and ASGP-DP2D
  slices only at full statement value roots. ASGP-DC covers `Int`, `Float`,
  and `String` roots: numeric roots may use an existing same-typed numeric-list
  source variable, while `String` roots traverse a `String` source as chars and
  rebuild with `singleton(index(...))` plus `concat`; literal-source fallback
  remains available. ASGP-DP1D and ASGP-DP2D now cover conservative `Int`,
  `Float`, and `String` root slices; `String` DP transitions use `concat`
  over memoized dependency results. GPU reproduction can transport existing
  ASGP subtrees and can synthesize conservative ASGP-DC, ASGP-DP1D, and
  ASGP-DP2D donors in `Int` / `Float` / `String` type-bucketed donor pools,
  while rebuilding ASGP side-table metadata during decode.
  Freer ASGP phase variation and broader source policies remain pending until
  the full ASGP typing policy lands.

### Evolution grammar configs
- `grammar-config` is the current schema.
- Checked-in `grammar-config` presets under `configs/grammar/` are accepted by Python and native loaders as compatibility input and translated into current search-space controls for fair comparisons.
- Generated native `compact` configs with `compat.num_list_mode="both"` preserve the old `NumList` input search-space shape by seeding exact `IntList` / `FloatList` fixture inputs as `Any` input variables. This does not reinterpret runtime fixture values.
- The config restricts random genome generation, CPU mutation donor synthesis, GPU reproduction preprocess candidate/donor generation, and seed replay regeneration.
- The config does not reject execution of existing ASTs or bytecode that use disabled constructs.
- CPU and GPU reproduction both respect non-default grammar configs.

### Builtins
Python, C++ CPU, and CUDA device runtime implementations cover the
runtime-supported current builtin set below. Native AST arity rules, compiler
lowering, typed-expression analysis, grammar-config gating, and GPU
reproduction child metadata parsing also recognize these source-call nodes.

Scalar builtins:
- `abs`
- `min`
- `max`
- `clip`
- `idiv0`
- `imod0`
- `char_to_string`
- `string_to_char`
- `ord`
- `chr`
- `is_letter`
- `is_digit`
- `is_space`
- `is_vowel`
- `to_lower`
- `to_upper`
- `to_string`

Container builtins:
- `len`
- `concat`
- `slice`
- `index`
- `append`
- `prepend`
- `reverse`
- `find`
- `contains`
- `singleton`

### Payload execution
Container values use payload-backed execution.
- CPU runtime keeps decoded `String`, `IntList`, `FloatList`, and `StringList` payloads in a registry after native current migration.
- The CPU payload registry can be swept down to a live-root closure between generations so dead container payloads from discarded individuals do not accumulate indefinitely.
- GPU runtime keeps a session-local host payload cache, lazily fills it by packed token from the process-global registry, and then builds compact per-eval payload packs for only the tokens needed by the current accepted population plus shared cases.
- GPU payload evaluation always launches one production `Mixed` eval kernel across the full accepted population.
- the finer `StringOnly` / `ListOnly` / `Mixed` flavor classifier is still kept for experiment tooling and offline bucketing studies
- GPU exact payload operations use bounded per-thread scratch.
- When exact output materialization does not fit, GPU transform builtins return deterministic fallback transport instead of aborting the full evaluation. CPU may still materialize larger host payloads, so CPU/GPU parity is guaranteed only within GPU payload limits.
- The native CLI defaults `retain_final_population` to `off`; the final scored population is not materialized unless explicitly requested, but `result.best` and history remain available.

See also:
- [CPP_RUNTIME_PAYLOAD.md](CPP_RUNTIME_PAYLOAD.md) for the C++ container token, payload registry, exact/fallback split, and collision tradeoffs.

## Fitness Model

The current scoring model is defined in [fitness.md](../spec/fitness.md).

Operational summary:
- numeric expected + numeric actual => negative absolute error
- numeric expected + non-numeric actual => `-penalty`
- `Bool` / `Char` / `String` / `IntList` / `FloatList` / `StringList` => exact match `1`, same-type mismatch `0`, type mismatch `-penalty`
- runtime error => `-penalty`

This keeps numeric tasks dense while keeping container semantics exact and simple.

## Python Module Map

### `python/src/g3p_vm_gpu/core/`
- `ast.py`: prefix AST definitions and traversal helpers
- `errors.py`: runtime outcome and error types
- `value_semantics.py`: shared scalar comparison and exact numeric operand rules

### `python/src/g3p_vm_gpu/runtime/`
- `builtins.py`: reference builtin semantics
- `compiler.py`: AST to bytecode compiler
- `interp.py`: direct AST interpreter
- `vm.py`: Python bytecode VM

### `python/src/g3p_vm_gpu/evolution/`
- `genome.py`: genome container and compile-for-eval helpers
- `grammar_config.py`: Python-side `grammar-config` parser and legacy compatibility translator
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
- `convert_psb_to_fitness_cases.py`: convert PSB1/PSB2 JSON Lines into `fitness-cases` compatibility fixtures or `fitness-cases` direct-list fixtures with schema hashes and optional field-schema overrides

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
- `spec/grammar.md`
- `spec/bytecode_isa.md`
- `spec/bytecode_format.md` if the wire format changed
- this file

### Builtin, type, or payload changes
Update:
- `spec/builtins_base.md` or `spec/builtins_runtime.md`
- `spec/bytecode_isa.md` if opcode behavior changed
- this file

### Fitness or evolution-arg changes
Update:
- `spec/fitness.md`
- `docs/DEVELOPMENT.md`
- `README.md` if the main workflow or key defaults changed

### Repo structure or entrypoint changes
Update:
- this file
- `docs/FILE_STRUCTURE.md`
- repo skill references under `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/references/`
