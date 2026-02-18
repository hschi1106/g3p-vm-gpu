# g3p-vm-gpu — Full Architecture & Structure

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Layout](#2-directory-layout)
3. [Core Data Models](#3-core-data-models)
4. [Python Reference Implementation](#4-python-reference-implementation)
5. [C++ Execution Engine](#5-c-execution-engine)
6. [GPU Acceleration Path](#6-gpu-acceleration-path)
7. [Evolution Pipeline](#7-evolution-pipeline)
8. [Fixture & Data Formats](#8-fixture--data-formats)
9. [Bytecode ISA Specification](#9-bytecode-isa-specification)
10. [Tooling & Scripts](#10-tooling--scripts)
11. [Testing Architecture](#11-testing-architecture)
12. [Build & Run](#12-build--run)
13. [Performance Measurement Pipeline](#13-performance-measurement-pipeline)
14. [Cross-Cutting Design Patterns](#14-cross-cutting-design-patterns)
15. [Design Constraints & Decisions](#15-design-constraints--decisions)

---

## 1. Project Overview

g3p-vm-gpu is a **prefix-AST genetic programming virtual machine** providing:

- **Python reference implementation** — interpreter, compiler, bytecode VM, evolution loop
- **C++ CPU/GPU evolution backends** — high-performance bytecode VM and genetic algorithm core

**Primary KPI**: fair CPU vs GPU speedup measurement with strictly identical fitness computation on both paths.

### End-to-End Pipeline

```
Prefix AST (AstProgram)
    │
    ▼
Compiler (AST → Bytecode)
    │
    ▼
BytecodeProgram
    │
    ├──► Python VM (reference)
    ├──► C++ CPU VM
    └──► C++ GPU VM (CUDA kernel)
          │
          ▼
    Fitness Evaluation → Evolution Loop (selection / crossover / mutation)
```

---

## 2. Directory Layout

```
g3p-vm-gpu/
├── python/                        # Python reference implementation
│   ├── src/g3p_vm_gpu/            # Core package
│   │   ├── ast.py                 # AST node definitions (frozen dataclasses)
│   │   ├── interp.py              # Tree-walking reference interpreter
│   │   ├── compiler.py            # AST → bytecode compiler
│   │   ├── vm.py                  # Bytecode stack-machine VM
│   │   ├── semantics.py           # Shared type promotion / comparison logic
│   │   ├── builtins.py            # Built-in function implementations
│   │   ├── errors.py              # Error codes and outcome types
│   │   ├── fuzz.py                # Random program generator
│   │   ├── evo_encoding.py        # Genome type, genetic operators, validation
│   │   ├── evolve.py              # Full generational evolution loop
│   │   └── demo.py                # Manual smoke test
│   └── tests/                     # Python test suite
│       ├── test_eval.py           # Interpreter tests + 1000-program fuzz
│       ├── test_vm_equiv.py       # Interpreter vs VM equivalence
│       ├── test_evo_encoding.py   # Genome manipulation tests
│       ├── test_evolve.py         # Evolution loop tests
│       ├── test_cpp_vm_equiv.py   # Python vs C++ VM equivalence
│       ├── test_simple_evo_fixtures.py  # Fixture format validation
│       └── test_run_cpp_evolution_tool.py  # Tool integration tests
│
├── cpp/                           # C++ high-performance engine
│   ├── CMakeLists.txt             # Build configuration
│   ├── include/g3pvm/             # Public headers
│   │   ├── value.hpp              # Tagged union Value (host+device)
│   │   ├── value_semantics.hpp    # Shared CPU/GPU arithmetic & comparison
│   │   ├── bytecode.hpp           # BytecodeProgram + Instr
│   │   ├── evo_ast.hpp            # AstProgram, ProgramGenome, genetic ops
│   │   ├── evolve.hpp             # EvolutionConfig, FitnessCase, evolution API
│   │   ├── vm_cpu.hpp             # CPU VM execution API
│   │   ├── vm_gpu.hpp             # GPU VM execution API + GPUFitnessSession
│   │   ├── errors.hpp             # ErrCode enum
│   │   └── builtins.hpp           # Built-in function dispatch
│   ├── src/
│   │   ├── vm_cpu.cpp             # CPU VM implementation
│   │   ├── vm_gpu.cu              # CUDA kernel + GPU VM implementation
│   │   ├── vm_gpu/                # GPU internals
│   │   │   ├── host_pack.hpp      # Program/case packing for GPU upload
│   │   │   ├── types.hpp          # Device-side DInstr, DResult, DProgramMeta
│   │   │   └── constants.hpp      # GPU constants (MAX_STACK, MAX_LOCALS, opcodes)
│   │   ├── evo_ast.cpp            # AST operations, compiler, genetic operators
│   │   ├── evolve.cpp             # Evolution loop + profiled variant
│   │   ├── evolve_cli.cpp         # Evolution CLI entry point
│   │   ├── builtins.cpp           # Built-in function implementations
│   │   ├── vm_cpu_cli.cpp         # CPU VM CLI entry point
│   │   ├── vm_cpu_cli/            # CLI support (JSON codec)
│   │   │   ├── codec.cpp          # JSON ↔ BytecodeProgram conversion
│   │   │   └── json.cpp           # JSON parser
│   │   ├── vm_cpu_multi_bench.cpp # CPU multi-program benchmark
│   │   └── vm_gpu_multi_bench.cpp # GPU multi-program benchmark
│   ├── tests/
│   │   ├── test_vm_smoke.cpp          # CPU VM basic tests
│   │   ├── test_vm_edges.cpp          # VM edge-case tests
│   │   ├── test_vm_gpu_smoke.cpp      # GPU VM basic tests
│   │   ├── test_vm_gpu_multi_batch.cpp# GPU batch execution tests
│   │   ├── test_fitness_cpu_gpu_parity.cpp # CPU/GPU fitness parity
│   │   ├── test_evo_ast.cpp           # Evolution AST unit tests
│   │   └── test_evolve.cpp            # Evolution loop tests
│   ├── build/                     # Debug build artifacts
│   └── build_release/             # Release build artifacts
│
├── spec/                          # Specifications (behavioral source of truth)
│   ├── subset_v0_1.md             # Python-like subset language spec
│   ├── bytecode_isa.md            # Bytecode instruction set architecture
│   ├── bytecode_format_v0_1.md    # Bytecode JSON transport format
│   └── builtins.md                # Built-in function contracts
│
├── data/fixtures/                 # Unified input fixtures (fitness-cases-v1)
│   ├── simple_evo_x_plus_1_1024.json
│   ├── simple_evo_affine_2x_plus_3_1024.json
│   ├── simple_evo_square_x2_1024.json
│   └── speedup_cases_bouncing_balls_1024.json
│
├── tools/                         # Benchmark & experiment scripts
│   ├── run_cpp_evolution.py       # C++ evolution launcher (CPU/GPU)
│   └── run_cpu_gpu_speedup_experiment.sh  # CPU vs GPU speedup experiment
│
├── scripts/                       # Operational wrappers
│   ├── run_gpu_command.sh         # GPU device auto-retry wrapper
│   └── run_triplet_checks.sh     # Python + C++ build + ctest triple check
│
├── docs/                          # Development documentation
│   ├── ARCHITECTURE.md            # Architecture summary
│   └── DEVELOPMENT.md             # Developer runbook
│
├── logs/                          # Experiment outputs (not version-controlled)
│
├── AGENTS.md                      # Repository guidelines for AI agents
└── README.md                      # Project entry point
```

---

## 3. Core Data Models

### 3.1 Prefix AST (`AstProgram`)

The project uses a **linear prefix AST** as its single canonical program representation. There are no pointer-based tree nodes — the tree structure is implicit via node arities.

**Python** — frozen dataclasses in `python/src/g3p_vm_gpu/ast.py`:

```python
@dataclass(frozen=True)
class AstNode:
    kind: NodeKind
    i0: int = 0
    i1: int = 0

@dataclass(frozen=True)
class AstProgram:
    nodes: tuple[AstNode, ...]
    names: tuple[str, ...]
    consts: tuple[Val, ...]
    version: str = "ast-prefix-v1"
```

**C++** — structs in `cpp/include/g3pvm/evo_ast.hpp`:

```cpp
struct AstNode  { NodeKind kind; int i0 = 0, i1 = 0; };
struct AstProgram {
    std::vector<AstNode> nodes;
    std::vector<std::string> names;
    std::vector<Value> consts;
    std::string version = "ast-prefix-v1";
};
```

**NodeKind** (29 variants, identical in both languages):

| Category         | NodeKind values                                                  |
|------------------|------------------------------------------------------------------|
| Structure        | `PROGRAM`, `BLOCK_NIL`, `BLOCK_CONS`                             |
| Statements       | `ASSIGN`, `IF_STMT`, `FOR_RANGE`, `RETURN`                       |
| Leaf expressions | `CONST`, `VAR`                                                   |
| Unary ops        | `NEG`, `NOT`                                                     |
| Binary ops       | `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`, `AND`, `OR` |
| Ternary          | `IF_EXPR`                                                        |
| Built-in calls   | `CALL_ABS`, `CALL_MIN`, `CALL_MAX`, `CALL_CLIP`                  |

Tree traversal is arity-based: `prefix_subtree_end()` and `prefix_child_index()` compute subtree boundaries using `NODE_ARITY`.

### 3.2 Bytecode Program (`BytecodeProgram`)

**Python** — `python/src/g3p_vm_gpu/compiler.py`:

```python
@dataclass(frozen=True)
class BytecodeProgram:
    consts: tuple[Val, ...]
    code: tuple[Instr, ...]
    n_locals: int
    var2idx: dict[str, int]
```

**C++** — `cpp/include/g3pvm/bytecode.hpp`:

```cpp
struct Instr { std::string op; int a = 0, b = 0; bool has_a = false, has_b = false; };
struct BytecodeProgram {
    std::vector<Value> consts;
    std::vector<Instr> code;
    int n_locals = 0;
    std::unordered_map<std::string, int> var2idx;
};
```

### 3.3 Value Type

```
Value ::= Int(int64) | Float(double) | Bool(bool) | None
```

The C++ `Value` (in `value.hpp`) is a tagged union annotated `__host__ __device__` for use in both CPU and GPU code:

```cpp
struct Value {
    union { int64_t i; double f; };
    bool b = false;
    ValueTag tag = ValueTag::None;
    // factory statics: from_int(), from_float(), from_bool(), none()
};
static_assert(std::is_trivially_copyable_v<Value>);
static_assert(sizeof(Value) <= 16);
```

**Critical rule**: `Bool` is **not** a numeric type. Conditions must be strictly `Bool` — no implicit truthiness.

### 3.4 Program Genome (`ProgramGenome`)

The unit of evolution, defined in both `evo_encoding.py` and `evo_ast.hpp`:

```cpp
struct GenomeMeta {
    int node_count = 0;
    int max_depth = 0;
    bool uses_builtins = false;
    std::string hash_key;         // SHA1 prefix for deduplication
};

struct ProgramGenome {
    AstProgram ast;
    GenomeMeta meta;
};
```

### 3.5 Error Model

```
ErrCode ::= Name | Type | ZeroDiv | Value | Timeout
```

Execution outcomes (Python): `Normal()`, `Returned(value)`, `Failed(err: Err)`.
C++ VMResult: `{is_error, value, err}`.

On any error the VM halts immediately — no continuation after errors.

---

## 4. Python Reference Implementation

Located in `python/src/g3p_vm_gpu/`. Serves as the **behavioral reference** — all C++ implementations must produce identical results.

| Module              | Purpose                                              |
|---------------------|------------------------------------------------------|
| `ast.py`            | AST node definitions, prefix tree traversal, `build_program()` DSL |
| `interp.py`         | Tree-walking interpreter (directly evaluates prefix AST) |
| `compiler.py`       | AST → bytecode compiler (label-based jumps, FOR_RANGE unrolling) |
| `vm.py`             | Bytecode stack-machine VM                            |
| `semantics.py`      | Numeric promotion, comparison logic (shared rules)   |
| `builtins.py`       | Built-in function implementations (`abs`, `min`, `max`, `clip`) |
| `errors.py`         | Error codes and execution outcome types              |
| `fuzz.py`           | Random program generator (`make_random_program`)     |
| `evo_encoding.py`   | Genome type, genetic operators (mutate, crossover), validation |
| `evolve.py`         | Full generational evolution loop with selection strategies |
| `demo.py`           | Manual smoke test entry point                        |

### Execution Paths

```python
# Path 1: Direct interpretation (reference semantics)
env, outcome = run_program(prog, inputs, fuel=10000)

# Path 2: Compile + VM execution
bytecode = compile_program(prog)
result = run_bytecode(bytecode, inputs, fuel=10000)
```

### Interpreter Design

The interpreter (`interp.py`) uses an **env-passing functional style** — environment is copied on mutation. It walks the prefix AST recursively with fuel checking at every node. Short-circuit evaluation is implemented for `AND`, `OR`, and `IF_EXPR` via subtree-skipping functions (`_skip_expr_end`, etc.).

### Compiler Design

The compiler (`compiler.py`) uses an internal `_Compiler` class that:
- Emits instructions into a flat `code[]` list
- Uses **label-based forward jumps** resolved at `finalize()`
- Compiles `FOR_RANGE` into counter-based loops with hidden temp locals
- Implements `AND`/`OR` via short-circuit conditional jumps

---

## 5. C++ Execution Engine

### 5.1 CPU VM

- Header: `cpp/include/g3pvm/vm_cpu.hpp`
- Implementation: `cpp/src/vm_cpu.cpp`

```cpp
VMResult run_bytecode(const BytecodeProgram& prog,
                      const std::vector<LocalBinding>& inputs,
                      int fuel);

std::vector<int> run_bytecode_cpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& cases,
    const std::vector<Value>& answer,
    int fuel);
```

The CPU VM mirrors the Python VM exactly: stack machine with `ip`, `stack[]`, `locals[]`, `consts[]`, fuel-per-instruction.

### 5.2 AST Compiler (C++)

The `Compiler` class in `cpp/src/evo_ast.cpp` compiles `ProgramGenome` → `BytecodeProgram`:

```cpp
BytecodeProgram compile_for_eval(const ProgramGenome& genome);
BytecodeProgram compile_for_eval_with_preset_locals(
    const ProgramGenome& genome,
    const std::vector<std::pair<std::string, int>>& preset_locals);
```

Compilation conventions follow `spec/bytecode_isa.md §4`:
- Binary ops: `compile(lhs); compile(rhs); OP`
- Ternary `if_expr`: `compile(cond); JMP_IF_FALSE else; compile(then); JMP end; else: compile(else); end:`
- Short-circuit `AND`/`OR` via conditional jumps

### 5.3 Shared Value Semantics

`cpp/include/g3pvm/value_semantics.hpp` contains **all arithmetic and comparison logic** as `__host__ __device__ inline` functions — a single source of truth compiled for both CPU and GPU:

```cpp
CompareStatus compare_values(CmpOp op, Value a, Value b, bool& out);
void to_numeric_pair(Value a, Value b, double& a_out, double& b_out, bool& any_float);
double py_float_mod(double a, double b);
int64_t py_int_mod(int64_t a, int64_t b);
bool values_equal_for_fitness(Value a, Value b);  // strict, float epsilon 1e-12
```

This ensures **CPU/GPU arithmetic parity** by construction.

### 5.4 Evolution CLI

`cpp/src/evolve_cli.cpp` — the main C++ evolution entry point supporting:
- `--engine cpu|gpu` engine selection
- Configurable population size, generations, mutation/crossover rates
- Structured stdout output (`GEN`, `FINAL`, `TIMING` lines) parsed by `tools/run_cpp_evolution.py`
- Fine-grained timing breakdown per generation

---

## 6. GPU Acceleration Path

### 6.1 Device Data Structures

Defined in `cpp/src/vm_gpu/types.hpp`:

```cpp
struct DInstr {                     // Packed GPU instruction (8 bytes)
    uint8_t op;                     // DeviceOp enum
    uint8_t flags;                  // DINSTR_HAS_A, DINSTR_HAS_B
    int32_t a, b;
};

struct DProgramMeta {               // Per-program metadata for dispatch
    int code_offset, code_len;
    int const_offset, const_len;
    int n_locals;
    int case_offset, case_count;
    int case_local_offset;
    int is_valid, err_code;
};

struct DResult { int is_error, err_code; Value value; };
```

Constants: `MAX_STACK = 64`, `MAX_LOCALS = 64`, 21 opcodes (`OP_PUSH_CONST` .. `OP_RETURN`).

### 6.2 Host Packing Layer

`cpp/src/vm_gpu/host_pack.hpp` flattens multi-program + shared-cases into contiguous GPU-friendly arrays:

```cpp
struct PackResult {
    std::vector<DProgramMeta> metas;
    std::vector<DInstr>       all_code;
    std::vector<Value>        all_consts;
    std::vector<Value>        packed_case_local_vals;
    std::vector<unsigned char> packed_case_local_set;
    std::size_t total_cases, max_code_len;
};

struct DeviceArena {                // RAII CUDA memory holder
    Value* d_consts; DInstr* d_code; DProgramMeta* d_metas;
    Value* d_shared_case_local_vals; unsigned char* d_shared_case_local_set;
    DResult* d_out; Value* d_expected; int* d_fitness;
    // destructor frees all device memory
};
```

### 6.3 GPU Execution Model

1. **Host packing** — flatten N programs × M cases into contiguous `PackResult`
2. **Upload** — single `cudaMemcpy` to device via `DeviceArena`
3. **Kernel launch** — one thread per (program, case) pair; thread-local stack + locals in registers/local memory
4. **Copyback** — fitness scores (one `int` per program) back to host
5. **Session reuse** — `GPUFitnessSession` (pimpl pattern) keeps cases/answers on device across generations, amortizing upload cost

```cpp
class GPUFitnessSession {
    void init(const std::vector<InputCase>& cases,
              const std::vector<Value>& answer, int fuel, int blocksize);
    std::vector<int> eval_programs(const std::vector<BytecodeProgram>& programs);
    bool is_ready() const;
    // move-only, pimpl hides CUDA types
};
```

### 6.4 Fitness Scoring Formula

Same on CPU and GPU:

```
score = exact_match_count - round(mean_abs_error) + runtime_error_count * (-10)
```

Where `exact_match_count` uses strict equality (`values_equal_for_fitness`: tag must match, float epsilon 1e-12).

### 6.5 GPU Device Management

All GPU commands must go through `scripts/run_gpu_command.sh`:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper auto-retries with `CUDA_VISIBLE_DEVICES=0` then `=1` on device-unavailable failures.

---

## 7. Evolution Pipeline

### 7.1 Configuration

Defined in `cpp/include/g3pvm/evolve.hpp` (and mirrored in `python/src/g3p_vm_gpu/evolve.py`):

```cpp
struct EvolutionConfig {
    int population_size    = 64;
    int generations        = 40;
    int elitism            = 2;
    double mutation_rate   = 0.5;
    double crossover_rate  = 0.9;
    CrossoverMethod crossover_method = CrossoverMethod::Hybrid;
    SelectionMethod selection_method = SelectionMethod::Tournament;
    EvalEngine eval_engine = EvalEngine::CPU;
    int gpu_blocksize      = 256;
    int tournament_k       = 3;
    double truncation_ratio = 0.5;
    int fuel               = 20000;
    Limits limits;
    // float tolerances, reward/penalty weights ...
};
```

### 7.2 AST Size Limits

```cpp
struct Limits {
    int max_expr_depth       = 5;
    int max_stmts_per_block  = 6;
    int max_total_nodes      = 80;
    int max_for_k            = 16;
    int max_call_args        = 3;
};
```

### 7.3 Genetic Operators

All defined in `cpp/src/evo_ast.cpp` (public API in `evo_ast.hpp`), with Python equivalents in `evo_encoding.py`:

| Operator              | Function                        | Description                                    |
|-----------------------|---------------------------------|------------------------------------------------|
| Random generation     | `make_random_genome(seed, limits)` | Type-driven recursive prefix AST generation |
| Mutation              | `mutate(genome, seed, limits)`  | Replace random subtree (80%) or regenerate     |
| Top-level crossover   | `crossover_top_level(a, b, seed, limits)` | One-point splice of top-level statement lists |
| Typed-subtree crossover | `crossover_typed_subtree(a, b, seed, limits)` | Same-type subtree exchange |
| Hybrid crossover      | `crossover(a, b, seed, limits, method)` | Dispatches by `CrossoverMethod` |
| Validation            | `validate_genome(genome, limits)` | Check size, depth, structure, compilability |

### 7.4 Random Program Generation

Uses recursive descent with type guidance:
- `RType`: `Num`, `Bool`, `NoneType`, `Any`, `Invalid`
- Leaf nodes: 45% chance of selecting a known same-type variable, otherwise generate constant
- Constants: integers in `[-8, 8]`, floats in `[-8.0, 8.0]` with 0.001 precision

### 7.5 Selection Methods

| Method       | Enum value    | Description                          |
|--------------|---------------|--------------------------------------|
| Tournament   | `Tournament`  | Pick best of `k` random individuals  |
| Roulette     | `Roulette`    | Fitness-proportional selection       |
| Rank         | `Rank`        | Rank-proportional selection          |
| Truncation   | `Truncation`  | Uniform from top `ratio` fraction    |
| Random       | `Random`      | Uniform random (baseline)            |

### 7.6 Evolution Loop

```
Initialize population (make_random_genome × population_size)
    │
    ▼
for generation in 0..generations:
    │
    ├── Compile genomes → BytecodePrograms
    ├── Evaluate fitness (CPU VM or GPU kernel)
    ├── Sort by fitness (descending)
    ├── Record best/mean fitness + timing
    ├── Preserve elite (top elitism individuals)
    ├── Fill rest via:
    │   ├── Crossover (with crossover_rate probability)
    │   └── Mutation (with mutation_rate probability)
    └── Validate new genomes
```

The profiled variant (`evolve_population_profiled`) returns an `EvolutionRun` with detailed `EvolutionTiming` covering every phase.

---

## 8. Fixture & Data Formats

### 8.1 Unified Format: `fitness-cases-v1`

All evolution/benchmark inputs use a single JSON schema:

```json
{
  "format_version": "fitness-cases-v1",
  "meta": {
    "description": "...",
    "var_names": ["x"],
    "target_function": "2*x + 3"
  },
  "cases": [
    {
      "inputs": { "x": { "type": "int", "value": 42 } },
      "expected": { "type": "int", "value": 87 }
    }
  ]
}
```

### 8.2 Committed Fixtures

| File                                       | Target function    | Cases |
|--------------------------------------------|--------------------|-------|
| `simple_evo_x_plus_1_1024.json`            | $f(x) = x + 1$    | 1024  |
| `simple_evo_affine_2x_plus_3_1024.json`    | $f(x) = 2x + 3$   | 1024  |
| `simple_evo_square_x2_1024.json`           | $f(x) = x^2$      | 1024  |
| `speedup_cases_bouncing_balls_1024.json`   | Bouncing ball sim  | 1024  |

### 8.3 Bytecode JSON Transport Format

Used by the CPU/GPU VM CLIs, defined in `spec/bytecode_format_v0_1.md`:

```json
{
  "format_version": "bytecode-json-v0.1",
  "engine": "cpu",
  "fuel": 20000,
  "bytecode": {
    "n_locals": 3,
    "consts": [{"type": "int", "value": 1}],
    "code": [{"op": "LOAD", "a": 0}]
  },
  "inputs": [{"idx": 0, "value": {"type": "int", "value": 5}}]
}
```

---

## 9. Bytecode ISA Specification

Full specification in `spec/bytecode_isa.md` (v0.1).

### 9.1 VM State

| Component | Description                                     |
|-----------|-------------------------------------------------|
| `ip`      | Instruction pointer                              |
| `stack`   | Operand stack                                    |
| `locals`  | Local variable array (UNSET sentinel for unset)  |
| `consts`  | Constant pool                                    |
| `fuel`    | Instruction budget (checked before each instruction; ≤ 0 → Timeout) |

### 9.2 Instruction Set

| Category     | Instructions                                          |
|--------------|-------------------------------------------------------|
| Const/Var    | `PUSH_CONST k`, `LOAD i`, `STORE i`                   |
| Unary        | `NEG`, `NOT`                                           |
| Arithmetic   | `ADD`, `SUB`, `MUL`, `DIV`, `MOD`                     |
| Comparison   | `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`                    |
| Control flow | `JMP addr`, `JMP_IF_FALSE addr`, `JMP_IF_TRUE addr`   |
| Built-ins    | `CALL_BUILTIN bid argc`                                |
| Return       | `RETURN`                                               |

### 9.3 Built-in Functions

Whitelist-only, defined in `spec/builtins.md`:

| bid | Name   | Signature                                    | Notes                  |
|-----|--------|----------------------------------------------|------------------------|
| 0   | `abs`  | `abs(x: Int\|Float) → Int\|Float`            | Preserves type         |
| 1   | `min`  | `min(x, y: Numeric) → Numeric`               | Numeric promotion      |
| 2   | `max`  | `max(x, y: Numeric) → Numeric`               | Numeric promotion      |
| 3   | `clip` | `clip(x, lo, hi: Numeric) → Numeric`         | Requires $lo \leq hi$ |

All are pure (no side effects). Bool is not numeric. Args pushed left-to-right, last at TOS.

### 9.4 Type Promotion Rules

Following Python semantics:
- `Int ⊕ Float → Float` (any float promotes to float domain)
- `Bool` does **not** participate in arithmetic
- Comparison: `Bool` supports only `EQ`/`NE`; `None` supports only `EQ`/`NE`; numerics support all six

### 9.5 Error Kinds

| Error        | Trigger                                          |
|--------------|--------------------------------------------------|
| `NameError`  | Undefined variable, unknown builtin              |
| `TypeError`  | Type mismatch on operation                       |
| `ZeroDiv`    | Division/modulo by zero                          |
| `ValueError` | Domain violation (e.g., `clip` with `lo > hi`)   |
| `Timeout`    | Fuel exhausted                                   |

---

## 10. Tooling & Scripts

### 10.1 Evolution Launcher

`tools/run_cpp_evolution.py` — Python orchestrator for C++ evolution runs:

```bash
python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/simple_evo_x_plus_1_1024.json \
  --cpp-cli cpp/build/g3pvm_evolve_cli \
  --engine gpu \
  --population-size 1024 \
  --generations 40 \
  --cpp-timing all
```

Features:
- Loads fixture → launches C++ CLI → parses structured stdout (`GEN`, `FINAL`, `TIMING` regex patterns)
- `StageTracker` class for per-phase wall-clock timing
- Generates `*.summary.json`, `*.evolution.json`, `*.timings.log`

### 10.2 CPU vs GPU Speedup Experiment

`tools/run_cpu_gpu_speedup_experiment.sh`:

```bash
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
```

- Runs CPU and GPU evolution on identical fixture with identical config
- Produces comparison report: `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.md` and `.json`
- Defaults: `popsize=4096`, `generations=40`, `blocksize=256`, `selection=tournament`, `crossover=hybrid`

### 10.3 Operational Scripts

| Script                        | Purpose                                         |
|-------------------------------|------------------------------------------------|
| `scripts/run_gpu_command.sh`  | GPU device auto-retry (mandatory for all GPU commands) |
| `scripts/run_triplet_checks.sh` | Python tests + C++ build + ctest triple check |

---

## 11. Testing Architecture

### 11.1 Python Tests

Framework: `unittest`. Run from repo root:

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
```

| Test file                      | Coverage                                          |
|-------------------------------|---------------------------------------------------|
| `test_eval.py`                | Interpreter correctness + 1000-program fuzz       |
| `test_vm_equiv.py`            | Interpreter vs VM equivalence                     |
| `test_evo_encoding.py`        | Genome manipulation, validation, genetic operators |
| `test_evolve.py`              | Full evolution loop, selection, scoring            |
| `test_cpp_vm_equiv.py`        | Python vs C++ VM equivalence                      |
| `test_simple_evo_fixtures.py` | Fixture format validation                         |
| `test_run_cpp_evolution_tool.py` | Tool integration tests                         |

The fuzz test (`test_fuzz_random_programs` in `test_eval.py`) generates 1000 random programs via `make_random_program()` and asserts all produce `Returned` or `Failed` (including `Timeout`) — never crash.

### 11.2 C++ Tests

Framework: ctest. Run:

```bash
ctest --test-dir cpp/build --output-on-failure
```

| Test file                        | Coverage                              |
|----------------------------------|---------------------------------------|
| `test_vm_smoke.cpp`              | CPU VM basic operations               |
| `test_vm_edges.cpp`              | VM edge cases and error paths         |
| `test_vm_gpu_smoke.cpp`          | GPU VM basic operations               |
| `test_vm_gpu_multi_batch.cpp`    | GPU batch execution correctness       |
| `test_fitness_cpu_gpu_parity.cpp`| CPU vs GPU fitness result identity    |
| `test_evo_ast.cpp`               | AST operations, compilation, genetic ops |
| `test_evolve.cpp`                | Evolution loop correctness            |

### 11.3 Key Testing Invariant

**CPU/GPU fitness parity** is enforced: `test_fitness_cpu_gpu_parity.cpp` runs the same programs through both CPU and GPU fitness evaluation and asserts identical integer fitness scores.

---

## 12. Build & Run

### 12.1 Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compiler
- CUDA Toolkit + NVIDIA driver (for GPU path)

### 12.2 Build C++

```bash
# Debug build
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j

# Release build
cmake -S cpp -B cpp/build_release -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build_release -j
```

### 12.3 Run Tests

```bash
# All Python tests
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v

# Single Python module
PYTHONPATH=python python3 -m unittest python.tests.test_vm_equiv -v

# All C++ tests
ctest --test-dir cpp/build --output-on-failure

# Full triple check (Python + build + ctest)
bash scripts/run_triplet_checks.sh
```

### 12.4 Run Demo

```bash
PYTHONPATH=python/src python3 -m g3p_vm_gpu.demo
```

### 12.5 Key Build Artifacts

| Binary                                    | Purpose                       |
|-------------------------------------------|-------------------------------|
| `g3pvm_evolve_cli`                        | Evolution CLI (CPU + GPU)     |
| `g3pvm_vm_cpu_cli`                        | CPU VM CLI (JSON bytecode)    |
| `g3pvm_vm_gpu_batch_cli`                  | GPU batch VM CLI              |
| `g3pvm_vm_cpu_multi_bench`                | CPU multi-program benchmark   |
| `g3pvm_vm_gpu_multi_bench`                | GPU multi-program benchmark   |
| `g3pvm_vm_cpu_fitness_multi_bench`        | CPU fitness benchmark         |
| `g3pvm_vm_gpu_fitness_multi_bench`        | GPU fitness benchmark         |

---

## 13. Performance Measurement Pipeline

### 13.1 Timing Dimensions

Each generation in the evolution loop records fine-grained timing via `EvolutionTiming`:

| Metric              | Description                                |
|---------------------|--------------------------------------------|
| `eval_ms`           | Fitness evaluation time                    |
| `repro_ms`          | Total reproduction phase time              |
| `selection_ms`      | Parent selection time                      |
| `crossover_ms`      | Crossover operator time                    |
| `mutation_ms`       | Mutation operator time                     |
| `elite_ms`          | Elite preservation time                    |
| `cpu_compile_ms`    | CPU-side compilation time                  |

GPU-specific additional metrics:

| Metric              | Description                                |
|---------------------|--------------------------------------------|
| `pack_upload_ms`    | Program packing + device upload time       |
| `kernel_ms`         | Kernel execution time                      |
| `copyback_ms`       | Result copy-back time                      |

### 13.2 Report Structure

```
logs/cpu_gpu_compare_pop1024_YYYYMMDD_HHMMSS/
├── cpu_gpu_compare.report.md          # Human-readable comparison
├── cpu_gpu_compare.report.json        # Machine-readable comparison
├── cpp_evo_*_cpu_*.summary.json       # CPU run summary
├── cpp_evo_*_gpu_*.summary.json       # GPU run summary
└── *.stdout.log / *.stderr.log        # Raw outputs
```

### 13.3 Speedup Metrics

The comparison report includes multi-level speedup:

- **End-to-end speedup** — total wall-clock time ratio
- **Inner total speedup** — evolution loop internal time ratio
- **Eval-only speedup** — pure fitness evaluation time ratio

### 13.4 Profiling Rules

- GPU profiling must use `nsys` only (not `ncu` — GPU performance counters unavailable)
- On device contention, retry with `CUDA_VISIBLE_DEVICES=0` then `=1`

---

## 14. Cross-Cutting Design Patterns

| Pattern                           | Where / How                                                                     |
|-----------------------------------|---------------------------------------------------------------------------------|
| **Prefix-encoded AST**            | Core representation in both languages; no tree pointers, arity-based traversal   |
| **Tagged union Value**            | Trivially copyable, ≤ 16 bytes, `__host__ __device__` annotated                 |
| **Single-source semantics**       | `value_semantics.hpp` — `__host__ __device__ inline` for CPU/GPU parity         |
| **Session-based GPU evaluation**  | `GPUFitnessSession` — upload cases once, reuse across generations               |
| **Pimpl for CUDA isolation**      | `GPUFitnessSession::Impl` hides CUDA types from non-CUDA translation units      |
| **Contiguous packing layer**      | `host_pack.hpp` flattens multi-program + shared-cases into GPU-friendly arrays   |
| **Round-trip AST ↔ tuple DSL**    | `evo_encoding.py` converts prefix AST to/from nested tuples for genetic ops     |
| **Frozen / immutable data**       | All Python data types are frozen dataclasses; env-passing in interpreter         |
| **Fuel-bounded execution**        | Both interpreter and VM check fuel before every instruction/node                 |
| **CPU/GPU fitness parity**        | Same scoring formula in both paths; enforced by dedicated parity tests           |

---

## 15. Design Constraints & Decisions

### 15.1 Core Constraints

1. **Prefix-only AST** — single canonical representation across Python and C++; no tree-pointer structures
2. **CPU/GPU fitness identity** — both paths must produce identical integer fitness scores
3. **Spec-driven behavior** — `spec/` documents are the authoritative source of truth
4. **Version-frozen specs** — v0.1 specs are frozen; semantic changes require new version documents

### 15.2 Key Design Decisions

| Decision                      | Rationale                                               |
|-------------------------------|--------------------------------------------------------|
| Prefix AST (not tree)         | Contiguous memory layout; GPU-friendly; trivial copy    |
| `Bool` is not numeric         | Avoids implicit conversion bugs                         |
| Fuel mechanism                | Guarantees VM termination; prevents infinite loops       |
| Whitelist-only built-ins      | Controls search space complexity                         |
| Unified fixture format        | Simplifies fair CPU/GPU comparison                       |
| GPU session reuse             | Amortizes device memory allocation across generations    |
| Pimpl for GPU session         | Allows non-CUDA code to use GPU API without CUDA headers |
| `__host__ __device__` semantics | Single source eliminates CPU/GPU divergence by construction |

### 15.3 Language Subset Boundaries

Supported (per `spec/subset_v0_1.md`):
- Expressions: constants, variables, unary/binary ops, ternary conditional, built-in calls
- Statements: assignment, if-else, bounded for-range, return

**Not supported** (by design):
- Containers (list, dict, set, tuple)
- User-defined functions or classes
- While loops, break, continue
- Exception handling
- Import, I/O, strings
- Nested function definitions
