# g3p-vm-gpu

Bytecode VM project with Python reference/runtime and C++ CPU/GPU backends.

## What is in this repo

- `python/src/g3p_vm_gpu/`
  - language AST, compiler, interpreter, Python VM, fuzz generator
- `cpp/`
  - C++ CPU VM
  - CUDA GPU VM (`run_bytecode_gpu_multi_batch`, fitness)
  - CLIs and C++ tests
- `spec/`
  - language, bytecode ISA, builtins, and test contracts
- `tools/`
  - fixture generation and validation scripts

## Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compiler
- CUDA toolkit + NVIDIA driver (for GPU path)

## Python test commands

Run all Python tests:

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
```

Run one module:

```bash
PYTHONPATH=python python3 -m unittest python.tests.test_vm_equiv -v
```

Run demo:

```bash
PYTHONPATH=python/src python3 -m g3p_vm_gpu.demo
```

## Evolution Encoding (AST Genome)

The repo now includes a typed AST encoding layer for evolutionary workflows:

- module: `python/src/g3p_vm_gpu/evo_encoding.py`
- key APIs:
  - `make_random_genome(seed, limits)`
  - `mutate(genome, seed, limits)`
  - `crossover(parent_a, parent_b, seed, limits, method=...)`
  - `crossover_top_level(parent_a, parent_b, seed, limits)`
  - `crossover_typed_subtree(parent_a, parent_b, seed, limits)`
  - `validate_genome(genome, limits)`
  - `compile_for_eval(genome)` (AST -> bytecode)

`crossover(..., method=...)` supports:

- `"top_level_splice"`: one-point splice on top-level statements
- `"typed_subtree"`: typed subtree swap (`Expr[NUM]`, `Expr[BOOL]`, etc.)
- `"hybrid"`: mostly typed subtree with occasional top-level splice

Default limits are designed to control bloat and keep programs VM-safe:

- `max_expr_depth=5`
- `max_stmts_per_block=6`
- `max_total_nodes=80`
- `max_for_k=16`
- `max_call_args=3`

Minimal usage:

```python
from src.g3p_vm_gpu.evo_encoding import Limits, make_random_genome, mutate, crossover, compile_for_eval
from src.g3p_vm_gpu.vm import run_bytecode

limits = Limits()
g0 = make_random_genome(seed=0, limits=limits)
g1 = mutate(g0, seed=1, limits=limits)
g2 = crossover(g0, g1, seed=2, limits=limits, method="typed_subtree")

bc = compile_for_eval(g2)
out = run_bytecode(bc, {}, fuel=20_000)
```

Run encoding tests:

```bash
PYTHONPATH=python python3 -m unittest python.tests.test_evo_encoding -v
```

Run full evolution loop tests (AST -> bytecode eval -> selection -> crossover/mutation):

```bash
PYTHONPATH=python python3 -m unittest python.tests.test_evolve -v
```

Minimal end-to-end evolve loop:

```python
from src.g3p_vm_gpu.evolve import EvolutionConfig, FitnessCase, SelectionMethod, evolve_population

cases = [
    FitnessCase(inputs={"x": 1, "y": 2}, expected=3),
    FitnessCase(inputs={"x": -1, "y": 4}, expected=3),
]

cfg = EvolutionConfig(
    population_size=32,
    generations=20,
    selection_method=SelectionMethod.TOURNAMENT,
    crossover_method="hybrid",
    seed=0,
)

result = evolve_population(cases, cfg)
print(result.best.fitness, result.best.genome.meta.hash_key)
```

Run evolution from JSON cases via CLI:

```bash
PYTHONPATH=python python3 tools/run_evolution.py \
  --cases data/fixtures/evolution_cases.json \
  --population-size 64 \
  --generations 40 \
  --selection tournament \
  --crossover-method hybrid \
  --show-program none \
  --out-json data/fixtures/evolution_run_summary.json
```

Run directly from bouncing-balls PSB2 fixture (`shared_cases/shared_answer`):

```bash
PYTHONPATH=python python3 tools/run_evolution.py \
  --cases data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cases-format psb2_fixture \
  --input-indices 1 \
  --input-names x \
  --population-size 64 \
  --generations 40 \
  --selection tournament \
  --crossover-method hybrid
```

`cases` JSON minimal schema:

```json
{
  "cases": [
    {"inputs": {"x": 1, "y": 2}, "expected": 3},
    {"inputs": {"x": {"type": "int", "value": -1}}, "expected": {"type": "int", "value": -1}}
  ]
}
```

## C++ build and tests

Configure and build:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

Run C++ tests:

```bash
ctest --test-dir cpp/build --output-on-failure
```

## GPU API status

GPU execution is multi/fitness only:

- `run_bytecode_gpu_multi_batch(programs, shared_cases, fuel, blocksize)`
- `run_bytecode_gpu_multi_fitness_shared_cases(programs, shared_cases, shared_answer, fuel, blocksize)`

GPU runtime auto-selects the least-used visible CUDA device by memory usage.
If you want to pin manually, set `CUDA_VISIBLE_DEVICES`.

## CPU/GPU multi-program benchmark

The repo includes in-memory benchmark for non-fitness multi-program workloads:

- GPU: `g3pvm_vm_gpu_multi_bench`
- CPU: `g3pvm_vm_cpu_multi_bench`

Argument order:

`<program_count> <cases_per_program> <pass_programs> <fail_programs> <timeout_programs> <fuel> [blocksize(gpu only)]`

Example for your target workload (`4096 programs`, each `1024 cases`, program buckets `2048/1024/1024`):

```bash
# GPU (auto-selects least-used visible device; pin with CUDA_VISIBLE_DEVICES to override)
cpp/build/g3pvm_vm_gpu_multi_bench 4096 1024 2048 1024 1024 64 256

# CPU
cpp/build/g3pvm_vm_cpu_multi_bench 4096 1024 2048 1024 1024 64
```

## Data directory behavior

`data/` is git-ignored on purpose. Generated fixtures are local artifacts.

Generation scripts automatically create parent directories for output paths:

- `tools/gen_psb2_fitness_multi_bench_inputs.py`

So commands like `--out data/fixtures/...` work even if directories do not exist yet.

Generate PSB2 (bouncing-balls) fitness multi-bench input JSON:

```bash
PYTHONPATH=python python3 tools/gen_psb2_fitness_multi_bench_inputs.py \
  --psb2-root data/psb2_datasets \
  --out data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --require-psb2-fetch
```

Validate CPU/GPU fitness against expected scores from that generated JSON:

```bash
PYTHONPATH=python python3 tools/check_fitness_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cli cpp/build_release/g3pvm_vm_cpu_cli
```

Benchmark CPU/GPU fitness speed from the same generated JSON:

```bash
PYTHONPATH=python python3 tools/bench_fitness_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cli cpp/build_release/g3pvm_vm_cpu_cli \
  --runs 5 \
  --blocksize 256
```

## Fitness CLI (CPU/GPU aligned)

Use `g3pvm_vm_cpu_cli` with a `bytecode_program_inputs` payload.
It returns one fitness value per program (`error=-10`, `wrong=0`, `correct=1` per case):

Canonical request schema (use this one format for all tooling/scripts):
- top-level key is always `bytecode_program_inputs`
- `programs` + `shared_cases` are always required
- `shared_answer` is included when you want fitness output

```json
{
  "bytecode_program_inputs": {
    "format_version": "bytecode-json-v0.1",
    "fuel": 64,
    "programs": [ { "n_locals": 1, "consts": [], "code": [] } ],
    "shared_cases": [ [] ],
    "shared_answer": [ { "type": "int", "value": 0 } ]
  }
}
```

Run:

```bash
cpp/build/g3pvm_vm_cpu_cli < your_bytecode_program_inputs.json
```

Optional runtime selection is via CLI args (not JSON fields):

```bash
cpp/build/g3pvm_vm_cpu_cli --engine gpu --blocksize 256 < your_bytecode_program_inputs.json
```

Output format:

```text
OK fitness_count <N>
FIT <program_idx> <fitness>
```
