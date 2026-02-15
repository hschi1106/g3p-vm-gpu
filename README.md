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

- `tools/gen_fitness_multi_bench_inputs.py`

So commands like `--out data/fixtures/...` work even if directories do not exist yet.

Generate canonical fitness multi-bench input JSON:

```bash
PYTHONPATH=python python3 tools/gen_fitness_multi_bench_inputs.py \
  --out data/fixtures/fitness_multi_bench_inputs.json
```

Validate CPU/GPU fitness against expected scores from that generated JSON:

```bash
PYTHONPATH=python python3 tools/check_fitness_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs.json \
  --cli cpp/build_release/g3pvm_vm_cpu_cli
```

Benchmark CPU/GPU fitness speed from the same generated JSON:

```bash
PYTHONPATH=python python3 tools/bench_fitness_fixture_cpu_gpu.py \
  --fixture data/fixtures/fitness_multi_bench_inputs.json \
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
