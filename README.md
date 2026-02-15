# g3p-vm-gpu

Bytecode VM project with Python reference/runtime and C++ CPU/GPU backends.

## What is in this repo

- `python/src/g3p_vm_gpu/`
  - language AST, compiler, interpreter, Python VM, fuzz generator
- `cpp/`
  - C++ CPU VM
  - CUDA GPU VM (`run_bytecode_gpu_batch`)
  - CLIs and C++ tests
- `spec/`
  - language, bytecode ISA, builtins, and test contracts
- `tools/`
  - fixture generation and cross-backend comparison scripts

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

GPU execution is batch-only now:

- `run_bytecode_gpu_batch(program, cases, fuel, blocksize)`
- `run_bytecode_gpu_multi_batch(programs, shared_cases, fuel, blocksize)`

Single-case GPU API has been removed from the public interface. Use one-case batch input if needed.

## Batch fixture generation

Generate deterministic batch fixture (default: 2048 pass / 1024 fail / 1024 timeout):

```bash
PYTHONPATH=python python3 tools/gen_gpu_batch_cases.py --out data/fixtures/gpu_batch_cases.json
```

Run fixture with C++ GPU batch CLI:

```bash
cpp/build/g3pvm_vm_gpu_batch_cli data/fixtures/gpu_batch_cases.json
```

GPU runtime now auto-selects the least-used visible CUDA device by memory usage.
If you still want to pin manually:

```bash
CUDA_VISIBLE_DEVICES=1 cpp/build/g3pvm_vm_gpu_batch_cli data/fixtures/gpu_batch_cases.json
```

## CPU/GPU multi-program benchmark

The repo includes in-memory benchmarks for large multi-program workloads:

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

- `tools/gen_bytecode_fixture_set.py`
- `tools/gen_gpu_batch_cases.py`

So commands like `--out data/fixtures/...` work even if directories do not exist yet.

## Useful comparison tools

- Python VM vs C++ CPU fixtures:

```bash
PYTHONPATH=python python3 tools/compare_vm_py_cpp_fixtures.py --fixture data/fixtures/bytecode_cases.json
```

- Python VM vs C++ GPU fixtures:

```bash
PYTHONPATH=python python3 tools/compare_vm_py_gpu_fixtures.py --fixture data/fixtures/bytecode_cases.json
```

## Fitness CLI (CPU/GPU aligned)

Use `g3pvm_vm_cpu_cli` with a `fitness_request` payload.
It returns one fitness value per program (`error=-10`, `wrong=0`, `correct=1` per case):

```json
{
  "fitness_request": {
    "format_version": "bytecode-json-v0.1",
    "engine": "gpu",
    "fuel": 64,
    "blocksize": 256,
    "programs": [ { "n_locals": 1, "consts": [], "code": [] } ],
    "shared_cases": [ [] ],
    "shared_answer": [ { "type": "int", "value": 0 } ]
  }
}
```

Run:

```bash
cpp/build/g3pvm_vm_cpu_cli < your_fitness_request.json
```

Output format:

```text
OK fitness_count <N>
FIT <program_idx> <fitness>
```
