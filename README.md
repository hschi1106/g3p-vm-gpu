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

If GPU0 is busy, pin to GPU1:

```bash
CUDA_VISIBLE_DEVICES=1 cpp/build/g3pvm_vm_gpu_batch_cli data/fixtures/gpu_batch_cases.json
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
