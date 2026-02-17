# g3p-vm-gpu

Prefix-AST genetic programming VM with Python reference implementation and C++ CPU/GPU evolution backends.

## Project Status

- AST pipeline is **prefix-only** across Python and C++.
- Tree AST legacy APIs are intentionally removed.
- Evolution, compile, and runtime paths all consume prefix `AstProgram`.

## Repository Layout

- `python/src/g3p_vm_gpu/`
  - `ast.py`: prefix AST model, validators, builders
  - `compiler.py`: prefix AST -> bytecode compiler
  - `interp.py`: prefix AST interpreter
  - `vm.py`: bytecode VM
  - `evo_encoding.py`: genome generation/mutation/crossover/validation
  - `evolve.py`: evolution loop and selection
- `cpp/`
  - CPU VM, GPU VM, evolution core, CLIs, tests
- `tools/`
  - end-to-end runners and benchmarking scripts
- `scripts/`
  - operational wrappers (notably GPU retry wrapper)
- `spec/`
  - language and bytecode contracts
- `docs/`
  - architecture and development runbook

## Quick Start

### Build C++

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

### Run all tests

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
ctest --test-dir cpp/build --output-on-failure
```

### Run demo

```bash
PYTHONPATH=python/src python3 -m g3p_vm_gpu.demo
```

## Evolution / Benchmark Entrypoints

### Python evolution CLI

```bash
PYTHONPATH=python python3 tools/run_evolution.py \
  --cases data/fixtures/evolution_cases.json \
  --population-size 64 \
  --generations 40 \
  --selection tournament \
  --crossover-method hybrid
```

### C++ evolution runner (CPU/GPU)

```bash
python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/fitness_multi_bench_inputs_psb2.json \
  --cases-format psb2_fixture \
  --input-indices 1 \
  --input-names x \
  --cpp-cli cpp/build/g3pvm_evolve_cli \
  --engine gpu \
  --blocksize 256 \
  --cpp-timing all
```

### CPU vs GPU speedup experiment

```bash
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 4096 --generations 40
```

Output reports are written to `logs/cpu_gpu_compare_pop*_*/cpu_gpu_compare.report.md`.

## GPU Runbook

Always run GPU commands via:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper retries `CUDA_VISIBLE_DEVICES=0` then `1` on device-unavailable failures.

## Docs

- Architecture: `docs/ARCHITECTURE.md`
- Development/Test/Benchmark runbook: `docs/DEVELOPMENT.md`
- Language and VM contracts: `spec/subset_v0_1.md`, `spec/bytecode_isa.md`, `spec/builtins.md`
