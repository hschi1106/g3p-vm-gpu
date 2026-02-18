# g3p-vm-gpu

Prefix-AST genetic programming VM with Python reference implementation and C++ CPU/GPU evolution backends.

## Project Status

- AST pipeline is prefix-only across Python and C++.
- Primary project KPI is fair CPU/GPU speedup measurement.
- Input fixture schema is unified as `fitness-cases-v1`.

## Repository Layout

- `python/src/g3p_vm_gpu/`: Python AST/compiler/interpreter/VM/evolution reference.
- `cpp/`: CPU VM, GPU VM, evolution core, CLIs, and tests.
- `tools/`: speedup-focused runners.
- `scripts/`: operational wrappers (`run_gpu_command.sh`).
- `data/fixtures/`: unified input fixtures (`fitness-cases-v1`).

## Quick Start

### Build C++

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

### Run tests

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
ctest --test-dir cpp/build --output-on-failure
```

## Evolution / Benchmark Entrypoints

### C++ evolution runner (CPU/GPU)

```bash
python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/speedup_cases_bouncing_balls_1024.json \
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

Output reports are written to `logs/cpu_gpu_compare_pop*_*/cpu_gpu_compare.report.md` and `.json`.

## GPU Runbook

Always run GPU commands via:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper retries `CUDA_VISIBLE_DEVICES=0` then `1` on device-unavailable failures.
