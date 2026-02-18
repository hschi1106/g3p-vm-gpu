# Development Runbook

## 1. Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compiler
- CUDA toolkit + NVIDIA driver (for GPU paths)

## 2. Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

## 3. Tests

### Python

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
```

### C++

```bash
ctest --test-dir cpp/build --output-on-failure
```

### Recommended full check

```bash
bash scripts/run_triplet_checks.sh
```

## 4. GPU Command Policy

Use wrapper for all GPU commands:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

This retries device 0 then 1 when GPU is unavailable/contended.

## 5. CPU/GPU Speedup Benchmark

```bash
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
```

Inputs must use `fitness-cases-v1` schema (example: `data/fixtures/speedup_cases_bouncing_balls_1024.json`).

Outputs:

- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.md`
- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.json`
- per-run CPU/GPU summary JSON logs

## 6. Contribution Rules

1. Keep APIs prefix-native (`AstProgram`).
2. Preserve CPU/GPU fitness parity and timing report compatibility.
3. Run Python and C++ tests after behavior changes.
4. For speedup changes, attach generated comparison report.
