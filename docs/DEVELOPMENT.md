# Development Runbook

## 1. Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compiler
- CUDA toolkit + NVIDIA driver (GPU paths)

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

## 5. Evolution Benchmarking

### CPU/GPU comparison

```bash
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 1024 --generations 40
bash tools/run_cpu_gpu_speedup_experiment.sh --popsize 4096 --generations 40
```

Outputs:

- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.md`
- `logs/cpu_gpu_compare_pop*/cpu_gpu_compare.report.json`
- per-run CPU/GPU summary JSON logs

## 6. Prefix-Only Contribution Rules

When changing AST/evolution logic:

1. Keep APIs prefix-native (`AstProgram` as source of truth).
2. Do not add tree compatibility layers.
3. Preserve `ast-prefix-v1` validation invariants.
4. Run interpreter-vm equivalence tests and C++ parity tests.

## 7. Pull Request Checklist

- [ ] Python tests pass
- [ ] C++ tests pass
- [ ] For performance-sensitive changes: attach CPU/GPU compare report
- [ ] Updated docs if behavior or workflows changed
- [ ] No new tree legacy symbols/APIs introduced
