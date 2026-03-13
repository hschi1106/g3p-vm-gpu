# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `python/src/g3p_vm_gpu/`.
  Current structure:
  - `core/`: AST, error types, shared value semantics
  - `runtime/`: builtins, compiler, interpreter, Python VM
  - `evolution/`: genome, random generation, mutation, crossover, evolution loop
- Native implementation lives under `cpp/`.
  Current structure:
  - `include/g3pvm/`: public C++ headers
  - `src/runtime/`: CPU runtime, GPU runtime, payload support
  - `src/evolution/`: compiler, genome generation, operators, evolution loop
  - `src/cli/`: native CLIs such as `g3pvm_evolve_cli` and `g3pvm_population_bench_cli`
  - `tests/`: runtime, GPU smoke, parity, and evolution tests
- Tests live in `python/tests/`.
- Native tests live in `cpp/tests/`.
- Normative behavior is documented in `spec/`:
  - `grammar_v1_0.md`
  - `bytecode_isa_v1_0.md`
  - `bytecode_format_v1_0.md`
  - `builtins_base_v1_0.md`
  - `builtins_runtime_v1_0.md`
  - `fitness_v1_0.md`
  Treat these files as the behavioral source of truth.
- Operational and structural docs live in:
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/CPP_RUNTIME_PAYLOAD.md`
  - `structure.md`

## Build, Test, and Development Commands
- Build native binaries:
  ```bash
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
  cmake --build cpp/build -j
  ```
- Run all Python tests:
  ```bash
  PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
  ```
- Run one test module:
  ```bash
  PYTHONPATH=python python3 -m unittest python.tests.test_vm_equiv -v
  ```
- Run the demo program:
  ```bash
  PYTHONPATH=python/src python3 -m g3p_vm_gpu.demo
  ```
- Run all native tests:
  ```bash
  ctest --test-dir cpp/build --output-on-failure
  ```
- Run the main native GPU/parity regression set:
  ```bash
  ctest --test-dir cpp/build -R 'g3pvm_test_vm_gpu_smoke|g3pvm_test_fitness_cpu_gpu_parity|g3pvm_test_evolution_cpu_gpu_parity' --output-on-failure
  ```
- Run the fixed-population CPU/GPU benchmark sweep:
  ```bash
  python3 scripts/speedup_experiment.py
  ```
- If imports fail, verify you are running from repo root and using the matching `PYTHONPATH` shown above.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and type hints where practical.
- Follow existing naming patterns: `snake_case` for functions/variables/modules, `PascalCase` for dataclasses/classes, `UPPER_CASE` for constants.
- Keep modules small and single-purpose; place reusable runtime logic in package modules, not tests.
- No formatter/linter config is committed yet; match the style already present in neighboring files.

## Testing Guidelines
- Framework: `unittest`.
- Test files use `test_*.py`; test classes use `Test*`; test methods use `test_*`.
- Native tests are built with CMake and run through `ctest`.
- Add or update tests with every behavior change, especially for:
  - error code behavior (`ErrCode` paths),
  - interpreter vs VM parity,
  - CPU vs GPU fitness parity when touching runtime, payload, or GPU execution,
  - edge cases around fuel/timeouts and numeric/type operations.

## Profiling Guidelines
- GPU profiling must use `nsys` only.
- Do not use `ncu` in this project environment.
- GPU-capable C++ paths select the least-used visible CUDA device internally.
- To force a specific visible-device index for a run, use `G3PVM_CUDA_DEVICE=0` or `G3PVM_CUDA_DEVICE=1`.

## GPU Device Runbook
- Run GPU-capable binaries directly.
- The C++ GPU runtime selects the least-used visible CUDA device automatically.
- Recommended examples:
  ```bash
  ctest --test-dir cpp/build -R g3pvm_test_vm_gpu --output-on-failure -V
  cpp/build/g3pvm_evolve_cli --cases data/fixtures/bouncing_balls_1024.json --engine gpu --blocksize 256 --population-size 64 --generations 2 --out-json logs/bouncing_balls_1024.run.json
  cpp/build/g3pvm_population_bench_cli --cases data/fixtures/bouncing_balls_1024.json --engine gpu --blocksize 256 --population-size 1024 --probe-cases 32 --min-success-rate 0.10 --out-population-json logs/bouncing_balls_1024.population.json
  python3 scripts/speedup_experiment.py --fixtures bouncing_balls_1024 --population-sizes 1024
  ```

## Commit & Pull Request Guidelines
- Follow commit style: `<type>: <summary>`.
- Preferred types in this repo: `feat`, `fix`, `init`, `docs`, `test`, `refactor`, `chore`.
- PRs should include:
  - a clear behavior summary,
  - linked issue/task if available,
  - test evidence,
  - spec updates in `spec/` when semantics change.

## Skills
A skill is a set of local instructions stored in a `SKILL.md` file.

Repo-stable skill:
- `g3p-vm-gpu-repo`: repository-specific guidance for architecture, parity constraints, benchmark workflow, payload behavior, and change impact

Use repo skills this way:
- If the task clearly matches the repo skill, open `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/SKILL.md` and read only the referenced material you need.
- Keep context small; avoid bulk-loading unrelated references.
- System-provided skills may vary by session, so treat the live session skill list as the source of truth for non-repo skills.
