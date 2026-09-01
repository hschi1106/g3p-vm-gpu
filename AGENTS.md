# Repository Guidelines

## Project Structure & Module Organization
- The implementation lives under `cpp/`.
  Current structure:
  - `include/g3pvm/`: public C++ headers
  - `src/runtime/`: CPU runtime, GPU runtime, payload support
  - `src/evolution/`: compiler, genome generation, operators, evolution loop
  - `src/cli/`: native CLIs such as `g3pvm_evolve_cli`
  - `tests/`: runtime, GPU smoke, parity, and evolution tests
- Native tests live in `cpp/tests/`.
- Operational tool tests live in `tools/tests/`; runtime-independent repository
  checks live in `tests/repository/`.
- Normative behavior is documented in `spec/`:
  - `grammar.md`
  - `bytecode_isa.md`
  - `bytecode_format.md`
  - `builtins_base.md`
  - `builtins_runtime.md`
  - `fitness.md`
  - `fitness_cases.md`
  - `grammar_config.md`
  Treat these files as the release 1.0.0 behavioral source of truth. Historical spec
  files are not kept in-tree after the breaking refactor. Release details are
  recorded only in `VERSION.md`.
- Maintained documentation is indexed by `docs/README.md` and grouped by
  ownership:
  - `docs/design/`: architecture, dataflow, payload, and GPU reproduction
  - `docs/guides/`: development, benchmarking, PSB, grammar-config, and
    experiment workflows
  - `docs/reference/`: checked timing, tooling, and repository-layout records
  - `docs/refactor/`: historical refactor evidence

## Build, Test, and Development Commands
- Build native binaries:
  ```bash
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
  cmake --build cpp/build -j
  ```
- Run operational tool and repository checks:
  ```bash
  python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
  python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
  ```
- Run all native tests:
  ```bash
  ctest --test-dir cpp/build --output-on-failure
  ```
- Run the main native GPU/parity regression set:
  ```bash
  ctest --test-dir cpp/build -R 'g3pvm_test_vm_gpu_smoke|g3pvm_test_fitness_cpu_gpu_parity|g3pvm_test_evolution_cpu_gpu_parity' --output-on-failure
  ```
- Run commands from the repository root so relative fixtures and configs resolve.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and type hints where practical.
- Follow existing naming patterns: `snake_case` for functions/variables/modules, `PascalCase` for dataclasses/classes, `UPPER_CASE` for constants.
- Keep modules small and single-purpose; place reusable runtime logic in package modules, not tests.
- No formatter/linter config is committed yet; match the style already present in neighboring files.

## Testing Guidelines
- Python tool/repository-check framework: `unittest`.
- Test files use `test_*.py`; test classes use `Test*`; test methods use `test_*`.
- Native tests are built with CMake and run through `ctest`.
- Add or update tests with every behavior change, especially for:
  - error code behavior (`ErrCode` paths),
  - compiler/runtime contract parity,
  - CPU vs GPU fitness parity when touching runtime, payload, or GPU execution,
  - `IntList` / `FloatList` / `StringList` typed-list behavior when touching sequence values, fixture conversion, payloads, or generation,
  - `grammar-config` search-space behavior when touching generation, mutation, reproduction, or seed replay,
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
  cpp/build/g3pvm_evolve_cli --cases data/fixtures/simple_exp_1024.json --engine gpu --repro-backend gpu --repro-overlap on --blocksize 1024 --population-size 64 --generations 2 --out-json logs/simple_exp_1024.run.json
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
