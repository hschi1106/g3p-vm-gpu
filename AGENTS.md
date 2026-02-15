# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `python/src/g3p_vm_gpu/`.
  Example modules: `ast.py` (AST nodes), `interp.py` (reference interpreter), `compiler.py` (AST -> bytecode), `vm.py` (bytecode VM), `fuzz.py` (random program generator), `demo.py` (manual smoke demo).
- Tests live in `python/tests/` and currently include interpreter and VM equivalence coverage.
- Language and VM contracts are documented in `spec/` (`subset_v0_1.md`, `bytecode_isa.md`, `builtins.md`). Treat these files as the behavioral source of truth.

## Build, Test, and Development Commands
- Run all tests:
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
- If imports fail, verify you are running from repo root and using the matching `PYTHONPATH` shown above.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and type hints where practical.
- Follow existing naming patterns: `snake_case` for functions/variables/modules, `PascalCase` for dataclasses/classes, `UPPER_CASE` for constants.
- Keep modules small and single-purpose; place reusable runtime logic in package modules, not tests.
- No formatter/linter config is committed yet; match the style already present in neighboring files.

## Testing Guidelines
- Framework: `unittest`.
- Test files use `test_*.py`; test classes use `Test*`; test methods use `test_*`.
- Add or update tests with every behavior change, especially for:
  - error code behavior (`ErrCode` paths),
  - interpreter vs VM parity,
  - edge cases around fuel/timeouts and numeric/type operations.

## Profiling Guidelines
- GPU profiling must use `nsys` only.
- Do not use `ncu` in this project environment (GPU performance counters are not available).
- If GPU execution fails due to device contention/occupancy, explicitly retry with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.

## Commit & Pull Request Guidelines
- Follow observed commit style: `<type>: <summary>` (for example, `feat: add bytecode jump validation`, `fix: handle unary negation type errors`).
- Preferred types in this repo: `feat`, `fix`, `init` , `docs`, `test`, `refactor`, `chore`.
- PRs should include:
  - a clear behavior summary,
  - linked issue/task (if available),
  - test evidence (command + pass result),
  - spec updates in `spec/` when semantics change.
