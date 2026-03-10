# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `python/src/g3p_vm_gpu/`.
  Current structure:
  - `core/`: AST, error types, shared value semantics
  - `runtime/`: builtins, compiler, interpreter, Python VM
  - `evolution/`: genome, random generation, mutation, crossover, evolution loop
- Tests live in `python/tests/`.
- Normative behavior is documented in `spec/`:
  - `grammar_v1_0.md`
  - `bytecode_isa_v1_0.md`
  - `bytecode_format_v1_0.md`
  - `builtins_base_v1_0.md`
  - `builtins_runtime_v1_0.md`
  - `fitness_v1_0.md`
  Treat these files as the behavioral source of truth.

## Build, Test, and Development Commands
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
- Do not use `ncu` in this project environment.
- If GPU execution fails due to device contention or occupancy, explicitly retry with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.

## GPU Device Runbook
- Always run GPU commands through:
  ```bash
  scripts/run_gpu_command.sh -- <gpu_command> [args...]
  ```
- The wrapper automatically tries `CUDA_VISIBLE_DEVICES=0` then `CUDA_VISIBLE_DEVICES=1`.
- Recommended examples:
  ```bash
  scripts/run_gpu_command.sh -- ctest --test-dir cpp/build -R g3pvm_test_vm_gpu --output-on-failure -V
  scripts/run_gpu_command.sh -- python3 tools/run_cpp_evolution.py --cases data/fixtures/bouncing_balls_1024.json --cpp-cli cpp/build/g3pvm_evolve_cli --engine gpu --blocksize 256 --population-size 64 --generations 2
  scripts/run_gpu_command.sh -- bash scripts/run_cpu_gpu_speedup_experiment.sh --cases data/fixtures/bouncing_balls_1024.json --popsize 1024
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
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.
### Available skills
- g3p-vm-gpu-repo: Repository-specific guidance for working in the g3p-vm-gpu codebase. Use when Codex needs to understand this repo's architecture, GP/VM theory, CPU/GPU parity constraints, performance goals, benchmark workflow, change impact, or where to edit for AST, bytecode, VM, payload, fitness, PSB2, and evolution tasks. (file: /home/hschi1106/.codex/skills/g3p-vm-gpu-repo/SKILL.md)
- skill-creator: Guide for creating effective skills. Use when users want to create or update a skill. (file: /home/hschi1106/.codex/skills/.system/skill-creator/SKILL.md)
- skill-installer: Install Codex skills into `$CODEX_HOME/skills` from a curated list or a GitHub repo path. (file: /home/hschi1106/.codex/skills/.system/skill-installer/SKILL.md)
- slides: Build, edit, render, import, and export presentation decks. (file: /home/hschi1106/.codex/skills/.system/slides/SKILL.md)
- spreadsheets: Build, edit, recalculate, import, and export spreadsheet workbooks. (file: /home/hschi1106/.codex/skills/.system/spreadsheets/SKILL.md)
### How to use skills
- If the user names a skill or the task clearly matches a skill's description, you must use that skill for that turn.
- Read only the skill files needed for the task.
- Keep context small and avoid bulk-loading unrelated references.
