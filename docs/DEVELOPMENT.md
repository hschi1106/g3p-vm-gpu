# Development

## Build And Test

### Build C++

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

### Python tests

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
```

### C++ tests

```bash
ctest --test-dir cpp/build --output-on-failure
```

### Recommended full check

```bash
bash scripts/run_triplet_checks.sh
```

## GPU Run Policy

All GPU commands must go through:

```bash
scripts/run_gpu_command.sh -- <gpu_command> [args...]
```

The wrapper retries `CUDA_VISIBLE_DEVICES=0` and then `1` on device-unavailable failures.

## Public CLI And Script Arguments

This section documents the adjustable arguments that affect public workflows.

### `cpp/build/g3pvm_evolve_cli`

Core execution args:
- `--cases PATH`: input `fitness-cases-v1` file
- `--engine {cpu|gpu}`: evaluation backend
- `--blocksize N`: CUDA block size for GPU evaluation
- `--out-json PATH`: write evolution history JSON
- `--timing {none|summary|per_gen|all}`: timing verbosity from the native CLI
- `--show-program {none|ast|bytecode|both}`: include final-program dumps in output

Evolution args:
- `--population-size N`: individuals per generation
- `--generations N`: number of generations to run
- `--elitism N`: number of elites copied without mutation/crossover
- `--mutation-rate F`: probability that a non-elite child is mutated
- `--mutation-subtree-prob F`: internal mutation operator mix; probability of typed-subtree mutation instead of constant perturbation
- `--crossover-rate F`: probability that a child is built via crossover instead of direct reproduction
- `--selection-pressure N`: tournament size; larger values increase selection pressure
- `--seed N`: RNG seed for deterministic replay

Fitness args:
- `--numeric-type-penalty F`: penalty used when a numeric-expected case produces a non-numeric actual; must be `>= 0`
- `--fuel N`: per-program execution budget

Genome-shape args:
- `--max-expr-depth N`: maximum generated expression depth
- `--max-stmts-per-block N`: maximum statements per block
- `--max-total-nodes N`: maximum total AST nodes in one genome
- `--max-for-k N`: maximum constant loop bound used by generated `for range(K)`
- `--max-call-args N`: maximum allowed builtin call arity during generation/compilation

### `tools/run_cpp_evolution.py`

This is the main human-facing wrapper around `g3pvm_evolve_cli`.

Execution args:
- `--cases PATH`: input `fitness-cases-v1` file
- `--cpp-cli PATH`: native evolve CLI executable
- `--engine {cpu|gpu}`: CPU or GPU evaluation
- `--blocksize N`: GPU block size when `--engine gpu`
- `--timeout-sec N`: subprocess timeout in seconds; `0` disables timeout

Evolution and fitness args:
- `--population-size N`
- `--generations N`
- `--elitism N`
- `--mutation-rate F`
- `--mutation-subtree-prob F`
- `--crossover-rate F`
- `--selection-pressure N`
- `--numeric-type-penalty F`
- `--seed N`
- `--fuel N`
- `--max-expr-depth N`
- `--max-stmts-per-block N`
- `--max-total-nodes N`
- `--max-for-k N`
- `--max-call-args N`

Output and diagnostics args:
- `--show-program {none|ast|bytecode|both}`: request final-program rendering from native CLI
- `--cpp-timing {none|summary|per_gen|all}`: timing detail requested from native CLI
- `--log-dir PATH`: directory for stdout, stderr, timing, summary, and evolution JSON artifacts
- `--run-tag TAG`: suffix used in generated log filenames
- `--print-command`: print the exact subprocess command before execution

### `cpp/build/g3pvm_runtime_cli`

Runtime execution args:
- `--engine {cpu|gpu}`: select CPU raw execution or GPU fitness mode
- `--blocksize N`: CUDA block size when `--engine gpu` and `shared_answer` is present

Input shape notes:
- stdin JSON must follow `spec/bytecode_format_v1_0.md`
- GPU raw execution without `shared_answer` is rejected in the current CLI

### `tools/fetch_psb2_datasets.py`

Dataset-fetch args:
- `--out-dir PATH`: target root for downloaded datasets
- `--problems LIST|all`: comma-separated task names or `all`
- `--splits LIST`: comma-separated subset of `edge,random`
- `--retries N`: retry count per file
- `--timeout-sec N`: per-request timeout
- `--force`: overwrite existing files
- `--dry-run`: print planned work without downloading

### `tools/audit_psb2_tasks.py`

Audit args:
- `--datasets-root PATH`: PSB2 dataset root to inspect
- `--out-json PATH`: summary JSON output path

### `tools/convert_psb2_to_fitness_cases.py`

Dataset-conversion args:
- `--edge-file PATH`: PSB2 edge JSONL file
- `--random-file PATH`: PSB2 random JSONL file
- `--n-train N`: number of training rows to emit
- `--n-test N`: number of test rows to emit
- `--seed N`: sampling seed
- `--out PATH`: output training fixture path
- `--out-test PATH`: optional test fixture output path
- `--summary-json PATH`: optional conversion summary JSON path

### `tools/run_psb2_all_tasks.py`

Dataset selection args:
- `--datasets-root PATH`: root containing `data/psb2_datasets/<task>/...`
- `--tasks LIST|all`: comma-separated task list or `all`
- `--n-train N`: training cases per task after conversion
- `--n-test N`: test cases per task after conversion
- `--seed N`: conversion and evolution seed

Evolution args passed through to `run_cpp_evolution.py`:
- `--engine {cpu|gpu}`
- `--blocksize N`
- `--population-size N`
- `--generations N`
- `--selection-pressure N`
- `--numeric-type-penalty F`
- `--cpp-cli PATH`
- `--run-cpp-tool PATH`
- `--log-dir PATH`

### `scripts/run_cpu_gpu_speedup_experiment.sh`

Benchmark args:
- `--cases PATH`: benchmark fixture, usually `data/fixtures/speedup_cases_bouncing_balls_1024.json`
- `--popsize N`: population size used in both CPU and GPU runs
- `--generations N`: generations used in both runs
- `--blocksize N`: GPU block size
- `--selection-pressure N`: tournament size for both runs
- `--cpp-cli PATH`: evolve CLI executable
- `--outdir PATH`: output directory for compare reports and raw run logs

### `tools/run_v1_release_gate.py`

Gate orchestration args:
- `--out-dir PATH`: root output directory for gate artifacts
- `--cpp-build-dir PATH`: build tree passed to `ctest`
- `--cpp-cli PATH`: native evolve CLI
- `--baseline-speed-report PATH`: reference speedup report used by the speed gate
- `--speedup-threshold-ratio F`: minimum acceptable fraction of baseline speedup

Speed benchmark args:
- `--speed-population-size N`
- `--speed-generations N`
- `--speed-blocksize N`

Evolution gate args:
- `--exp-cases PATH`
- `--exp-population-size N`
- `--exp-generations N`
- `--exp-engine {cpu|gpu}`

PSB2 gate args:
- `--psb2-datasets-root PATH`
- `--psb2-n-train N`
- `--psb2-n-test N`
- `--psb2-population-size N`
- `--psb2-generations N`

Optional skips:
- `--skip-python-tests`
- `--skip-cpp-tests`

## Benchmark Workflow

### Canonical speed benchmark

```bash
scripts/run_gpu_command.sh -- bash scripts/run_cpu_gpu_speedup_experiment.sh \
  --cases data/fixtures/speedup_cases_bouncing_balls_1024.json \
  --popsize 1024 \
  --generations 40
```

Primary metrics to track:
- `cpu.inner_total_ms`
- `gpu.inner_total_ms`
- `gpu.gpu_kernel_total_ms`
- `speedup.inner_total_cpu_over_gpu`
- `speedup.eval_only_cpu_over_gpu`

### Canonical evolution-progress run

```bash
scripts/run_gpu_command.sh -- python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/simple_evo_exp_1024.json \
  --cpp-cli cpp/build/g3pvm_evolve_cli \
  --engine gpu \
  --blocksize 256 \
  --population-size 1024 \
  --generations 20
```

## PSB2 Workflow

### Convert one task

```bash
python3 tools/convert_psb2_to_fitness_cases.py \
  --edge-file data/psb2_datasets/bouncing-balls/bouncing-balls-edge.json \
  --random-file data/psb2_datasets/bouncing-balls/bouncing-balls-random.json \
  --n-train 1024 \
  --n-test 1024 \
  --seed 0 \
  --out logs/psb2/converted/bouncing-balls.train.json \
  --out-test logs/psb2/converted/bouncing-balls.test.json \
  --summary-json logs/psb2/converted/bouncing-balls.summary.json
```

### Run all tasks

```bash
python3 tools/run_psb2_all_tasks.py \
  --datasets-root data/psb2_datasets \
  --tasks all \
  --engine gpu \
  --population-size 1024 \
  --generations 20
```

## Document Sync Rules

Update the following documents whenever the matching code changes.

- grammar, typing, AST, control flow, bytecode lowering:
  - `spec/grammar_v1_0.md`
  - `spec/bytecode_isa_v1_0.md`
  - `docs/ARCHITECTURE.md`
- builtin or payload behavior:
  - `spec/builtins_base_v1_0.md` or `spec/builtins_runtime_v1_0.md`
  - `docs/ARCHITECTURE.md`
- fitness formulas or scoring args:
  - `spec/fitness_v1_0.md`
  - this file
  - `README.md` if the main workflow or defaults changed
- public CLI or script args:
  - this file
  - `README.md` if the quick-start command or default workflow changed
- repo layout or module ownership:
  - `docs/ARCHITECTURE.md`
  - `structure.md`
  - repo skill references
