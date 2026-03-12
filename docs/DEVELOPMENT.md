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
- `--mutation-rate F`: probability that a selected child is mutated
- `--mutation-subtree-prob F`: internal mutation operator mix; probability of typed-subtree mutation instead of constant perturbation
- `--crossover-rate F`: probability that a selected child is rebuilt via crossover
- `--selection-pressure N`: tournament size; larger values increase selection pressure
- `--seed N`: RNG seed for deterministic replay

Fitness args:
- `--penalty F`: penalty used for runtime errors and type-mismatch outputs that cannot be compared directly; must be `>= 0`
- `--fuel N`: per-program execution budget

Genome-shape args:
- `--max-expr-depth N`: maximum generated expression depth
- `--max-stmts-per-block N`: maximum statements per block
- `--max-total-nodes N`: maximum total AST nodes in one genome
- `--max-for-k N`: maximum constant loop bound used by generated `for range(K)`
- `--max-call-args N`: maximum allowed builtin call arity during generation/compilation

### `cpp/build/g3pvm_population_bench_cli`

Fixed-population benchmark args:
- `--cases PATH`: input `fitness-cases-v1` file
- `--population-json PATH`: optional input `population-seeds-v1` JSON; if omitted, the CLI generates a fixed population first
- `--out-population-json PATH`: optional output path when generating a new fixed population
- `--engine {cpu|gpu}`: evaluation backend
- `--blocksize N`: GPU block size when `--engine gpu`
- `--fuel N`: per-program execution budget
- `--population-size N`: number of accepted genomes when generating a fixed population
- `--seed-start N`: first RNG seed considered during generation
- `--probe-cases N`: number of leading cases used to reject programs that error too often
- `--min-success-rate F`: required non-error ratio across probe cases
- `--max-attempts N`: hard cap on candidate seeds inspected before failing
- `--max-expr-depth N`
- `--max-stmts-per-block N`
- `--max-total-nodes N`
- `--max-for-k N`
- `--max-call-args N`
- `--mutation-rate F`
- `--mutation-subtree-prob F`
- `--crossover-rate F`
- `--penalty F`
- `--selection-pressure N`

Metric semantics:
- `compile_ms`: compile-cache lookup plus any required genome-to-bytecode compilation
- `eval_ms`: fitness execution only; compile time is excluded
- `repro_ms`: one-generation reproduction work after the current population has been scored
- `selection_ms`: tournament-selection pool construction for the full offspring pool
- `crossover_ms`: crossover pass over the selected parent pool
- `mutation_ms`: mutation pass over the post-crossover offspring pool
- `total_ms`: the full one-generation benchmark wall time
- GPU runs also report `pack_upload_ms`, `kernel_ms`, and `copyback_ms`

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
- `--mutation-rate F`
- `--mutation-subtree-prob F`
- `--crossover-rate F`
- `--selection-pressure N`
- `--penalty F`
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
- `--penalty F`
- `--cpp-cli PATH`
- `--run-cpp-tool PATH`
- `--log-dir PATH`

### `scripts/run_cpu_gpu_speedup_experiment.sh`

Benchmark args:
- `--cases PATH`: benchmark fixture, usually `data/fixtures/bouncing_balls_1024.json`
- `--popsize N`: population size used in both CPU and GPU runs
- `--generations N`: accepted for compatibility but ignored by the fixed-population benchmark flow
- `--blocksize N`: GPU block size
- `--generator-cli PATH`: fixed-population generator executable
- `--bench-cli PATH`: fixed-population benchmark executable
- `--seed-start N`: first candidate seed used during population generation
- `--probe-cases N`: fixture cases probed while filtering candidate programs
- `--min-success-rate F`: minimum accepted non-error probe ratio
- `--fuel N`: shared execution budget
- `--max-expr-depth N`
- `--max-stmts-per-block N`
- `--max-total-nodes N`
- `--max-for-k N`
- `--max-call-args N`
- `--mutation-rate F`
- `--mutation-subtree-prob F`
- `--crossover-rate F`
- `--penalty F`
- `--selection-pressure N`: tournament size for both runs
- `--cpp-cli PATH`: alias for `--bench-cli` kept for compatibility
- `--outdir PATH`: output directory for compare reports and raw run logs

Report shape:
- `cpu.compile_ms`, `gpu.compile_ms`
- `cpu.eval_ms`, `gpu.eval_ms`
- `cpu.repro_ms`, `gpu.repro_ms`
- `cpu.one_gen_e2e_total_ms`, `gpu.one_gen_e2e_total_ms`
- `speedup.compile_cpu_over_gpu`
- `speedup.eval_cpu_over_gpu`
- `speedup.repro_cpu_over_gpu`
- `speedup.one_gen_e2e_cpu_over_gpu`

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
bash scripts/run_cpu_gpu_speedup_experiment.sh \
  --cases data/fixtures/bouncing_balls_1024.json \
  --popsize 1024
```

This benchmark uses one fixed population for both CPU and GPU runs.
The canonical interpretation is:
- `compile`: genome-to-bytecode preparation and compile-cache lookup
- `eval`: fitness execution only
- `repro`: one-generation selection, crossover, and mutation work
- `total`: the full one-generation benchmark cost

`eval` is intentionally narrower than the old mixed metric. It excludes `compile`.

Pre-generated populations for the canonical fixtures are checked in under `data/fixtures/programs/`.

### Benchmark one fixed population directly

```bash
cpp/build/g3pvm_population_bench_cli \
  --cases data/fixtures/bouncing_balls_1024.json \
  --population-json data/fixtures/programs/bouncing_balls_1024_pop1024.json \
  --engine cpu
```

### Generate and benchmark one fixed population in one step

```bash
cpp/build/g3pvm_population_bench_cli \
  --cases data/fixtures/bouncing_balls_1024.json \
  --out-population-json logs/fixed_population.json \
  --population-size 1024 \
  --probe-cases 32 \
  --min-success-rate 0.10 \
  --engine cpu
```

Primary metrics to track:
- `cpu.compile_ms`
- `gpu.compile_ms`
- `cpu.eval_ms`
- `gpu.eval_ms`
- `cpu.repro_ms`
- `gpu.repro_ms`
- `cpu.one_gen_e2e_total_ms`
- `gpu.one_gen_e2e_total_ms`
- `speedup.compile_cpu_over_gpu`
- `speedup.eval_cpu_over_gpu`
- `speedup.repro_cpu_over_gpu`
- `speedup.one_gen_e2e_cpu_over_gpu`

### Canonical evolution-progress run

```bash
scripts/run_gpu_command.sh -- python3 tools/run_cpp_evolution.py \
  --cases data/fixtures/simple_exp_1024.json \
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
- fitness formulas, type/runtime failure scoring, or scoring args:
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
