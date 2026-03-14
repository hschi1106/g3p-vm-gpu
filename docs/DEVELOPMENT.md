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
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
cmake --build cpp/build -j4
ctest --test-dir cpp/build --output-on-failure
python3 scripts/speedup_experiment.py --fixtures bouncing_balls_1024 --population-sizes 64 --probe-cases 8 --min-success-rate 0.0
```

## GPU Run Policy

GPU-capable C++ paths select the least-used visible CUDA device internally.
To force a specific visible-device index for a run, set:

```bash
G3PVM_CUDA_DEVICE=0
```

## Public CLI And Script Arguments

This section documents the adjustable arguments that affect public workflows.

### `cpp/build/g3pvm_evolve_cli`

Core execution args:
- `--cases PATH`: input `fitness-cases-v1` file
- `--engine {cpu|gpu}`: evaluation backend
- `--repro-backend {cpu|gpu}`: reproduction backend; `gpu` is the formal GPU reproduction path and does not promise CPU child identity
- `--repro-overlap {on|off}`: when `--engine gpu --repro-backend gpu`, overlap reproduction input prep with GPU evaluation
- `--blocksize N`: CUDA block size for GPU evaluation; current native CLI default is `1024`
- `--out-json PATH`: write evolution history JSON
- `--timing {none|summary|per_gen|all}`: timing verbosity from the native CLI
- `--show-program {none|ast|bytecode|both}`: include final-program dumps in output

Evolution args:
- `--population-size N`: individuals per generation
- `--generations N`: number of generations to run
- `--mutation-rate F`: probability that a selected child is mutated
- `--mutation-subtree-prob F`: internal mutation operator mix; probability of typed-subtree mutation instead of constant perturbation
- `--crossover-rate F`: probability that a selected child is rebuilt via crossover
- `--selection-pressure N`: tournament size for each round-based without-replacement pass; larger values increase selection pressure
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
- `--out-population-json PATH`: optional output path for the generated `population-seeds-v1` JSON
- `--engine {cpu|gpu}`: evaluation backend
- `--repro-backend {cpu|gpu}`: reproduction backend for the one-generation reproduction phase
- `--repro-overlap {on|off}`: overlap `gpu` prepare/pack work with GPU eval; ignored by the CPU backend
- `--blocksize N`: GPU block size when `--engine gpu`; current native CLI default is `1024`
- `--fuel N`: per-program execution budget
- `--population-size N`: number of accepted genomes in the generated fixed population
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
- `selection_ms`: round-based without-replacement tournament selection to fill the offspring pool
- `crossover_ms`: crossover pass over the selected parent pool
- `mutation_ms`: mutation pass over the post-crossover offspring pool
- `repro_prepare_inputs_ms`: host-side extraction of genomes and fitness into the internal `gpu` input vectors plus config synthesis
- `repro_setup_ms`: CUDA device selection, context warmup, and reproduction arena allocation for `gpu`
- `repro_preprocess_ms`: extracted host-side subtree/candidate/donor preprocessing for `gpu`
- `repro_pack_ms`: host flattening into the internal packed upload schema for `gpu`
- `repro_upload_ms`: H2D upload time for packed reproduction buffers plus the per-generation fitness upload
- `repro_kernel_ms`: total GPU reproduction kernel time
- `repro_copyback_ms`: host buffer preparation plus D2H child copyback for `gpu`
- `repro_decode_ms`: host-side reconstruction of `ProgramGenome` children from copied-back packed data
- `repro_teardown_ms`: GPU arena teardown for `gpu`
- `repro_selection_kernel_ms`: GPU selection kernel subset of `repro_kernel_ms`
- `repro_variation_kernel_ms`: GPU variation kernel subset of `repro_kernel_ms`
- `gpu` now reuses its device arena and pinned host staging within a process, so `repro_setup_ms` is typically a first-use cost and `repro_teardown_ms` may remain `0` during steady-state generations
- with `--repro-overlap on`, `repro_prepare_inputs_ms` / `repro_preprocess_ms` / `repro_pack_ms` may be partially hidden by `eval_ms`; steady-state generation timing is the relevant comparison
- `total_ms`: the full one-generation benchmark wall time
- GPU runs also report `pack_upload_ms`, `kernel_ms`, and `copyback_ms`

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

### `scripts/speedup_experiment.py`

Benchmark args:
- `--config PATH`: override the default JSON config in `scripts/`
- `--fixtures LIST`: comma-separated fixture stems or paths to run; default runs all configured fixtures
- `--outdir PATH`: output directory for reports
- `--bench-cli PATH`: override the benchmark CLI executable
- `--population-sizes LIST`: override configured population sizes
- `--max-expr-depths LIST`: override configured `max_expr_depths` or `max_expr_depth`; runs one full sweep per listed depth
- `--modes LIST`: comma-separated benchmark modes chosen from `cpu,gpu_eval,gpu_repro,gpu_repro_overlap`
- `--probe-cases N`: override configured probe case count
- `--min-success-rate F`: override configured acceptance threshold

Default config lookup:
- prefer local `scripts/speedup_experiment.json`
- otherwise use tracked `scripts/speedup_experiment.example.json`

Config keys in `scripts/speedup_experiment.example.json`:
- `bench_cli`
- `fixtures`
- `modes`
- `population_sizes`
- `blocksize`
- `seed_start`
- `probe_cases`
- `min_success_rate`
- `fuel`
- `max_expr_depths` or `max_expr_depth`
- `max_stmts_per_block`
- `max_total_nodes`
- `max_for_k`
- `max_call_args`
- `mutation_rate`
- `mutation_subtree_prob`
- `crossover_rate`
- `penalty`
- `selection_pressure`
- `outdir_prefix`

Report shape:
- per fixture, `mode_compare.report.json` / `.md`
- `modes.cpu`, `modes.gpu_eval`, `modes.gpu_repro`, `modes.gpu_repro_overlap`
- `speedup_vs_cpu.gpu_eval.*`
- `speedup_vs_cpu.gpu_repro.*`
- `speedup_vs_cpu.gpu_repro_overlap.*`

## Benchmark Workflow

### Canonical speed benchmark

```bash
python3 scripts/speedup_experiment.py
```

This benchmark sweep generates one fixed population per fixture run and uses that same generation configuration for every selected mode.
The canonical interpretation is:
- `compile`: genome-to-bytecode preparation and compile-cache lookup
- `eval`: fitness execution only
- `repro`: one-generation selection, crossover, and mutation work
- `total`: the full one-generation benchmark cost

`eval` is intentionally narrower than the old mixed metric. It excludes `compile`.
The canonical public modes are:
- `cpu`
- `gpu_eval`
- `gpu_repro`
- `gpu_repro_overlap`

### Run one small benchmark smoke

```bash
python3 scripts/speedup_experiment.py \
  --fixtures bouncing_balls_1024 \
  --modes cpu,gpu_eval,gpu_repro,gpu_repro_overlap \
  --max-expr-depths 5,7 \
  --population-sizes 64 \
  --probe-cases 8 \
  --min-success-rate 0.0
```

Primary metrics to track:
- `modes.cpu.*`
- `modes.gpu_eval.*`
- `modes.gpu_repro.*`
- `modes.gpu_repro_overlap.*`
- `speedup_vs_cpu.gpu_eval.*`
- `speedup_vs_cpu.gpu_repro.*`
- `speedup_vs_cpu.gpu_repro_overlap.*`

### Canonical evolution-progress run

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --population-size 1024 \
  --generations 20 \
  --out-json logs/simple_exp_1024.run.json
```

### Canonical evolution-progress run

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --engine gpu \
  --blocksize 1024 \
  --population-size 1024 \
  --generations 20 \
  --out-json logs/simple_exp_1024.run.json
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
