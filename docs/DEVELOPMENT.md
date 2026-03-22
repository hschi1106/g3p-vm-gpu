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
python3 scripts/speedup_experiment.py
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
- `--population-json PATH`: load a pre-generated `population-seeds-v1` JSON instead of generating a fresh population
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

Timing semantics:
- Canonical timing names, scope boundaries, and CLI/JSON mappings are defined in [TIMING.md](TIMING.md).
- `compile_ms`, `eval_ms`, `repro_ms`, and `total_ms` are the coarse benchmark rollups used by scripts such as `speedup_experiment.py`.
- GPU evaluation detail is reported with the `gpu_eval_*` family, including `gpu_eval_init_ms`, `gpu_eval_call_ms`, `gpu_eval_pack_ms`, `gpu_eval_launch_prep_ms`, `gpu_eval_upload_ms`, `gpu_eval_pack_upload_ms`, `gpu_eval_kernel_ms`, `gpu_eval_copyback_ms`, and `gpu_eval_teardown_ms`.
- Reproduction detail is reported with the existing `repro_*` family, including selection/crossover/mutation plus the GPU backend phases `repro_prepare_inputs_ms`, `repro_setup_ms`, `repro_preprocess_ms`, `repro_pack_ms`, `repro_upload_ms`, `repro_kernel_ms`, `repro_copyback_ms`, `repro_decode_ms`, `repro_teardown_ms`, `repro_selection_kernel_ms`, and `repro_variation_kernel_ms`.
- With `--repro-overlap on`, `repro_prepare_inputs_ms`, `repro_preprocess_ms`, and `repro_pack_ms` may be partially hidden behind GPU evaluation wall time; compare steady-state generation metrics rather than only the coarse `repro_ms`.

### `cpp/build/g3pvm_population_bucket_cli`

Exact-bucket population generation args:
- `--cases PATH`: input `fitness-cases-v1` file used for probe filtering and payload classification
- `--out-population-json PATH`: output `population-seeds-v1` JSON
- `--out-metadata-json PATH`: output per-program metadata summary
- `--target-depth N`: require `genome.meta.max_depth == N`
- `--target-node-count N`: require `genome.meta.node_count == N`; when set, `--target-depth` becomes optional
- `--target-payload-flavor {any|none|string|list|mixed}`: require the runtime GPU payload classifier to match this flavor; `any` accepts all flavors
- `--generator-mode {native|synthetic}`: use the standard GP generator with rejection sampling or a synthetic exact-depth / exact-node generator
- `--generator-root-type {any|num|bool|none|string|list}`: optional root-type hint for generation
- `--population-size N`
- `--seed-start N`
- `--probe-cases N`
- `--min-success-rate F`
- `--max-attempts N`
- `--fuel N`
- `--max-expr-depth N`
- `--max-stmts-per-block N`
- `--max-total-nodes N`
- `--max-for-k N`
- `--max-call-args N`

Typical exact-depth bucket generation:

```bash
cpp/build/g3pvm_population_bucket_cli \
  --cases data/fixtures/simple_x_plus_1_1024.json \
  --target-depth 9 \
  --target-payload-flavor mixed \
  --generator-mode synthetic \
  --population-size 1024 \
  --probe-cases 32 \
  --min-success-rate 0.5 \
  --max-attempts 500000 \
  --out-population-json data/exp/depth09_mixed.population.json \
  --out-metadata-json data/exp/depth09_mixed.metadata.json
```

Typical exact-node bucket generation:

```bash
cpp/build/g3pvm_population_bucket_cli \
  --cases data/fixtures/simple_x_plus_1_1024.json \
  --target-node-count 40 \
  --target-payload-flavor any \
  --generator-mode synthetic \
  --population-size 1024 \
  --probe-cases 32 \
  --min-success-rate 0.5 \
  --max-attempts 500000 \
  --max-expr-depth 13 \
  --max-total-nodes 80 \
  --out-population-json data/exp/node40.population.json \
  --out-metadata-json data/exp/node40.metadata.json
```

Notes:
- `synthetic + --target-node-count` builds exact-size trees directly instead of relying on rejection sampling alone.
- `--target-payload-flavor any` is only meaningful with synthetic exact-node generation; it uses round-robin underfilled-flavor scheduling to keep `none/string/list/mixed` buckets as balanced as feasibility allows.
- generated metadata includes per-program `node_count`, `actual_depth`, `code_len`, and classified `payload_flavor`.

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
This script no longer accepts command-line benchmark overrides.
It always reads its full configuration from:

- `scripts/speedup_experiment.example.json`

Run it as:

```bash
python3 scripts/speedup_experiment.py
```

Config keys in `scripts/speedup_experiment.example.json`:
- `bench_cli`
- `fixtures`
- `population_jsons`
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

Fixed-population mode:
- each `population_jsons` entry may be either a string path or an object with:
  - `population_json`
  - optional `label`
  - optional `cases`
- when `population_jsons` is used, set these generation axes to empty lists:
  - `fixtures`
  - `population_sizes`
  - `max_expr_depths`
- and set these generation-time controls to `null`:
  - `seed_start`
  - `probe_cases`
  - `min_success_rate`
  - `max_expr_depth`
  - `max_stmts_per_block`
  - `max_total_nodes`
  - `max_for_k`
  - `max_call_args`
- the script validates this so the config clearly shows which controls are inactive under fixed-population benchmarking
- each population JSON must be `population-seeds-v1`; `cases_path` is loaded from the file unless overridden by `cases`

Example node-bucket sweep:

Edit `scripts/speedup_experiment.example.json` to:

```json
{
  "bench_cli": "cpp/build/g3pvm_population_bench_cli",
  "population_jsons": [
    "data/exp/node_simple_x_plus_1_1024/node20.population.json",
    "data/exp/node_simple_x_plus_1_1024/node30.population.json",
    "data/exp/node_simple_x_plus_1_1024/node40.population.json",
    "data/exp/node_simple_x_plus_1_1024/node50.population.json",
    "data/exp/node_simple_x_plus_1_1024/node60.population.json",
    "data/exp/node_simple_x_plus_1_1024/node70.population.json"
  ],
  "modes": ["cpu", "gpu_eval", "gpu_repro", "gpu_repro_overlap"],
  "fixtures": [],
  "population_sizes": [],
  "seed_start": null,
  "probe_cases": null,
  "min_success_rate": null,
  "fuel": 20000,
  "blocksize": 1024,
  "max_expr_depths": [],
  "max_expr_depth": null,
  "max_stmts_per_block": null,
  "max_total_nodes": null,
  "max_for_k": null,
  "max_call_args": null,
  "mutation_rate": 0.5,
  "mutation_subtree_prob": 0.8,
  "crossover_rate": 0.9,
  "penalty": 1.0,
  "selection_pressure": 3,
  "outdir_prefix": "logs/node_simple_x_plus_1_speedup"
}
```

Then run:

```bash
python3 scripts/speedup_experiment.py
```

Report shape:
- per fixture, `mode_compare.report.json` / `.md`
- top-level summary, `summary.json` / `summary.md`
- per fixture JSON includes:
  - `modes`: raw `BENCH` fields from `g3pvm_population_bench_cli`
  - `speedup_vs_cpu`: coarse CPU-vs-mode speedups
  - `timing_analysis.mode_breakdown`: per-mode coarse timing, GPU eval breakdown, and reproduction breakdown
  - `timing_analysis.mode_comparisons`: derived comparisons such as `gpu_eval_vs_cpu`, `gpu_repro_vs_gpu_eval`, and `gpu_repro_overlap_vs_gpu_repro`
- summary JSON includes:
  - `average_speedup_vs_cpu`
  - `average_mode_timings`
  - `average_timing_analysis`
  - depth-bucketed variants when multiple `max_expr_depth` values are run

Timing-analysis intent:
- separate coarse benchmark wall clocks from detailed GPU eval timing using canonical `gpu_eval_*` names
- expose GPU eval cold-start tax versus steady-state eval cost
- expose GPU reproduction primary-phase breakdown and kernel subphases
- quantify hidden overlap in `gpu_repro_overlap` as `hidden_overlap_ms`

### `scripts/kernel_bucket_experiment.py`

Experiment args:
- `--cases PATH`: fixture used for payload classification and evaluation
- `--bucket-cli PATH`: override `g3pvm_population_bucket_cli`
- `--bench-cli PATH`: override `g3pvm_population_bench_cli`
- `--population-size N`
- `--replicates N`: number of fixed populations per grid cell
- `--bench-repeats N`: benchmark repetitions per fixed population
- `--depths LIST`: comma-separated exact depths
- `--payload-flavors LIST`: comma-separated payload flavors
- `--probe-cases N`
- `--min-success-rate F`
- `--fuel N`
- `--blocksize N`
- `--max-stmts-per-block N`
- `--max-total-nodes N`
- `--max-for-k N`
- `--max-call-args N`
- `--max-attempts N`
- `--seed-base N`
- `--population-outdir PATH`
- `--outdir PATH`

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

Edit `scripts/speedup_experiment.example.json` to a small sweep such as:

```json
{
  "bench_cli": "cpp/build/g3pvm_population_bench_cli",
  "fixtures": ["data/fixtures/bouncing_balls_1024.json"],
  "modes": ["cpu", "gpu_eval", "gpu_repro", "gpu_repro_overlap"],
  "population_sizes": [64],
  "blocksize": 1024,
  "seed_start": 0,
  "probe_cases": 8,
  "min_success_rate": 0.0,
  "fuel": 20000,
  "max_expr_depths": [5, 7],
  "max_stmts_per_block": 6,
  "max_total_nodes": 80,
  "max_for_k": 16,
  "max_call_args": 3,
  "mutation_rate": 0.5,
  "mutation_subtree_prob": 0.8,
  "crossover_rate": 0.9,
  "penalty": 1.0,
  "selection_pressure": 3,
  "outdir_prefix": "logs/fixture_speedup_smoke"
}
```

Then run:

```bash
python3 scripts/speedup_experiment.py
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
  - `docs/FILE_STRUCTURE.md`
  - repo skill references
