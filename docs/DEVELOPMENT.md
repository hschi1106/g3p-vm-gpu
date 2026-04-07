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
```

## GPU Run Policy

GPU-capable C++ paths select the least-used visible CUDA device internally.
To force a specific visible-device index for a run, set:

```bash
G3PVM_CUDA_DEVICE=0
```

Current GPU eval runtime behavior:
- one production `Mixed` eval kernel launch per accepted population
- payload flavor labels remain available for offline analysis, but are not used for production eval dispatch

## Public CLI Arguments

This section documents the adjustable arguments that affect supported public workflows.

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
- `--population-json PATH`: load a fixed `population-seeds-v1` initial population instead of generating from `--seed`
- `--grammar-config PATH`: load a `grammar-config-v1` JSON file that restricts evolution generation and reproduction donor synthesis
- `--skip-final-eval {on|off}`: skip the post-loop final scoring pass; fixed-population timing runs should set this to `on`

Evolution args:
- `--population-size N`: individuals per generation
- `--generations N`: number of generations to run
- selected parent pairs always attempt `typed_subtree` crossover; mutation is applied afterward per child
- `--mutation-rate F`: probability that a post-crossover child is mutated
- `--mutation-subtree-prob F`: internal mutation operator mix; probability of typed-subtree mutation instead of constant perturbation
- `--selection-pressure N`: tournament size for each round-based without-replacement pass; larger values increase selection pressure
- `--seed N`: RNG seed for deterministic replay

Fitness args:
- `--penalty F`: penalty used for runtime errors and type-mismatch outputs that cannot be compared directly; must be `>= 0`
- `--fuel N`: per-program execution budget

Genome-shape args:
- `--max-expr-depth N`: maximum generated expression depth
- `--max-stmts-per-block N`: maximum statements per block
- `--max-total-nodes N`: maximum total AST nodes in one genome
- `--max-for-k N`: maximum integer constant used when the random generator seeds `ForRange(x, e, ...)` bounds with `Const(K)`; it is a generator limit, not a general static bound on every loop expression
- `--max-call-args N`: maximum allowed builtin call arity during generation/compilation

## Grammar Configs

Checked-in presets live under `configs/grammar/`:
- `all.json`: all public grammar constructs enabled; this is the default when `--grammar-config` is omitted
- `scalar.json`: scalar numeric / boolean / `None` search space; sequence values and container builtins disabled
- `string.json`: scalar plus `String` and string-compatible builtins
- `num_list.json`: scalar plus `NumList` and numeric-list builtins
- `string_list.json`: scalar plus `String` / `StringList` and string-list builtins
- `sequence.json`: broad sequence profile, currently equivalent to all first-wave sequence support

The config is a search-space control only. It does not reject bytecode execution or loading of already-materialized ASTs that contain disabled constructs.
The full config schema and replay rules are defined in [GRAMMAR_CONFIG.md](GRAMMAR_CONFIG.md).

Current backend support:
- CPU reproduction respects non-default grammar configs.
- GPU reproduction respects non-default grammar configs during host-side preprocess by filtering typed candidates and building config-aware donor buckets.

Example:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --grammar-config configs/grammar/scalar.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --population-size 64 \
  --generations 5 \
  --out-json logs/simple_exp_1024.scalar.json
```

## Fixed-Pop Benchmark Mode

The supported fixed-population benchmark workflow is:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/bouncing_balls_1024.json \
  --population-json logs/fixed_population.seeds.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap off \
  --blocksize 1024 \
  --generations 1 \
  --skip-final-eval on \
  --timing all \
  --out-json logs/fixed_population.run.json
```

For fair comparisons:
- reuse the same `population-seeds-v1` file across all modes
- compare generation-0 timing fields only
- treat `total_ms` as the primary wall-clock metric, then inspect subphases

Canonical timing names, scope boundaries, and CLI/JSON mappings are defined in [TIMING.md](TIMING.md).

Important timing interpretations:
- `generation_eval_ms` includes compile, scoring, canonicalization, and scored-population rebuild for that generation
- GPU evaluation detail is reported with the `gpu_eval_*` family, including `gpu_eval_init_ms`, `gpu_eval_call_ms`, `gpu_eval_pack_ms`, `gpu_eval_launch_prep_ms`, `gpu_eval_upload_ms`, `gpu_eval_pack_upload_ms`, `gpu_eval_kernel_ms`, `gpu_eval_copyback_ms`, and `gpu_eval_teardown_ms`
- reproduction detail is reported with the `repro_*` family, including selection/crossover/mutation plus the GPU backend phases `repro_prepare_inputs_ms`, `repro_setup_ms`, `repro_preprocess_ms`, `repro_pack_ms`, `repro_upload_ms`, `repro_kernel_ms`, `repro_copyback_ms`, `repro_decode_ms`, `repro_teardown_ms`, `repro_selection_kernel_ms`, and `repro_variation_kernel_ms`
- with `--repro-overlap on`, `repro_prepare_inputs_ms`, `repro_preprocess_ms`, and `repro_pack_ms` may be partially hidden behind GPU evaluation wall time

## Canonical Runbooks

### Evolution progress run

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

### Fixed-population timing smoke

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/bouncing_balls_1024.json \
  --population-json logs/fixed_population.seeds.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --generations 1 \
  --skip-final-eval on \
  --timing all \
  --out-json logs/fixed_population.run.json
```

## PSB Dataset Workflow

### `tools/fetch_psb_datasets.py`

Unified dataset-fetch args:
- `--suite psb1|psb2`: dataset suite to fetch
- `--out-dir PATH`: optional target datasets directory; default depends on `--suite`
- `--problems LIST|all`: comma-separated task names or `all`
- `--splits LIST`: comma-separated subset of `edge,random`
- `--retries N`: retry count per file
- `--timeout-sec N`: per-request timeout
- `--force`: overwrite existing files
- `--dry-run`: print planned work without downloading

### `tools/convert_psb_to_fitness_cases.py`

Unified dataset-conversion args:
- `--suite psb1|psb2`: dataset suite to convert
- `--problem NAME`: problem name under `--datasets-root`
- `--datasets-root PATH`: optional dataset root; default depends on `--suite`
- `--edge-file PATH`: explicit edge JSONL file; use with `--random-file` instead of `--problem`
- `--random-file PATH`: explicit random JSONL file
- `--n-train N`: number of training rows to emit
- `--n-test N`: number of test rows to emit
- `--seed N`: sampling seed
- `--out PATH`: output training fixture path
- `--out-test PATH`: optional test fixture output path
- `--summary-json PATH`: optional conversion summary JSON path

Conversion behavior:
- PSB1 and PSB2 conversion both infer list fields column-wise and emit explicit `num_list` or `string_list` values
- empty list values use the inferred column schema
- mixed numeric/string list columns are rejected
- multi-output PSB rows are rejected for now and are not encoded as list values
