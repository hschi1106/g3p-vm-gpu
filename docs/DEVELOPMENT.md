# Development

## Build And Test

### Build C++

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

The default graph builds the product CLI and maintained test targets. Auxiliary
executables are opt-in:

```bash
cmake -S cpp -B cpp/build-aux \
  -DG3PVM_BUILD_BENCHMARKS=ON \
  -DG3PVM_BUILD_EXPERIMENTS=ON
cmake --build cpp/build-aux -j \
  --target g3pvm_runtime_multi_bench g3pvm_simple_exp_population_probe
```

The experiment probe requires CUDA. Ownership, support status, and consumers
for all scripts and auxiliary binaries are recorded in
[TOOLING_INVENTORY.md](TOOLING_INVENTORY.md).

### Tool and repository checks

Operational tools and runtime-independent repository contracts use the Python
standard library only. No product/runtime Python package or `PYTHONPATH` is
required:

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
```

For the unified operational command, install the independent package in a
virtual environment:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/pip install -e tools
.venv-tools/bin/g3pvm-tools --help
```

See [tools/README.md](../tools/README.md) for the pipeline, every subcommand,
compatibility wrappers, and artifact policy.

### C++ tests

```bash
ctest --test-dir cpp/build --output-on-failure
```

The native semantic corpus is split into independently runnable contracts:

```bash
ctest --test-dir cpp/build -R g3pvm_test_runtime_scalar --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_runtime_control_flow --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_runtime_builtins --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_runtime_typed_values --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_compiler_lowering --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_fixture_codec --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_structured_semantics --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_asgp_semantics --output-on-failure
```

The JSON corpus lives under `cpp/tests/fixtures/runtime/`. Each scenario names
its semantic intent and declares an exact expected value or error code. The
fixture executable is built normally by CMake; Python tests do not compile C++
sources ad hoc.

### Property, sanitizer, and fuzz gates

The deterministic native property layer covers generation, mutation,
crossover, CPU reproduction, GPU reproduction decode, grammar restrictions,
and payload retention:

```bash
ctest --test-dir cpp/build -L property --output-on-failure
```

The default build includes a bounded 1,000-input malformed-data smoke test.
ASan/UBSan are opt-in and CPU-only so the normal CUDA build is unchanged:

```bash
cmake -S cpp -B /tmp/g3pvm-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DG3PVM_ENABLE_CUDA=OFF \
  -DG3PVM_ENABLE_SANITIZERS=ON
cmake --build /tmp/g3pvm-sanitize -j4
ctest --test-dir /tmp/g3pvm-sanitize \
  -L 'unit|contract|property' --output-on-failure
```

Clang builds additionally expose two libFuzzer targets. Always fuzz a copied
corpus so newly minimized inputs do not appear in the source directory:

```bash
cmake -S cpp -B /tmp/g3pvm-fuzz \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DG3PVM_ENABLE_CUDA=OFF \
  -DG3PVM_BUILD_FUZZERS=ON
cmake --build /tmp/g3pvm-fuzz -j4 \
  --target g3pvm_fuzz_bytecode_json g3pvm_fuzz_ast_verify
cp -a cpp/tests/fuzz/corpus/json /tmp/g3pvm-json-corpus
/tmp/g3pvm-fuzz/g3pvm_fuzz_bytecode_json \
  -runs=1000 /tmp/g3pvm-json-corpus
/tmp/g3pvm-fuzz/g3pvm_fuzz_ast_verify \
  -runs=1000 /tmp/g3pvm-json-corpus
```

If a campaign finds a defect, preserve its smallest reproducer under a
descriptive name in `cpp/tests/fuzz/corpus/` or as a named deterministic seed
regression in the relevant property target.

### Recommended full check

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
cmake --build cpp/build -j4
ctest --test-dir cpp/build --output-on-failure
```

## Spec Freeze Gate

The current breaking-refactor spec freeze is recorded in
`benchmarks/spec_freeze.json`. It hashes the authoritative current
spec set, with `spec/grammar.md` as the latest grammar entrypoint.

When changing any listed current spec, update the manifest in the same semantic
change and run:

```bash
python3 -m unittest tests.repository.test_spec_freeze -v
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

The executable and native harnesses link `g3pvm_cli_support`; its option
parser is the single owner of the flags, defaults, and validation below.
`g3pvm_test_cli_options` locks that command-line contract independently of
evolution execution. `g3pvm_test_evolve_cli_contract` locks representative
evolution stdout, top-level JSON sections, and the intentionally unsupported
`--help`/no-argument error behavior.

Core execution args:
- `--cases PATH`: input fitness-cases file; current validation uses
  `fitness-cases`, while baseline comparisons may use
  `fitness-cases`
- `--engine {cpu|gpu}`: evaluation backend
- `--repro-backend {cpu|gpu}`: reproduction backend; `gpu` is the formal GPU reproduction path and does not promise CPU child identity
- `--repro-overlap {on|off}`: when `--engine gpu --repro-backend gpu`, overlap reproduction input prep with GPU evaluation
- `--cpu-repro-ablation {none|gpu_selection|gpu_candidates|gpu_coupled_donor}`: experimental CPU reproduction ablation flag for isolating GPU reproduction behaviors; valid only with `--repro-backend cpu`
- `--blocksize N`: CUDA block size for GPU evaluation; current native CLI default is `1024`
- `--out-json PATH`: write evolution history JSON
- `--eval-ast-json PATH`: evaluate a materialized AST JSON file against
  `--cases` and write an `ast-eval-result`; this is currently a CPU
  evaluation path used by PSB test-set reporting
- `--timing {none|summary|per_gen|all}`: timing verbosity from the native CLI
- `--show-program {none|ast|bytecode|both}`: include final-program dumps in output
- `--population-json PATH`: load a fixed `population-seeds` initial population instead of generating from `--seed`
- `--grammar-config PATH`: load a grammar config JSON file that restricts evolution generation and reproduction donor synthesis; current is the current schema, and checked-in base presets are translated as compatibility input for fair comparisons
- `--skip-final-eval {on|off}`: skip the post-loop final scoring pass; fixed-population timing runs should set this to `on`
- `--retain-final-population {on|off}`: when `off`, keep only `result.best` after the final scoring pass instead of materializing the full final scored population; the native CLI default is `off`

When `--population-json` is not supplied, native evolution infers a homogeneous expected output type from the loaded fitness cases. Under current, generation 0 should seed with that return type for payload return types (`String`, `IntList`, `FloatList`, and `StringList`) when the active grammar allows the inferred type. Other expected output types, mixed or unsupported expected output types, and grammar configs that disable the inferred payload type use generic random generation. A fixed `--population-json` file bypasses this inference and is replayed exactly.

Evolution args:
- `--population-size N`: individuals per generation
- `--generations N`: number of generations to run
- selected parent pairs always attempt `typed_subtree` crossover; mutation is applied afterward per child
- `--mutation-rate F`: probability that a post-crossover child is mutated
- `--mutation-subtree-prob F`: internal mutation operator mix; probability of typed-subtree mutation instead of constant perturbation
- `--selection-pressure N`: tournament size for each round-based without-replacement pass; larger values increase selection pressure; default is `2`
- `--seed N`: RNG seed for deterministic replay

Fitness args:
- `--penalty F`: penalty used for runtime errors and type-mismatch outputs that cannot be compared directly; must be `>= 0`
- `--fuel N`: per-program execution budget

Genome-shape args:
- `--max-expr-depth N`: maximum generated expression depth; default is `7`
- `--max-stmts-per-block N`: maximum statements per block
- `--max-total-nodes N`: maximum total AST nodes in one genome
- `--max-for-k N`: maximum integer constant used when the random generator seeds `ForRange(x, e, ...)` bounds with `Const(K)`; it is a generator limit, not a general static bound on every loop expression
- `--max-call-args N`: maximum allowed builtin call arity during generation/compilation

## Grammar Configs

Checked-in presets live under `configs/grammar/`:
- `all.json`: all public grammar constructs enabled; this is the default when `--grammar-config` is omitted
- `scalar.json`: scalar numeric / boolean search space; sequence values and container builtins disabled
- `string.json`: scalar plus `String` and string-compatible builtins
- `num_list.json`: legacy numeric-list profile; translated to current `IntList` and `FloatList` in current generation
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
python3 tools/make_population_seeds.py \
  --cases data/fixtures/simple_exp_1024.json \
  --count 1024 \
  --out logs/fixed_population.seeds.json

cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
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
- reuse the same `population-seeds` file across all modes
- generate seed files with `tools/make_population_seeds.py` so `limits`,
  `cases_path`, and optional grammar-config identity are recorded consistently
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
  --cases data/fixtures/simple_exp_1024.json \
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
- `--format-version fitness-cases|fitness-cases`: fixture schema to emit; default remains `fitness-cases` for baseline compatibility
- `--schema-json PATH`: optional field schema overrides, required for ambiguous current empty-list fields or explicit mixed numeric-list normalization
- `--n-train N`: number of training rows to emit
- `--n-test N`: number of test rows to emit
- `--seed N`: sampling seed
- `--out PATH`: output training fixture path
- `--out-test PATH`: optional test fixture output path
- `--summary-json PATH`: optional conversion summary JSON path

Conversion behavior:
- `fitness-cases` mode emits legacy `num_list` or `string_list` values for compatibility baselines
- `fitness-cases` mode emits direct `int_list`, `float_list`, or `string_list` values and records a deterministic schema hash
- empty list values use the inferred column schema when a non-empty value exists, otherwise require `--schema-json`
- mixed numeric/string list columns are rejected
- mixed `Int` / `Float` list columns are rejected in current unless `--schema-json` explicitly normalizes the field, for example to `float_list`
- multi-output PSB rows are rejected for now and are not encoded as list values

### `tools/materialize_psb_fixtures.py`

Batch fixture materialization prepares a cases root for PSB regression and
writes a machine-readable support manifest. It uses
`tools/convert_psb_to_fitness_cases.py` internally, including per-problem
schema overrides under `configs/psb_schemas/SUITE/PROBLEM.json` when present.

Common args:
- `--suite psb1|psb2`: dataset suite to convert
- `--format-version fitness-cases|fitness-cases`: fixture schema to emit
- `--datasets-root PATH`: mirrored PSB dataset root; defaults to the selected suite root
- `--schema-root PATH`: schema override root; defaults to `configs/psb_schemas`
- `--problems LIST`: optional comma-separated problem subset; omit for the whole suite
- `--n-train N`, `--n-test N`, `--seed N`: sampling controls
- `--fallback-n-train N`, `--fallback-n-test N`: optional retry sizes for
  tasks whose mirrored random split is too small for the primary sample size
- `--out-dir PATH`: output directory for `PROBLEM.train.json`,
  `PROBLEM.test.json`, per-problem summaries, and `manifest.json`

Example current materialization for the current PSB1 gate root:

```bash
python3 tools/materialize_psb_fixtures.py \
  --suite psb1 \
  --format-version fitness-cases \
  --n-train 256 \
  --n-test 256 \
  --fallback-n-train 64 \
  --fallback-n-test 64 \
  --out-dir data/fixtures/psb1-current
```

The manifest records `ok_problems` and `excluded_problems` with categories
such as `multi_output`, `schema_required`, `mixed_numeric_list`, and
`insufficient_random_rows`. Use the same materialized cases root for
`compat` and `compact` comparisons; generate baseline roots with
the same sampling controls when comparing against baseline fixtures.

The Grammar release-gate exclusion record is compacted separately in
`benchmarks/psb_release_exclusions.json`. It references the PSB1/PSB2
materialization manifests and records the unresolved multi-output exclusions
without requiring raw per-problem fixture files to be inspected during release
audit.

## PSB Regression Workflow

`tools/run_psb_regression.py` runs repeated native evolution jobs across PSB
problem fixtures and seeds, then writes a compact summary for baseline and
candidate comparison. It is intended for preserving speed and solution-quality
evidence before large grammar/runtime changes.

Common args:
- `--suite psb1|psb2`: benchmark suite label recorded in the summary
- `--profile NAME`: run profile label, such as `baseline` or `compat`
- `--cases-root PATH`: directory containing `*.train.json` fitness-case files
- `--problems LIST`: optional comma-separated problem subset; omit to run all train fixtures under `--cases-root`
- `--seeds LIST`: comma-separated seed list
- `--binary PATH`: native `g3pvm_evolve_cli` path; defaults to `cpp/build/g3pvm_evolve_cli`
- `--grammar-config PATH`: grammar config passed to the native CLI
- `--base-grammar-config PATH`: old `grammar-config` source used for
  compatibility-profile reporting; when `--profile compat` or
  `--profile compact` is used without `--grammar-config`, the runner
  generates a per-problem `grammar-config` under
  `OUT_DIR/_grammar_configs/` and passes it to the native CLI
- `--engine cpu|gpu`: evaluation backend
- `--repro-backend cpu|gpu`: reproduction backend
- `--repro-overlap [on|off]`: overlap GPU reproduction preparation with GPU evaluation
- `--eval-test`: after each successful train run, save the final best AST and
  evaluate it on the matching `*.test.json` fixture with the native CPU
  evaluation path; summary rows then include `test_best_fitness`,
  `test_solved`, and `test_status`
- `--population-size N`, `--generations N`, `--selection-pressure N`, `--mutation-rate F`, `--mutation-subtree-prob F`, `--fuel N`: native evolution budget
- `--out-dir PATH`: output directory for per-run logs plus `summary.json`
- `--dry-run`: write planned run metadata without invoking the native CLI

Tiny runner smoke:

```bash
python3 tools/run_psb_regression.py \
  --suite psb1 \
  --profile smoke \
  --cases-root data/fixtures/psb1 \
  --problems count-odds \
  --seeds 0 \
  --engine cpu \
  --repro-backend cpu \
  --population-size 16 \
  --generations 1 \
  --out-dir logs/psb_smoke
```

Generate a standalone current compact grammar config using the same numeric-list
shape as the base grammar config:

```bash
python3 tools/grammar_config_profiles.py \
  --profile compact \
  --base-grammar-config configs/grammar/num_list.json \
  --fixture-cases logs/psb_current_cases/count-odds.train.json \
  --out logs/count_odds.compact.json
```

Full baseline capture template:

```bash
python3 tools/run_psb_regression.py \
  --suite psb1 \
  --profile baseline \
  --cases-root data/fixtures/psb1 \
  --grammar-config configs/grammar/all.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --population-size 8192 \
  --generations 100 \
  --selection-pressure 2 \
  --mutation-rate 0.5 \
  --mutation-subtree-prob 0.8 \
  --fuel 20000 \
  --seeds 0,1,2,3,4 \
  --out-dir logs/psb_baselines/baseline_gpu_repro_overlap
```

For fair `compact` comparisons, always pass the same base grammar config to
the baseline via `--grammar-config` and to the current candidate via
`--base-grammar-config`. Do not compare a candidate translated from
`configs/grammar/all.json` against a baseline that relies on CLI defaults.
When `compact` is generated with `num_list_mode=both`, native generation
uses legacy `NumList` input compatibility for search only: exact current
`IntList`/`FloatList` fixture inputs are seeded as `Any` input variables, while
runtime fixture values and fitness execution stay exact current values.

Full current compact comparison template using the same base grammar-config shape
as the baseline:

```bash
python3 tools/run_psb_regression.py \
  --suite psb1 \
  --profile compact \
  --base-grammar-config configs/grammar/all.json \
  --cases-root data/fixtures/psb1-current \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --population-size 8192 \
  --generations 100 \
  --selection-pressure 2 \
  --mutation-rate 0.5 \
  --mutation-subtree-prob 0.8 \
  --fuel 20000 \
  --seeds 0,1,2,3,4 \
  --eval-test \
  --out-dir logs/psb_compact/psb1_gpu_repro_overlap
```

`tools/compare_psb_baseline.py` compares two regression summaries and exits
non-zero when a gate fails.

Comparison template:

```bash
python3 tools/compare_psb_baseline.py \
  --baseline logs/psb_baselines/baseline_gpu_repro_overlap/summary.json \
  --candidate logs/psb_compact/psb1_gpu_repro_overlap/summary.json \
  --out logs/psb_compact/psb1_gpu_repro_overlap/comparison.json
```

For formal gates that use problem-specific quality tolerances, pass a
versioned tolerance policy. The comparison output records the policy path,
hash, and format version under `thresholds.problem_tolerances`, and each
problem records its effective train/test median fitness tolerances:

```bash
python3 tools/compare_psb_baseline.py \
  --baseline logs/psb_baselines/baseline_gpu_repro_overlap/summary.json \
  --candidate logs/psb_compact/psb1_gpu_repro_overlap/summary.json \
  --problem-tolerances configs/psb_tolerances/psb1_compact_fullbudget.json \
  --out logs/psb_compact/psb1_gpu_repro_overlap/comparison.json
```

Default comparison gates:
- solved-count regression tolerance: `1` seed
- stable solved threshold: baseline solved at least `4/5`
- stable candidate minimum: candidate solved at least `3/5`
- median best-fitness regression tolerance: `0` unless a versioned
  problem-specific tolerance policy is passed
- median total runtime ratio: `1.15x`
- p90 total runtime ratio: `1.25x`
- median GPU eval kernel ratio: `1.20x`

When a problem fails, the comparison JSON includes:
- `speed_failure_attribution`: speed phases implicated by the failed metrics,
  such as `gpu_kernel`, `eval`, `repro`, `repro_decode`, or `total_wall`
- `quality_failure_categories`: train/test solved or fitness categories
- `failure_context`: seeds, schema hashes, grammar-config metadata, and compact
  per-seed train/test fitness/program summaries

`tools/write_psb_manifest.py` turns baseline summary, candidate summary, and
comparison JSON into a compact manifest suitable for keeping under
`benchmarks/` without committing raw per-run logs.

The baseline full-budget supported-PSB1 baseline used by the current compact gate is
recorded separately as
`benchmarks/psb1_supported_all_config_fullbudget_baseline.json`.

Manifest template:

```bash
python3 tools/write_psb_manifest.py \
  --baseline logs/psb_baselines/baseline_gpu_repro_overlap/summary.json \
  --candidate logs/psb_compact/psb1_gpu_repro_overlap/summary.json \
  --comparison logs/psb_compact/psb1_gpu_repro_overlap/comparison.json \
  --excluded-manifest data/fixtures/psb1-current/manifest.json \
  --out benchmarks/psb1_compact_manifest.json
```
