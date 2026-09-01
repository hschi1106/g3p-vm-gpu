# PSB Workflow

This guide owns dataset materialization and native PSB regression procedures.
The unified command surface and artifact policy are in
[`../../tools/README.md`](../../tools/README.md).

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
