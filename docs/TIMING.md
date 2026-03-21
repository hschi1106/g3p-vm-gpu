# Timing

This document defines the canonical timing metric names used by the native benchmark and evolution CLIs.

The goal is to keep timing output:

- explicit about scope,
- consistent across CLI, JSON, and internal structs,
- safe to aggregate without residual guessing,
- easy to extend when a new phase needs its own timer.

## Naming Rules

- Every duration is reported in milliseconds and ends with `_ms`.
- A metric without `_total` is one measurement for one call or one generation row.
- A metric with `_total` is the sum of the corresponding per-generation or per-call values over the full run.
- `generation_*` names are aligned by generation index in `evolve_cli` JSON.
- When a metric family is inactive for a run, its per-generation arrays are zero-filled rather than omitted.
- `gpu_eval_*` is reserved for GPU fitness evaluation.
- `repro_*` is reserved for reproduction.
- `compile_ms`, `eval_ms`, `repro_ms`, and `total_ms` are coarse wall-clock rollups.

## Measurement Layers

There are three timing layers in the repo.

### 1. Runtime-internal phase timing

These are the direct timers produced by runtime code.

#### `FitnessSessionInitTiming`

Produced by `FitnessSessionGpu::init(...)`.

- `device_select_ms`: device selection and first-touch CUDA setup
- `shared_case_pack_ms`: host packing of shared cases and expected values
- `payload_cache_warm_ms`: payload cache preparation for shared inputs
- `upload_ms`: upload of shared-case buffers to device memory
- `total_ms`: full `init(...)` wall-clock

#### `FitnessEvalTiming`

Produced by `FitnessSessionGpu::eval_programs(...)`.

- `pack_ms`: host packing of bytecode programs and payload token lookup
- `launch_prep_ms`: launch-shape preparation and host-side eval setup after packing
- `upload_ms`: per-eval device allocation, upload, and fitness buffer initialization
- `kernel_ms`: eval kernel launch plus synchronization
- `copyback_ms`: device-to-host fitness copyback
- `teardown_ms`: teardown of temporary eval allocations after copyback
- `total_ms`: full `eval_programs(...)` wall-clock, including teardown

#### `ReproductionStats`

Produced by the reproduction backend.

- `selection_ms`: top-level selection phase
- `crossover_ms`: top-level crossover phase
- `mutation_ms`: top-level mutation phase
- `prepare_inputs_ms`: host extraction of genomes, fitness, and backend config
- `setup_ms`: GPU reproduction setup, device selection, and reusable arena preparation
- `preprocess_ms`: subtree/candidate/donor preprocessing
- `pack_ms`: host flattening into the packed GPU upload schema
- `upload_ms`: H2D upload of reproduction inputs
- `kernel_ms`: total GPU reproduction kernel time
- `copyback_ms`: child copyback to host staging
- `decode_ms`: reconstruction of `ProgramGenome` children from packed copyback buffers
- `teardown_ms`: reproduction arena teardown
- `selection_kernel_ms`: selection-kernel subset of `kernel_ms`
- `variation_kernel_ms`: variation-kernel subset of `kernel_ms`

## Canonical Metric Families

### Coarse wall-clock metrics

These are the first metrics to use for end-to-end benchmark comparison.

| Metric | Meaning |
| --- | --- |
| `compile_ms` | Fixed-population compile-cache lookup plus any genome-to-bytecode compilation |
| `eval_ms` | One benchmark eval stage wall-clock |
| `repro_ms` | One benchmark reproduction stage wall-clock |
| `total_ms` | Full benchmark or full evolution run wall-clock |
| `generation_eval_ms` | One generation scoring-stage wall-clock in `evolve_cli` |
| `generation_repro_ms` | One generation reproduction-stage wall-clock in `evolve_cli` |
| `generation_total_ms` | One generation total wall-clock in `evolve_cli` |
| `final_eval_ms` | Final post-generation scoring wall-clock in `evolve_cli` |

Important notes:

- In `population_bench_cli`, CPU `eval_ms` is just the CPU fitness call, while GPU `eval_ms` is `gpu_eval_init_ms + gpu_eval_call_ms`.
- In `evolve_cli`, `generation_eval_ms` includes compile, scoring, canonicalization, and scored-population rebuild. It is intentionally broader than the device kernel path.
- `repro_ms` can be lower than the sum of some `repro_*` prep phases when overlap is enabled, because part of reproduction is hidden behind evaluation.

### GPU eval metrics

These are the canonical metrics for GPU fitness attribution.

| Metric | Meaning | Direct or derived |
| --- | --- | --- |
| `gpu_eval_init_ms` | Full `FitnessSessionGpu::init(...)` wall-clock | Direct |
| `gpu_eval_call_ms` | Full `FitnessSessionGpu::eval_programs(...)` wall-clock | Direct |
| `gpu_eval_pack_ms` | Host program packing | Direct |
| `gpu_eval_launch_prep_ms` | Launch preparation and host-side pre-launch setup | Direct |
| `gpu_eval_upload_ms` | Per-eval upload/allocation phase | Direct |
| `gpu_eval_pack_upload_ms` | `gpu_eval_pack_ms + gpu_eval_upload_ms` | Derived convenience aggregate |
| `gpu_eval_kernel_ms` | GPU eval kernel time | Direct |
| `gpu_eval_copyback_ms` | Fitness copyback | Direct |
| `gpu_eval_teardown_ms` | Temporary eval teardown after copyback | Direct |

### Reproduction metrics

These are the canonical metrics for reproduction attribution.

| Metric | Meaning |
| --- | --- |
| `selection_ms` | Host top-level selection phase |
| `crossover_ms` | Host top-level crossover phase |
| `mutation_ms` | Host top-level mutation phase |
| `repro_prepare_inputs_ms` | Reproduction input extraction and config setup |
| `repro_setup_ms` | GPU reproduction setup and reusable arena prep |
| `repro_preprocess_ms` | Preprocessing for subtree/donor metadata |
| `repro_pack_ms` | Host flattening into packed reproduction buffers |
| `repro_upload_ms` | Reproduction H2D upload |
| `repro_kernel_ms` | Total GPU reproduction kernel time |
| `repro_copyback_ms` | Reproduction D2H copyback |
| `repro_decode_ms` | Host decode of copied-back children |
| `repro_teardown_ms` | Reproduction teardown |
| `repro_selection_kernel_ms` | Selection-kernel subset of `repro_kernel_ms` |
| `repro_variation_kernel_ms` | Variation-kernel subset of `repro_kernel_ms` |

## `g3pvm_population_bench_cli`

The benchmark CLI emits one `BENCH ...` line.

### Always-present timing fields

| Field | Meaning |
| --- | --- |
| `compile_ms` | Fixed-population compile stage |
| `eval_ms` | Benchmark eval stage wall-clock |
| `repro_ms` | Benchmark reproduction stage wall-clock |
| `selection_ms` | Host selection phase |
| `crossover_ms` | Host crossover phase |
| `mutation_ms` | Host mutation phase |
| `repro_prepare_inputs_ms` | Reproduction input extraction |
| `repro_setup_ms` | GPU reproduction setup |
| `repro_preprocess_ms` | Reproduction preprocessing |
| `repro_pack_ms` | Reproduction pack |
| `repro_upload_ms` | Reproduction upload |
| `repro_kernel_ms` | Reproduction kernel total |
| `repro_copyback_ms` | Reproduction copyback |
| `repro_decode_ms` | Reproduction decode |
| `repro_teardown_ms` | Reproduction teardown |
| `repro_selection_kernel_ms` | Selection-kernel subset |
| `repro_variation_kernel_ms` | Variation-kernel subset |
| `total_ms` | Full one-generation benchmark wall-clock |

### GPU-only detail fields

When `--engine gpu` is used, the canonical GPU eval fields are:

- `gpu_eval_init_ms`
- `gpu_eval_call_ms`
- `gpu_eval_pack_ms`
- `gpu_eval_launch_prep_ms`
- `gpu_eval_upload_ms`
- `gpu_eval_pack_upload_ms`
- `gpu_eval_kernel_ms`
- `gpu_eval_copyback_ms`
- `gpu_eval_teardown_ms`

No GPU eval residual field is produced. In particular, the runtime does not emit a fake "session init" value computed by subtraction.

## `g3pvm_evolve_cli`

The evolution CLI exposes the same timing families in three forms:

- summary `TIMING phase=...`
- per-generation `TIMING gen=...` and `TIMING gpu_gen=...`
- JSON in `meta.timing` and top-level `timing`

### Summary `TIMING phase=...`

Common summary phases:

- `init_population`
- `generations_eval_total`
- `generations_repro_total`
- `generations_selection_total`
- `generations_crossover_total`
- `generations_mutation_total`
- `generations_repro_prepare_inputs_total`
- `generations_repro_setup_total`
- `generations_repro_preprocess_total`
- `generations_repro_pack_total`
- `generations_repro_upload_total`
- `generations_repro_kernel_total`
- `generations_repro_copyback_total`
- `generations_repro_decode_total`
- `generations_repro_teardown_total`
- `generations_repro_selection_kernel_total`
- `generations_repro_variation_kernel_total`
- `cpu_compile_total`
- `final_eval`
- `total`

GPU summary phases:

- `gpu_eval_init`
- `gpu_compile_total`
- `gpu_eval_call_total`
- `gpu_eval_pack_total`
- `gpu_eval_launch_prep_total`
- `gpu_eval_upload_total`
- `gpu_eval_pack_upload_total`
- `gpu_eval_kernel_total`
- `gpu_eval_copyback_total`
- `gpu_eval_teardown_total`

### Per-generation console lines

`TIMING gen=...` reports the coarse per-generation wall clocks and reproduction breakdown:

- `eval_ms`
- `repro_ms`
- `total_ms`
- `selection_ms`
- `crossover_ms`
- `mutation_ms`
- `repro_prepare_inputs_ms`
- `repro_setup_ms`
- `repro_preprocess_ms`
- `repro_pack_ms`
- `repro_upload_ms`
- `repro_kernel_ms`
- `repro_copyback_ms`
- `repro_decode_ms`
- `repro_teardown_ms`
- `repro_selection_kernel_ms`
- `repro_variation_kernel_ms`
- `cpu_compile_ms`

`TIMING gpu_gen=...` reports GPU scoring detail for the same generation:

- `gpu_compile_ms`
- `gpu_eval_call_ms`
- `gpu_eval_pack_ms`
- `gpu_eval_launch_prep_ms`
- `gpu_eval_upload_ms`
- `gpu_eval_pack_upload_ms`
- `gpu_eval_kernel_ms`
- `gpu_eval_copyback_ms`
- `gpu_eval_teardown_ms`

### JSON timing keys

`meta.timing` stores run-level totals:

- `init_population_ms`
- `gpu_eval_init_ms`
- `final_eval_ms`
- `cpu_compile_ms_total`
- `gpu_compile_ms_total`
- `gpu_eval_call_ms_total`
- `gpu_eval_pack_ms_total`
- `gpu_eval_launch_prep_ms_total`
- `gpu_eval_upload_ms_total`
- `gpu_eval_pack_upload_ms_total`
- `gpu_eval_kernel_ms_total`
- `gpu_eval_copyback_ms_total`
- `gpu_eval_teardown_ms_total`
- `generations_selection_ms_total`
- `generations_crossover_ms_total`
- `generations_mutation_ms_total`
- `generations_repro_prepare_inputs_ms_total`
- `generations_repro_setup_ms_total`
- `generations_repro_preprocess_ms_total`
- `generations_repro_pack_ms_total`
- `generations_repro_upload_ms_total`
- `generations_repro_kernel_ms_total`
- `generations_repro_copyback_ms_total`
- `generations_repro_decode_ms_total`
- `generations_repro_teardown_ms_total`
- `generations_repro_selection_kernel_ms_total`
- `generations_repro_variation_kernel_ms_total`
- `total_ms`

The top-level `timing` object stores per-generation arrays:

- `generation_eval_ms`
- `generation_repro_ms`
- `generation_cpu_compile_ms`
- `generation_gpu_compile_ms`
- `generation_gpu_eval_call_ms`
- `generation_gpu_eval_pack_ms`
- `generation_gpu_eval_launch_prep_ms`
- `generation_gpu_eval_upload_ms`
- `generation_gpu_eval_pack_upload_ms`
- `generation_gpu_eval_kernel_ms`
- `generation_gpu_eval_copyback_ms`
- `generation_gpu_eval_teardown_ms`
- `generation_selection_ms`
- `generation_crossover_ms`
- `generation_mutation_ms`
- `generation_repro_prepare_inputs_ms`
- `generation_repro_setup_ms`
- `generation_repro_preprocess_ms`
- `generation_repro_pack_ms`
- `generation_repro_upload_ms`
- `generation_repro_kernel_ms`
- `generation_repro_copyback_ms`
- `generation_repro_decode_ms`
- `generation_repro_teardown_ms`
- `generation_repro_selection_kernel_ms`
- `generation_repro_variation_kernel_ms`
- `generation_total_ms`

## Accounting Notes

- `gpu_eval_init_ms` depends on shared cases, payload setup, CUDA context state, and shared-buffer upload. It does not scale with program depth in the same way as `gpu_eval_call_ms`.
- `gpu_eval_call_ms` includes `gpu_eval_teardown_ms`. This is intentional so temporary eval allocation cleanup is no longer hidden in an unlabelled residual.
- `gpu_eval_pack_upload_ms` is a convenience aggregate. The direct timers remain `gpu_eval_pack_ms` and `gpu_eval_upload_ms`.
- `repro_kernel_ms` should equal the sum of `repro_selection_kernel_ms` and `repro_variation_kernel_ms` up to normal timer rounding.
- With `repro_overlap=on`, the coarse `repro_ms` value is a visibility metric, not a full-accounting sum of all reproduction work.
- New timing work should extend the direct phase structs first, then thread the names through CLI and JSON unchanged.

## Removed Legacy Names

The repo no longer emits the old GPU eval aliases:

- `pack_upload_ms`
- `kernel_ms`
- `copyback_ms`
- residual-derived fake GPU init metrics
- `gpu_session_init_ms`
- `gpu_scoring_*`
- `gpu_generations_*`

Use the canonical `gpu_eval_*` family instead.
