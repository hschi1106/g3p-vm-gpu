# Experiment Plan

## Scope

- `blocksize sweep` has already been completed and is excluded from this plan.
- All GPU-capable runs should pin the same visible device with `G3PVM_CUDA_DEVICE`.
- All experiments should use the same machine, the same build, and the same fixed benchmark parameters unless stated otherwise.
- When reporting aggregate results, use `median + IQR`.

## Common Settings

- Keep the previously selected `blocksize` fixed across all runs.
- Keep `fuel`, shape limits, `mutation_rate`, `mutation_subtree_prob`, and `selection_pressure` fixed within each experiment.
- Use the same population seed sets or RNG seeds across modes whenever the comparison requires matched runs.
- GPU profiling must use `nsys` only.

## Output Layout

All generated artifacts should live under `logs/` with one timestamped root per experiment run.

Recommended naming:

- Experiment 1 root: `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/`
- Experiment 2 root: `logs/exp2_closed_loop_effectiveness_<timestamp>/`
- Experiment 3 root: `logs/exp3_gpu_resource_boundary_<timestamp>/`

Within each experiment root, use the same top-level layout:

- `raw/`: console logs and direct CLI outputs
- `reports/`: parsed JSON / Markdown summaries
- `plots/`: generated figures
- `populations/`: fixed population JSONs or generated stress populations when applicable

If a human-written narrative note is needed after the run, place it under `meeting/<date>/` and include the source log directory path.

## Experiment 1: Fixed-Pop Adjusted Speedup

### Goal

Measure performance on the same fixed workload without closed-loop trajectory drift.

### Tooling

- Canonical unit run:
  - `g3pvm_evolve_cli --generations 1 --population-json ... --skip-final-eval on --timing all --out-json ...`
- Driver:
  - any wrapper is acceptable as long as it produces the same fixed populations, runs the same four modes, and aggregates results according to this section
- Fixture: `data/fixtures/bouncing_balls_1024.json`
- Modes:
  - `cpu`
  - `gpu_eval`
  - `gpu_repro`
  - `gpu_repro_overlap`

### Population Sizes

- `1024`
- `2048`
- `4096`
- `8192`
- `16384`

### Sampling Plan

- For each `population_size`, generate `5` fixed populations as `population-seeds-v1`.
- Fixed populations should be deterministic seed sets, not on-the-fly random generations inside the benchmark run.
- Do not use probe filtering as part of the benchmark definition; the benchmark should measure the exact fixed population it is given.
- Reuse the same fixed populations across all four modes.
- For each fixed population and each mode, run `3` repeats.
- For each fixed population, take the median over the `3` repeats.
- For each `population_size`, report `median + IQR` over the `5` fixed populations.

### Output Paths

- Fixed population files:
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/populations/pop<population_size>/population_<index>.seeds.json`
- Raw CLI logs:
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/raw/pop<population_size>/population_<index>/<mode>/repeat_<index>.console.log`
- Native one-generation JSON:
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/raw/pop<population_size>/population_<index>/<mode>/repeat_<index>.run.json`
- Parsed per-repeat metrics:
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/raw/pop<population_size>/population_<index>/<mode>/repeat_<index>.metrics.json`
- Top-level experiment summary:
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/reports/summary.json`
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/reports/summary.md`
- Figures:
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/plots/popsize_vs_steady_eval_speedup.png`
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/plots/popsize_vs_warm_total_proxy_speedup.png`
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/plots/popsize_vs_warm_total_proxy_ms.png`
  - `logs/exp1_fixed_pop_adjusted_speedup_<timestamp>/plots/repro_breakdown_pop<population_size>.png`

### Metrics

Per-run raw quantities:

- `generation_eval_ms[0]`
- `generation_repro_ms[0]`
- `generation_total_ms[0]`
- `generation_cpu_compile_ms[0]`
- `generation_gpu_compile_ms[0]`
- `gpu_eval_init_ms`
- `gpu_eval_call_ms = generation_gpu_eval_call_ms[0]`
- `repro_setup_ms`

Derived benchmark metrics:

- `compile_ms(cpu) = generation_cpu_compile_ms[0]`
- `compile_ms(gpu_*) = generation_gpu_compile_ms[0]`
- `eval_ms(cpu) = generation_eval_ms[0] - generation_cpu_compile_ms[0]`
- `eval_ms(gpu_*) = gpu_eval_init_ms + generation_gpu_eval_call_ms[0]`
- `cold_total_ms(cpu) = generation_total_ms[0]`
- `cold_total_ms(gpu_*) = generation_total_ms[0] + gpu_eval_init_ms`
- `steady_eval_ms(cpu) = eval_ms(cpu)`
- `steady_eval_ms(gpu_*) = gpu_eval_call_ms`
- `warm_total_proxy_ms(cpu) = cold_total_ms(cpu)`
- `warm_total_proxy_ms(gpu_eval) = cold_total_ms(gpu_eval) - gpu_eval_init_ms`
- `warm_total_proxy_ms(gpu_repro) = cold_total_ms(gpu_repro) - gpu_eval_init_ms - repro_setup_ms`
- `warm_total_proxy_ms(gpu_repro_overlap) = cold_total_ms(gpu_repro_overlap) - gpu_eval_init_ms - repro_setup_ms`

Speedup formulas:

- `cold_eval_speedup_vs_cpu = cpu.eval_ms / eval_ms(mode)`
- `steady_eval_speedup_vs_cpu = cpu.steady_eval_ms / steady_eval_ms(mode)`
- `cold_total_speedup_vs_cpu = cpu.cold_total_ms / cold_total_ms(mode)`
- `warm_total_proxy_speedup_vs_cpu = cpu.warm_total_proxy_ms / warm_total_proxy_ms(mode)`

### Interpretation Rules

- The fixed-pop benchmark uses `evolve_cli` generation-0 timing arrays plus `meta.timing.gpu_eval_init_ms`; it does not rely on a separate benchmark-only CLI.
- `skip-final-eval` must be enabled so that a post-loop final scoring pass does not contaminate the one-generation benchmark.
- `generation_total_ms[0]` is the generation wall time before adding any cold GPU eval init tax.
- `gpu_repro_overlap` is allowed to subtract both `gpu_eval_init_ms` and `repro_setup_ms`.
- Do not subtract `hidden_overlap_ms` again; that would double count hidden work.
- `warm_total_proxy_ms` must be labeled as a `proxy`, not as a full closed-loop per-generation cost.
- The primary benchmark claims should be based on:
  - `steady_eval_speedup_vs_cpu`
  - `warm_total_proxy_speedup_vs_cpu`
- `cold_*` metrics should still be reported, but they are startup-sensitive and should not be the headline result.

### Additional Breakdown To Report

- `gpu_eval_kernel_ms`
- `gpu_eval_pack_upload_ms`
- `repro_ms`
- `repro_prepare_inputs_ms`
- `repro_decode_ms`
- `repro_kernel_ms`
- `hidden_overlap_ms`

### Plots

- `population_size vs steady_eval_speedup_vs_cpu`
- `population_size vs warm_total_proxy_speedup_vs_cpu`
- `population_size vs warm_total_proxy_ms`
- `population_size vs cold_total_ms`
- Reproduction breakdown stacked bar for `gpu_repro` and `gpu_repro_overlap`

## Experiment 2: Closed-Loop Evolution Effectiveness

### Goal

Compare real evolution quality and efficiency under closed-loop runs without requiring trajectory identity under GPU reproduction.

### Tooling

- CLI: `g3pvm_evolve_cli --out-json --timing all`
- Modes:
  - `cpu`
  - `gpu_eval`
  - `gpu_repro`
  - `gpu_repro_overlap`

### Tasks

- `data/fixtures/simple_exp_1024.json`
- `data/fixtures/solve_boolean_1024.json`
- `data/fixtures/middle_character_1024.json`

### Rationale

- `simple_exp_1024`: numeric / non-binary task
- `solve_boolean_1024`: binary / boolean task
- `middle_character_1024`: string / payload task

### Sampling Plan

- Use at least `10` matched seeds, recommended `0..9`.
- Run `40` generations per seed and per mode.
- Keep all non-mode hyperparameters fixed across tasks and modes.

### Output Paths

- Raw CLI logs:
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/raw/<task>/<mode>/seed_<seed>.console.log`
- Native evolution JSON:
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/raw/<task>/<mode>/seed_<seed>.run.json`
- Parsed per-task reports:
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/reports/<task>.summary.json`
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/reports/<task>.summary.md`
- Top-level experiment summary:
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/reports/summary.json`
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/reports/summary.md`
- Figures:
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/plots/<task>_best_fitness_vs_generation.png`
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/plots/<task>_mean_fitness_vs_generation.png`
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/plots/<task>_final_best_fitness_boxplot.png`
  - `logs/exp2_closed_loop_effectiveness_<timestamp>/plots/<task>_quality_vs_cumulative_time.png`

### Comparison Rules

- `cpu vs gpu_eval`: can be used as a parity / sanity-check comparison.
- `gpu_eval vs gpu_repro vs gpu_repro_overlap`: compare result distributions, not per-generation identity.

### Metrics

- `final_best_fitness`
- `final_best_fitness - gen0_best_fitness`
- `final_mean_fitness - gen0_mean_fitness`
- `AUC(history_best_fitness)`
- `AUC(history_mean_fitness)`
- `avg(generation_total_ms[1:])`
- `avg(generation_eval_ms[1:])`
- `avg(generation_repro_ms[1:])`

### Reporting Views

- Fixed generations:
  - compare final quality and total runtime
- Fixed wall-clock:
  - compare `quality-at-time`

### Plots

- `best_fitness vs generation`
- `mean_fitness vs generation`
- `final_best_fitness` distribution by mode
- `quality vs cumulative time`

## Experiment 3: Current GPU Scaling, Reproduction Cost, And Boundary

### Goal

Characterize the current production GPU path rather than legacy kernel-bucket behavior:

- how GPU eval scales with fixed program shape
- where GPU reproduction time is currently spent
- where the real stability boundary shows up under fixed shape and growing workload

### Rationale

This experiment should follow the current implementation, not the old bucket-dispatch mental model:

- production GPU eval now runs a single `Mixed` kernel family for the accepted population
- payload flavor is still useful as metadata, but it is no longer the main experimental axis
- the repo already contains exact-depth and exact-node fixed populations under `data/exp/`

The main design principle is:

- use existing fixed populations first
- use `g3pvm_evolve_cli --generations 1 --population-json ... --skip-final-eval on` as the canonical measurement unit
- only synthesize new stress populations when the existing fixed sets are insufficient

### Inputs

Primary fixture:

- `data/fixtures/simple_x_plus_1_1024.json`

Primary fixed-population sources:

- `data/exp/depth_simple_x_plus_1_1024/`
- `data/exp/node_simple_x_plus_1_1024/`

Optional stress-population sources:

- a narrow helper that emits `population-seeds-v1`
- direct synthesis of larger fixed populations that preserve the selected depth or node bucket

### Canonical Unit Run

- `g3pvm_evolve_cli --generations 1 --population-json ... --skip-final-eval on --timing all --out-json ...`

Use the same derived fixed-pop metrics as Experiment 1:

- `compile_ms`
- `eval_ms`
- `steady_eval_ms`
- `cold_total_ms`
- `warm_total_proxy_ms`

### Replication Policy

- If a bucket directory already contains multiple fixed populations, treat those files as matched replicates.
- If only one fixed population exists for a bucket, either:
  - accept it as a single-cell case study, or
  - synthesize additional fixed populations for that same bucket before treating it as a statistical sweep
- For repeated timings on the same fixed population, use at least `3` repeats and aggregate with the median.
- When multiple fixed populations exist for the same bucket, report `median + IQR` across those fixed populations.

### Experiment 3A: Eval Scaling Sweep

Goal:

- measure how the current `gpu_eval` path responds to fixed program shape

Sweep axes:

- exact depth buckets from `data/exp/depth_simple_x_plus_1_1024/`
- exact node-count buckets from `data/exp/node_simple_x_plus_1_1024/`

Recommended bucket values:

- depth: `5 / 7 / 9 / 11 / 13 / 15` if available
- node count: `20 / 30 / 40 / 50 / 60 / 70` if available

Modes:

- `cpu`
- `gpu_eval`

Primary metrics:

- `eval_ms`
- `steady_eval_ms`
- `gpu_eval_init_ms`
- `gpu_eval_call_ms`
- `gpu_eval_pack_upload_ms`
- `gpu_eval_kernel_ms`
- `gpu_eval_teardown_ms`
- `steady_eval_speedup_vs_cpu`

Interpretation rules:

- treat `gpu_eval_init_ms` as startup cost, not as the scaling signal
- treat `gpu_eval_call_ms` and `gpu_eval_kernel_ms` as the main steady-state GPU eval metrics
- use exact depth and exact node count as the primary explanatory variables
- do not reconstruct or report legacy payload-family buckets as if they were production dispatch buckets

### Experiment 3B: Reproduction Cost Sweep

Goal:

- measure where GPU reproduction cost lives for the same fixed populations

Sweep axes:

- reuse the same exact-depth and exact-node fixed populations from Experiment 3A

Modes:

- `gpu_eval`
- `gpu_repro`
- `gpu_repro_overlap`

Primary metrics:

- `repro_ms`
- `repro_prepare_inputs_ms`
- `repro_setup_ms`
- `repro_preprocess_ms`
- `repro_pack_ms`
- `repro_upload_ms`
- `repro_selection_kernel_ms`
- `repro_variation_kernel_ms`
- `repro_copyback_ms`
- `repro_decode_ms`
- `hidden_overlap_ms`
- `warm_total_proxy_ms`

Interpretation rules:

- compare `gpu_repro` and `gpu_repro_overlap` against `gpu_eval` on the same fixed populations
- for overlap mode, do not expect the sum of reproduction subphases to equal visible `repro_ms`
- use this sweep to determine whether the present bottleneck is:
  - host preprocessing / packing
  - device kernels
  - copyback
  - decode
- use `warm_total_proxy_ms` as the main end-to-end proxy when comparing `gpu_repro` and `gpu_repro_overlap`

### Experiment 3C: Boundary Stress Sweep

Goal:

- locate the real stability boundary of the current GPU path

Method:

- choose one or two heaviest stable fixed populations from Experiments 3A and 3B
- hold program shape fixed
- increase `population_size` until one of the following happens:
  - hard failure
  - obvious resource saturation
  - clear performance cliff

Required stress axis:

- `population_size` growth under fixed shape

Optional targeted stress axes:

- high-name-count populations
- high-const-count populations

These are preferred over payload-flavor buckets because they map more directly to current bounded GPU buffers, remap cost, and decode cost.

Payload flavor:

- record it if metadata is already available
- do not use it as a primary sweep axis for the main boundary study

Boundary definitions:

- `hard failure`: non-zero exit, CUDA/runtime error, invalid JSON output, or correctness failure
- `performance cliff`: a size step whose wall time jumps much more than the neighboring trend would predict
- `last stable configuration`: largest configuration without hard failure and without a severe cliff

### Output Paths

- Fixed or stress population files:
  - `logs/exp3_gpu_resource_boundary_<timestamp>/populations/<population_label>/population.seeds.json`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/populations/<population_label>/metadata.json`
- Raw CLI logs:
  - `logs/exp3_gpu_resource_boundary_<timestamp>/raw/<population_label>/<mode>/repeat_<index>.console.log`
- Native one-generation JSON:
  - `logs/exp3_gpu_resource_boundary_<timestamp>/raw/<population_label>/<mode>/repeat_<index>.run.json`
- Parsed per-run metrics:
  - `logs/exp3_gpu_resource_boundary_<timestamp>/raw/<population_label>/<mode>/repeat_<index>.metrics.json`
- Aggregated reports:
  - `logs/exp3_gpu_resource_boundary_<timestamp>/reports/scaling_summary.json`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/reports/scaling_summary.md`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/reports/boundary_summary.json`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/reports/boundary_summary.md`
- Figures:
  - `logs/exp3_gpu_resource_boundary_<timestamp>/plots/eval_scaling_vs_depth.png`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/plots/eval_scaling_vs_nodes.png`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/plots/repro_breakdown_vs_depth.png`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/plots/repro_breakdown_vs_nodes.png`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/plots/population_size_stress_curve.png`
  - `logs/exp3_gpu_resource_boundary_<timestamp>/plots/failure_boundary_heatmap.png`

### Metrics

- success / failure kind
- first visibly failing phase
- last stable configuration
- `gpu_eval_init_ms`
- `gpu_eval_call_ms`
- `gpu_eval_pack_upload_ms`
- `gpu_eval_kernel_ms`
- `gpu_eval_teardown_ms`
- `repro_prepare_inputs_ms`
- `repro_setup_ms`
- `repro_preprocess_ms`
- `repro_pack_ms`
- `repro_selection_kernel_ms`
- `repro_variation_kernel_ms`
- `repro_copyback_ms`
- `repro_decode_ms`
- `hidden_overlap_ms`

### Outputs

- eval-scaling summary by exact depth and exact node count
- reproduction-cost summary by exact depth and exact node count
- resource-boundary table by stress axis
- failure-mode summary with the first failing phase called out explicitly

## Execution Order

1. Run Experiment 1 to establish fixed-workload adjusted speedup baselines.
2. Run Experiment 2 to compare closed-loop search quality and efficiency.
3. Run Experiment 3 to locate GPU resource cliffs and failure modes.
