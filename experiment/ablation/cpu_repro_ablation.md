# CPU Reproduction Ablation Plan

## Goal

Identify which GPU reproduction behavior most likely explains the observed
`gpu_repro_overlap` advantage over `gpu_eval` in PSB1 closed-loop
`mean_fitness` curves.

This experiment keeps GPU fitness evaluation fixed and changes only the CPU
reproduction path through explicit ablation modes.

## Motivation

Existing PSB1 runs under `logs/psb1_pop1024` through `logs/psb1_pop16384`
show that `gpu_repro_overlap` generally improves `history.mean_fitness`
relative to `gpu_eval`.

The first ablation round focuses on the most likely operator-level causes:

- `3`: GPU-style selection / permutation / pairing
- `6`: GPU-style bounded typed crossover candidate tables
- `78`: GPU-style subtree mutation target coupling plus type-bucket donor pool

The `7` and `8` effects are intentionally grouped because the current GPU
reproduction implementation applies subtree mutation by replacing the selected
crossover site with a donor-pool subtree. Splitting those two behaviors would
create a less natural experimental operator.

## Experimental Modes

Run these five modes:

- `baseline`: `--engine gpu --repro-backend cpu`
- `full`: `--engine gpu --repro-backend gpu --repro-overlap on`
- `cpu+3`: `--engine gpu --repro-backend cpu --cpu-repro-ablation gpu_selection`
- `cpu+6`: `--engine gpu --repro-backend cpu --cpu-repro-ablation gpu_candidates`
- `cpu+78`: `--engine gpu --repro-backend cpu --cpu-repro-ablation gpu_coupled_donor`

The `cpu+78` report label is a compact shorthand. The CLI value is
`gpu_coupled_donor`.

## Proposed CLI Contract

Add an experimental flag to `g3pvm_evolve_cli`:

```bash
--cpu-repro-ablation {none,gpu_selection,gpu_candidates,gpu_coupled_donor}
```

Rules:

- default: `none`
- valid only with `--repro-backend cpu`
- rejected with `--repro-backend gpu`
- emitted in output JSON as `meta.cpu_repro_ablation`

This flag is experimental. It should not change the public definitions of
selection, crossover, or mutation.

## Ablation Semantics

### `gpu_selection`

Keep CPU crossover and CPU mutation unchanged.

Change only parent selection and pairing to host-emulate the GPU reproduction
selection model:

- use GPU-style deterministic per-round permutation
- use tournament chunks controlled by `selection_pressure`
- use GPU-style parent slot ordering
- pair adjacent parent slots directly, as the GPU kernel does

This isolates the effect of selection RNG, ordering, tie behavior, and mating
pair structure.

### `gpu_candidates`

Keep CPU selection and CPU mutation unchanged.

Change only typed crossover site selection to use GPU-style bounded candidate
tables:

- build subtree end positions for each parent
- collect typed expression roots
- filter roots through active `grammar-config-v1`
- keep at most `16` candidates per program using the same deterministic
  spacing as GPU reproduction preprocess
- choose the common result type and candidate pair from those bounded tables

The crossover should still produce normal `ProgramGenome` children on the host.

### `gpu_coupled_donor`

Keep CPU selection unchanged.

Change the host variation step to emulate the GPU coupling between typed
crossover and subtree mutation:

- first choose a typed crossover site pair for the selected parents
- for a child with no mutation, construct the normal typed-crossover child by
  replacing the selected site with the opposite parent's selected subtree
- for a child with constant mutation, construct the same normal typed-crossover
  child, then apply CPU-style constant perturbation
- for a child with subtree mutation, use the selected crossover site as the
  mutation target and replace it directly with a donor from a type-bucketed
  donor pool instead of the opposite parent's selected subtree
- build the donor pool once per generation using the active grammar config

This is the host-side analogue of the current GPU reproduction `7+8` behavior:
subtree mutation is coupled to the crossover site, and its replacement subtree
comes from the donor pool.

If no valid typed crossover site pair exists, or the donor replacement cannot
produce a valid child, use the same parent-fallback policy as the GPU
reproduction decode path. Do not change the public crossover API contract.

## First Run Matrix

Task:

- `data/fixtures/psb1/median.train.json`

Grammar:

- `configs/grammar/scalar.json`

Run settings:

- population size: `8192`
- generations: `20`
- seeds: `0..9`
- engine: `gpu`
- blocksize: `1024`
- mutation rate: CLI default unless explicitly overridden
- mutation subtree probability: CLI default unless explicitly overridden
- selection pressure: CLI default unless explicitly overridden
- shape limits: CLI defaults unless explicitly overridden

Rationale:

- existing `pop8192/median` data shows a clear `mean_fitness` gap within the
  first 20 generations
- `20` generations is intended as a screening run, not a final-quality claim
- matched seeds are required across all modes

## Primary Metrics

Use `history.mean_fitness` as the primary outcome:

- `AUC(history_mean_fitness[0:20])`
- `gen19_mean_fitness`
- `gen19_mean_fitness - gen0_mean_fitness`
- per-generation mean fitness curve

Secondary metrics:

- `history.best_fitness`
- `final.best_fitness`
- average generation wall time
- average reproduction time

Do not treat best-fitness metrics as the main conclusion for this ablation.
The observed effect is primarily a mean-fitness effect.

## Output Layout

Write one timestamped root per run:

```text
logs/ablation_cpu_repro_median_pop8192_gen20_<timestamp>/
  raw/<mode>/seed_<seed>.json
  raw/<mode>/seed_<seed>.stdout.txt
  reports/summary.json
  reports/summary.md
  plots/mean_fitness_vs_generation.png
  plots/auc_mean_fitness_boxplot.png
```

Mode directory names:

- `baseline`
- `full`
- `cpu_gpu_selection`
- `cpu_gpu_candidates`
- `cpu_gpu_coupled_donor`

Report display labels:

- `baseline`
- `full`
- `cpu+3`
- `cpu+6`
- `cpu+78`

## Interpretation

Compare each ablation against `baseline` and `full`.

Expected readout:

- if `cpu+3` moves toward `full`, selection / pairing is a likely driver
- if `cpu+6` moves toward `full`, bounded crossover candidate sampling is a
  likely driver
- if `cpu+78` moves toward `full`, GPU-style coupled donor variation is a
  likely driver
- if none move much alone, run a second-stage combined mode such as
  `gpu_selection + gpu_candidates + gpu_coupled_donor`

Only promote a finding after checking at least the matched-seed distribution of
`AUC(history_mean_fitness[0:20])`.
