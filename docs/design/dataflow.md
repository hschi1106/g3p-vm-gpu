# Native Dataflow

This document explains how inputs move through the maintained native system.
Normative AST, bytecode, fitness, and fixture behavior remains in
[`../../spec/`](../../spec/README.md).

## Command boundary

`g3pvm_evolve_cli` is a thin process boundary. The CLI support library parses
and validates options, loads JSON cases, grammar configuration, and optional
population seeds, then dispatches either AST evaluation or evolution. The
command layer writes stable console and JSON output; it does not own evolution
semantics.

```text
arguments + JSON inputs
        |
        v
CLI parser / codec / command dispatcher
        |
        v
CaseSet + GrammarConfig + EvolutionConfig
        |
        v
population initialization or fixed-population replay
```

External AST and bytecode inputs are verified before execution. Generated and
reproduced genomes are verified at their construction boundaries, so the
compiler and evaluators consume structurally valid, typed programs.

## Evaluation loop

`CaseSet` normalizes fixture bindings, expected values, input names, and input
types once. `initialize_population` either checks and replays an explicitly
loaded population or generates genomes using the active grammar and inferred
case schema.

For each generation:

```text
ProgramGenome population
        |
        v
compile_for_eval + bytecode verification
        |
        +---------------------+
        |                     |
        v                     v
CPU fitness evaluator    GPU FitnessSession
        |                     |
        +----------+----------+
                   v
        canonical fitness vector
                   |
                   v
       shared ranking and statistics
                   |
                   v
       CPU or GPU reproduction backend
                   |
                   v
     verified next-generation genomes
```

Both evaluators return the same population-shaped fitness vector. One shared
ranking path canonicalizes values, records timing, computes generation
statistics, and materializes scored genomes. This keeps evaluation backend
choice out of selection semantics.

The reproduction backend receives the ranked population and the same grammar
and shape limits. CPU reproduction performs selection, typed-subtree
crossover, and mutation on the host. GPU reproduction prepares typed candidate
metadata, packs it, performs device selection and variation, copies results
back, decodes them, and verifies the resulting genomes.

When GPU evaluation and GPU reproduction overlap are enabled, reproduction
input preparation starts while the current population is evaluated. Selection
still waits for the completed fitness vector; overlap changes scheduling, not
the ranking or operator contract. `PayloadLifetimeManager` retains container
values referenced by cases, active genomes, history, and final results across
these boundaries.

After the configured generations, the engine either evaluates and ranks the
final population or marks that pass skipped. `EvolutionResult` owns the best
genome, history, optional final population, and nested timing aggregates. The
CLI output adapter maps those structures to the stable flat JSON keys described
in [`../reference/timing.md`](../reference/timing.md).

## Operational artifact flow

The independent Python tool package orchestrates data and reports; it does not
implement runtime or evolution semantics.

```text
PSB source rows
  -> fetch -> convert -> materialize fixtures
  -> derive grammar profile / population seeds
  -> native CLI runs
  -> compare summaries
  -> compact committed manifest
```

Generated datasets, populations, per-run JSON, and profiler output stay in
artifact directories. Only compact manifests with provenance belong in
`benchmarks/`. See [`../../tools/README.md`](../../tools/README.md) for the
command surface and [`../guides/experiment-protocol.md`](../guides/experiment-protocol.md)
for experiment controls.

