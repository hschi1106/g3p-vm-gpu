# Native CLI Reference

This table is mechanically checked against `CliOptions` and the parser in
`cpp/include/g3pvm/cli/options.hpp` and `cpp/src/cli/options.cpp`. Update all
three together when a flag or default changes.

`g3pvm_evolve_cli` requires `--cases`. It supports evolution by default and
switches to one-AST evaluation when `--eval-ast-json` is supplied. `--help` is
not supported; it follows the unknown-argument error contract.

| Flag | Field | Default | Meaning |
| --- | --- | --- | --- |
| `--cases` | `cases_path` | `required` | Fitness-case input path |
| `--population-json` | `population_json` | `unset` | Fixed `population-seeds` replay path |
| `--grammar-config` | `grammar_config_path` | `unset` | Evolution search-space config path |
| `--eval-ast-json` | `eval_ast_json` | `unset` | Evaluate one materialized AST on CPU |
| `--engine` | `engine` | `cpu` | Fitness backend: `cpu` or `gpu` |
| `--repro-backend` | `repro_backend` | `cpu` | Reproduction backend: `cpu` or `gpu` |
| `--cpu-repro-ablation` | `cpu_repro_ablation` | `none` | CPU experiment mode: `none`, `gpu_selection`, `gpu_candidates`, or `gpu_coupled_donor` |
| `--repro-overlap` | `repro_overlap` | `off` | Overlap GPU reproduction preparation with GPU evaluation |
| `--skip-final-eval` | `skip_final_eval` | `off` | Skip the post-generation final scoring pass |
| `--retain-final-population` | `retain_final_population` | `off` | Materialize the full final scored population |
| `--blocksize` | `blocksize` | `1024` | CUDA evaluation block size |
| `--population-size` | `population_size` | `64` | Generated individuals per generation |
| `--generations` | `generations` | `40` | Evolution generations |
| `--mutation-rate` | `mutation_rate` | `0.5` | Per-child post-crossover mutation probability |
| `--mutation-subtree-prob` | `mutation_subtree_prob` | `0.8` | Subtree versus constant mutation probability |
| `--penalty` | `penalty` | `1.0` | Non-negative fitness penalty |
| `--selection-pressure` | `selection_pressure` | `2` | Round-based tournament size |
| `--seed` | `seed` | `0` | Deterministic RNG seed |
| `--fuel` | `fuel` | `20000` | Per-program execution budget |
| `--max-expr-depth` | `max_expr_depth` | `7` | Generated expression-depth limit |
| `--max-stmts-per-block` | `max_stmts_per_block` | `6` | Generated statements-per-block limit |
| `--max-total-nodes` | `max_total_nodes` | `80` | Generated AST node limit |
| `--max-for-k` | `max_for_k` | `16` | Largest constant seeded for generated loop bounds |
| `--max-call-args` | `max_call_args` | `3` | Generated builtin-call arity limit |
| `--show-program` | `show_program` | `none` | Program output: `none`, `ast`, `bytecode`, or `both` |
| `--timing` | `timing` | `summary` | Timing output: `none`, `summary`, `per_gen`, or `all` |
| `--out-json` | `out_json` | `unset` | Optional result JSON path |

Fixed-population timing should use `--population-json`, `--generations 1`,
`--skip-final-eval on`, and `--timing all`. See
[`../guides/benchmarking.md`](../guides/benchmarking.md) for the complete fair
comparison procedure.

