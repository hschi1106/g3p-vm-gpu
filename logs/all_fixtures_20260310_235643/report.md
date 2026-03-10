# All Fixtures Report

Directory: `logs/all_fixtures_20260310_235643`

## Summary

- Checked 16 CPU/GPU fixture pairs.
- `best_fitness` history matched for all 16/16 pairs.
- `program_key` history matched for all 16/16 pairs.
- Final `best_fitness` matched for all 16/16 pairs.
- Final `program_key` matched for all 16/16 pairs.
- `stdout.log` was not byte-for-byte identical because timing output differed.
- All generation lines matched exactly, so `best`, `mean`, and `program_key` outputs were aligned.

## Speedup Definitions

- `inner_total`: `[inner_cpp_summary] total` from `timings.log`
- `eval_only`: `generations_eval_total + final_eval` from `timings.log`
- `end_to_end`: `run.total_elapsed_ms` from `summary.json`
- Speedup is reported as `cpu_ms / gpu_ms`

## Results

| Fixture | Popsize | Inner Total | Eval Only | End-to-End |
|---|---:|---:|---:|---:|
| `bouncing_balls_1024` | 1024 | 114.37x | 278.71x | 111.81x |
| `bouncing_balls_1024` | 2048 | 111.29x | 263.99x | 109.26x |
| `gcd_1024` | 1024 | 104.92x | 252.99x | 102.40x |
| `gcd_1024` | 2048 | 105.73x | 252.92x | 103.66x |
| `middle_character_1024` | 1024 | 101.78x | 240.68x | 99.39x |
| `middle_character_1024` | 2048 | 137.46x | 335.71x | 134.99x |
| `simple_affine_2x_plus_3_1024` | 1024 | 110.26x | 269.69x | 107.86x |
| `simple_affine_2x_plus_3_1024` | 2048 | 110.20x | 261.57x | 108.21x |
| `simple_exp_1024` | 1024 | 49.29x | 116.81x | 47.88x |
| `simple_exp_1024` | 2048 | 46.64x | 103.16x | 45.55x |
| `simple_square_x2_1024` | 1024 | 194.23x | 396.44x | 187.58x |
| `simple_square_x2_1024` | 2048 | 80.22x | 192.17x | 78.42x |
| `simple_x_plus_1_1024` | 1024 | 114.00x | 275.33x | 111.58x |
| `simple_x_plus_1_1024` | 2048 | 52.28x | 105.96x | 50.98x |
| `solve_boolean_1024` | 1024 | 95.93x | 227.89x | 93.57x |
| `solve_boolean_1024` | 2048 | 101.10x | 236.08x | 99.08x |

## Aggregate

- Average `inner_total` speedup: `101.86x`
- Average `eval_only` speedup: `238.13x`
- Average `end_to_end` speedup: `99.51x`

## Notes

- Lowest speedups in this batch were `simple_exp_1024` and `simple_x_plus_1_1024 pop2048`.
- Highest `inner_total` speedup was `simple_square_x2_1024 pop1024` at `194.23x`.
- Highest `eval_only` speedup was `simple_square_x2_1024 pop1024` at `396.44x`.

## Popsize Interpretation

Increasing `popsize` did not consistently increase speedup in this batch. The results split into three patterns:

- Speedup increased for some fixtures, such as `middle_character_1024`:
  `pop1024 101.78x -> pop2048 137.46x`
- Speedup stayed roughly flat for others, such as `bouncing_balls_1024`:
  `pop1024 114.37x -> pop2048 111.29x`
- Speedup dropped for some fixtures, such as:
  `simple_exp_1024 49.29x -> 46.64x`
  `simple_x_plus_1_1024 114.00x -> 52.28x`

This is expected for the current system:

- `pop1024` and `pop2048` are different evolution trajectories, so they do not execute the same average program mix.
- A larger population can evolve longer bytecode, deeper control flow, more expensive builtins, or fewer early exits.
- GPU execution is not only kernel time. Host-side work such as packing, compile/cache lookup, sorting, selection, mutation, crossover, and copy-back still contributes to `inner_total`.
- Once GPU occupancy is already high enough, doubling population size does not guarantee a proportional throughput gain.
- The gap between `eval_only` and `inner_total` confirms that non-kernel work remains significant in end-to-end runs.

Practical takeaway:

- Larger `popsize` increases total work, but does not guarantee higher CPU-over-GPU speedup.
- For this codebase, speedup depends on both the evolved program distribution and the share of host-side overhead, not only on GPU kernel utilization.

## Phase Appendix

To separate GPU evaluation throughput from host-side orchestration, the runs were also split into:

- `eval`: `generations_eval_total + final_eval`
- `non-eval`: `inner_total - eval`

The main pattern is consistent across all fixtures:

- `eval` speedup is large and highly variable across fixtures.
- `non-eval` speedup stays close to `1.0x`.

This means host-side work is not a GPU acceleration win today. Packing, compile/cache lookup, sorting, selection, crossover, mutation, and other orchestration stages cost roughly the same on both CPU and GPU runs. The overall speedup is therefore dominated by how much of the total run is spent inside fitness evaluation.

Representative breakdown:

| Fixture | Popsize | Eval Speedup | Non-Eval Speedup |
|---|---:|---:|---:|
| `bouncing_balls_1024` | 1024 | 278.71x | 0.96x |
| `bouncing_balls_1024` | 2048 | 263.99x | 0.98x |
| `middle_character_1024` | 1024 | 240.68x | 0.96x |
| `middle_character_1024` | 2048 | 335.71x | 0.97x |
| `simple_exp_1024` | 1024 | 116.81x | 0.97x |
| `simple_exp_1024` | 2048 | 103.16x | 0.98x |
| `simple_square_x2_1024` | 1024 | 396.44x | 0.96x |
| `simple_square_x2_1024` | 2048 | 192.17x | 0.98x |
| `simple_x_plus_1_1024` | 1024 | 275.33x | 0.96x |
| `simple_x_plus_1_1024` | 2048 | 105.96x | 0.98x |

Interpretation:

- When `pop2048` improves `eval` speedup, overall speedup tends to improve as well, as seen in `middle_character_1024`.
- When `pop2048` reduces `eval` speedup, overall speedup falls even if the host-side ratio stays almost unchanged, as seen in `simple_exp_1024` and `simple_x_plus_1_1024`.
- Since `non-eval` is near `1.0x`, increasing population size mostly changes the end result through the evolved workload mix inside evaluation, not through better amortization of host overhead.
