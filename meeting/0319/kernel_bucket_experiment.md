# 2026-03-19 Eval Kernel Bucket Experiment

## Scope

- Log dir:
  - [summary.md](/home/hschi1106/g3p-vm-gpu/logs/kernel_bucket_grid_simple_x_plus_1_1024/summary.md)
  - [summary.json](/home/hschi1106/g3p-vm-gpu/logs/kernel_bucket_grid_simple_x_plus_1_1024/summary.json)
- Generated populations:
  - [data/exp/kernel_bucket_grid_simple_x_plus_1_1024](/home/hschi1106/g3p-vm-gpu/data/exp/kernel_bucket_grid_simple_x_plus_1_1024)
- Fixture:
  - `simple_x_plus_1_1024`
- Experiment type:
  - exact `actual_depth`
  - exact runtime `payload_flavor`
  - GPU-only eval benchmark
- Grid:
  - `depth = 5, 7, 9, 11, 13`
  - `payload = none, string, list, mixed`
  - `3` population replicates per cell
  - `3` benchmark repeats per population
- Population size:
  - `1024`
- Generator mode:
  - `synthetic`

Total populations: `5 depths x 4 payload flavors x 3 replicates = 60`

## Cell Summary

Median of replicate medians:

| depth | payload | kernel ms | eval ms | pack/upload ms | mean code len | mean node count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `5` | `none` | `36.045` | `149.149` | `3.382` | `8.680` | `11.680` |
| `5` | `string` | `29.201` | `144.918` | `3.311` | `11.983` | `14.983` |
| `5` | `list` | `31.856` | `142.161` | `3.375` | `12.023` | `15.023` |
| `5` | `mixed` | `30.401` | `142.209` | `3.310` | `13.490` | `16.490` |
| `7` | `none` | `25.971` | `149.888` | `2.658` | `11.967` | `14.967` |
| `7` | `string` | `27.111` | `137.994` | `2.635` | `16.995` | `19.995` |
| `7` | `list` | `25.469` | `146.894` | `2.704` | `16.957` | `19.957` |
| `7` | `mixed` | `31.574` | `149.398` | `2.696` | `23.527` | `26.527` |
| `9` | `none` | `22.978` | `165.080` | `2.172` | `15.374` | `18.374` |
| `9` | `string` | `25.104` | `166.047` | `2.289` | `22.002` | `25.002` |
| `9` | `list` | `32.442` | `177.335` | `2.252` | `22.010` | `25.010` |
| `9` | `mixed` | `24.880` | `156.675` | `2.234` | `33.484` | `36.484` |
| `11` | `none` | `24.616` | `224.067` | `1.769` | `18.653` | `21.653` |
| `11` | `string` | `23.105` | `236.725` | `1.876` | `26.997` | `29.997` |
| `11` | `list` | `25.875` | `228.753` | `1.883` | `26.995` | `29.995` |
| `11` | `mixed` | `24.993` | `242.671` | `1.856` | `43.557` | `46.557` |
| `13` | `none` | `24.606` | `349.331` | `2.042` | `22.035` | `25.035` |
| `13` | `string` | `30.757` | `350.597` | `1.889` | `31.979` | `34.979` |
| `13` | `list` | `24.451` | `353.872` | `2.006` | `32.053` | `35.053` |
| `13` | `mixed` | `33.250` | `360.608` | `2.063` | `53.526` | `56.526` |

## Feature Analysis

Using population-level medians across all `60` populations:

### Kernel Time

- `corr(depth, kernel_ms) = -0.255`
- `corr(mean_code_len, kernel_ms) = -0.005`
- `corr(mean_node_count, kernel_ms) = -0.005`

Grouped `R^2` for `kernel_ms`:

- `payload = 0.041`
- `depth = 0.098`
- `payload + depth = 0.382`
- `payload + code_len_bucket = 0.291`
- `payload + node_count_bucket = 0.291`

Interpretation:

1. `payload` alone explains very little of the observed kernel variance in this exact-depth synthetic grid.
2. `depth` alone is also weak.
3. `payload + depth` is the best of the tested coarse groupings, but it still only explains about `38%` of the variance.
4. This is not strong enough evidence to justify a new production dispatch tree that multiplies kernel count by depth buckets.

### Eval Wall Time

Additional population-level analysis on `eval_ms`:

- `corr(depth, eval_ms) = 0.899`
- `corr(mean_code_len, eval_ms) = 0.692`
- `corr(mean_node_count, eval_ms) = 0.692`

Grouped `R^2` for `eval_ms`:

- `payload = 0.001`
- `depth = 0.988`
- `payload + depth = 0.992`
- `payload + code_len_bucket = 0.795`
- `payload + node_count_bucket = 0.795`

Interpretation:

1. Depth is an extremely strong predictor of end-to-end GPU eval wall time in this grid.
2. Payload flavor contributes almost nothing once depth is known.
3. The strongest signal in this experiment is therefore not “which kernel family should exist”, but “how expensive a population is likely to be to evaluate”.

## Design Reading

This experiment does **not** say that the current payload split was a mistake.

The current production payload split solved a different problem:

1. It removed unnecessary payload thread state from programs that did not need it.
2. It materially reduced resource usage for `None/StringOnly/ListOnly` relative to the old monolithic payload kernel.
3. It already delivered large real speedups on production workloads.

What this experiment says is narrower:

1. `payload` is not a strong enough **second-stage performance bucket** by itself for these exact-depth synthetic populations.
2. `depth` is clearly related to overall eval cost, but not strongly enough to justify splitting production kernels by depth just for kernel-time reasons.
3. If depth is used at all, it should be used first as an analysis or launch hint, not as a new axis that explodes the number of specialized kernels.

## Recommendation

1. Keep the current payload-flavor specialization. It still has clear structural value.
2. Do **not** add `payload x depth` kernel specializations based on this experiment alone.
3. If we want to use depth operationally, use it as:
   - a profiling label,
   - a launch-policy hint,
   - or an eval-cost predictor.
4. The next better experiment would not be another larger dispatch grid. It would be a matched-population study that controls `code_len` and `node_count` more tightly to test whether depth still has independent predictive value.

## Bottom Line

1. For `kernel_ms`, depth is too weak to justify more kernel families.
2. For `eval_ms`, depth is very strong and should be treated as a cost signal.
3. The correct short-term conclusion is:
   - keep payload specialization,
   - do not add depth-based kernel bucketing yet,
   - and use depth as analysis/heuristic information rather than a new production dispatch dimension.
