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

## Eval Time Decomposition

For this benchmark, GPU `eval_ms` is not just device execution. In the historical fixed-pop benchmark path used at that time, the GPU path timed:

- `session.init(...)`
- `session.eval_programs(...)`

and only then writes `eval_ms`.

Within `session.eval_programs(...)`, the reported GPU subphases are:

- `pack_upload_ms`
- `kernel_ms`
- `copyback_ms`

So the missing residual is:

- `session_init_ms = eval_ms - pack_upload_ms - kernel_ms - copyback_ms`

Depth-averaged decomposition:

| depth | eval ms | session init ms | pack/upload ms | kernel ms | copyback ms | session init share | kernel share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `144.609` | `107.140` | `3.344` | `31.876` | `0.026` | `74.9%` | `22.6%` |
| `7` | `146.044` | `115.573` | `2.673` | `27.531` | `0.026` | `79.7%` | `18.5%` |
| `9` | `166.284` | `135.534` | `2.237` | `26.351` | `0.027` | `82.7%` | `15.9%` |
| `11` | `233.054` | `205.599` | `1.846` | `24.647` | `0.027` | `88.5%` | `10.7%` |
| `13` | `353.602` | `324.669` | `2.000` | `28.266` | `0.027` | `91.3%` | `8.1%` |

Population-level depth correlations for the decomposed pieces:

- `corr(depth, session_init_ms) = 0.913`
- `corr(depth, non_kernel_ms) = 0.911`
- `corr(depth, kernel_ms) = -0.381`
- `corr(depth, pack_upload_ms) = -0.918`

Interpretation:

1. The strong `depth -> eval_ms` relationship is overwhelmingly a `depth -> session_init_ms` relationship.
2. `kernel_ms` does not rise with depth in the same way. In this synthetic grid it is roughly flat to weakly negative.
3. `pack_upload_ms` is small and actually trends downward with depth.
4. `copyback_ms` is negligible.

So if the design question is “should we split eval kernels by depth to reduce device execution time,” this experiment is weak evidence. If the question is “is depth a good predictor of benchmark-level GPU eval wall time,” the answer is yes, but mostly because `session.init(...)` gets more expensive for deeper populations.

### Steady-State View

For an actual evolution run, `session.init(...)` is paid once per session, not once per generation. So the more relevant recurring cost is:

- `steady_eval_ms = eval_ms - session_init_ms = pack_upload_ms + kernel_ms + copyback_ms`

Depth-averaged steady-state comparison:

| depth | steady eval ms | relative to depth 5 |
| --- | ---: | ---: |
| `5` | `35.247` | `1.000x` |
| `7` | `30.231` | `1.166x` |
| `9` | `28.615` | `1.232x` |
| `11` | `26.520` | `1.329x` |
| `13` | `30.293` | `1.164x` |

Interpretation:

1. Once `session_init` is removed, the depth trend mostly disappears.
2. In this synthetic grid, deeper buckets are not slower in steady-state GPU eval. `depth=7/9/11` are actually a bit faster than `depth=5`.
3. `depth=13` is only slightly slower than the best steady-state bucket and still faster than `depth=5`.
4. This makes the case for depth-based kernel bucketing even weaker. The main observed depth penalty in raw `eval_ms` is startup/setup cost, not recurring execution cost.

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
2. For `eval_ms`, depth is very strong, but most of that signal comes from `session.init(...)`, not from the eval kernel itself.
3. The correct short-term conclusion is:
   - keep payload specialization,
   - do not add depth-based kernel bucketing yet,
   - and use depth as analysis/heuristic information rather than a new production dispatch dimension.
