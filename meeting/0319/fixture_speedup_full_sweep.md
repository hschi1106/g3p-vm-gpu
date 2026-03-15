# 2026-03-19 Fixture Speedup Full Sweep

## Scope

- Log dir: `/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260314_215409`
- Summary:
  - [summary.md](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260314_215409/summary.md)
  - [summary.json](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260314_215409/summary.json)
- Fixtures: `bouncing_balls_1024`, `gcd_1024`, `middle_character_1024`, `simple_affine_2x_plus_3_1024`, `simple_exp_1024`, `simple_square_x2_1024`, `simple_x_plus_1_1024`, `solve_boolean_1024`
- Modes: `cpu`, `gpu_eval`, `gpu_repro`, `gpu_repro_overlap`
- Population sizes: `1024`, `2048`, `4096`, `8192`, `16384`
- Shared settings:
  - `blocksize=1024`
  - `max_expr_depth=7`
  - `probe_cases=32`
  - `min_success_rate=0.5`
  - `fuel=20000`
  - `selection_pressure=3`

Total benchmark count: `8 fixtures x 5 population sizes x 4 modes = 160 runs`

## Overall Result

Average speedup vs `cpu`:

| Mode | Avg eval speedup | Avg repro speedup | Avg total speedup |
| --- | ---: | ---: | ---: |
| `gpu_eval` | `62.176x` | `1.035x` | `38.759x` |
| `gpu_repro` | `65.354x` | `1.489x` | `43.590x` |
| `gpu_repro_overlap` | `65.559x` | `2.630x` | `45.415x` |

Main takeaways:

1. `gpu_eval` already gives very large gains, but reproduction stays close to CPU.
2. `gpu_repro` lifts total speedup by moving reproduction work onto GPU.
3. `gpu_repro_overlap` is the best overall mode. The main gain comes from hiding part of the reproduction tail, not from materially higher eval throughput.

## Population Scaling

Average speedup across all 8 fixtures:

| Population | `gpu_eval` eval | `gpu_eval` total | `gpu_repro` eval | `gpu_repro` total | `gpu_repro_overlap` eval | `gpu_repro_overlap` total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | `23.683x` | `20.804x` | `29.324x` | `25.341x` | `29.431x` | `25.819x` |
| `2048` | `36.506x` | `28.924x` | `45.087x` | `35.520x` | `44.730x` | `35.961x` |
| `4096` | `56.800x` | `38.952x` | `59.584x` | `42.587x` | `59.749x` | `44.136x` |
| `8192` | `82.743x` | `48.303x` | `82.238x` | `52.283x` | `83.319x` | `55.131x` |
| `16384` | `111.151x` | `56.814x` | `110.536x` | `62.219x` | `110.564x` | `66.028x` |

Scaling observations:

1. All GPU modes scale strongly with population size.
2. Eval speedup is already very close between `gpu_repro` and `gpu_repro_overlap` at large populations.
3. The main differentiator at large populations is total time, where overlap continues to pull ahead.
4. At `16384`, `gpu_repro_overlap` reaches the best average total speedup: `66.028x`.

## Representative Fixtures

### `simple_x_plus_1_1024`

- `pop=1024`
  - `gpu_eval`: `eval 23.089x`, `total 20.300x`
  - `gpu_repro`: `eval 30.602x`, `total 26.007x`
  - `gpu_repro_overlap`: `eval 27.970x`, `total 24.574x`
- `pop=16384`
  - `gpu_eval`: `eval 109.022x`, `total 57.349x`
  - `gpu_repro`: `eval 112.993x`, `total 63.619x`
  - `gpu_repro_overlap`: `eval 114.779x`, `total 67.968x`

### `bouncing_balls_1024`

- `pop=1024`
  - `gpu_eval`: `eval 24.276x`, `total 21.196x`
  - `gpu_repro`: `eval 28.001x`, `total 24.254x`
  - `gpu_repro_overlap`: `eval 26.950x`, `total 23.784x`
- `pop=16384`
  - `gpu_eval`: `eval 121.524x`, `total 59.363x`
  - `gpu_repro`: `eval 117.023x`, `total 62.603x`
  - `gpu_repro_overlap`: `eval 116.585x`, `total 67.947x`

### `middle_character_1024`

- `pop=1024`
  - `gpu_eval`: `eval 25.653x`, `total 22.250x`
  - `gpu_repro`: `eval 30.334x`, `total 26.536x`
  - `gpu_repro_overlap`: `eval 33.297x`, `total 29.045x`
- `pop=16384`
  - `gpu_eval`: `eval 104.129x`, `total 52.573x`
  - `gpu_repro`: `eval 106.533x`, `total 61.128x`
  - `gpu_repro_overlap`: `eval 107.283x`, `total 64.700x`

## Decision Summary

1. `gpu_repro_overlap` is the best production benchmark mode when optimizing total wall time.
2. `gpu_repro` is still clearly worthwhile even without overlap.
3. At large populations, reproduction and overlap matter more to total speedup than small differences in eval speedup.
4. Future optimization should continue to focus on the reproduction tail, especially decode and overlap interaction, rather than evaluation throughput alone.
