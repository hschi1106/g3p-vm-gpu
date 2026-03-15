# 2026-03-19 Fixture Speedup Depth/Population Grid Sweep

## Scope

- Log dir: `/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_004923`
- Summary:
  - [summary.md](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_004923/summary.md)
  - [summary.json](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_004923/summary.json)
- Fixtures: `bouncing_balls_1024`, `gcd_1024`, `middle_character_1024`, `simple_affine_2x_plus_3_1024`, `simple_exp_1024`, `simple_square_x2_1024`, `simple_x_plus_1_1024`, `solve_boolean_1024`
- Modes: `cpu`, `gpu_eval`, `gpu_repro`, `gpu_repro_overlap`
- Population sizes: `1024`, `2048`, `4096`, `8192`, `16384`
- Depth sweep: `5`, `7`, `9`, `11`, `13`
- Shared settings:
  - `blocksize=1024`
  - `probe_cases=32`
  - `min_success_rate=0.5`
  - `fuel=20000`
  - `max_stmts_per_block=6`
  - `max_total_nodes=80`
  - `max_for_k=16`
  - `max_call_args=3`
  - `selection_pressure=3`

Total benchmark count: `8 fixtures x 5 depths x 5 population sizes x 4 modes = 200 runs`

## Overall Result

Average speedup vs `cpu` across the full grid:

| Mode | Avg eval speedup | Avg repro speedup | Avg total speedup |
| --- | ---: | ---: | ---: |
| `gpu_eval` | `43.887x` | `1.021x` | `27.533x` |
| `gpu_repro` | `46.005x` | `1.478x` | `30.511x` |
| `gpu_repro_overlap` | `46.124x` | `2.569x` | `31.722x` |

Main takeaways:

1. `gpu_repro_overlap` is still the best overall mode on average.
2. Population scaling remains strongly positive across the whole grid.
3. Depth scaling remains strongly negative; by `depth=11/13`, reproduction backend choice matters much less.
4. The new grid confirms the main bottleneck split:
   - large `pop` rewards GPU eval and overlap
   - large `depth` pushes the bottleneck back into eval throughput

## Population Scaling

Average speedup across all 8 fixtures and all 5 depths:

| Population | `gpu_eval` eval | `gpu_eval` total | `gpu_repro` eval | `gpu_repro` total | `gpu_repro_overlap` eval | `gpu_repro_overlap` total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | `17.391x` | `15.238x` | `21.397x` | `18.374x` | `21.062x` | `18.445x` |
| `2048` | `27.282x` | `21.709x` | `30.965x` | `24.687x` | `31.343x` | `25.353x` |
| `4096` | `41.506x` | `28.789x` | `44.775x` | `31.679x` | `44.538x` | `32.769x` |
| `8192` | `57.715x` | `33.853x` | `57.340x` | `36.315x` | `57.564x` | `38.032x` |
| `16384` | `75.540x` | `38.076x` | `75.549x` | `41.498x` | `76.113x` | `44.010x` |

Scaling observations:

1. Total speedup keeps improving as population rises, even after averaging over all depths.
2. The repro backend contributes more at large populations to `total` than to `eval`.
3. By `8192/16384`, `eval` speedup is already very close across the three GPU modes; remaining differences are mainly reproduction-tail effects.

## Depth Scaling

Average speedup across all 8 fixtures and all 5 populations:

| Depth | `gpu_eval` eval | `gpu_eval` total | `gpu_repro` eval | `gpu_repro` total | `gpu_repro_overlap` eval | `gpu_repro_overlap` total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `93.799x` | `50.491x` | `101.008x` | `58.657x` | `101.961x` | `62.596x` |
| `7` | `59.541x` | `37.265x` | `62.362x` | `41.725x` | `62.165x` | `43.232x` |
| `9` | `37.177x` | `26.837x` | `37.655x` | `28.561x` | `37.615x` | `29.053x` |
| `11` | `19.897x` | `15.743x` | `19.991x` | `16.200x` | `19.852x` | `16.265x` |
| `13` | `9.020x` | `7.331x` | `9.011x` | `7.410x` | `9.027x` | `7.461x` |

Depth observations:

1. The speedup collapse with depth persists even after population scaling is added.
2. The biggest drop is still from `depth=5 -> 7 -> 9`.
3. At `depth=13`, all three GPU modes are effectively tied; repro optimization becomes secondary.

## Focus Slice: `pop=16384`, `depth=13`

This is the specific corner you asked about. Average across all 8 fixtures:

| Mode | Avg eval speedup | Avg total speedup |
| --- | ---: | ---: |
| `gpu_eval` | `10.700x` | `8.237x` |
| `gpu_repro` | `10.705x` | `8.364x` |
| `gpu_repro_overlap` | `10.718x` | `8.411x` |

Average raw timings across all 8 fixtures:

| Mode | Avg eval ms | Avg repro ms | Avg total ms |
| --- | ---: | ---: | ---: |
| `cpu` | `38507.675` | `180.695` | `38816.964` |
| `gpu_eval` | `3600.319` | `178.248` | `4714.360` |
| `gpu_repro` | `3598.925` | `111.627` | `4642.852` |
| `gpu_repro_overlap` | `3594.434` | `62.522` | `4617.096` |

Interpretation:

1. GPU evaluation is still much faster than CPU, but it has already degraded to only about `10.7x` on average.
2. Reproduction still improves:
   - `gpu_eval -> gpu_repro` saves about `66.6 ms`
   - `gpu_repro -> gpu_repro_overlap` saves about `25.8 ms`
3. But these savings are small relative to `~3.6 s` of GPU eval and `~38.5 s` of CPU eval, so total speedup only moves from `8.237x` to `8.411x`.
4. This is the clearest evidence that `depth=13` is now eval-bound, not repro-bound.

Per-fixture total speedup at `pop=16384`, `depth=13`:

| Fixture | `gpu_eval` | `gpu_repro` | `gpu_repro_overlap` |
| --- | ---: | ---: | ---: |
| `bouncing_balls_1024` | `8.381x` | `8.556x` | `8.541x` |
| `gcd_1024` | `8.170x` | `8.301x` | `8.364x` |
| `middle_character_1024` | `7.923x` | `8.085x` | `8.182x` |
| `simple_affine_2x_plus_3_1024` | `8.363x` | `8.452x` | `8.628x` |
| `simple_exp_1024` | `8.347x` | `8.445x` | `8.505x` |
| `simple_square_x2_1024` | `8.293x` | `8.466x` | `8.416x` |
| `simple_x_plus_1_1024` | `8.393x` | `8.515x` | `8.546x` |
| `solve_boolean_1024` | `8.029x` | `8.095x` | `8.105x` |

The spread is very narrow. That reinforces the same point: this corner is dominated by deep-program eval cost rather than fixture-specific repro effects.

## Representative Case: `simple_x_plus_1_1024`

Strong corner, `depth=5`, `pop=16384`:

- [mode_compare.report.md](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_004923/depth5/simple_x_plus_1_1024_pop16384/mode_compare.report.md)
- CPU: `eval 60363.687 ms`, `repro 266.099 ms`, `total 60861.832 ms`
- `gpu_eval`: `eval 332.668 ms`, `repro 276.490 ms`, `total 843.673 ms`
- `gpu_repro`: `eval 330.401 ms`, `repro 191.716 ms`, `total 755.990 ms`
- `gpu_repro_overlap`: `eval 329.445 ms`, `repro 91.311 ms`, `total 693.765 ms`
- Speedup:
  - `gpu_eval total = 72.139x`
  - `gpu_repro total = 80.506x`
  - `gpu_repro_overlap total = 87.727x`

Weak corner, `depth=13`, `pop=16384`:

- [mode_compare.report.md](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_004923/depth13/simple_x_plus_1_1024_pop16384/mode_compare.report.md)
- CPU: `eval 38288.088 ms`, `repro 167.599 ms`, `total 38582.928 ms`
- `gpu_eval`: `eval 3521.822 ms`, `repro 167.120 ms`, `total 4596.889 ms`
- `gpu_repro`: `eval 3516.182 ms`, `repro 110.918 ms`, `total 4530.940 ms`
- `gpu_repro_overlap`: `eval 3514.752 ms`, `repro 60.382 ms`, `total 4514.665 ms`
- Speedup:
  - `gpu_eval total = 8.393x`
  - `gpu_repro total = 8.515x`
  - `gpu_repro_overlap total = 8.546x`

This one case already shows the whole pattern:

1. Repro optimization still works numerically.
2. But the savings are drowned out by deep-program eval time.
3. The ceiling is no longer reproduction; it is evaluation throughput on deep accepted populations.

## Decision Summary

1. `gpu_repro_overlap` remains the best production mode over the full depth/population grid.
2. Large populations continue to help a lot.
3. Large depths continue to hurt a lot, and by `depth=13` the system is largely eval-bound again.
4. The most important next optimization target is not more repro specialization; it is eval throughput on deep programs.
