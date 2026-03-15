# 2026-03-19 Fixture Speedup Depth Sweep

## Scope

- Log dir: `/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_001623`
- Summary:
  - [summary.md](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_001623/summary.md)
  - [summary.json](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_001623/summary.json)
- Fixtures: `bouncing_balls_1024`, `gcd_1024`, `middle_character_1024`, `simple_affine_2x_plus_3_1024`, `simple_exp_1024`, `simple_square_x2_1024`, `simple_x_plus_1_1024`, `solve_boolean_1024`
- Modes: `cpu`, `gpu_eval`, `gpu_repro`, `gpu_repro_overlap`
- Population size: `1024`
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

Total benchmark count: `8 fixtures x 5 depths x 4 modes = 40 runs`

## Overall Result

Average speedup vs `cpu` across all depths:

| Mode | Avg eval speedup | Avg repro speedup | Avg total speedup |
| --- | ---: | ---: | ---: |
| `gpu_eval` | `18.101x` | `1.033x` | `15.779x` |
| `gpu_repro` | `20.780x` | `1.230x` | `17.921x` |
| `gpu_repro_overlap` | `21.408x` | `2.021x` | `18.686x` |

Main takeaways:

1. Speedup drops monotonically as `max_expr_depth` rises.
2. `gpu_repro_overlap` remains the best overall mode, but its edge shrinks at high depth.
3. At high depth, the dominant regression is in GPU evaluation throughput rather than reproduction.

## Depth Scaling

Average speedup across all 8 fixtures:

| Depth | `gpu_eval` eval | `gpu_eval` total | `gpu_repro` eval | `gpu_repro` total | `gpu_repro_overlap` eval | `gpu_repro_overlap` total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `35.312x` | `29.962x` | `42.714x` | `35.664x` | `46.022x` | `38.913x` |
| `7` | `23.413x` | `20.494x` | `28.471x` | `24.573x` | `27.986x` | `24.597x` |
| `9` | `14.327x` | `12.768x` | `15.619x` | `13.907x` | `15.575x` | `14.027x` |
| `11` | `11.194x` | `10.073x` | `10.939x` | `9.916x` | `11.149x` | `10.188x` |
| `13` | `6.260x` | `5.598x` | `6.158x` | `5.547x` | `6.306x` | `5.703x` |

Scaling observations:

1. The large drop happens between `depth=5 -> 7` and again `7 -> 9`.
2. By `depth=11`, all three GPU modes are already close.
3. By `depth=13`, reproduction backend choice has only small impact on total speedup.
4. This suggests that at high depth, GPU eval becomes the primary limiter again.

## Representative Fixture: `simple_x_plus_1_1024`

Per-depth speedup:

| Depth | `gpu_eval` eval | `gpu_eval` total | `gpu_repro` eval | `gpu_repro` total | `gpu_repro_overlap` eval | `gpu_repro_overlap` total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `36.685x` | `31.127x` | `41.454x` | `34.952x` | `46.447x` | `39.482x` |
| `7` | `22.463x` | `19.722x` | `27.466x` | `23.601x` | `28.040x` | `24.517x` |
| `9` | `15.974x` | `14.227x` | `14.818x` | `13.392x` | `17.507x` | `15.753x` |
| `11` | `10.856x` | `9.824x` | `11.285x` | `10.180x` | `11.435x` | `10.414x` |
| `13` | `6.174x` | `5.538x` | `6.312x` | `5.669x` | `6.355x` | `5.740x` |

Detailed reports:

- [depth5 simple_x](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_001623/depth5/simple_x_plus_1_1024_pop1024/mode_compare.report.md)
- [depth13 simple_x](/home/hschi1106/g3p-vm-gpu/logs/fixture_speedup_20260315_001623/depth13/simple_x_plus_1_1024_pop1024/mode_compare.report.md)

Useful raw timing comparison:

- `depth=5`
  - CPU: `eval 6169.028 ms`, `total 6199.513 ms`
  - `gpu_repro_overlap`: `eval 132.818 ms`, `repro 7.927 ms`, `total 157.023 ms`
- `depth=13`
  - CPU: `eval 2165.076 ms`, `total 2182.067 ms`
  - `gpu_repro_overlap`: `eval 340.694 ms`, `repro 5.039 ms`, `total 380.162 ms`

Interpretation:

1. High depth does not make CPU slower here; CPU time actually drops because the generated accepted population changes.
2. GPU eval time rises sharply with depth on the same benchmark shape.
3. The speedup collapse is therefore mainly a GPU eval throughput problem, not a reproduction problem.

## Representative Fixture Trends

### `bouncing_balls_1024`

- `depth=5`
  - `gpu_eval`: `eval 36.752x`, `total 31.263x`
  - `gpu_repro`: `eval 42.594x`, `total 35.755x`
  - `gpu_repro_overlap`: `eval 46.545x`, `total 39.432x`
- `depth=13`
  - `gpu_eval`: `eval 6.329x`, `total 5.665x`
  - `gpu_repro`: `eval 6.222x`, `total 5.599x`
  - `gpu_repro_overlap`: `eval 6.216x`, `total 5.627x`

### `middle_character_1024`

- `depth=5`
  - `gpu_eval`: `eval 33.082x`, `total 28.051x`
  - `gpu_repro`: `eval 36.025x`, `total 31.041x`
  - `gpu_repro_overlap`: `eval 46.935x`, `total 39.356x`
- `depth=13`
  - `gpu_eval`: `eval 6.413x`, `total 5.666x`
  - `gpu_repro`: `eval 6.172x`, `total 5.553x`
  - `gpu_repro_overlap`: `eval 6.164x`, `total 5.584x`

## Decision Summary

1. For `pop=1024`, low depth remains the sweet spot for GPU acceleration.
2. With `max_total_nodes=80`, increasing `max_expr_depth` from `5` to `13` is already enough to collapse speedup from roughly `30-39x` total to roughly `5.5-5.7x`.
3. At high depth, optimizing reproduction further will have limited payoff until GPU eval throughput is improved.
4. A follow-up experiment should keep the same depth sweep but increase `max_total_nodes` only if the goal is to study larger-and-deeper programs rather than depth pressure alone.
