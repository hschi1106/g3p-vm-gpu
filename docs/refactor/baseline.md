# Refactor Baseline

Captured on 2026-08-31 (Asia/Taipei) at
`5f2b840724cfd06d3ec70c1515e7ea827d265e15` (`before goal`). Raw timing files
are intentionally local under `logs/`; their hashes below make the recorded
measurements auditable without committing large run output.

## Toolchain and hardware

| Component | Baseline |
| --- | --- |
| CMake | 3.22.1 |
| C++ | GCC 11.4.0 |
| CUDA compiler | 12.2.140 |
| NVIDIA driver | 595.84 |
| GPUs | 2 x NVIDIA GeForce RTX 4090, 24,564 MiB each |
| Build | `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug` |

The Debug build completed with CUDA architecture 89 and built all default
product, benchmark, probe, and test targets.

## Correctness baseline

| Gate | Command | Result |
| --- | --- | --- |
| Native suite | `ctest --test-dir cpp/build --output-on-failure` | 9/9 passed in 4.31 s |
| Python suite | `PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v` | 161/161 passed in 14.127 s |
| VM GPU smoke | CTest target `g3pvm_test_vm_gpu_smoke` | passed |
| CPU/GPU fitness parity | CTest target `g3pvm_test_fitness_cpu_gpu_parity` | passed |
| CPU/GPU evolution parity | CTest target `g3pvm_test_evolution_cpu_gpu_parity` | passed |

The Python fuzz-equivalence test reported 153 executable programs and 847
generation/compile rejections. That rejection rate is baseline evidence for
the planned native verifier and typed property tests, not a passing-program
rate target.

## Fixed-population timing baseline

The seed set was created with:

```text
python3 tools/make_population_seeds.py
  --cases data/fixtures/simple_exp_1024.json
  --count 1024
  --out logs/refactor_baseline.seeds.json
```

Each mode used the same seeds, Debug build, 1,024 fixture cases, population
1,024, one generation, `--skip-final-eval on`, and `--timing all`.
GPU measurements use block size 256; the CPU block-size field is inert.
Times are milliseconds and are single-run observations, suitable as a
regression alarm rather than a benchmark claim.

| Mode | Eval | Reproduction | Generation total | Compile | GPU call | GPU kernel | GPU init |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU eval / CPU repro | 21,524.731 | 52.198 | 21,581.430 | 22.309 | 0 | 0 | 0 |
| GPU eval / CPU repro | 183.057 | 46.825 | 232.244 | 19.700 | 157.180 | 123.819 | 447.355 |
| GPU eval / GPU repro | 224.862 | 81.218 | 308.534 | 19.558 | 199.476 | 159.992 | 427.627 |
| GPU eval / GPU repro overlap | 222.366 | 62.340 | 288.243 | 19.686 | 195.859 | 154.334 | 434.049 |

For GPU reproduction without overlap, the recorded prepare/preprocess/pack,
kernel, and decode times were 9.530/23.733/3.971, 7.335, and 19.895 ms. With
overlap they were 11.376/25.116/4.179, 14.000, and 22.083 ms.

The documented fixed-pop examples currently select `--blocksize 1024`. With
this seed set on the recorded RTX 4090, the first GPU run failed with
`cuda kernel execution failure launch=too many resources requested for launch`.
Documentation must not present that block size as universally valid.

Raw evidence hashes:

| Local artifact | SHA-256 |
| --- | --- |
| `logs/refactor_baseline.seeds.json` | `7e8dc4ba5e7823d5735a5a8f02a05a38b63085da232deee0ca95327c9cd4e2ab` |
| `logs/refactor_baseline.cpu.json` | `fa641f06e51c5561f2dbd1356f29ba58cdb6f700bd9e1a129a2384a0455db918` |
| `logs/refactor_baseline.gpu_eval.json` | `d33e861c977b2462a18e77b8c099668ed24a7da934909672854b6d4a50394c20` |
| `logs/refactor_baseline.gpu_repro.json` | `67f0741a628ee26130ec8254afdd1c65c288fc46e656d2e1f94bbaddcee3b399` |
| `logs/refactor_baseline.gpu_repro_overlap.json` | `1b379249405e5a2c3a2ce39cab4913dd2c003fc074507bb34b8e6faf9d323d7e` |

Re-capture rather than infer performance from a copied raw file if a local
artifact is absent.

## PSB quality baseline

The checked-in representative gate is
[`benchmarks/psb1_supported_compact_5seed_pop512.json`](../../benchmarks/psb1_supported_compact_5seed_pop512.json).
It records 28 supported PSB1 problems, five seeds, population 512, ten
generations, GPU evaluation/reproduction with overlap, and test-set evaluation.
Its status is `passed`: 140/140 baseline and 140/140 candidate runs succeeded,
with zero failed problems. This manifest remains the quality comparison anchor
until a stage deliberately re-captures an equivalent manifest.

## CLI baseline

The product CLI has no discoverability output: `g3pvm_evolve_cli --help`
returns exit failure with `unknown argument: --help`, and invoking it without
arguments returns `--cases is required`. At baseline, the actual parser and
defaults were in `cpp/src/cli/evolve_cli.cpp`. Stage 11 moved that interface
unchanged to `cpp/src/cli/options.cpp` and added
`g3pvm_test_cli_options` contract coverage.

Public flags at baseline are: `--cases`, `--population-json`,
`--grammar-config`, `--eval-ast-json`, `--engine`, `--repro-backend`,
`--cpu-repro-ablation`, `--repro-overlap`, `--skip-final-eval`,
`--retain-final-population`, `--blocksize`, `--population-size`,
`--generations`, `--mutation-rate`, `--mutation-subtree-prob`, `--penalty`,
`--selection-pressure`, `--seed`, `--fuel`, `--max-expr-depth`,
`--max-stmts-per-block`, `--max-total-nodes`, `--max-for-k`,
`--max-call-args`, `--show-program`, `--timing`, and `--out-json`.

Defaults that later CLI stages must preserve are CPU evaluation/reproduction,
overlap off, final eval retained, final population not retained, block size
1,024, population 64, generations 40, mutation rate 0.5, subtree probability
0.8, penalty 1.0, selection pressure 2, seed 0, fuel 20,000, limits
7/6/80/16/3, program display `none`, and timing `summary`.
