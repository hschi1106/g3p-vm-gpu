# Benchmarking

This guide owns supported fixed-population timing and run commands. Timing
field definitions remain in [`../reference/timing.md`](../reference/timing.md),
and formal study constraints remain in
[`experiment-protocol.md`](experiment-protocol.md).

## Fixed-Pop Benchmark Mode

The supported fixed-population benchmark workflow is:

```bash
python3 tools/make_population_seeds.py \
  --cases data/fixtures/simple_exp_1024.json \
  --count 1024 \
  --out logs/fixed_population.seeds.json

cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --population-json logs/fixed_population.seeds.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap off \
  --blocksize 1024 \
  --generations 1 \
  --skip-final-eval on \
  --timing all \
  --out-json logs/fixed_population.run.json
```

For fair comparisons:
- reuse the same `population-seeds` file across all modes
- generate seed files with `tools/make_population_seeds.py` so `limits`,
  `cases_path`, and optional grammar-config identity are recorded consistently
- compare generation-0 timing fields only
- treat `total_ms` as the primary wall-clock metric, then inspect subphases

Canonical timing names, scope boundaries, and CLI/JSON mappings are defined in [timing.md](../reference/timing.md).

Important timing interpretations:
- `generation_eval_ms` includes compile, scoring, canonicalization, and scored-population rebuild for that generation
- GPU evaluation detail is reported with the `gpu_eval_*` family, including `gpu_eval_init_ms`, `gpu_eval_call_ms`, `gpu_eval_pack_ms`, `gpu_eval_launch_prep_ms`, `gpu_eval_upload_ms`, `gpu_eval_pack_upload_ms`, `gpu_eval_kernel_ms`, `gpu_eval_copyback_ms`, and `gpu_eval_teardown_ms`
- reproduction detail is reported with the `repro_*` family, including selection/crossover/mutation plus the GPU backend phases `repro_prepare_inputs_ms`, `repro_setup_ms`, `repro_preprocess_ms`, `repro_pack_ms`, `repro_upload_ms`, `repro_kernel_ms`, `repro_copyback_ms`, `repro_decode_ms`, `repro_teardown_ms`, `repro_selection_kernel_ms`, and `repro_variation_kernel_ms`
- with `--repro-overlap on`, `repro_prepare_inputs_ms`, `repro_preprocess_ms`, and `repro_pack_ms` may be partially hidden behind GPU evaluation wall time

## Canonical Runbooks

### Evolution progress run

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --population-size 1024 \
  --generations 20 \
  --out-json logs/simple_exp_1024.run.json
```

### Fixed-population timing smoke

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --population-json logs/fixed_population.seeds.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --generations 1 \
  --skip-final-eval on \
  --timing all \
  --out-json logs/fixed_population.run.json
```

