# GAGP Operational Toolchain

The tools are a standard-library Python package organized by responsibility:

```text
PSB upstream JSONL
  -> psb fetch
  -> psb convert / psb materialize
  -> fitness-cases fixtures and support manifests
  -> grammar profile (when compatibility shaping is needed)
  -> benchmark population-seeds (for fixed-population runs)
  -> psb run -> gagp_evolve_cli
  -> psb compare
  -> report psb-manifest / report simple-manifest
  -> reviewed compact evidence under benchmarks/
```

Install an editable command in an isolated environment:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/pip install -e tools
.venv-tools/bin/gagp-tools --help
```

The historical `tools/*.py` paths remain thin compatibility wrappers during
the migration. They and the unified command execute the same package functions.

## Dataset commands

```bash
gagp-tools psb fetch --suite psb1 --problems count-odds --dry-run
gagp-tools psb convert --suite psb1 --problem count-odds --out /tmp/count-odds.train.json
gagp-tools psb materialize --suite psb1 --datasets-root data/psb1_datasets --out-dir /tmp/psb1
```

Dataset commands own acquisition and conversion only. They emit typed
`fitness-cases` fixtures; they do not implement runtime semantics.

## Experiment commands

```bash
gagp-tools grammar profile --help
gagp-tools benchmark population-seeds \
  --cases data/fixtures/simple_exp_1024.json --count 1024 \
  --out logs/fixed_population.seeds.json
gagp-tools psb run --suite psb1 --cases-root data/fixtures/psb1 --dry-run
```

Experiment execution records cases/config hashes, seeds, native binary path,
backend choices, and run status needed for replay.

## Report commands

```bash
gagp-tools psb compare --baseline baseline.json --candidate candidate.json
gagp-tools report psb-manifest --help
gagp-tools report simple-manifest --help
```

Comparisons reject incompatible run metadata before computing ratios. Report
commands compact reviewed raw runs into versioned evidence manifests.

## Artifact policy

- `data/psb*_datasets/`: mirrored upstream source data
- `data/fixtures/`: materialized, runtime-consumable fixtures
- `logs/`: ignored raw runs, generated configs, and local reports
- `benchmarks/`: intentionally committed compact evidence and freeze manifests
- `configs/`: versioned input policies, not generated run output

Run the independent suite without the retired runtime package:

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
```
