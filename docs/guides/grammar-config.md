# Grammar Config Guide

`grammar-config` restricts what evolution may generate or synthesize. It does
not change execution semantics and does not reject an already-materialized AST
or bytecode program solely because a construct is disabled. The complete schema
and validation rules are normative in
[`../../spec/grammar_config.md`](../../spec/grammar_config.md).

## Choose a checked preset

Presets live under `configs/grammar/`:

| Preset | Intended search space |
| --- | --- |
| `all.json` | All enabled source constructs |
| `scalar.json` | Numeric/boolean scalar programs |
| `string.json` | Scalar plus string-compatible constructs |
| `num_list.json` | Legacy numeric-list comparison profile, translated on load |
| `string_list.json` | Scalar plus string/string-list constructs |
| `sequence.json` | Broad sequence profile |

Pass one to the native CLI:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --grammar-config configs/grammar/scalar.json \
  --engine gpu --repro-backend gpu --repro-overlap on \
  --population-size 64 --generations 5
```

The selection affects initial generation, CPU mutation donors, GPU
reproduction candidate/donor preparation, and seed replay that regenerates
genomes. CPU/GPU runtime semantics and fitness remain unchanged.

## Compatibility profiles

Checked presets from the legacy search-space shape are accepted as comparison
inputs. The loader translates legacy numeric-list enablement into the explicit
typed-list search space and rejects unknown keys. This preserves benchmark
search-space comparability; it does not reintroduce legacy runtime values.

The tool package can derive deterministic profiles:

```bash
g3pvm-tools grammar profile \
  --profile compact \
  --base-grammar-config configs/grammar/num_list.json \
  --fixture-cases data/fixtures/psb1/count-odds.train.json \
  --out logs/count-odds.compact.json
```

- `compat` derives the native typed-list choices from the fixture schema.
- `compact` preserves the base profile's numeric-list search shape for direct
  comparisons and records its compatibility metadata.
- `full` emits the full native profile for experiments.

`g3pvm-tools psb run --profile compat|compact --base-grammar-config PATH`
creates per-problem configs under the run output directory and records base,
generated-config, and fixture-schema hashes.

## Seed replay

Population-seed artifacts should record the grammar-config path and hash. A
replay with recorded config identity requires matching content; an artifact
without config metadata uses the all-enabled default unless the caller supplies
`--grammar-config`.

Use the same cases, grammar config, limits, and population-seed file across
backend comparisons. The native CLI records the selected config identity in
its output JSON.

## Implementation and tests

- `cpp/include/g3pvm/evolution/grammar_config.hpp` and
  `cpp/src/evolution/grammar_config.cpp`: native configuration model/validation
- `cpp/src/evolution/genome_generation.cpp` and reproduction modules: search
  gating consumers
- `tools/g3pvm_tools/experiments/grammar_profiles.py`: profile generation
- `cpp/tests/evolution/test_genome_properties.cpp`: deterministic generation
  conformance
- `cpp/tests/evolution/test_repro_prep.cpp`: reproduction preprocessing
  conformance

