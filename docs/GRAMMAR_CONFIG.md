# Grammar Config

`grammar-config` controls which grammar constructs evolution is allowed to
generate. It is a search-space control, not a runtime language mode.

The interpreter, VM, CPU runtime, and GPU fitness runtime execute the public current
grammar. Grammar config only affects generated or regenerated programs.

## Compatibility Input

Checked-in presets under `configs/grammar/` are still `grammar-config` files
during the current migration. The native loader accepts those files as
compatibility input and translates them into current search-space controls:

- `values.none` is discarded because `None` is not a public current value.
- `values.num_list` enables both `values.int_list` and `values.float_list`.
- current-only `char` values, char builtins, `prepend`, `singleton`, and structured
  forms are disabled unless the input is a native current config.
- unknown keys are still rejected before translation.

This compatibility path exists for fair benchmark comparisons such as
`compact` versus the base grammar profile. It is not a runtime execution
mode and does not reintroduce public `None` or `NumList` values. Legacy
`fitness-cases` records with `type: "num_list"` are decoded as `IntList`
when all elements are integral and `FloatList` otherwise.

## Scope

Grammar config affects:

- random genome generation
- C++ CPU mutation donor synthesis
- C++ GPU reproduction candidate filtering and mutation donor pool generation
  for the implemented current source-form subset
- seed-population replay when replay regenerates genomes from seeds

Grammar config does not affect:

- execution of already-materialized AST programs
- bytecode decoding
- interpreter or VM semantics
- CPU/GPU fitness semantics
- loading existing programs that contain disabled constructs

## Presets

Checked-in presets live under `configs/grammar/`:

- `all.json`: all legacy profile constructs enabled; translated to current
  direct list values for native generation
- `scalar.json`: numeric / boolean scalar search space; sequence values and
  container builtins disabled
- `string.json`: scalar plus `String` and string-compatible builtins
- `num_list.json`: scalar plus legacy numeric-list profile; translated to
  `IntList` and `FloatList`
- `string_list.json`: scalar plus `String` / `StringList`
- `sequence.json`: broad legacy sequence profile

## Current Format

Every native current config file must explicitly define every known key.

```json
{
  "format_version": "grammar-config",
  "statements": {
    "assign": true,
    "if_stmt": true,
    "for_range": true,
    "return": true
  },
  "expressions": {
    "const": true,
    "var": true,
    "bound_var": true,
    "unary": true,
    "binary": true,
    "if_expr": true,
    "call": true,
    "map_list": true,
    "filter_list": true,
    "linear_rec": true,
    "asgp_dc": true,
    "asgp_dp1d": true,
    "asgp_dp2d": true
  },
  "builtins": {
    "abs": true,
    "min": true,
    "max": true,
    "clip": true,
    "idiv0": true,
    "imod0": true,
    "len": true,
    "concat": true,
    "slice": true,
    "index": true,
    "append": true,
    "prepend": true,
    "reverse": true,
    "find": true,
    "contains": true,
    "singleton": true,
    "char_to_string": true,
    "string_to_char": true,
    "ord": true,
    "chr": true,
    "is_letter": true,
    "is_digit": true,
    "is_space": true,
    "is_vowel": true,
    "to_lower": true,
    "to_upper": true,
    "to_string": true
  },
  "values": {
    "int": true,
    "float": true,
    "bool": true,
    "char": true,
    "string": true,
    "int_list": true,
    "float_list": true,
    "string_list": true
  }
}
```

## Validation

Invalid configs fail before an evolution run starts.

Required validation:

- native current configs use `format_version=grammar-config`
- compatibility inputs use `format_version=grammar-config` and are
  translated before use
- all known statement, expression, builtin, and value keys must be present
- unknown keys are rejected
- `statements.return` must be enabled
- `expressions.const` must be enabled
- `expressions.call=false` disables generated builtin calls even when individual
  builtin keys are true
- `values.int` or `values.float` must be enabled
- `statements.for_range` requires `values.int`
- native current configs do not include `none` or `num_list`

Generation treats disabled operations as unavailable choices. It should not
generate disabled syntax and rely on compilation or runtime validation to reject
it later.

## Seed Replay

`population-seeds` replay regenerates genomes from RNG seeds, so seed-set
writers should record grammar config identity:

```json
{
  "format_version": "population-seeds",
  "grammar_config": {
    "path": "configs/grammar/scalar.json",
    "hash": "fnv1a64:..."
  }
}
```

Replay behavior:

- old seed files without `grammar_config` metadata replay with the all-enabled
  default unless the user supplies `--grammar-config`
- seed files with `grammar_config.hash` require the same `--grammar-config`
  content
- seed files with only `grammar_config.path` require a matching
  `--grammar-config` path

The native CLI reports the selected config path and hash in `out-json`
metadata.

## Compatibility Profile Generation

`tools/grammar_config_profiles.py` materializes deterministic
`grammar-config` payloads:

- `compat`: derive a current config from an old `grammar-config` file plus a
  current fixture schema. Legacy `num_list` is mapped to `int_list` and/or
  `float_list` according to the fixture schema, and current-only forms stay
  disabled.
- `compact`: derive the same current subset but preserve the base grammar-config
  numeric-list shape by mapping legacy `num_list` to both `int_list` and
  `float_list`. Use this profile for the fairest direct comparison against
  old runs that used the same checked-in grammar config.
  Native `compact` generation also treats exact `IntList` and `FloatList`
  fixture inputs as `Any` for input-variable seeding when the generated config
  records `compat.num_list_mode="both"`. Runtime values remain exact current
  `IntList` / `FloatList`; this compatibility rule only preserves the old
  `NumList` search-space shape for generation and reproduction.
- `full`: emit the full current profile shape for experiments. Native generation
  currently uses only the implemented source-form subset.

`tools/run_psb_regression.py --profile compat|compact
--base-grammar-config PATH` automatically writes per-problem generated configs
under `OUT_DIR/_grammar_configs/` when `--grammar-config` is not supplied, then
records generated config hash, base config hash, fixture schema hash, and
`num_list_mode` in the summary.

## Implementation Points

Operational profile generation:

- `tools/grammar_config_profiles.py`
- `tools/run_psb_regression.py`

Native implementation:

- `cpp/include/g3pvm/evolution/grammar_config.hpp`
- `cpp/src/evolution/grammar_config.cpp`
- `cpp/src/evolution/genome_generation.cpp`
- `cpp/src/evolution/subtree_utils.cpp`
- `cpp/src/evolution/mutation.cpp`
- `cpp/src/evolution/repro/backend.cpp`
- `cpp/src/evolution/repro/prep.cpp`
- `cpp/src/evolution/repro/gpu.cpp`
- `cpp/src/evolution/evolve.cpp`
- `cpp/src/cli/evolve_cli.cpp`

Native C++ currently maps spec-level `expressions.unary=true` to all
implemented unary operators and `expressions.binary=true` to all implemented
binary operators. current-only builtin keys are real search-space gates for source
calls that have native AST/compiler/runtime support. Structured-form keys are
also accepted and gated, but generation remains conservative and only emits
source forms covered by the current native AST/runtime migration slice.

Tests:

- native generator/config coverage in `cpp/tests/evolution/test_genome.cpp`
- deterministic grammar properties in
  `cpp/tests/evolution/test_genome_properties.cpp`
- native GPU reproduction preprocess/config coverage in
  `cpp/tests/evolution/test_repro_prep.cpp`
