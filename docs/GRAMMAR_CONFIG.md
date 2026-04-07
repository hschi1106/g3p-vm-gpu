# Grammar Config

`grammar-config-v1` is an external JSON format for controlling the grammar constructs that evolution is allowed to generate.

It is a search-space control, not a runtime language mode. The interpreter, VM, CPU runtime, and GPU fitness runtime still execute the full public grammar superset.

## Scope

Grammar config affects:

- random genome generation
- Python mutation and crossover fallback generation
- C++ CPU mutation donor synthesis
- native seed-population replay when replay regenerates genomes from seeds

Grammar config does not affect:

- execution of already-materialized AST programs
- bytecode decoding
- interpreter or VM semantics
- CPU/GPU fitness semantics
- loading existing programs that contain disabled constructs

Current backend support:

- CPU reproduction respects non-default grammar configs.
- GPU reproduction rejects non-default configs until GPU donor buckets become config-aware.

## CLI Usage

Use `--grammar-config PATH` with `g3pvm_evolve_cli`:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --grammar-config configs/grammar/scalar.json \
  --engine cpu \
  --repro-backend cpu \
  --population-size 64 \
  --generations 5 \
  --out-json logs/simple_exp_1024.scalar.json
```

If `--grammar-config` is omitted, the native CLI uses the embedded all-enabled default. This is equivalent to `configs/grammar/all.json`.

Non-default configs currently require CPU reproduction:

```bash
--repro-backend cpu
```

Using a non-default config with `--repro-backend gpu` fails fast instead of silently violating the requested grammar.

## Presets

Checked-in presets live under `configs/grammar/`:

- `all.json`: all public grammar constructs enabled; default behavior
- `scalar.json`: scalar numeric / boolean / `None` search space; sequence values and container builtins disabled
- `string.json`: scalar plus `String` and string-compatible builtins
- `num_list.json`: scalar plus `NumList` and numeric-list builtins
- `string_list.json`: scalar plus `String` / `StringList` and string-list builtins
- `sequence.json`: broad first-wave sequence support

## Format

Every config file must explicitly define every known key.

```json
{
  "format_version": "grammar-config-v1",
  "statements": {
    "assign": true,
    "if_stmt": true,
    "for_range": true,
    "return": true
  },
  "expressions": {
    "const": true,
    "var": true,
    "if_expr": true,
    "unary": {
      "neg": true,
      "not": true
    },
    "binary": {
      "add": true,
      "sub": true,
      "mul": true,
      "div": true,
      "mod": true,
      "lt": true,
      "le": true,
      "gt": true,
      "ge": true,
      "eq": true,
      "ne": true,
      "and": true,
      "or": true
    },
    "builtins": {
      "abs": true,
      "min": true,
      "max": true,
      "clip": true,
      "len": true,
      "concat": true,
      "slice": true,
      "index": true,
      "append": true,
      "reverse": true,
      "find": true,
      "contains": true
    }
  },
  "values": {
    "int": true,
    "float": true,
    "bool": true,
    "none": true,
    "string": true,
    "num_list": true,
    "string_list": true
  }
}
```

## Validation

Invalid configs fail before an evolution run starts.

Required validation:

- `format_version` must be `grammar-config-v1`
- all known statement, expression, unary, binary, builtin, and value keys must be present
- unknown keys are rejected
- `statements.return` must be enabled
- `expressions.const` must be enabled
- `values.int` or `values.float` must be enabled
- `statements.for_range` requires `values.int`

Generation treats disabled operations as unavailable choices. It should not generate disabled syntax and rely on compilation or runtime validation to reject it later.

## Seed Replay

`population-seeds-v1` replay regenerates genomes from RNG seeds, so new seed-set writers should record grammar config identity:

```json
{
  "format_version": "population-seeds-v1",
  "grammar_config": {
    "path": "configs/grammar/scalar.json",
    "hash": "fnv1a64:..."
  }
}
```

Replay behavior:

- old seed files without `grammar_config` metadata replay with the all-enabled default unless the user supplies `--grammar-config`
- seed files with `grammar_config.hash` require the same `--grammar-config` content
- seed files with only `grammar_config.path` require a matching `--grammar-config` path

The native CLI reports the selected config path and hash in `out-json` metadata.

## Implementation Points

Python:

- `python/src/g3p_vm_gpu/evolution/grammar_config.py`
- `python/src/g3p_vm_gpu/evolution/random_tree.py`
- `python/src/g3p_vm_gpu/evolution/random_genome.py`
- `python/src/g3p_vm_gpu/evolution/mutation.py`
- `python/src/g3p_vm_gpu/evolution/crossover.py`
- `python/src/g3p_vm_gpu/evolution/evolve.py`

C++:

- `cpp/include/g3pvm/evolution/grammar_config.hpp`
- `cpp/src/evolution/grammar_config.cpp`
- `cpp/src/evolution/genome_generation.cpp`
- `cpp/src/evolution/subtree_utils.cpp`
- `cpp/src/evolution/mutation.cpp`
- `cpp/src/evolution/repro/backend.cpp`
- `cpp/src/evolution/evolve.cpp`
- `cpp/src/cli/evolve_cli.cpp`

Tests:

- Python generator/config coverage in `python/tests/test_evolution_ops.py`
- native generator/config coverage in `cpp/tests/evolution/test_genome.cpp`
