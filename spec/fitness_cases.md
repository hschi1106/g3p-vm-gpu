# Fitness Cases

This document defines the `fitness-cases` fixture schema used by current
evolution, parity tests, PSB conversion, and benchmark entrypoints.

See also:

- [grammar.md](./grammar.md)
- [fitness.md](./fitness.md)
- [grammar_config.md](./grammar_config.md)

## Top-Level Schema

```json
{
  "format_version": "fitness-cases",
  "meta": {
    "suite": "psb1",
    "problem": "count-odds",
    "case_count": 2
  },
  "schema": {
    "inputs": {
      "xs": "int_list"
    },
    "expected": "int"
  },
  "cases": [
    {
      "inputs": {
        "xs": {"type": "int_list", "value": [1, 2, 3]}
      },
      "expected": {"type": "int", "value": 2}
    },
    {
      "inputs": {
        "xs": {"type": "int_list", "value": []}
      },
      "expected": {"type": "int", "value": 0}
    }
  ]
}
```

Required fields:

- `format_version` must be `fitness-cases`.
- `cases` must be a non-empty array.
- each case must contain an `inputs` object and one `expected` value.

Optional fields:

- `meta`: descriptive metadata that does not affect execution.
- `schema`: explicit type declarations used by conversion tools and
  compatibility-profile generation.

When `schema` is present, every case must conform exactly to it.

## Typed Value Encoding

Supported values:

```json
{"type": "bool", "value": true}
{"type": "int", "value": 123}
{"type": "float", "value": 1.5}
{"type": "char", "value": "a"}
{"type": "string", "value": "abc"}
{"type": "int_list", "value": [1, 2, 3]}
{"type": "float_list", "value": [1, 2.5, 3]}
{"type": "string_list", "value": ["a", "bc"]}
```

Rules:

- `bool.value` must be a JSON boolean.
- `int.value` must be an integer JSON number; booleans are rejected.
- `float.value` must be a JSON number and decodes as runtime `Float`.
- `char.value` must be a JSON string containing exactly one supported Unicode
  scalar value.
- `string.value` must be a JSON string.
- `int_list.value` must be an array of integer JSON numbers; booleans are
  rejected.
- `float_list.value` must be an array of JSON numbers; every element decodes
  as runtime `Float`, including integer-looking JSON numbers.
- `string_list.value` must be an array of JSON strings.
- empty lists are valid because the direct list tag is explicit.

Invalid current value tags:

- `none`
- `num_list`
- generic `list`
- `char_list`

## Input Type Consistency

Input names define the ordinary local environment used by typed generation and
variation.

Required behavior:

- every case must provide the same set of input names.
- one input name must have the same runtime type in every case.
- `IntList` and `FloatList` are different input types.
- `Char` and `String` are different input types.
- grammar config must not reinterpret an input value as another runtime type.

## Expected Output Types

Expected output values are scored according to
[fitness.md](./fitness.md).

When all cases have the same expected runtime type, evolution may use that
exact type for expected-output-aware initial generation if the active grammar
config enables it.

## Old Fixture Migration

Public `None` has no current representation.

Old `num_list` migration requires an explicit destination tag:

- all-`Int` non-empty lists may become `int_list`.
- all-`Float` non-empty lists may become `float_list`.
- mixed numeric lists must be normalized explicitly by the dataset schema or
  rejected.
- empty numeric lists require an explicit destination tag from the problem
  schema.

Migration must not use the active grammar config to infer runtime value types.

## PSB Schema Requirements

PSB conversion tools must use per-problem schemas whenever raw data contains
ambiguous empty lists or numeric lists.

Required metadata for generated PSB fixtures:

- source suite
- source problem
- source files
- sampling seed
- train/test split
- schema hash
- field schemas

The compatibility profile uses the fixture schema to translate old
`NumList` search-space choices into `IntList` or `FloatList`.
