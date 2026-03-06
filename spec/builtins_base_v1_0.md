# Builtins Base v1.0

This document defines the scalar builtin whitelist.

Semantics must remain consistent across the Python interpreter, Python VM, C++ CPU runtime, and C++ GPU runtime.

See also:
- [grammar_v1_0.md](./grammar_v1_0.md)
- [bytecode_isa_v1_0.md](./bytecode_isa_v1_0.md)

## Builtin Set

Scalar builtins are:
- `abs(x)`
- `min(x, y)`
- `max(x, y)`
- `clip(x, lo, hi)`

Only these names are valid in the scalar builtin set.

## Shared Rules

- Builtins are pure.
- Builtins do not access external state.
- `Bool` is not numeric.
- Numeric arguments must be `Int` or `Float`.
- `None` is never a valid numeric argument.

## Numeric Promotion

When a builtin combines numeric values:
- if any participating operand is `Float`, operate in `Float`
- otherwise operate in `Int`

## Definitions

### `abs(x)`
- arity: `1`
- input: numeric
- result: absolute value of `x`
- non-numeric input => `TypeError`

### `min(x, y)`
- arity: `2`
- inputs: both numeric
- result: smaller value after numeric promotion
- non-numeric input => `TypeError`

### `max(x, y)`
- arity: `2`
- inputs: both numeric
- result: larger value after numeric promotion
- non-numeric input => `TypeError`

### `clip(x, lo, hi)`
- arity: `3`
- inputs: all numeric
- result: `min(max(x, lo), hi)` after numeric promotion
- if `lo > hi` => `ValueError`
- non-numeric input => `TypeError`

## Builtin IDs

The scalar builtin id mapping is:
- `0`: `abs`
- `1`: `min`
- `2`: `max`
- `3`: `clip`

Additional container builtins are defined in [builtins_runtime_v1_0.md](./builtins_runtime_v1_0.md).
