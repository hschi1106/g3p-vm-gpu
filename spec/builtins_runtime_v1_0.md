# Builtins Runtime v1.0

This document defines runtime-only builtin behavior that extends the scalar builtin set with container operations and payload-backed execution rules.

See also:
- [builtins_base_v1_0.md](./builtins_base_v1_0.md)
- [bytecode_isa_v1_0.md](./bytecode_isa_v1_0.md)
- [fitness_v1_0.md](./fitness_v1_0.md)

## Added Builtins

- `len(x)`
- `concat(a, b)`
- `slice(x, lo, hi)`
- `index(x, i)`

Builtin ids:
- `4`: `len`
- `5`: `concat`
- `6`: `slice`
- `7`: `index`

## `len(x)`

- arity: `1`
- input: `String` or `List`
- result: `Int`
- non-container input => `TypeError`

The returned length must match the exact payload length.

## `concat(a, b)`

- arity: `2`
- valid inputs:
  - `String`, `String`
  - `List`, `List`
- result: same tag as inputs
- mixed tags => `TypeError`

Execution behavior:
- CPU exact path concatenates exact payloads
- GPU exact path concatenates payloads when scratch capacity is sufficient
- GPU fallback path returns deterministic compact transport when exact materialization does not fit

## `slice(x, lo, hi)`

- arity: `3`
- valid inputs:
  - `x`: `String` or `List`
  - `lo`: `Int`
  - `hi`: `Int`
- invalid types => `TypeError`
- result tag matches `x`

Semantics:
- Python-like negative-index normalization
- bounds clamp behavior consistent with Python slicing
- exact path returns payload-backed slice
- GPU fallback uses deterministic compact transport when exact materialization does not fit

## `index(x, i)`

- arity: `2`
- valid inputs:
  - `x`: `String` or `List`
  - `i`: `Int`
- invalid types => `TypeError`
- out-of-range index => `ValueError`

Semantics:
- Python-like negative-index normalization
- Python runtime returns the exact indexed element
- C++ CPU returns exact payload value when payload registry is available
- C++ GPU returns exact payload value when uploaded payload and scratch capacity are sufficient
- GPU fallback returns deterministic compact transport when exact materialization does not fit

## Container Comparison

The runtime extends comparison semantics as follows:
- `EQ` and `NE` are defined for `String` with `String`
- `EQ` and `NE` are defined for `List` with `List`
- ordering comparisons on containers remain invalid and yield `TypeError`

## Payload Contract

### CPU
- decoded `String` and `List` payloads are stored in a registry
- exact builtin results may allocate new payload entries

### GPU
- payload snapshots are uploaded into global memory before evaluation
- exact container builtins use bounded per-thread scratch
- if exact materialization does not fit, the runtime must fall back to deterministic compact transport rather than failing the whole job

## Fitness Link

Scoring rules are defined in [fitness_v1_0.md](./fitness_v1_0.md).
This file only defines execution semantics, not scoring formulas.
