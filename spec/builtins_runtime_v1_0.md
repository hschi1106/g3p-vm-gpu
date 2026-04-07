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
- `append(xs, x)`
- `reverse(x)`
- `find(s, sub)`
- `contains(s, sub)`

Builtin ids:
- `4`: `len`
- `5`: `concat`
- `6`: `slice`
- `7`: `index`
- `8`: `append`
- `9`: `reverse`
- `10`: `find`
- `11`: `contains`

Container tags:
- `String`
- `NumList`
- `StringList`

`NumList` elements are numeric scalar values (`Int` or `Float`, never `Bool`).
`StringList` elements are strings.
Nested lists and heterogeneous lists are not part of the public runtime contract.

## `len(x)`

- arity: `1`
- input: `String`, `NumList`, or `StringList`
- result: `Int`
- non-container input => `TypeError`

The returned length must match the exact payload length.

## `concat(a, b)`

- arity: `2`
- valid inputs:
  - `String`, `String`
  - `NumList`, `NumList`
  - `StringList`, `StringList`
- result: same tag as inputs
- mixed tags => `TypeError`

Execution behavior:
- CPU exact path concatenates exact payloads
- GPU exact path concatenates payloads when scratch capacity is sufficient
- any fallback path returns `FallbackToken`

## `slice(x, lo, hi)`

- arity: `3`
- valid inputs:
  - `x`: `String`, `NumList`, or `StringList`
  - `lo`: `Int`
  - `hi`: `Int`
- invalid types => `TypeError`
- result tag matches `x`

Semantics:
- Python-like negative-index normalization
- bounds clamp behavior consistent with Python slicing
- exact path returns payload-backed slice
- any fallback path returns `FallbackToken`

## `index(x, i)`

- arity: `2`
- valid inputs:
  - `x`: `String`, `NumList`, or `StringList`
  - `i`: `Int`
- invalid types => `TypeError`
- out-of-range index => `ValueError`

Semantics:
- Python-like negative-index normalization
- `index(String, i)` returns a length-1 `String`
- `index(NumList, i)` returns the indexed numeric element
- `index(StringList, i)` returns the indexed `String`
- C++ CPU returns exact payload value when payload registry is available
- C++ GPU returns exact payload value when uploaded payload and scratch capacity are sufficient
- any fallback path returns `FallbackToken`

## `append(xs, x)`

- arity: `2`
- valid inputs:
  - `NumList`, numeric scalar
  - `StringList`, `String`
- result: same list tag as `xs`
- invalid types => `TypeError`

Execution behavior:
- exact path returns a newly registered list payload
- fallback path returns `FallbackToken`

## `reverse(x)`

- arity: `1`
- valid input: `String`, `NumList`, or `StringList`
- result: same tag as `x`
- non-container input => `TypeError`

Execution behavior:
- exact path returns a newly registered reversed payload
- fallback path returns `FallbackToken`

## `find(s, sub)`

- arity: `2`
- valid input: `String`, `String`
- result: `Int`
- returns the first substring index or `-1` when absent
- invalid types => `TypeError`
- missing exact payload in the C++ CPU/GPU path => `ValueError`

## `contains(s, sub)`

- arity: `2`
- valid input: `String`, `String`
- result: `Bool`
- invalid types => `TypeError`
- missing exact payload in the C++ CPU/GPU path => `ValueError`

## Container Comparison

The runtime extends comparison semantics as follows:
- `EQ` and `NE` are defined for `String` with `String`
- `EQ` and `NE` are defined for `NumList` with `NumList`
- `EQ` and `NE` are defined for `StringList` with `StringList`
- ordering comparisons on containers remain invalid and yield `TypeError`

## Payload Contract

### CPU
- decoded `String`, `NumList`, and `StringList` payloads are stored in a registry
- exact builtin results may allocate new payload entries

### GPU
- payload snapshots are uploaded into global memory before evaluation
- exact container builtins use bounded per-thread scratch
- if exact materialization does not fit, the runtime must return `FallbackToken` rather than failing the whole job

## Fitness Link

Scoring rules are defined in [fitness_v1_0.md](./fitness_v1_0.md).
This file only defines execution semantics, not scoring formulas.
