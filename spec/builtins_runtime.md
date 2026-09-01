# Builtins Runtime

This document defines release 1.0.0 sequence, list-construction, and payload-backed
builtin behavior.

See also:

- [grammar.md](./grammar.md)
- [builtins_base.md](./builtins_base.md)
- [bytecode_isa.md](./bytecode_isa.md)
- [fitness.md](./fitness.md)

## Builtin Set

Sequence builtins:

- `len(x)`
- `concat(a, b)`
- `slice(x, lo, hi)`
- `index(x, i)`
- `append(xs, x)`
- `prepend(xs, x)`
- `reverse(x)`
- `find(s, sub)`
- `contains(s, sub)`
- `singleton(x)`

Container tags:

- `String`
- `IntList`
- `FloatList`
- `StringList`

`Char` is a scalar. There is no `CharList`.

## Shared Rules

- Builtins are pure.
- Builtins do not access external state.
- Invalid argument types yield `TypeError`.
- Invalid values for a valid type yield `ValueError`.
- Payload fallback tokens are internal runtime artifacts and are not public
  source-language values.
- CPU and GPU exact paths must produce identical public results.
- GPU fallback paths must be deterministic.

## Numeric List Strictness

Current does not perform implicit numeric-list promotion.

- `IntList` accepts only `Int` elements.
- `FloatList` accepts only `Float` elements.
- operations combining `IntList` and `FloatList` yield `TypeError`.
- operations that would insert an `Int` into `FloatList` or a `Float` into
  `IntList` yield `TypeError`.

## `len(x)`

- arity: `1`
- valid input: `String`, `IntList`, `FloatList`, or `StringList`
- result type: `Int`
- result: exact sequence length

`len(String)` counts `Char` values, not bytes.

## `concat(a, b)`

- arity: `2`
- valid inputs:
  - `String`, `String`
  - `IntList`, `IntList`
  - `FloatList`, `FloatList`
  - `StringList`, `StringList`
- result type: same tag as inputs
- mixed tags: `TypeError`

## `slice(x, lo, hi)`

- arity: `3`
- valid inputs:
  - `x`: `String`, `IntList`, `FloatList`, or `StringList`
  - `lo`: `Int`
  - `hi`: `Int`
- result type: same tag as `x`

Semantics:

- negative indices are normalized like Python slicing.
- bounds are clamped like Python slicing.
- source order is preserved.
- an empty slice returns an empty value with the same runtime tag as `x`.

## `index(x, i)`

- arity: `2`
- valid inputs:
  - `String`, `Int`
  - `IntList`, `Int`
  - `FloatList`, `Int`
  - `StringList`, `Int`
- out-of-range index: `ValueError`

Result types:

- `index(String, Int) -> Char`
- `index(IntList, Int) -> Int`
- `index(FloatList, Int) -> Float`
- `index(StringList, Int) -> String`

Negative indices are normalized like Python indexing.

## `append(xs, x)`

- arity: `2`
- valid inputs:
  - `IntList`, `Int`
  - `FloatList`, `Float`
  - `StringList`, `String`
- result type: same list tag as `xs`
- result: new list with `x` after all original elements

`append(String, Char)` is not defined. Use `concat(s, char_to_string(c))`.

## `prepend(xs, x)`

- arity: `2`
- valid inputs:
  - `IntList`, `Int`
  - `FloatList`, `Float`
  - `StringList`, `String`
- result type: same list tag as `xs`
- result: new list with `x` before all original elements

The RSGP operation named `cons` maps to `prepend(xs, x)`.

## `reverse(x)`

- arity: `1`
- valid input: `String`, `IntList`, `FloatList`, or `StringList`
- result type: same tag as `x`
- result: reversed sequence

## `find(s, sub)`

- arity: `2`
- valid inputs: `String`, `String`
- result type: `Int`
- result: first substring index or `-1` when absent

## `contains(s, sub)`

- arity: `2`
- valid inputs: `String`, `String`
- result type: `Bool`
- result: true iff `sub` occurs in `s`

## `singleton(x)`

- arity: `1`
- valid inputs:
  - `Int`
  - `Float`
  - `String`
  - `Char`

Result types:

- `singleton(Int) -> IntList`
- `singleton(Float) -> FloatList`
- `singleton(String) -> StringList`
- `singleton(Char) -> String`

## Structured Expression Payload Behavior

`MapList`, `FilterList`, and `LinearRec` are grammar forms, not builtins. They
still depend on exact typed-list payload access.

Required behavior:

- source typed-list payload lookup occurs after the source expression is
  evaluated exactly once.
- missing exact payload in an exact path yields `ValueError`.
- fallback behavior must be deterministic.
- result builders must preserve exact result list tags.
- empty results must preserve statically selected list tags.

## ASGP Payload Behavior

`AsgpDC`, `AsgpDP1D`, and `AsgpDP2D` are grammar forms, not builtins.

Required behavior:

- `SeqOf(Char)` traverses a `String` as `Char` values.
- `SeqOf(Int)` traverses an `IntList`.
- `SeqOf(Float)` traverses a `FloatList`.
- `SeqOf(String)` traverses a `StringList`.
- ASGP frame-stack overflow is deterministic.
- ASGP memo-table overflow is deterministic.
- ASGP payload materialization overflow is deterministic.

GPU implementations must not depend on device recursion.
