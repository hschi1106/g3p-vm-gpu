# Builtins Base

This document defines the current scalar, character, and conversion builtin
whitelist.

Semantics must remain consistent across the C++ CPU and CUDA GPU runtimes.

See also:

- [grammar.md](./grammar.md)
- [bytecode_isa.md](./bytecode_isa.md)
- [builtins_runtime.md](./builtins_runtime.md)

## Builtin Set

Scalar numeric builtins:

- `abs(x)`
- `min(x, y)`
- `max(x, y)`
- `clip(x, lo, hi)`
- `idiv0(x, y)`
- `imod0(x, y)`

Character and conversion builtins:

- `char_to_string(c)`
- `string_to_char(s)`
- `ord(c)`
- `chr(i)`
- `is_letter(c)`
- `is_digit(c)`
- `is_space(c)`
- `is_vowel(c)`
- `to_lower(c)`
- `to_upper(c)`
- `to_string(x)`

Only names listed in this document and
[builtins_runtime.md](./builtins_runtime.md) are valid source
builtins.

## Shared Rules

- Builtins are pure.
- Builtins do not access external state.
- `Bool` is not numeric.
- `Char` is not numeric.
- Numeric arguments must be `Int` or `Float`.
- Protected integer operations accept only `Int`; `Bool`, `Char`, and `Float`
  are invalid.
- Invalid argument types yield `TypeError`.
- Invalid values for a valid type yield `ValueError`.

## Exact Numeric Operands

Scalar numeric types are exactly `Int` and `Float`.

```text
SameNumeric(A, B) is valid only when A and B are the same runtime type.
```

No scalar numeric builtin performs implicit `Int` / `Float` conversion.
`idiv0` and `imod0` require `Int` operands.

## Numeric Builtins

### `abs(x)`

- arity: `1`
- input: `A`, where `A` is `Int` or `Float`
- result type: `A`
- result: absolute value of `x`

### `min(x, y)`

- arity: `2`
- inputs: `A`, `A`, where `A` is `Int` or `Float`
- result type: `A`
- result: smaller value

### `max(x, y)`

- arity: `2`
- inputs: `A`, `A`, where `A` is `Int` or `Float`
- result type: `A`
- result: larger value

### `clip(x, lo, hi)`

- arity: `3`
- inputs: `A`, `A`, `A`, where `A` is `Int` or `Float`
- result type: `A`
- result: `min(max(x, lo), hi)`

When `lo > hi`, the result is `hi`. This matches nested `min(max(...), ...)`
evaluation and does not swap bounds.

### `idiv0(x, y)`

- arity: `2`
- inputs: `Int`, `Int`
- result type: `Int`
- result: truncating integer division when `y != 0`
- result when `y == 0`: `0`

### `imod0(x, y)`

- arity: `2`
- inputs: `Int`, `Int`
- result type: `Int`
- result: integer modulo when `y != 0`
- result when `y == 0`: `0`

## Character and Conversion Builtins

### Supported Characters

`Char` values are Unicode scalar values encoded as JSON strings containing
exactly one scalar value. Surrogate code points are invalid.

Implementations may reject characters they cannot transport losslessly through
their payload representation. Rejection must be deterministic and reported as
`ValueError`.

### `char_to_string(c)`

- arity: `1`
- input: `Char`
- result type: `String`
- result: a one-character string containing `c`

### `string_to_char(s)`

- arity: `1`
- input: `String`
- result type: `Char`
- result: the only character in `s`
- if `len(s) != 1`: `ValueError`

### `ord(c)`

- arity: `1`
- input: `Char`
- result type: `Int`
- result: Unicode scalar value code point

### `chr(i)`

- arity: `1`
- input: `Int`
- result type: `Char`
- valid when `i` is a supported Unicode scalar value code point
- invalid code point: `ValueError`

### `is_letter(c)`

- arity: `1`
- input: `Char`
- result type: `Bool`
- result: true when `c` is classified as a letter by the implementation's
  supported character table

### `is_digit(c)`

- arity: `1`
- input: `Char`
- result type: `Bool`
- result: true when `c` is an ASCII digit `0` through `9`

### `is_space(c)`

- arity: `1`
- input: `Char`
- result type: `Bool`
- result: true when `c` is a supported whitespace character

### `is_vowel(c)`

- arity: `1`
- input: `Char`
- result type: `Bool`
- result: true for ASCII vowels `a`, `e`, `i`, `o`, `u` and their uppercase
  forms

### `to_lower(c)`

- arity: `1`
- input: `Char`
- result type: `Char`
- result: lowercase form when supported, otherwise `c`

### `to_upper(c)`

- arity: `1`
- input: `Char`
- result type: `Char`
- result: uppercase form when supported, otherwise `c`

### `to_string(x)`

- arity: `1`
- inputs:
  - `Int`
  - `Float`
- result type: `String`
- result: canonical deterministic string representation

For `Int`, the result is the base-10 integer spelling with no leading zeroes
except the value zero itself.

For finite `Float`, implementations format in fixed decimal with six
fractional digits, round to nearest according to the implementation's normal
fixed-format conversion, remove trailing fractional zeroes, remove a trailing
decimal point, and normalize negative zero to `0`.

For non-finite `Float`, the canonical spellings are `nan`, `inf`, and `-inf`.

`to_string(Char)` is not defined. Use `char_to_string`.
