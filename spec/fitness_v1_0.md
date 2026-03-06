# Fitness v1.0

This document defines the scoring rules used by evolution.

## Case Classification

A case is classified by the type of `expected`.

### Numeric case
A case is numeric when `expected` is `Int` or `Float`.
`Bool` is not numeric.

### Binary case
A case is binary when `expected` is one of:
- `Bool`
- `None`
- `String`
- `List`

## Adjustable Parameter

`numeric_type_penalty` is a non-negative scalar parameter.

Constraint:

```text
numeric_type_penalty >= 0
```

Default runtime value:

```text
numeric_type_penalty = 1.0
```

Meaning:
- it is the penalty applied when a numeric-expected case produces a non-numeric actual value

## Case-Level Scoring

### Numeric case

For numeric expected values:

```text
if actual is numeric:
    score_case(actual, expected) = -|actual - expected|
else:
    score_case(actual, expected) = -numeric_type_penalty
```

Properties:

```text
score_case = 0    iff actual = expected and actual is numeric
score_case < 0    otherwise
```

Monotonicity inside the numeric branch:

```text
if |actual1 - expected| > |actual2 - expected|
then score_case(actual1, expected) < score_case(actual2, expected)
```

Examples with `numeric_type_penalty = 1.0`:
- `expected = 3`, `actual = 3` => `0`
- `expected = 3`, `actual = 2.5` => `-0.5`
- `expected = 3`, `actual = 10` => `-7`
- `expected = 3`, `actual = "abc"` => `-1`
- `expected = 3`, runtime error => `0`

Important consequence:
- non-numeric output on a numeric case is penalized
- it must not outperform a near-correct numeric answer unless the configured penalty is smaller than that numeric error

### Binary case

For `Bool`, `None`, `String`, and `List` expected values:

```text
score_case(actual, expected) =
  1  if actual == expected
  0  otherwise
```

Examples:
- exact match => `1`
- mismatch => `0`
- runtime error => `0`

## Total Program Fitness

Program fitness is the sum over all cases:

```text
fitness(program) = Σ score_case(actual_i, expected_i)
```

For a mixed benchmark:

```text
fitness(program) =
    Σ_numeric score_numeric_case(actual_i, expected_i)
  + exact_binary_match_count
```

Value range:

```text
fitness(program) ∈ (-∞, N_bin]
```

where `N_bin` is the number of binary cases.

## Solved Criteria

### Pure numeric benchmark
A pure numeric benchmark is solved iff total fitness is exactly:

```text
0
```

Reason:
- every numeric case is maximized by an exact numeric hit, which contributes `0`

### Pure binary benchmark
A pure binary benchmark with `N` cases is solved iff total fitness is:

```text
N
```

Reason:
- every exact match contributes `1`

### Mixed benchmark
A mixed benchmark with `N_bin` binary cases is solved iff total fitness is:

```text
N_bin
```

Reason:
- every numeric exact hit contributes `0`
- every binary exact hit contributes `1`

## Runtime Errors

Runtime error handling is distinct from type-mismatch scoring inside numeric cases.

Rules:
- runtime error => `0`
- numeric expected + non-numeric actual => `-numeric_type_penalty`
- binary case mismatch => `0`

This is intentional. A completed but mistyped numeric output is penalized differently from a hard runtime failure.
