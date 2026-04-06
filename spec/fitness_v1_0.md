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

`penalty` is a non-negative scalar parameter.

Constraint:

```text
penalty >= 0
```

Default runtime value:

```text
penalty = 1.0
```

Meaning:
- it is the penalty applied when evaluation cannot produce a directly comparable result
- this includes runtime errors and type mismatches

## Case-Level Scoring

### Numeric case

For numeric expected values:

```text
if actual is numeric:
    score_case(actual, expected) = -min(|actual - expected|, penalty)
else:
    score_case(actual, expected) = -penalty
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

Examples with `penalty = 1.0`:
- `expected = 3`, `actual = 3` => `0`
- `expected = 3`, `actual = 2.5` => `-0.5`
- `expected = 3`, `actual = 10` => `-1`
- `expected = 3`, `actual = "abc"` => `-1`
- `expected = 3`, runtime error => `-1`

Consequences:
- numeric outliers are capped at the same per-case penalty used for runtime/type failures
- a directly comparable numeric result is never worse than `-penalty`

### Binary case

For `Bool`, `None`, `String`, and `List` expected values:

```text
if actual has the same runtime type as expected:
    score_case(actual, expected) = 1 if actual == expected else 0
else:
    score_case(actual, expected) = -penalty
```

Examples with `penalty = 1.0`:
- `expected = "ab"`, `actual = "ab"` => `1`
- `expected = "ab"`, `actual = "ac"` => `0`
- `expected = "ab"`, `actual = 7` => `-1`
- `expected = True`, `actual = 1` => `-1`
- binary case runtime error => `-1`

## Runtime Errors

Runtime errors always contribute:

```text
-penalty
```

This rule is independent of case type.

## Total Program Fitness

Program fitness is the sum over all cases:

```text
fitness(program) = Σ score_case(actual_i, expected_i)
```

For a mixed benchmark:

```text
fitness(program) =
    Σ_numeric score_numeric_case(actual_i, expected_i)
  + Σ_binary score_binary_case(actual_i, expected_i)
```

Value range:

```text
fitness(program) ∈ [-penalty * N_cases, N_bin]
```

where:
- `N_cases` is the total number of cases
- `N_bin` is the number of binary cases

## Solved Criteria

### Pure numeric benchmark
A pure numeric benchmark is solved iff total fitness is exactly:

```text
0
```

### Pure binary benchmark
A pure binary benchmark with `N` cases is solved iff total fitness is:

```text
N
```

### Mixed benchmark
A mixed benchmark with `N_bin` binary cases is solved iff total fitness is:

```text
N_bin
```

Reason:
- every numeric exact hit contributes `0`
- every binary exact hit contributes `1`
- any runtime error or type mismatch reduces fitness by `penalty`

## Design Consequences

- numeric tasks still have a dense gradient through capped negative absolute error
- binary tasks remain exact-match driven for same-type outputs
- invalid-type outputs and runtime failures are explicitly worse than ordinary binary mismatches
- `penalty` also upper-bounds the damage from extreme but directly comparable numeric outputs
