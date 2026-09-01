# Fitness

This document defines release 1.0.0 scoring rules used by evolution and benchmark
entrypoints.

Current preserves the old scoring shape, removes public `None`, removes public
`NumList`, adds public `Char`, and scores direct list tags exactly.

See also:

- [grammar.md](./grammar.md)
- [fitness_cases.md](./fitness_cases.md)
- [bytecode_format.md](./bytecode_format.md)

## Case Classification

A case is classified by the runtime type of `expected`.

### Numeric Case

A case is numeric when `expected` is `Int` or `Float`.

`Bool` is not numeric.
`Char` is not numeric.

### Exact-Match Case

A case is exact-match when `expected` is one of:

- `Bool`
- `Char`
- `String`
- `IntList`
- `FloatList`
- `StringList`

`IntList` and `FloatList` are different runtime types.
`Char` and `String` are different runtime types.

## Adjustable Parameter

`penalty` is a non-negative scalar parameter.

```text
penalty >= 0
```

Default runtime value:

```text
penalty = 1.0
```

The penalty is applied when evaluation cannot produce a directly comparable
result, including runtime errors and type mismatches.

## Case-Level Scoring

### Numeric Case

For numeric expected values:

```text
if actual is Int or Float:
    score_case(actual, expected) = -min(abs(actual - expected), penalty)
else:
    score_case(actual, expected) = -penalty
```

Properties:

```text
score_case = 0    iff actual = expected and actual is numeric
score_case < 0    otherwise
```

### Exact-Match Case

For exact-match expected values:

```text
if runtime_type(actual) = runtime_type(expected):
    score_case(actual, expected) = 1 if actual == expected else 0
else:
    score_case(actual, expected) = -penalty
```

Runtime type equality is exact:

- `Char("a")` and `String("a")` are a type mismatch.
- `IntList([1])` and `FloatList([1.0])` are a type mismatch.

## Runtime Errors

Runtime errors always contribute:

```text
-penalty
```

This includes:

- `NameError`
- `TypeError`
- `ZeroDiv`
- `ValueError`
- `Timeout`
- deterministic ASGP stack, memo, or payload overflow errors

The first runtime error for a case terminates that case execution.

## Total Program Fitness

Program fitness is the sum over all cases:

```text
fitness(program) = sum(score_case(actual_i, expected_i))
```

For a mixed benchmark:

```text
fitness(program) =
    sum_numeric(score_numeric_case(actual_i, expected_i))
  + sum_exact(score_exact_match_case(actual_i, expected_i))
```

Value range:

```text
fitness(program) in [-penalty * N_cases, N_exact]
```

where:

- `N_cases` is the total number of cases
- `N_exact` is the number of exact-match cases

## Solved Criteria

### Pure Numeric Benchmark

A pure numeric benchmark is solved iff total fitness is exactly:

```text
0
```

### Pure Exact-Match Benchmark

A pure exact-match benchmark with `N` cases is solved iff total fitness is:

```text
N
```

### Mixed Benchmark

A mixed benchmark with `N_exact` exact-match cases is solved iff total fitness
is:

```text
N_exact
```

Reason:

- every numeric exact hit contributes `0`
- every exact-match hit contributes `1`
- any mismatch, runtime error, or type mismatch prevents the target score

## PSB Regression Interpretation

PSB regression reports must record:

- train solved flag
- test solved flag when a test fixture is available
- best train fitness
- best test fitness when available
- best-fitness history
- first solved generation when available
- runtime error, timeout, and type error rates when available

For `compat`, compare these values against the baseline and gates recorded in
the relevant benchmark manifest.
