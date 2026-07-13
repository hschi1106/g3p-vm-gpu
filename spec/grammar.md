# Grammar

This document defines the source grammar, type rules, structured expression
forms, and ASGP scheme forms used by `g3p-vm-gpu`.

All programs use prefix `AstProgram` representation. The grammar is statically
typed for generation, crossover, mutation, compilation, and runtime
validation.

Related current specifications:

- [builtins_base.md](./builtins_base.md)
- [builtins_runtime.md](./builtins_runtime.md)
- [bytecode_format.md](./bytecode_format.md)
- [bytecode_isa.md](./bytecode_isa.md)
- [fitness_cases.md](./fitness_cases.md)
- [fitness.md](./fitness.md)
- [grammar_config.md](./grammar_config.md)

## Value Domain

Runtime values are:

```text
Value ::= Int
        | Float
        | Bool
        | Char
        | String
        | IntList
        | FloatList
        | StringList
```

Value rules:

- `Char` is a scalar character value.
- `String` is a sequence of `Char`.
- `IntList` contains only `Int` elements.
- `FloatList` contains only `Float` elements.
- `StringList` contains only `String` elements.
- Heterogeneous lists, nested lists, generic lists, char lists, product values,
  and null values are not part of the value domain.

## Type Notation

Expression types:

```text
T ::= Int | Float | Bool | Char | String
    | IntList | FloatList | StringList
```

List element types:

```text
ListElem ::= Int | Float | String
```

List type constructor:

```text
ListOf(Int)    = IntList
ListOf(Float)  = FloatList
ListOf(String) = StringList
```

Sequence types:

```text
Seq ::= String | IntList | FloatList | StringList
```

ASGP source element types:

```text
AsgpElem ::= Int | Float | String | Char
```

ASGP source sequence constructor:

```text
SeqOf(Int)    = IntList
SeqOf(Float)  = FloatList
SeqOf(String) = StringList
SeqOf(Char)   = String
```

`SeqOf(Char)` denotes string traversal by character. It does not introduce a
`CharList` runtime value.

## Conversion Policy

No subtype relation or implicit conversion is defined.

Explicit conversions:

```text
char_to_string: Char -> String
string_to_char: String -> Char
ord:            Char -> Int
chr:            Int -> Char
to_string:      Int -> String
to_string:      Float -> String
```

Conversion rules:

- `string_to_char(s)` succeeds only when `len(s) == 1`; otherwise it yields
  `ValueError`.
- `chr(i)` succeeds only for supported character code points; otherwise it
  yields `ValueError`.
- `Char` is not accepted where `String` is required.
- `String` is not accepted where `Char` is required.
- `IntList` and `FloatList` are not interchangeable.

## Lexical Environments

Expression typing is written:

```text
E[T; L; Q]
```

where:

- `T` is the expression result type.
- `L` maps ordinary local variable names to grammar types.
- `Q` maps immutable structured or scheme binder names to grammar types.

Ordinary locals and binders are distinct:

- `Var(x)` reads an ordinary local from `L`.
- `BoundVar(b)` reads an immutable binder from `Q`.
- `Assign` may update ordinary locals.
- Binders cannot be assigned.

Implementations must represent binder references in a capture-safe way.

## Program Grammar

```text
Program[R; L0] ::= Block[R; L0]

Block[R; L] ::= [Stmt[R; L], ...]

Stmt[R; L] ::= Assign(x:T, E[T; L; empty])
             | IfStmt(E[Bool; L; empty],
                      Block[R; L],
                      Block[R; L])
             | ForRange(i, E[Int; L; empty],
                        Block[R; L + {i:Int}])
             | Return(E[R; L; empty])
```

Statement rules:

- `Assign` evaluates its expression and stores the result in the ordinary local
  environment.
- `IfStmt` requires a `Bool` condition and executes only the selected block.
- `ForRange` evaluates its bound exactly once, requires a non-negative `Int`,
  and iterates with indices `0, 1, ...` while the index is less than the bound.
- `Return` evaluates its expression and terminates execution.
- Reading an undefined ordinary local yields `NameError`.
- Reaching the end of a program without `Return` yields `ValueError`.

## Expression Grammar

```text
E[T; L; Q] ::= Const(v)                         if type(v) = T
             | Var(x)                           if L(x) = T
             | BoundVar(b)                      if Q(b) = T
             | Unary(op, E[T1; L; Q])           if op: T1 -> T
             | Binary(op, E[T1; L; Q],
                           E[T2; L; Q])          if op: T1 x T2 -> T
             | IfExpr(E[Bool; L; Q],
                      E[T; L; Q],
                      E[T; L; Q])
             | Call(name, args)                 if name(args) -> T

             | MapList[A, C](
                   E[ListOf(A); L; Q],
                   body(u:A): E[C; L; Q + {u:A}]
               )

             | FilterList[A](
                   E[ListOf(A); L; Q],
                   pred(u:A): E[Bool; L; Q + {u:A}]
               )

             | LinearRec[A, R](
                   E[ListOf(A); L; Q],
                   E[Int; L; Q],
                   E[R; L; Q],
                   step(u:A, v:R, idx:Int):
                       E[R; L; Q + {u:A, v:R, idx:Int}],
                   last(u:A, idx:Int):
                       E[R; L; Q + {u:A, idx:Int}]
               )

             | AsgpDC[A, T](
                   source: E[SeqOf(A); L; Q],
                   solve(xs:SeqOf(A), n:Int, lo:Int):
                       E[T; empty; Qdc_solve(A)],
                   ordivide(n:Int):
                       E[Int; empty; Qdc_divide],
                   andcombine(r1:T, r2:T):
                       E[T; empty; Qdc_combine(T)]
               )

             | AsgpDP1D[T](
                   state: E[Int; L; Q],
                   bounds: DpBounds1D[T],
                   solve(s:Int):
                       E[T; empty; Qdp1_solve],
                   depselect(s:Int):
                       DpDeps1D[K],
                   transition(s:Int, d1:T, ..., dK:T):
                       E[T; empty; Qdp1_transition(T, K)]
               )

             | AsgpDP2D[T](
                   state_i: E[Int; L; Q],
                   state_j: E[Int; L; Q],
                   bounds: DpBounds2D[T],
                   solve(i:Int, j:Int):
                       E[T; empty; Qdp2_solve],
                   depselect(i:Int, j:Int):
                       DpDeps2D[K],
                   transition(i:Int, j:Int, d1:T, ..., dK:T):
                       E[T; empty; Qdp2_transition(T, K)]
               )
```

`A` and `C` range over `ListElem`. `R` and `T` range over expression types.

## Operators

Exact numeric operands:

```text
SameNumeric(A, B)       is valid only when A and B are the same runtime type
SameNumeric(A, B, C)    is valid only when A, B, and C are the same runtime type
```

Unary operators:

```text
NEG: A -> A       where A is Int or Float
NOT: Bool -> Bool
```

Arithmetic operators:

```text
ADD, SUB, MUL, MOD:
  A x A -> A              where A is Int or Float

DIV:
  A x A -> Float          where A is Int or Float
```

Comparison operators:

```text
LT, LE, GT, GE: A x A -> Bool
                  where A is Int or Float

EQ, NE:         T x T -> Bool
                  for exact same runtime type T

AND, OR:        Bool x Bool -> Bool
```

Ordering comparisons are valid only for numeric scalar types. Equality and
inequality require exact same runtime type.

## Builtins

Scalar builtins:

```text
abs:      A -> A
            where A is Int or Float

min:      A x A -> A
max:      A x A -> A
            where A is Int or Float

clip:     A x A x A -> A
            where A is Int or Float

idiv0:    Int x Int -> Int
imod0:    Int x Int -> Int
```

Sequence builtins:

```text
len:      Seq -> Int
concat:   Seq x Seq -> Seq                         for matching sequence tags
slice:    Seq x Int x Int -> Seq
index:    String x Int -> Char
index:    ListOf(A) x Int -> A
append:   ListOf(A) x A -> ListOf(A)
prepend:  ListOf(A) x A -> ListOf(A)
reverse:  Seq -> Seq
find:     String x String -> Int
contains: String x String -> Bool
```

Character and string builtins:

```text
char_to_string: Char -> String
string_to_char: String -> Char
ord:            Char -> Int
chr:            Int -> Char
is_letter:      Char -> Bool
is_digit:       Char -> Bool
is_space:       Char -> Bool
is_vowel:       Char -> Bool
to_lower:       Char -> Char
to_upper:       Char -> Char
to_string:      Int -> String
to_string:      Float -> String
```

Construction helpers:

```text
singleton: A -> ListOf(A)
             where A is Int, Float, or String

singleton: Char -> String
```

Builtin rules:

- Builtins are pure.
- Builtins do not access external state.
- Invalid argument types yield `TypeError`.
- Invalid values for a valid type yield `ValueError`.
- Protected integer operations return `0` when the divisor is `0`.

## Structured List Expressions

Structured list expressions are syntax forms with lexical bodies. They are not
first-class functions and cannot be returned, stored, or passed to ordinary
builtins.

### `MapList`

```text
MapList[A, C](
  xs: E[ListOf(A); L; Q],
  body(u:A): E[C; L; Q + {u:A}]
) -> ListOf(C)
```

Rules:

- Evaluate `xs` exactly once.
- `xs` must evaluate to `ListOf(A)`.
- Visit source elements left to right.
- Evaluate `body` once per element with `u` bound to the current element.
- Every body result must have type `C`.
- Return `ListOf(C)`.
- Empty input returns an empty list with type `ListOf(C)`.

### `FilterList`

```text
FilterList[A](
  xs: E[ListOf(A); L; Q],
  pred(u:A): E[Bool; L; Q + {u:A}]
) -> ListOf(A)
```

Rules:

- Evaluate `xs` exactly once.
- `xs` must evaluate to `ListOf(A)`.
- Visit source elements left to right.
- Evaluate `pred` once per element with `u` bound to the current element.
- Preserve elements whose predicate result is `True`.
- Preserve source order.
- Return `ListOf(A)`.

### `LinearRec`

```text
LinearRec[A, R](
  xs: E[ListOf(A); L; Q],
  start_idx: E[Int; L; Q],
  empty_case: E[R; L; Q],
  step(u:A, v:R, idx:Int): E[R; L; Q + {u:A, v:R, idx:Int}],
  last(u:A, idx:Int): E[R; L; Q + {u:A, idx:Int}]
) -> R
```

Rules:

- Evaluate `xs` exactly once.
- Evaluate `start_idx` exactly once.
- `xs` must evaluate to `ListOf(A)`.
- `start_idx` must evaluate to `Int`.
- Empty input evaluates only `empty_case`.
- Singleton input evaluates only `last`.
- Longer input evaluates `last` for the final element, then evaluates `step`
  from right to left.
- `step`, `last`, and `empty_case` must all produce `R`.

Mathematical definition:

```text
LinearRec([], idx, empty, step, last) =
  empty

LinearRec([u], idx, empty, step, last) =
  last(u, idx)

LinearRec(u :: rest, idx, empty, step, last) =
  step(u,
       LinearRec(rest, ADD(idx, 1), empty, step, last),
       idx)
```

## ASGP-DC

ASGP-DC is a fixed divide-and-conquer scheme. The evolvable phase grammar is:

```text
AsgpDC[A, T] ::= source
               + solve(xs:SeqOf(A), n:Int, lo:Int): T
               + ordivide(n:Int): Int
               + andcombine(r1:T, r2:T): T
```

The fixed template semantics are:

```text
base = SizeAtMost(1)
split = clamp(ordivide(n), 1, n - 1)
left/right recursive evaluation
fuel accounting
phase visibility
```

### DC Phase Grammar

Solve phase:

```text
Qdc_solve(A) = {
  xs: SeqOf(A),
  n: Int,
  lo: Int
}

solve(xs:SeqOf(A), n:Int, lo:Int): E[T; empty; Qdc_solve(A)]
```

Divide phase:

```text
Qdc_divide = {
  n: Int
}

ordivide(n:Int): E[Int; empty; Qdc_divide]
```

Combine phase:

```text
Qdc_combine(T) = {
  r1: T,
  r2: T
}

andcombine(r1:T, r2:T): E[T; empty; Qdc_combine(T)]
```

Visibility rules:

- `solve` may read only `xs`, `n`, `lo`, and constants.
- `ordivide` may read only `n` and constants.
- `andcombine` may read only `r1`, `r2`, and constants.
- `ordivide` must not inspect source elements.
- `andcombine` must not inspect the original source or subproblem sequence.
- ASGP phase bodies cannot assign locals.
- ASGP phase bodies cannot recursively call ASGP schemes.

### DC Evaluation

```text
eval_dc(xs, lo):
  n = len(xs)

  if n <= 1:
    return solve(xs, n, lo)

  raw = ordivide(n)
  k = clamp(raw, 1, n - 1)

  r1 = eval_dc(slice(xs, 0, k), lo)
  r2 = eval_dc(slice(xs, k, n), lo + k)

  return andcombine(r1, r2)

AsgpDC(source, solve, ordivide, andcombine):
  return eval_dc(source, 0)
```

`lo` is the source index offset of the current subproblem.

## ASGP-DP

ASGP-DP is a fixed memoized dynamic-programming scheme. The evolvable phase
grammar is:

```text
solve
depselect
transition
```

`depselect` produces a scheme-level dependency pattern. It is not an ordinary
runtime expression value.

### ASGP-DP 1D

```text
AsgpDP1D[T] ::= state
              + bounds
              + solve(s:Int): T
              + depselect(s:Int): DpDeps1D[K]
              + transition(s:Int, d1:T, ..., dK:T): T
```

Node form:

```text
E[T; L; Q] ::= AsgpDP1D[T](
  state: E[Int; L; Q],
  bounds: DpBounds1D[T],

  solve(s:Int):
    E[T; empty; Qdp1_solve],

  depselect(s:Int):
    DpDeps1D[K],

  transition(s:Int, d1:T, ..., dK:T):
    E[T; empty; Qdp1_transition(T, K)]
)
```

Dependency grammar:

```text
DpDeps1D[K] ::= Backward1(c1)
              | Backward2(c1, c2)
              | Backward3(c1, c2, c3)
              | Forward1(c1)
              | Forward2(c1, c2)
              | Forward3(c1, c2, c3)
```

Dependency constraints:

- `c_i` is a positive `Int` constant.
- `c_i <= max_step`.
- All offsets in one dependency pattern use the same direction.
- `K` is the number of dependencies selected by the pattern.

Dependency expansion:

```text
BackwardK(c1, ..., cK) => [s - c1, ..., s - cK]
ForwardK(c1, ..., cK)  => [s + c1, ..., s + cK]
```

Phase grammar:

```text
Qdp1_solve = {
  s: Int
}

solve(s:Int): E[T; empty; Qdp1_solve]

Qdp1_transition(T, K) = {
  s: Int,
  d1: T,
  ...,
  dK: T
}

transition(s:Int, d1:T, ..., dK:T):
  E[T; empty; Qdp1_transition(T, K)]
```

Evaluation:

```text
eval_dp1d(s):
  if out_of_bounds(s):
    return boundary_value

  if is_base(s):
    return solve(s)

  if memo[s] is defined:
    return memo[s]

  deps = depselect(s)
  values = [eval_dp1d(s2) for s2 in deps]
  out = transition(s, values...)
  memo[s] = out
  return out
```

`bounds`, `is_base`, and `boundary_value` are fixed scheme parameters.

### ASGP-DP 2D

```text
AsgpDP2D[T] ::= state_i
              + state_j
              + bounds
              + solve(i:Int, j:Int): T
              + depselect(i:Int, j:Int): DpDeps2D[K]
              + transition(i:Int, j:Int, d1:T, ..., dK:T): T
```

Node form:

```text
E[T; L; Q] ::= AsgpDP2D[T](
  state_i: E[Int; L; Q],
  state_j: E[Int; L; Q],
  bounds: DpBounds2D[T],

  solve(i:Int, j:Int):
    E[T; empty; Qdp2_solve],

  depselect(i:Int, j:Int):
    DpDeps2D[K],

  transition(i:Int, j:Int, d1:T, ..., dK:T):
    E[T; empty; Qdp2_transition(T, K)]
)
```

Dependency grammar:

```text
DpDeps2D[K] ::= CrossBackward
              | CrossForward
              | DiagonalBackward
              | DiagonalForward
              | NeighborhoodBackward3
              | NeighborhoodForward3
```

Dependency expansion:

```text
CrossBackward => [(i - 1, j), (i, j - 1)]

CrossForward => [(i + 1, j), (i, j + 1)]

DiagonalBackward => [(i - 1, j - 1)]

DiagonalForward => [(i + 1, j + 1)]

NeighborhoodBackward3 => [(i - 1, j), (i, j - 1), (i - 1, j - 1)]

NeighborhoodForward3 => [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
```

Dependency constraints:

- All dependencies must be monotone in the configured direction.
- Dependencies outside bounds use `boundary_value`.
- Dependency grammar must guarantee an acyclic memo graph.
- `K` is fixed by the selected dependency pattern.

Phase grammar:

```text
Qdp2_solve = {
  i: Int,
  j: Int
}

solve(i:Int, j:Int): E[T; empty; Qdp2_solve]

Qdp2_transition(T, K) = {
  i: Int,
  j: Int,
  d1: T,
  ...,
  dK: T
}

transition(i:Int, j:Int, d1:T, ..., dK:T):
  E[T; empty; Qdp2_transition(T, K)]
```

Evaluation:

```text
eval_dp2d(i, j):
  if out_of_bounds(i, j):
    return boundary_value

  if is_base(i, j):
    return solve(i, j)

  if memo[i, j] is defined:
    return memo[i, j]

  deps = depselect(i, j)
  values = [eval_dp2d(i2, j2) for (i2, j2) in deps]
  out = transition(i, j, values...)
  memo[i, j] = out
  return out
```

`bounds`, `is_base`, and `boundary_value` are fixed scheme parameters.

## Generation And Variation Contract

Generation, crossover, and mutation must preserve:

- exact result type
- ordinary local scope
- binder scope
- ASGP scheme kind
- ASGP phase name
- ASGP visible environment
- dependency pattern arity for DP transition phases

General expression bucket key:

```text
bucket_key =
  result_type
  + scope_signature
```

ASGP phase bucket key:

```text
asgp_bucket_key =
  scheme_kind
  + phase_name
  + result_type
  + visible_env_signature
```

ASGP crossover is valid only when both donor and destination have the same
scheme kind, phase name, result type, and visible environment signature.

`depselect` mutation regenerates a dependency pattern from the dependency
grammar, not from ordinary runtime expressions.

## Runtime Requirements

- Evaluation is deterministic.
- Builtins and structured expressions are pure.
- Each executed instruction or equivalent lowered operation consumes fuel.
- Fuel exhaustion yields `Timeout`.
- The first runtime error terminates execution.
- ASGP runtime recursion must be bounded by fuel and implementation-defined
  depth or memo-table limits.
- GPU implementations must not depend on device recursion.
- Overflow of ASGP frame stacks, memo tables, or payload materialization must
  produce a deterministic runtime result.

## Unsupported Constructs

The grammar does not include:

- user-defined functions
- first-class closures
- first-class ASGP phase bodies
- classes or attributes
- exceptions
- imports
- I/O
- `while`
- `break`
- `continue`
- generic list values
- `CharList`
- tuple or product values
