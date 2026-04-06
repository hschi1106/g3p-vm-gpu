# Grammar v1.0

This document defines the current language grammar and evaluation rules used by `g3p-vm-gpu`.

## Goals

The language is designed to be:
- deterministic,
- fuel-bounded,
- bytecode-compilable,
- GPU-friendly.

## Value Domain

Base scalar values:
- `Int`
- `Float`
- `Bool`
- `None`

Runtime-extended values:
- `String`
- `List`

Type policy:
- `Bool` is not numeric.
- Conditions for `if`, `and`, `or`, and ternary expressions must be `Bool`.
- `String` and `List` are first-class runtime values, but only the whitelisted container builtins operate on them.

## Abstract Grammar

### Expressions

```text
Expr ::= Const(v)
       | Var(x)
       | Unary(op, e)
       | Binary(op, e1, e2)
       | IfExpr(cond, then_e, else_e)
       | Call(name, args)
```

Unary operators:
- `NEG`
- `NOT`

Binary operators:
- arithmetic: `ADD`, `SUB`, `MUL`, `DIV`, `MOD`
- comparisons: `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`
- boolean: `AND`, `OR`

Whitelisted call names:
- scalar: `abs`, `min`, `max`, `clip`
- container: `len`, `concat`, `slice`, `index`

### Statements

```text
Stmt ::= Assign(x, e)
       | IfStmt(cond, then_blk, else_blk)
       | ForRange(x, e, body_blk)
       | Return(e)

Block ::= [Stmt, ...]
Program ::= Block
```

## Syntactic Restrictions

- `ForRange(x, e, body)` requires `e` to evaluate to a non-negative `Int`.
- Builtin calls are restricted to the whitelist in this document and the builtin specs.
- No user-defined functions, closures, classes, attribute access, exceptions, imports, or I/O.
- No `while`, `break`, or `continue`.

## Evaluation Model

### Environment

The environment maps variable names to values.
- Reading an undefined variable yields `NameError`.
- Assignment mutates only the local environment.

### Fuel

Evaluation and execution are parameterized by `fuel`.
- Each expression evaluation or statement execution consumes at least one unit.
- If fuel is exhausted, evaluation stops with `Timeout`.

### Program result

A program returns one of:
- `Return(v)` when `return e` executes successfully
- `Error(kind, msg)` on the first runtime error
- `Error(ValueError, "program finished without return")` if control reaches the end without a return

## Expression Semantics

### Evaluation order

- Binary expressions evaluate left to right.
- `AND` and `OR` short-circuit.
- Ternary expressions evaluate the condition first and only the selected branch.

### Unary operators

- `NEG`: operand must be `Int` or `Float`; otherwise `TypeError`
- `NOT`: operand must be `Bool`; otherwise `TypeError`

### Arithmetic operators

`ADD`, `SUB`, `MUL`, `DIV`, `MOD` require numeric operands.
- Numeric means `Int` or `Float`, not `Bool`.
- Mixed `Int` and `Float` promotes to `Float`.
- `DIV` and `MOD` with zero divisor yield `ZeroDiv`.

### Boolean operators

`AND` and `OR` require boolean operands and return `Bool`.
- `AND`: if left is `False`, right is not evaluated
- `OR`: if left is `True`, right is not evaluated

The language does not use Python operand-returning truthiness behavior.

### Comparisons

All comparison operators return `Bool`.

Rules:
- Numeric comparisons require numeric operands and use numeric promotion.
- `None` supports only `EQ` and `NE`.
- `Bool` supports only `EQ` and `NE` with another `Bool`.
- `String` and `List` support only `EQ` and `NE` with the same tag.
- Ordering comparisons on `None`, `Bool`, `String`, and `List` are invalid and yield `TypeError`.

### Builtin calls

Builtin semantics are defined by:
- [builtins_base_v1_0.md](./builtins_base_v1_0.md)
- [builtins_runtime_v1_0.md](./builtins_runtime_v1_0.md)

Unknown builtin names yield `NameError`.

## Statement Semantics

### Assignment

`x = e`
- evaluates `e`
- stores the resulting value in `x`
- propagates any error from `e`

### Return

`return e`
- evaluates `e`
- terminates program execution with `Return(value)`

### If statement

`if cond: then_blk else: else_blk`
- `cond` must evaluate to `Bool`
- only the selected block executes

### ForRange

`for x in range(e): body`
- evaluates `e` exactly once before the first iteration
- `e` must evaluate to a non-negative `Int`; `Bool`, `Float`, negative integers, and all non-numeric values yield `TypeError`
- if `e == 0`, the body executes zero times
- executes `body` with integer `x = 0, 1, 2, ...` while `x < e`
- propagates `Return` and `Error` immediately

## Determinism

The language is pure except for local variable assignment.
Builtins must not depend on external state, randomness, or I/O.
