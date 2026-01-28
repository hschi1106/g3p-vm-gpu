# Python-like Subset v0.1 Specification

This document defines the **Python-like language subset (v0.1)** used by the `g3p-vm-gpu` project: syntax, restrictions, evaluation order, error behavior, and built-ins.

The goal is a language that is:
- **Deterministic**
- **Bounded** (via fuel/timeout)
- **Bytecode-compilable**
- **GPU-friendly** (batch evaluation)

---

## 1. Design Goals

### 1.1 Goals
- **Deterministic evaluation**: same inputs ⇒ same outputs (or same error kind).
- **Finite evaluation**: execution is bounded by a fuel counter; out-of-fuel yields `Timeout`.
- **GPU-friendly semantics**: restrict dynamic features that complicate parallel execution.
- **Single source of truth**: a reference interpreter defines semantics; VMs must conform.

### 1.2 Non-goals (Out of scope)
Not supported in v0.1:
- Containers: list/dict/set/tuple
- User-defined functions / lambdas / closures / recursion
- Classes / objects / attribute access
- Exceptions: try/except/raise
- while / break / continue
- import / I/O / randomness
- strings

---

## 2. Values and Types

### 2.1 Value domain
`Val ::= Int | Float | Bool | None`

- `Int`: mathematical integers (reference interpreter may use Python `int`).
- `Float`: IEEE-754 double (Python `float`).
- `Bool`: `True | False`
- `None`: singleton value

### 2.2 Type policy (normative)
- **Bool is not numeric**. Expressions like `True + 1` are **TypeError**.
- Conditions for `if`, `and`, `or`, and ternary must be **Bool** (no Python truthiness).

---

## 3. Abstract Syntax (AST)

### 3.1 Expressions
```
Expr e ::=
    Const(v)                      v ∈ {Int, Float, Bool, None}
  | Var(x)                        x ∈ Identifier
  | Unary(op, e)                  op ∈ {NEG, NOT}
  | Binary(op, e1, e2)            op ∈
      { ADD,SUB,MUL,DIV,MOD,
        LT,LE,GT,GE,EQ,NE,
        AND,OR }
  | IfExpr(cond, then_e, else_e)  (ternary)
  | Call(name, args)              name ∈ {abs, min, max, clip}
```

### 3.2 Statements / Blocks
```
Stmt s ::=
    Assign(x, e)                  (x = e)
  | IfStmt(cond, then_blk, else_blk)
  | ForRange(x, K, body_blk)      (for x in range(K): body)
  | Return(e)

Block b ::= [s1; s2; ...; sn]
Program p ::= Block
```

### 3.3 Syntactic restrictions (normative)
- In `ForRange(x, K, ...)`, `K` must be a **non-negative integer constant** and **must not be Bool**.
- `Call` is restricted to the built-in whitelist only.

---

## 4. Evaluation Model

### 4.1 Environment
- `Env`: mapping from variables to values: `x -> Val`.
- Reading an undefined variable yields `NameError`.

### 4.2 Results and Errors
Expression evaluation returns either:
- a `Val`, or
- an `EvalError(kind, msg)`.

Error kinds (v0.1):
- `NameError`: undefined variable; unknown builtin name (if chosen policy).
- `TypeError`: type mismatch (including non-bool conditions).
- `ZeroDiv`: division/modulo by zero.
- `ValueError`: invalid value constraints (e.g., `clip(lo > hi)`; program ends without return).
- `Timeout`: fuel exhausted.

### 4.3 Fuel / Timeout (normative)
- Evaluation/Execution is parameterized by `fuel ∈ ℕ`.
- Each AST node evaluation or statement execution consumes at least 1 unit of fuel.
- If fuel is exhausted, evaluation stops and returns `Timeout`.

---

## 5. Expression Semantics (Normative)

### 5.1 Evaluation order
- **Left-to-right** for binary operators: evaluate `e1` then `e2`.
- `AND`/`OR` are **short-circuiting** (see 5.4).
- Ternary evaluates `cond` first, then evaluates only the chosen branch.

### 5.2 Unary operators
- `NEG` (`-e`): `e` must be `Int` or `Float`, else `TypeError`.
- `NOT` (`not e`): `e` must be `Bool`, else `TypeError`.

### 5.3 Arithmetic operators
For `ADD/SUB/MUL/DIV/MOD`:
- Operands must be numeric (`Int` or `Float`), else `TypeError`.
- **Promotion rule**:
  - If either operand is `Float`, both are promoted to `Float`.
  - Otherwise operate in `Int`.

Special cases:
- `DIV`:
  - divisor `== 0` ⇒ `ZeroDiv`
  - result is the promoted-domain division output (typically `Float` if any operand was float).
- `MOD`:
  - divisor `== 0` ⇒ `ZeroDiv`

### 5.4 Boolean operators (short-circuit)
All boolean operators require boolean operands and return `Bool`.

- `AND` (`e1 and e2`):
  1. Evaluate `e1`; it must be `Bool` else `TypeError`.
  2. If `e1 == False`, result is `False` and `e2` is **not evaluated**.
  3. Otherwise evaluate `e2`; it must be `Bool` else `TypeError`, result is `e2`.

- `OR` (`e1 or e2`):
  1. Evaluate `e1`; it must be `Bool` else `TypeError`.
  2. If `e1 == True`, result is `True` and `e2` is **not evaluated**.
  3. Otherwise evaluate `e2`; it must be `Bool` else `TypeError`, result is `e2`.

> Note: v0.1 does **not** implement Python’s “return operand” behavior for `and/or`.

### 5.5 Comparisons
All comparisons return `Bool`.

Rules:
- If either operand is `None`:
  - Only `EQ/NE` are allowed.
  - `LT/LE/GT/GE` yield `TypeError`.
- If either operand is `Bool`:
  - Only `EQ/NE` are allowed and both must be `Bool`; otherwise `TypeError`.
  - `LT/LE/GT/GE` on bool yields `TypeError`.
- Numeric comparisons:
  - both operands must be `Int/Float`
  - mixed types are promoted using the arithmetic promotion rule

### 5.6 Ternary (IfExpr)
`then_e if cond else else_e`
- Evaluate `cond`; it must be `Bool` else `TypeError`.
- If `cond` is `True`, evaluate `then_e`; else evaluate `else_e`.
- Only the selected branch is evaluated.

### 5.7 Built-in calls (whitelist)
Allowed: `abs`, `min`, `max`, `clip`.

- `abs(x)`:
  - `x` must be numeric (`Int/Float`), else `TypeError`.
- `min(x, y)` / `max(x, y)`:
  - both must be numeric, else `TypeError`.
  - mixed types use promotion.
- `clip(x, lo, hi)`:
  - all must be numeric, else `TypeError`.
  - must satisfy `lo <= hi`, else `ValueError`.
  - returns `min(max(x, lo), hi)` under the promotion rule.

Unknown builtin name:
- Recommended policy: `NameError` (must be consistent across backends).

---

## 6. Statement Semantics (Normative)

### 6.1 Execution outcomes
Executing a statement/block returns:
- updated environment `Env'`
- an outcome `Out ∈ {Normal, Return(v), Error(err)}`

Block execution:
- executes statements sequentially
- stops immediately and propagates `Return` or `Error`

### 6.2 Assignment
`x = e`
- Evaluate `e` to `v`, then set `env[x] = v`.
- If evaluating `e` errors, propagate `Error(err)`.

### 6.3 Return
`return e`
- Evaluate `e` to `v`, return `Return(v)`.
- If evaluating `e` errors, propagate `Error(err)`.

### 6.4 If statement
`if cond: then_blk else: else_blk`
- Evaluate `cond`; it must be `Bool` else `TypeError`.
- Execute selected block and propagate its outcome.

### 6.5 ForRange
`for x in range(K): body`
- Restriction: `K` must be a non-negative integer constant and not Bool, else `TypeError`.
- Semantics:
  - For `i = 0..K-1`:
    - set `env[x] = i`
    - execute `body`
    - if `body` yields `Return` or `Error`, stop and propagate
- If loop finishes without return/error, outcome is `Normal`.

---

## 7. Program Result
A program is a `Block`.

- If execution yields `Return(v)`, program result is `v`.
- If execution yields `Error(err)`, program result is `err`.
- If execution finishes with `Normal` (no return), program result is `ValueError`
  with message like `"program finished without return"`.

---

## 8. Determinism and Side Effects
- The only side-effect is local variable assignment in `Env`.
- Built-ins are pure and must not touch external state.

---

## 9. Conformance Requirements
Implementations must satisfy (see `spec/test_contract.md`):
- `interp_py(ast)` == `vm_py(compile_py(ast))`
- `vm_py(bytecode)` == `vm_gpu(bytecode)`  
  (If float precision differs, specify an explicit tolerance policy in `test_contract.md`.)

---

## 10. Versioning
v0.1 is intended to “freeze” the subset + bytecode pipeline.  
Any extension must create a new version document (e.g., `subset_v0_2.md`) to avoid spec drift.
