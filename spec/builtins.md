# Built-ins v0.1 Specification

This document defines the **built-in function whitelist** for the `g3p-vm-gpu` v0.1 language subset.

It is a normative contract shared by:
- the reference interpreter,
- the compiler/VM(s) (`CALL_BUILTIN`),
- the GPU backend.

Semantics must be consistent with:
- `spec/subset_v0_1.md`
- `spec/bytecode_isa.md`

---

## 1. Scope and Principles

### 1.1 Whitelist only
Only the following built-ins exist in v0.1:

- `abs(x)`
- `min(x, y)`
- `max(x, y)`
- `clip(x, lo, hi)`

Any other function name is **not allowed**.

### 1.2 Purity
Built-ins are **pure**:
- No I/O, no randomness, no global state, no mutation.
- Output depends only on input arguments.

### 1.3 Value domain and type policy
`Value ::= Int | Float | Bool | None`

- **Bool is not numeric**.
- Numeric arguments must be `Int` or `Float`.
- `None` is never a valid numeric argument.

### 1.4 Errors
Built-ins can raise the following error kinds (via `EvalError`):
- `TypeError`: wrong arity or wrong argument type(s)
- `ValueError`: invalid value constraints (e.g., `clip(lo > hi)`)

> Note: The VM ISA also allows `NameError` for unknown builtin IDs/names. That behavior is specified in `bytecode_isa.md`.

---

## 2. Numeric Promotion (Normative)

When a built-in compares or combines numeric arguments, it uses the same promotion rule as the subset:

- If **any** numeric argument involved in the operation is `Float`, operate in `Float`.
- Otherwise operate in `Int`.

In float domain:
- Convert participating numeric inputs to `Float` (IEEE-754 double in Python reference mode).

---

## 3. Built-in Definitions (Normative)

### 3.1 `abs(x)`

**Signature**
- `abs(x: Int|Float) -> Int|Float`

**Arity**
- Exactly 1 argument.

**Semantics**
- Returns the absolute value of `x`.

**Type rules**
- If `x` is not numeric (`Int` or `Float`), return `TypeError`.

**Examples**
- `abs(-3) = 3`
- `abs(-2.5) = 2.5`
- `abs(True)` ⇒ `TypeError`

---

### 3.2 `min(x, y)`

**Signature**
- `min(x: Int|Float, y: Int|Float) -> Int|Float`

**Arity**
- Exactly 2 arguments.

**Semantics**
- Returns the smaller of `x` and `y` under the numeric promotion rule.

**Type rules**
- If either argument is not numeric, return `TypeError`.

**Examples**
- `min(1, 2) = 1`
- `min(1.0, 2) = 1.0` (float domain)
- `min(None, 2)` ⇒ `TypeError`

---

### 3.3 `max(x, y)`

**Signature**
- `max(x: Int|Float, y: Int|Float) -> Int|Float`

**Arity**
- Exactly 2 arguments.

**Semantics**
- Returns the larger of `x` and `y` under the numeric promotion rule.

**Type rules**
- If either argument is not numeric, return `TypeError`.

**Examples**
- `max(1, 2) = 2`
- `max(1.0, 2) = 2.0` (float domain)
- `max(False, 2)` ⇒ `TypeError` (bool is not numeric)

---

### 3.4 `clip(x, lo, hi)`

**Signature**
- `clip(x: Int|Float, lo: Int|Float, hi: Int|Float) -> Int|Float`

**Arity**
- Exactly 3 arguments.

**Semantics**
- Enforces the interval constraint `lo <= hi`.
- Returns:
  - `min(max(x, lo), hi)` under the numeric promotion rule.

**Type rules**
- If any argument is not numeric, return `TypeError`.

**Value rules**
- If `lo > hi`, return `ValueError`.

**Examples**
- `clip(5, 0, 3) = 3`
- `clip(-1, 0, 3) = 0`
- `clip(1.5, 0, 1) = 1.0` (float domain)
- `clip(1, 3, 0)` ⇒ `ValueError`
- `clip(True, 0, 1)` ⇒ `TypeError`

---

## 4. VM Calling Convention (Normative)

### 4.1 Bytecode instruction
Built-ins are invoked via:

- `CALL_BUILTIN bid argc`

The compiler pushes arguments left-to-right; the VM pops `argc` items from the stack.
The last argument is at the top of stack.

### 4.2 Built-in ID table (v0.1)
The following mapping is normative for backends that use numeric builtin IDs:

- `0: abs`
- `1: min`
- `2: max`
- `3: clip`

If a backend uses a different encoding, it must provide a deterministic translation layer, but the *logical* mapping must remain equivalent.

### 4.3 Error mapping
- Wrong `bid` ⇒ `NameError` (recommended; see ISA)
- Wrong `argc` ⇒ `TypeError`
- Type/value constraint violation ⇒ `TypeError` / `ValueError` as specified above

---

## 5. Notes for GPU Backends (Non-normative)

### 5.1 Float precision
If the GPU backend uses float32 internally:
- Either run a “correctness mode” that matches float64, or
- Specify a tolerance policy in `spec/test_contract.md` (e.g., relative/absolute epsilon).

### 5.2 Branch behavior
These built-ins should be implemented branch-minimally for performance, but correctness comes first.

---

## 6. Versioning
This is **Built-ins v0.1**. Any change to:
- whitelist contents,
- arities,
- type/value rules,
- numeric promotion rules,
requires a new versioned spec file (e.g., `builtins_v0_2.md`) to avoid silent drift.
