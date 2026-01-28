# Bytecode ISA v0.1 (Stack VM)

This document specifies the **bytecode instruction set architecture (ISA)** for the `g3p-vm-gpu` project, version v0.1.

It is the cross-language contract between:
- the **compiler** (AST → bytecode),
- the **CPU VM** (reference execution of bytecode),
- the **GPU VM** (accelerated execution of the same bytecode).

Semantics must conform to `spec/subset_v0_1.md`.

---

## 1. Execution Model

### 1.1 VM State
A VM instance maintains:
- `ip: int` — instruction pointer (index into `code[]`)
- `stack: Value[]` — operand stack
- `locals: Value[]` — local variables indexed by integer IDs
- `consts: Value[]` — constant pool
- `fuel: int` — remaining instruction budget (see §2.4)

### 1.2 Program Container
A bytecode program is:
- `consts: Value[]`
- `n_locals: int` (or `var2idx` mapping owned by compiler/front-end)
- `code: Instruction[]`

### 1.3 Value Domain
`Value ::= Int | Float | Bool | None`

Normative type policy:
- **Bool is not numeric** (even if host language represents it as an int subtype).
- Conditions for jumps must be **Bool** only (no truthiness).

### 1.4 Termination and Errors
Execution returns:
- `Return(v)` if `RETURN` executed successfully.
- `Error(kind, msg)` on the first error.
- `Error(ValueError, "program finished without return")` if `ip` reaches end of `code` without hitting `RETURN`.
- `Error(Timeout, "out of fuel")` if fuel is exhausted (see §2.4).

All backends must implement the same error kinds:
- `NameError`, `TypeError`, `ZeroDiv`, `ValueError`, `Timeout`

---

## 2. Instruction Format

### 2.1 Abstract Instruction
An instruction is conceptually:
- `op: Opcode`
- `a: int` (optional operand)
- `b: int` (optional operand)

Physical encoding (packing, bit-width, alignment) is implementation-defined, but **the abstract semantics are fixed**.

### 2.2 Stack Effects
Each instruction defines a **stack effect** in terms of:
- values popped from stack (top-first)
- values pushed to stack

If an instruction requires a value of a certain type and the stack top does not satisfy it, it yields `TypeError`.

### 2.3 Local Variable Indexing
- `LOAD i` / `STORE i` access `locals[i]`.
- `i` must be in range `[0, n_locals)`. Out-of-range access yields `NameError` (recommended).

### 2.4 Fuel / Timeout (normative)
- Before executing each instruction, the VM must check `fuel`.
- If `fuel <= 0`, execution stops with `Timeout`.
- Otherwise execute the instruction and decrement `fuel := fuel - 1`.

---

## 3. Opcode Set (v0.1)

### 3.1 Constants and Locals

#### `PUSH_CONST k`
- Operands: `k` (const index)
- Stack: `[] -> [consts[k]]`
- Errors:
  - if `k` out of range: `ValueError`

#### `LOAD i`
- Operands: `i` (local index)
- Stack: `[] -> [locals[i]]`
- Errors:
  - if `i` out of range: `NameError`
  - if `locals[i]` is uninitialized: `NameError`

#### `STORE i`
- Operands: `i` (local index)
- Stack: `[v] -> []`
- Effect: `locals[i] := v`
- Errors:
  - if `i` out of range: `NameError`

---

### 3.2 Unary Operators

#### `NEG`
- Stack: `[x] -> [(-x)]`
- Requires: `x` is numeric (`Int` or `Float`)
- Errors:
  - otherwise: `TypeError`

#### `NOT`
- Stack: `[b] -> [!b]`
- Requires: `b` is `Bool`
- Errors:
  - otherwise: `TypeError`

---

### 3.3 Arithmetic Operators
All arithmetic operators require numeric operands: `Int` or `Float`.
Promotion rule:
- if either operand is `Float`, promote both to `Float`;
- else operate as `Int`.

#### `ADD`, `SUB`, `MUL`
- Stack: `[a, b] -> [a ⊕ b]`
- Errors:
  - if any operand not numeric: `TypeError`

#### `DIV`
- Stack: `[a, b] -> [a / b]`
- Errors:
  - if any operand not numeric: `TypeError`
  - if `b == 0`: `ZeroDiv`

#### `MOD`
- Stack: `[a, b] -> [a % b]`
- Errors:
  - if any operand not numeric: `TypeError`
  - if `b == 0`: `ZeroDiv`

---

### 3.4 Comparison Operators
All comparison operators push a `Bool`.

Rules match `subset_v0_1.md`:
- `None` only supports `EQ/NE`. Ordering comparisons with `None` are `TypeError`.
- `Bool` only supports `EQ/NE` with `Bool`. Ordering comparisons on `Bool` are `TypeError`.
- Numeric comparisons operate on promoted numeric domain.

#### `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`
- Stack: `[a, b] -> [a ⋚ b]`
- Errors:
  - if type rules violated: `TypeError`

---

### 3.5 Control Flow

#### `JMP addr`
- Operands: `addr` (absolute instruction index)
- Stack: no change
- Effect: `ip := addr`
- Errors:
  - if `addr` out of range: `ValueError`

#### `JMP_IF_FALSE addr`
- Operands: `addr` (absolute instruction index)
- Stack: `[cond] -> []`
- Requires: `cond` is `Bool`
- Effect:
  - if `cond == False`: `ip := addr`
  - else: continue
- Errors:
  - if `cond` not Bool: `TypeError`
  - if `addr` out of range: `ValueError`

#### `JMP_IF_TRUE addr`
- Operands: `addr` (absolute instruction index)
- Stack: `[cond] -> []`
- Requires: `cond` is `Bool`
- Effect:
  - if `cond == True`: `ip := addr`
  - else: continue
- Errors:
  - if `cond` not Bool: `TypeError`
  - if `addr` out of range: `ValueError`

> Normative note: conditional jumps **pop** the condition.

---

### 3.6 Built-ins

Built-ins are pure and restricted to the whitelist:
- `abs`, `min`, `max`, `clip`

#### Builtin ID table (v0.1)
- `0: abs`
- `1: min`
- `2: max`
- `3: clip`

#### `CALL_BUILTIN bid argc`
- Operands: `bid` (builtin ID), `argc` (argument count)
- Stack: `[arg1, ..., argN] -> [ret]` where `N = argc`
  - Arguments are popped from the stack (top is the last argument).
- Errors:
  - if `bid` unknown: `NameError`
  - if `argc` mismatches builtin arity: `TypeError`
  - if type constraints violated: `TypeError`
  - `clip` with `lo > hi`: `ValueError`

Builtin semantics (normative):
- `abs(x)`:
  - `x` must be numeric
- `min(x,y)`, `max(x,y)`:
  - both numeric; promote mixed types
- `clip(x, lo, hi)`:
  - all numeric; promote mixed types
  - require `lo <= hi`
  - return `min(max(x, lo), hi)`

---

### 3.7 Return

#### `RETURN`
- Stack: `[v] -> []`
- Effect: terminate with `Return(v)`
- Errors:
  - if stack empty: `ValueError`

---

## 4. Compiler Conventions (Normative)

### 4.1 Expression Compilation (stack-based)
Typical patterns:
- Binary operator `e1 ⊕ e2`:
  - `compile(e1); compile(e2); OP_⊕`
- Ternary `t if c else f`:
  - `compile(c); JMP_IF_FALSE L_else; compile(t); JMP L_end; L_else: compile(f); L_end:`
- Short-circuit `AND`:
  - `compile(a); JMP_IF_FALSE L_false; compile(b); JMP L_end; L_false: PUSH_CONST false; L_end:`
- Short-circuit `OR`:
  - `compile(a); JMP_IF_TRUE L_true; compile(b); JMP L_end; L_true: PUSH_CONST true; L_end:`

### 4.2 `for x in range(K)` (K constant)
Recommended canonical lowering:
1. initialize loop var local to 0
2. loop condition `i < K`
3. increment `i := i + 1`

Exact lowering is implementation-defined, but must preserve v0.1 semantics.

---

## 5. Limits and GPU-Friendly Constraints (Recommended)
These are recommended constraints for GPU backends; they may be enforced by the compiler:
- `MAX_STACK_DEPTH` (e.g., 64)
- `MAX_LOCALS` (e.g., 16)
- `MAX_CODE_LEN` (e.g., 512)

If enforced, exceeding a limit should yield `ValueError` at compile time (preferred) or runtime.

---

## 6. Conformance Testing (Required)
Backends must satisfy:
- `interp_py(ast) == vm_py(compile_py(ast))`
- `vm_py(bytecode) == vm_gpu(bytecode)`

Comparison rules:
- Exact match for `Int/Bool/None`
- Float match policy is defined in `spec/test_contract.md`
- Error kind must match; message may differ unless frozen explicitly.

---

## 7. Versioning
This is **Bytecode ISA v0.1**. Any opcode/semantics change requires a new version file
(e.g., `bytecode_isa_v0_2.md`) to prevent silent drift.
