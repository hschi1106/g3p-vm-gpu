# Bytecode ISA v1.0

This document defines the bytecode execution contract shared by the compiler, CPU runtime, and GPU runtime.

Semantics must conform to [grammar_v1_0.md](./grammar_v1_0.md).

## Execution Model

A VM instance maintains:
- `ip`: instruction pointer
- `stack`: operand stack
- `locals`: local variable array
- `consts`: constant pool
- `fuel`: remaining instruction budget

A bytecode program contains:
- `consts`
- `n_locals`
- `code`

## Value Domain

Runtime values are:
- `Int`
- `Float`
- `Bool`
- `None`
- `String`
- `List`

Type rules:
- `Bool` is not numeric.
- Conditional jumps require `Bool`.
- Container ordering comparisons are invalid.

## Termination And Errors

Execution ends with one of:
- `Return(v)` when `RETURN` succeeds
- `Error(kind, msg)` on the first error
- `Error(ValueError, "program finished without return")` when `ip` reaches the end without `RETURN`
- `Error(Timeout, "out of fuel")` when fuel is exhausted

Supported error kinds:
- `NameError`
- `TypeError`
- `ZeroDiv`
- `ValueError`
- `Timeout`

## Fuel Rule

Before each instruction:
- if `fuel <= 0`, return `Timeout`
- otherwise execute one instruction and decrement fuel by one

## Instruction Format

Each instruction has:
- `op`: opcode
- `a`: optional integer operand
- `b`: optional integer operand

The physical packing is implementation-defined. The abstract semantics are fixed.

## Opcodes

### Constants and locals

#### `PUSH_CONST k`
Push `consts[k]`.
- invalid `k` => `ValueError`

#### `LOAD i`
Push `locals[i]`.
- invalid or uninitialized `i` => `NameError`

#### `STORE i`
Pop one value and assign it to `locals[i]`.
- invalid `i` => `NameError`

### Unary ops

#### `NEG`
Pop one numeric value and push its negation.
- non-numeric => `TypeError`

#### `NOT`
Pop one `Bool` and push logical negation.
- non-`Bool` => `TypeError`

### Arithmetic ops

Arithmetic ops require numeric operands and use `Int`/`Float` promotion.

#### `ADD`, `SUB`, `MUL`
Pop two numeric operands and push the result.
- any non-numeric operand => `TypeError`

#### `DIV`
Pop two numeric operands and push division result.
- any non-numeric operand => `TypeError`
- zero divisor => `ZeroDiv`

#### `MOD`
Pop two numeric operands and push modulo result.
- any non-numeric operand => `TypeError`
- zero divisor => `ZeroDiv`

### Comparisons

All comparison ops push `Bool`.

#### `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`
Rules:
- numeric comparisons require numeric operands and use promotion
- `None` only supports `EQ` and `NE`
- `Bool` only supports `EQ` and `NE` with `Bool`
- `String` and `List` only support `EQ` and `NE` with the same tag
- invalid comparison combinations => `TypeError`

### Control flow

#### `JMP addr`
Set `ip = addr`.
- invalid `addr` => `ValueError`

#### `JMP_IF_FALSE addr`
Pop one `Bool`.
- if value is `False`, jump to `addr`
- if value is `True`, continue
- non-`Bool` => `TypeError`
- invalid `addr` => `ValueError`

#### `JMP_IF_TRUE addr`
Pop one `Bool`.
- if value is `True`, jump to `addr`
- if value is `False`, continue
- non-`Bool` => `TypeError`
- invalid `addr` => `ValueError`

Compiler lowering notes:
- `ForRange(x, e, body)` first evaluates `e` once, stores the bound in a temporary local, validates that the bound is a non-negative integer, and then reuses that stored bound for every iteration.
- The compiler may emit internal builtin id `8` (`is_int`) while lowering `ForRange`; this builtin is not part of the source-language call whitelist.

### Builtins

Builtins are invoked by:
- `CALL_BUILTIN bid argc`

The VM pops `argc` arguments and pushes one return value.

Builtin ID mapping:
- `0`: `abs`
- `1`: `min`
- `2`: `max`
- `3`: `clip`
- `4`: `len`
- `5`: `concat`
- `6`: `slice`
- `7`: `index`
- `8`: `is_int` (compiler-internal helper used by `ForRange` lowering)

Errors:
- unknown builtin id => `NameError`
- wrong arity => `TypeError`
- type or value rule violation => builtin-specific error, usually `TypeError` or `ValueError`

Builtin semantics are defined in:
- [builtins_base_v1_0.md](./builtins_base_v1_0.md)
- [builtins_runtime_v1_0.md](./builtins_runtime_v1_0.md)

### Return

#### `RETURN`
Pop one value and terminate successfully.
- empty stack => `ValueError`
