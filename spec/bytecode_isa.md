# Bytecode ISA

This document defines the release 1.0.0 bytecode execution contract shared by the
compiler, CPU runtime, and GPU runtime.

Semantics must conform to:

- [grammar.md](./grammar.md)
- [builtins_base.md](./builtins_base.md)
- [builtins_runtime.md](./builtins_runtime.md)

The `bytecode-json` representation is defined by
[bytecode_format.md](./bytecode_format.md).

## Compatibility

Release 1.0.0 is a breaking bytecode contract relative to legacy artifacts.

- Old serialized values containing `None` are invalid.
- Old serialized values containing `NumList` are invalid.
- Old AST programs using `ast-prefix-old` are invalid as public `ast-prefix`
  inputs.
- Release 1.0.0 uses exact `Int`, `Float`, `Bool`, `Char`, `String`, `IntList`,
  `FloatList`, and `StringList` value tags.

Implementations may keep private migration tooling, but native runtime entrypoints
must not silently reinterpret old public values.

## Execution State

A VM instance has:

- instruction pointer
- operand stack
- local variable array
- constant pool
- remaining fuel
- internal local initialization state
- optional hidden locals
- optional structured-expression and ASGP frame state

Public `None` does not exist. Uninitialized storage must be tracked by an
internal validity bit, unset marker, or equivalent mechanism that cannot be
observed as a source-language value.

## Public Opcodes

The baseline public opcode set remains:

- `PUSH_CONST`
- `LOAD`
- `STORE`
- `NEG`
- `NOT`
- `ADD`
- `SUB`
- `MUL`
- `DIV`
- `MOD`
- `LT`
- `LE`
- `GT`
- `GE`
- `EQ`
- `NE`
- `JMP`
- `JMP_IF_FALSE`
- `JMP_IF_TRUE`
- `CALL_BUILTIN`
- `RETURN`

Only this public opcode set is portable between bytecode consumers. Source
forms may be lowered into it plus hidden locals and bytecode segments.

Current native private helper opcodes for structured-expression lowering are:

- `CHECK_LIST`
- `CHECK_INT`
- `EMPTY_LIST`
- `EMPTY_LIST_LIKE`
- `ASGP_DC`
- `ASGP_DP1D`
- `ASGP_DP2D`

These helpers belong to the native g3pvm private execution profile. The native
`bytecode-json` codec and verifier accept them with their required segment and
metadata checks, but another consumer may reject them. No private helper opcode
adds source-language behavior beyond the public grammar contract.

## Fuel and Errors

Before each instruction or equivalent lowered operation:

```text
if fuel == 0:
    return Timeout
else:
    fuel -= 1
    execute operation
```

Structured expressions and ASGP schemes must not bypass fuel accounting.

Runtime error kinds:

- `NameError`
- `TypeError`
- `ZeroDiv`
- `ValueError`
- `Timeout`

The first runtime error terminates the current case execution.

Deterministic ASGP stack, memo, or payload overflow must map to a runtime
error result and must be treated as `-penalty` by fitness evaluation.

## Verification Boundary

Native external bytecode decoding must run the verifier described in
[bytecode_format.md](./bytecode_format.md) before execution. Debug native
compiler builds also verify their completed output. This check is a boundary
and debug invariant, not a per-case execution step or a per-individual release
hot-path requirement.

Malformed operands, indices, CFG stack joins, and segment metadata are decode
errors. Well-formed instructions whose values cause a language runtime error
remain executable so fixture suites can assert the error behavior above.

## Type-Sensitive Operators

Arithmetic operators accept `Int` and `Float` only. `Bool` and `Char` are not
numeric.

Ordering comparisons accept `Int` and `Float` only.

Equality and inequality require exact same runtime type. Comparing different
runtime types is a `TypeError` in generated well-typed programs and must not be
used as an implicit conversion mechanism.

## Builtin IDs

The builtin ID mapping below is normative for release 1.0.0 bytecode serialization.

```text
0   abs
1   min
2   max
3   clip
4   len
5   concat
6   slice
7   index
8   append
9   reverse
10  find
11  contains
12  is_int          compiler-internal helper, not source syntax
13  idiv0
14  imod0
15  prepend
16  char_to_string
17  string_to_char
18  ord
19  chr
20  is_letter
21  is_digit
22  is_space
23  is_vowel
24  to_lower
25  to_upper
26  to_string
27  singleton
```

`is_int` remains compiler-internal and is not a source-language call.

## Bound Variable Lowering

`BoundVar` is a source AST distinction used to preserve lexical scope during
generation, variation, and compilation.

Required compiler behavior:

- each structured or scheme binder maps to a hidden immutable slot or
  equivalent internal location.
- ordinary source `Var` cannot access hidden binder slots.
- source `Assign` cannot write binder slots.
- nested binders must resolve capture-safely.
- hidden slots may be reused only when lifetimes do not overlap.

## Structured List Lowering

`MapList`, `FilterList`, and `LinearRec` must preserve the evaluation order
defined in [grammar.md](./grammar.md).

Required behavior:

- source expressions are evaluated exactly once.
- binder values are assigned before body evaluation.
- bodies are evaluated exactly once when selected by the grammar semantics.
- first selected body error terminates execution.
- empty list results preserve exact static list tags.
- `LinearRec` is equivalent to right-to-left recurrence and may be lowered to
  an iterative reverse traversal.

## ASGP Execution

ASGP schemes may be represented as bytecode phase segments or equivalent
runtime-internal segments.

Required phase isolation:

- DC `solve` reads only `xs`, `n`, `lo`, and constants.
- DC `ordivide` reads only `n` and constants.
- DC `andcombine` reads only `r1`, `r2`, and constants.
- DP solve phases read only their state binders and constants.
- DP transition phases read only their state binders, dependency values, and
  constants.

ASGP implementation requirements:

- DC split is `clamp(ordivide(n), 1, n - 1)`.
- DP dependency expansion must be acyclic by grammar construction.
- out-of-bounds DP dependencies use the configured boundary value.
- memo hits must not re-evaluate already memoized states.
- DP2D phase segments carry required bounds, base cell, boundary value, and
  dependency-pattern metadata equivalent to the source grammar parameters.
- CPU implementations may use bounded recursion or explicit stacks.
- GPU implementations must use explicit bounded stacks, iterative schedules,
  or equivalent non-recursive execution.
