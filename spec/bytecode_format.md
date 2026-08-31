# Bytecode Format

This document defines the current JSON wire format used by bytecode parity tooling,
runtime harnesses, and fixture-driven tests.

See also:

- [grammar.md](./grammar.md)
- [bytecode_isa.md](./bytecode_isa.md)

## Format Strings

- request payload: `bytecode-json`
- fixture payload: `bytecode-fixture`
- AST payload: `ast-prefix`

Old format strings are not valid current payloads.

## Harness Request Schema

```json
{
  "format_version": "bytecode-json",
  "engine": "cpu",
  "fuel": 20000,
  "programs": [
    {
      "n_locals": 3,
      "consts": [Value],
      "code": [Instr],
      "segments": {}
    }
  ],
  "shared_cases": [
    [{"idx": 0, "value": {"type": "int", "value": 1}}]
  ],
  "shared_answer": [{"type": "int", "value": 2}]
}
```

Fields:

- `format_version` must be `bytecode-json`.
- `engine` is an optional request hint.
- `fuel` is the required per-case execution budget.
- `programs` is the required list of bytecode programs.
- `shared_cases` is the required list of shared input cases.
- `shared_answer` is optional expected-output data for fitness paths.

## Value Encoding

Supported values:

```json
{"type": "bool", "value": true}
{"type": "int", "value": 123}
{"type": "float", "value": 1.5}
{"type": "char", "value": "a"}
{"type": "string", "value": "abc"}
{"type": "int_list", "value": [1, 2, 3]}
{"type": "float_list", "value": [1, 2.5, 3]}
{"type": "string_list", "value": ["a", "bc"]}
```

Invalid value tags:

- `none`
- `num_list`
- generic `list`
- `char_list`
- public `fallback_token`

Rules:

- `char.value` must contain exactly one supported Unicode scalar value.
- `string.value` is a sequence of `Char` values.
- list tags are direct and exact.
- empty lists are valid because the tag is explicit.

## Instruction Encoding

Instruction objects use opcode names from
[bytecode_isa.md](./bytecode_isa.md):

```json
{"op": "PUSH_CONST", "a": 0, "b": 0}
{"op": "CALL_BUILTIN", "a": 7, "b": 2}
```

Fields:

- `op`: required opcode string.
- `a`: integer operand; required by opcodes that consume operand `a` and
  otherwise optional/ignored.
- `b`: integer operand; required by `CALL_BUILTIN` and otherwise
  optional/ignored.

For `CALL_BUILTIN`, `a` is the builtin id and `b` is the arity.
Missing required operands are malformed bytecode; they are not interpreted as
zero.

## Bytecode Program Encoding

```json
{
  "n_locals": 4,
  "consts": [Value],
  "code": [Instr],
  "segments": {
    "dc_solve_0": Segment,
    "dc_divide_0": Segment,
    "dc_combine_0": Segment
  },
  "metadata": {}
}
```

Required fields:

- `n_locals`
- `consts`
- `code`

Optional fields:

- `segments`: ASGP or optimized structured-expression phase segments.
- `metadata`: non-semantic diagnostics.

Segment encoding:

```json
{
  "n_locals": 3,
  "consts": [Value],
  "code": [Instr],
  "binder_locals": [
    {"name": 0, "local": 0}
  ],
  "metadata": {}
}
```

`binder_locals` maps source binder name ids to phase-local slots. It may be
omitted for a phase with no binders, but every required ASGP phase role must
have an in-range mapping. ASGP phase execution uses this map to bind solve,
divide, combine, dependency, or transition parameters before executing the
segment code.

The implementation-defined ASGP segment payloads are encoded under
`segments`:

```json
{
  "segments": {
    "asgp_dc": [
      {
        "solve_xs_name": 0,
        "solve_n_name": 1,
        "solve_lo_name": 2,
        "divide_n_name": 3,
        "combine_left_name": 4,
        "combine_right_name": 5,
        "solve": Segment,
        "divide": Segment,
        "combine": Segment
      }
    ],
    "asgp_dp1d": [
      {
        "lo": 0,
        "hi": 10,
        "base_state": 0,
        "boundary_value": Value,
        "dep_kind": -1,
        "dep_offsets": [1],
        "solve_state_name": 0,
        "transition_state_name": 1,
        "transition_dep_names": [2],
        "solve": Segment,
        "transition": Segment
      }
    ],
    "asgp_dp2d": [
      {
        "i_lo": 0,
        "i_hi": 10,
        "j_lo": 0,
        "j_hi": 10,
        "base_i": 0,
        "base_j": 0,
        "boundary_value": Value,
        "dep_kind": 0,
        "solve_i_name": 0,
        "solve_j_name": 1,
        "transition_i_name": 2,
        "transition_j_name": 3,
        "transition_dep_names": [4, 5],
        "solve": Segment,
        "transition": Segment
      }
    ]
  }
}
```

Main bytecode refers to ASGP segment arrays by zero-based segment index in the
`a` operand of `ASGP_DC`, `ASGP_DP1D`, and `ASGP_DP2D`.

## Decode-Time Verification

Native JSON decoders verify each complete program before returning it. The
verification contract covers:

- public constant tags and Unicode scalar `Char` constants;
- required operands and constant/local/builtin/segment index ranges;
- builtin bytecode arity and jump targets;
- control-flow stack depth and compatible stack types at joins;
- reachable stack underflow and main-program fallthrough;
- local, variable-map, binder-local, and phase-program ranges;
- ASGP bounds, dependency metadata, phase binders, and forbidden nested ASGP
  calls inside phase bytecode;
- optional instruction, constant, local, segment, and stack resource limits.

Verification rejects malformed representation and unsafe control-flow shape.
It does not replace runtime semantics: bytecode that is structurally valid but
deliberately evaluates to `TypeError`, `NameError`, `ZeroDiv`, or `Timeout`
remains decodable and produces that runtime result.

ASGP-DP segment arity constraints:

- `asgp_dp1d.dep_kind` is `-1` for backward dependencies and `1` for forward
  dependencies.
- `asgp_dp1d.dep_offsets` must be non-empty.
- `asgp_dp1d.dep_offsets` and `asgp_dp1d.transition_dep_names` must have the
  same length.
- `asgp_dp2d.dep_kind` encodes the selected 2D dependency pattern:
  `0 = cross_backward`, `1 = cross_forward`, `2 = diagonal_backward`,
  `3 = diagonal_forward`, `4 = neighborhood_backward3`,
  `5 = neighborhood_forward3`.
- `asgp_dp2d.transition_dep_names` length must match the selected pattern:
  cross patterns use arity 2, diagonal patterns use arity 1, and neighborhood
  patterns use arity 3.
- Decoders must reject malformed ASGP-DP segment metadata before execution.

## AST Encoding

Public AST encoding uses:

```json
{
  "version": "ast-prefix",
  "nodes": [AstNode],
  "names": ["input1"],
  "consts": [Value],
  "linear_rec_binders": [],
  "asgp_dc_binders": [],
  "asgp_dp1d_specs": [],
  "asgp_dp2d_specs": []
}
```

Node encoding:

```json
{"kind": 0, "i0": 0, "i1": 0}
```

`kind` is the current numeric `NodeKind` value written by the native AST
serializer. `i0` and `i1` are required integers. The native verifier rejects
unknown kinds and invalid uses of the index fields before compilation.

Current optional side-table arrays are:

- `linear_rec_binders`: `node_index`, `elem_name`, `accum_name`, and
  `index_name`;
- `asgp_dc_binders`: `node_index`, `solve_xs_name`, `solve_n_name`,
  `solve_lo_name`, `divide_n_name`, `combine_left_name`, and
  `combine_right_name`;
- `asgp_dp1d_specs`: `node_index`, `lo`, `hi`, `base_state`,
  `boundary_const`, numeric dependency-node `dep_kind`, positive
  `dep_offsets`, `solve_state_name`, `transition_state_name`, and
  `transition_dep_names`;
- `asgp_dp2d_specs`: `node_index`, both dimension bounds, the base cell,
  `boundary_const`, numeric dependency-node `dep_kind`, solve/transition state
  names, and `transition_dep_names`.

Omitted side-table arrays are treated as empty. Every entry is owned by exactly
one matching node index. Their verified semantics satisfy:

- capture-safe `BoundVar` references.
- exact result-type reconstruction.
- ASGP phase and visibility reconstruction.
- deterministic AST cache keys.

The `--eval-ast-json` boundary derives exact input name/type declarations from
the selected `fitness-cases` file and runs the full structural, scope, binder,
and type verifier before building genome metadata or compiling.

## DP Bounds Encoding

`DpBounds1D`:

```json
{
  "kind": "dp_bounds_1d",
  "lo": 0,
  "hi": 10,
  "base": [{"state": 0}],
  "boundary_value": Value,
  "max_step": 3
}
```

`DpBounds2D`:

```json
{
  "kind": "dp_bounds_2d",
  "i_lo": 0,
  "i_hi": 10,
  "j_lo": 0,
  "j_hi": 10,
  "base": [{"i": 0, "j": 0}],
  "boundary_value": Value
}
```

Bounds are inclusive at `lo` and exclusive at `hi`.

The `boundary_value` type must equal the ASGP-DP result type.

## Fixture Schema

```json
{
  "format_version": "bytecode-fixture",
  "fuel": 20000,
  "scenarios": [
    {
      "intent": "integer addition preserves the Int tag",
      "program": BytecodeProgram,
      "fuel": 100,
      "cases": [
        {
          "inputs": [{"idx": 0, "value": Value}],
          "expected": Value
        },
        {
          "inputs": [],
          "expected_error": "TypeError"
        }
      ]
    }
  ]
}
```

`fuel` is required at the fixture root and may be overridden by a scenario.
`scenarios` must be non-empty. Every scenario has a non-empty semantic
`intent`, one verified `BytecodeProgram`, and a non-empty `cases` array. Each
case supplies input bindings and exactly one of:

- `expected`, compared by exact public value tag and payload content;
- `expected_error`, compared to the exact public error name (`NameError`,
  `TypeError`, `ZeroDivisionError`, `ValueError`, or `Timeout`).

The earlier single-program shape with top-level `program` and `cases` remains
accepted as a compatibility shorthand for one scenario. New corpus files use
the scenario form so every group carries its semantic intent.

Fixtures must not contain retired value tags. Program decoding runs the
bytecode verifier before any case is executed, so malformed representation is
covered by verifier/codec tests rather than encoded as a runtime-error case.
