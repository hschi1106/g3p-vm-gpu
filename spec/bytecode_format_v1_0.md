# Bytecode Format v1.0

This document defines the current JSON wire format used by runtime-facing tools and CLIs.

## Current Format Strings

The current implementation accepts these format identifiers:
- request payload: `bytecode-json-v0.1`
- fixture payload: `bytecode-fixture-v0.1`

These wire strings are part of the current compatibility contract.
The document filename is `v1_0` because it is the current repository spec set, not because the wire string changed.

## Runtime Request Schema

Top-level object:

```json
{
  "format_version": "bytecode-json-v0.1",
  "engine": "cpu",
  "fuel": 20000,
  "programs": [
    {
      "n_locals": 3,
      "consts": [Value, Value],
      "code": [Instr, Instr]
    }
  ],
  "shared_cases": [
    [{"idx": 0, "value": {"type": "int", "value": 1}}]
  ],
  "shared_answer": [{"type": "int", "value": 2}]
}
```

Fields:
- `format_version`: optional in some ad-hoc paths, but when present must be `bytecode-json-v0.1`
- `engine`: optional request hint; raw runtime execution is CPU-oriented, while GPU is primarily exposed through fitness mode
- `fuel`: required execution budget in the current runtime CLI request path
- `programs`: required list of bytecode programs
- `shared_cases`: required shared input cases for batched execution
- `shared_answer`: optional expected outputs used by the fitness path

## Value Encoding

Supported values:

```json
{"type": "none"}
{"type": "bool", "value": true}
{"type": "int", "value": 123}
{"type": "float", "value": 1.5}
{"type": "string", "value": "abc"}
{"type": "list", "value": [1, 2, 3]}
```

Rules:
- `int.value` must be an integer JSON number
- `float.value` must be a JSON number
- `bool.value` must be a JSON boolean
- `string.value` must be a JSON string
- `list.value` must be a JSON array of recursively encodable values supported by the current runtime/tooling path

## Instruction Encoding

Instruction object:

```json
{"op": "ADD"}
{"op": "PUSH_CONST", "a": 0}
{"op": "CALL_BUILTIN", "a": 3, "b": 3}
```

Rules:
- `op` is required
- `a` and `b` are optional integer operands
- missing or `null` means the operand is not present

## Runtime CLI Modes

### Raw execution mode
When `shared_answer` is absent:
- CPU engine executes the programs over the shared cases and returns raw execution results
- GPU raw execution is not exposed anymore in the current CLI; GPU requests without `shared_answer` are rejected

### Fitness mode
When `shared_answer` is present:
- CPU engine returns one fitness value per program
- GPU engine returns one fitness value per program

## Current Output

The runtime CLI uses line-oriented output.

Raw execution success:
- `OK <typed-value>`

Raw execution error:
- `ERR <ErrorCodeName>`
- optional `MSG <text>`

Fitness success:
- `OK fitness_count <N>`
- `FIT <idx> <score>` repeated per program

Fitness error:
- `ERR <ErrorCodeName>`
- optional `MSG <text>`
