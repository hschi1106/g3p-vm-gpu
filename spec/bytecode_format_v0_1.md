# Bytecode JSON Format v0.1

This document defines the stable JSON wire format used to send bytecode programs to C++/GPU runtimes.

## 1. Versioning

- Request payload must include:
  - `"format_version": "bytecode-json-v0.1"`
- Fixture files may include:
  - `"format_version": "bytecode-fixture-v0.1"`
  - `"meta.bytecode_format_version": "bytecode-json-v0.1"`

Any incompatible format change must use a new version string.

## 2. Run Request Schema

Top-level JSON object:

```json
{
  "format_version": "bytecode-json-v0.1",
  "fuel": 20000,
  "bytecode": {
    "n_locals": 3,
    "consts": [Value, Value, ...],
    "code": [Instr, Instr, ...]
  },
  "inputs": [
    {"idx": 0, "value": Value}
  ]
}
```

### 2.1 Value

```json
{"type": "none"}
{"type": "bool", "value": true}
{"type": "int", "value": 123}
{"type": "float", "value": 1.5}
```

Rules:
- `int.value` must be an integer JSON number (no fraction).
- `float.value` must be a JSON number.
- `bool.value` must be a JSON boolean.

### 2.2 Instr

```json
{"op": "ADD", "a": null, "b": null}
{"op": "PUSH_CONST", "a": 0, "b": null}
{"op": "CALL_BUILTIN", "a": 3, "b": 3}
```

Rules:
- `op` is required.
- `a`/`b` are optional operands.
  - absent or `null` means no operand.
  - integer number means operand present.

## 3. Fixture File Schema

Fixture root:

```json
{
  "format_version": "bytecode-fixture-v0.1",
  "meta": {...},
  "cases": [Case, Case, ...]
}
```

Each `Case` contains:
- `id`, `bucket`, `seed`, `depth`, `fuel`
- `bytecode` (same structure as request `bytecode`)
- `expected` (`{"kind":"return","value":Value}` or `{"kind":"error","code":"TypeError"}`)

## 4. Runtime Output

Current C++ CLI output is line-oriented:
- Success: `OK <typed-value>`
- Error: `ERR <ErrorCodeName>`

Where typed value is:
- `int N`
- `float X`
- `bool 0|1`
- `none`
