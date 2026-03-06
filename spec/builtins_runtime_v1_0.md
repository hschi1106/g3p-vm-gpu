# Built-ins v1.0 Extension

This file extends `spec/builtins_base_v1_0.md` and documents the current v1.0 runtime behavior.

Base rules from v0.1 remain unchanged for:
- `abs(x)`
- `min(x, y)`
- `max(x, y)`
- `clip(x, lo, hi)`

## Added built-in

- `len(x)`
- `concat(a, b)`
- `slice(x, lo, hi)`
- `index(x, i)`

### Signature
- `len(x: String|List) -> Int`

### Arity
- Exactly 1 argument.

### Semantics
- Returns container length.
- `len` reads container length from compact metadata and matches exact payload length.

### Type rules
- Non `String/List` argument => `TypeError`.

## Built-in ID mapping extension

On top of v0.1 mapping:
- `4: len`
- `5: concat`
- `6: slice`
- `7: index`

## `concat(a, b)` semantics

### Signature
- `concat(a: String|List, b: String|List) -> String|List`

### Arity
- Exactly 2 arguments.

### Semantics
- `String + String` returns `String`.
- `List + List` returns `List`.
- Mixed tags are not allowed.
- Exact path: concatenate payloads and return a same-tag container value.
- Compact path: use deterministic compact transport metadata for throughput and fallback.

### Type rules
- Non `(String,String)` / `(List,List)` inputs => `TypeError`.

## `slice(x, lo, hi)` semantics

### Signature
- `slice(x: String|List, lo: Int, hi: Int) -> String|List`

### Arity
- Exactly 3 arguments.

### Semantics
- Uses Python-like slicing behavior for index normalization and bounds clamp.
- Return tag is the same as `x`.
- Exact path: slice payload using normalized bounds.
- Compact path: use deterministic compact transport metadata.
- Result length follows normalized slice length.

### Type rules
- First arg must be `String` or `List`.
- `lo` and `hi` must be `Int`.

## `index(x, i)` semantics

### Signature
- `index(x: String|List, i: Int) -> Any`

### Arity
- Exactly 2 arguments.

### Semantics
- Uses Python indexing behavior with negative index normalization.
- Out-of-range index returns `ValueError`.
- Python interpreter/VM returns element payload (same as `x[i]`).
- C++ CPU path returns exact payload value when payload registry is available; otherwise fallback is deterministic compact token.
- C++ GPU path executes exact payload indexing when uploaded payload/scratch capacity is sufficient; otherwise fallback is deterministic compact token.

### Type rules
- First arg must be `String` or `List`.
- `i` must be `Int`.

## Comparison extension

For VM comparison ops:
- `EQ`/`NE` are defined for `String` vs `String`, and `List` vs `List`.
- Ordering (`LT/LE/GT/GE`) for container types remains unsupported (`TypeError` path).

## Payload execution model

- CPU runtime stores decoded `String/List` payloads in a payload registry.
- GPU runtime uploads payload snapshots into device global memory before kernel evaluation.
- Device builtins materialize exact payload results through bounded per-thread scratch.
- If exact GPU payload materialization cannot fit in bounded scratch, the runtime falls back to deterministic compact transport rather than failing the whole evaluation.

## Fitness rule

Evolution fitness is mixed per case:

- numeric expected/actual => `-abs(actual - expected)`
- `Bool/None/String/List` exact match => `+1`
- `Bool/None/String/List` mismatch => `+0`
- runtime error => `+0`
