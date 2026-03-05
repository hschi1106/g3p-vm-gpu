# Built-ins v1.0 Extension

This file extends `spec/builtins.md` for v1.0 runtime work.

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
- For current v1.0 transport, `String/List` are packed as hash+length in `Value`.
- `len` reads the encoded length field.

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
- For current v1.0 transport, result uses deterministic hash-combine and length add (saturating to 65535).

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
- For current v1.0 transport, result uses deterministic hash-combine with `(src_hash, src_len, lo, hi)`.
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
- C++ CPU path returns exact payload value when payload registry is available; otherwise fallback is deterministic `Int` token.
- C++ GPU path currently uses deterministic `Int` token for indexed element identity.

### Type rules
- First arg must be `String` or `List`.
- `i` must be `Int`.

## Comparison extension

For VM comparison ops:
- `EQ`/`NE` are defined for `String` vs `String`, and `List` vs `List`.
- Ordering (`LT/LE/GT/GE`) for container types remains unsupported (`TypeError` path).
