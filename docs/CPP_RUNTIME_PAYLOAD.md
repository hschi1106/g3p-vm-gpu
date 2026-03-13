# C++ Runtime Payload Model

This document explains how C++ runtime container values work under `cpp/` and how the exact and fallback paths interact.

## Why This Layer Exists

`g3pvm::Value` must stay compact and trivially copyable so it can move through:

- VM stacks
- bytecode const pools
- CPU/GPU runtime boundaries
- CUDA host/device transfers

That means `Value` does not directly own `std::string` or `std::vector<Value>` payloads.

Instead, `String` and `List` values use a two-layer model:

1. `Value` stores a compact packed token
2. the payload registry stores the real host-side container contents keyed by that token

## Base Container Representation

In [cpp/include/g3pvm/core/value.hpp](../cpp/include/g3pvm/core/value.hpp), `String` and `List` use `Value.i` as:

- upper 16 bits: saturated length
- lower 48 bits: deterministic hash

Helpers:

- `pack_container_payload()`
- `container_len()`
- `container_hash48()`
- `from_string_hash_len()`
- `from_list_hash_len()`

This compact representation is the public runtime transport form for containers.

## Payload Registry

The host-side registry lives in:

- [cpp/include/g3pvm/runtime/payload/payload.hpp](../cpp/include/g3pvm/runtime/payload/payload.hpp)
- [cpp/src/runtime/payload/payload.cpp](../cpp/src/runtime/payload/payload.cpp)

It stores:

- `unordered_map<PayloadKey, std::string>` for strings
- `unordered_map<PayloadKey, std::vector<Value>>` for lists

Where `PayloadKey` is:

- `ValueTag`
- packed `Value.i`

The registry is process-global and protected by a single mutex.

## Main APIs

### Construction

- `payload::make_string_value(s)`
- `payload::make_list_value(elems)`

These functions:

1. compute the packed token
2. register the real payload in the host registry
3. return the compact `Value`

Use these when exact container behavior should be available later.

### Manual registry access

- `register_string()`
- `register_list()`
- `lookup_string()`
- `lookup_list()`
- `clear()`

`clear()` drops the registry contents without invalidating existing `Value` tokens. After that, the same `Value` may still exist, but exact payload lookup will fail.

### Snapshot export

- `snapshot_strings()`
- `snapshot_lists()`

These export registry contents into flat vectors so the GPU path can upload them into device memory.

## GPU Session Handoff

The GPU runtime does not read the host registry directly.

Instead:

1. `payload::snapshot_strings()` and `payload::snapshot_lists()` take a host-side snapshot
2. `build_payload_pack()` in `cpp/src/runtime/gpu/fitness_gpu.cu` flattens those snapshots into:
   - `DStringPayloadEntry` plus one contiguous byte buffer
   - `DListPayloadEntry` plus one contiguous `Value` buffer
3. `FitnessSessionGpu::init()` uploads those flat tables once for the current shared-case session

This separation is important:

- the host registry is convenient for CPU exact behavior and test setup
- the GPU runtime consumes compact, read-only snapshot tables
- shared case locals and expected answers are packed separately from payload snapshots

## Exact Path vs Fallback Path

Container builtins in the CPU and GPU runtimes follow the same high-level policy:

1. try exact payload lookup
2. if lookup or exact materialization is not available, fall back to deterministic compact transport

Exact path:

- `concat` builds the real concatenated string/list
- `slice` builds the real sliced string/list
- `index(string, i)` returns a length-1 `String`
- `index(list, i)` returns the actual element value

Fallback path:

- `concat` returns a new container token from `combine_container_hash48()`
- `slice` returns a new container token from `slice_container_hash48()`
- `index` returns an `Int` token from `index_container_token64()`

This fallback is deterministic and parity-friendly, but not fully semantics-preserving.
The design target is:

- exact CPU/GPU parity when both sides have exact payload access
- deterministic fallback parity when exact payload access or materialization is unavailable

## When Exact Payload Can Be Missing

Exact payload lookup is not guaranteed.

Common cases:

- tests or helper code directly create `Value::from_string_hash_len()` / `Value::from_list_hash_len()` without calling `payload::make_*()`
- random constant generation creates container tokens directly
- registry state was cleared with `payload::clear()`
- GPU exact materialization exceeds bounded per-thread scratch and falls back to compact transport

Because of this, callers must not assume every `String` or `List` value has a recoverable payload behind it.

## `index` Return Type Behavior

`index` is type-stable only on the exact path.

On exact payload lookup:

- `index(string, i)` returns `String`
- `index(list, i)` returns the element's real `Value`, so any `ValueTag` is possible

On fallback:

- both string and list indexing return `Int` tokens

This means `index` result type depends on whether exact payload is available.

## List Hashing Is Shallow

`payload::make_list_value()` hashes list elements with a shallow helper.

For nested `String` and `List` elements, the hash includes their packed token, not a recursive deep re-hash of the full nested payload.

This keeps hashing cheap and stable across CPU/GPU paths, but it means list identity is based on element `Value` identity, not fully expanded recursive content.

## Collision Risk

Container identity is not collision-free.

Risk sources:

- container hashes are truncated to 48 bits in the packed transport form
- registry keys are based on `(tag, packed-token)`

If two different payloads produce the same tag + length + hash48, they alias to the same registry key. Later registration overwrites earlier contents for that key.

This is a compactness/performance tradeoff, not a cryptographic or collision-proof design.

## Design Tradeoffs

Benefits:

- compact `Value`
- cheap stack and bytecode transport
- CPU/GPU-friendly representation
- deterministic fallback when exact payload is unavailable

Costs:

- process-global registry lifetime
- possible token collision
- fallback behavior is not fully semantics-preserving
- `index` return type can degrade to `Int`
- list hashing is shallow

## Files To Read Together

For the full runtime picture, read these in order:

1. [cpp/include/g3pvm/core/value.hpp](../cpp/include/g3pvm/core/value.hpp)
2. [cpp/include/g3pvm/runtime/payload/payload.hpp](../cpp/include/g3pvm/runtime/payload/payload.hpp)
3. [cpp/src/runtime/payload/payload.cpp](../cpp/src/runtime/payload/payload.cpp)
4. [cpp/src/runtime/cpu/builtins_cpu.cpp](../cpp/src/runtime/cpu/builtins_cpu.cpp)
5. GPU mirrors under `cpp/src/runtime/gpu/`
