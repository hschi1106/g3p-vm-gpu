# C++ Runtime Payload Model

This document explains how C++ runtime container values work under `cpp/` and how the exact and fallback paths interact.

## Why This Layer Exists

`g3pvm::Value` must stay compact and trivially copyable so it can move through:

- VM stacks
- bytecode const pools
- CPU/GPU runtime boundaries
- CUDA host/device transfers

That means `Value` does not directly own `std::string` or `std::vector<Value>` payloads.

Instead, `String`, `NumList`, and `StringList` values use a two-layer model:

1. `Value` stores a compact packed token
2. the payload registry stores the real host-side container contents keyed by that token

## Base Container Representation

In [cpp/include/g3pvm/core/value.hpp](../cpp/include/g3pvm/core/value.hpp), `String`, `NumList`, and `StringList` use `Value.i` as:

- upper 16 bits: saturated length
- lower 48 bits: deterministic hash

Helpers:

- `pack_container_payload()`
- `container_len()`
- `container_hash48()`
- `from_string_hash_len()`
- `from_num_list_hash_len()`
- `from_string_list_hash_len()`

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
- `payload::make_num_list_value(elems)`
- `payload::make_string_list_value(elems)`

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

These export registry contents into flat vectors.
They remain useful for diagnostics and offline tooling, but the production GPU fitness path no longer snapshots the full registry at session start.

### By-token lookup

- `lookup_string_packed()`
- `lookup_list_packed()`

`lookup_list_packed()` takes the typed-list `ValueTag` as part of the lookup key, so `NumList` and `StringList` payloads with the same compact token do not alias across tags.
These let the GPU fitness session lazily fetch only the payload tokens it actually needs for the current accepted population.

## GPU Session Handoff

The GPU runtime does not read the host registry directly from device code.

Instead:

1. `FitnessSessionGpu::init()` caches shared-case payload tokens for the current session and lazily preloads only those shared tokens
2. `FitnessSessionGpu::eval_programs()` gathers the payload tokens actually needed by the accepted program subset
3. missing tokens are fetched from the process-global payload registry by packed token and inserted into a session-local host cache
4. `build_payload_pack()` in `cpp/src/runtime/gpu/fitness_gpu.cu` flattens only that needed-token closure into:
   - `DStringPayloadEntry` plus one contiguous byte buffer
   - `DListPayloadEntry` plus one contiguous `Value` buffer
5. the compact pack is uploaded for the current evaluation run

This separation is important:

- the host registry is convenient for CPU exact behavior and test setup
- the GPU runtime consumes compact, read-only per-eval payload tables instead of the full registry
- shared case locals and expected answers are packed separately from payload snapshots
- device lookup for global payload entries is done against compact sorted tables

## GPU Payload Flavors

The device path now uses one production eval kernel family.

Programs are still classifiable into four fine-grained payload flavors in `cpp/src/runtime/gpu/host_pack_gpu.cu`:

- `None`
- `StringOnly`
- `ListOnly`
- `Mixed`

The current production GPU fitness path always launches a single `Mixed` eval kernel over the full accepted population.

The finer `StringOnly` / `ListOnly` labels are kept for experiment tooling and offline bucket studies rather than the production eval dispatch tree.

Exact string/typed-list builtins still use bounded per-thread scratch and still fall back to compact transport when exact materialization does not fit.

Operationally, this means production GPU eval no longer maintains a runtime dispatch split between payload-free and payload-bearing programs. Timing and benchmark analysis should treat `gpu_eval_kernel_ms` as one kernel family rather than reconstructing legacy `None` / `Mixed` launch buckets.

## Exact Path vs Fallback Path

Container builtins in the CPU and GPU runtimes follow the same high-level policy:

1. try exact payload lookup
2. if lookup or exact materialization is not available, return an opaque fallback result

Exact path:

- `concat` builds the real concatenated string or typed list
- `slice` builds the real sliced string or typed list
- `append` builds the real typed list with one additional element
- `reverse` builds the real reversed string or typed list
- `find` and `contains` inspect exact string payloads
- `index(string, i)` returns a length-1 `String`
- `index(NumList, i)` returns the numeric element value
- `index(StringList, i)` returns the string element value

Fallback path:

- `concat` returns `FallbackToken`
- `slice` returns `FallbackToken`
- `append` returns `FallbackToken`
- `reverse` returns `FallbackToken`
- `index` returns `FallbackToken`
- `find` and `contains` return `ValueError` when exact string payload lookup is unavailable

This fallback is deterministic and parity-friendly, but not fully semantics-preserving.
The design target is:

- exact CPU/GPU parity when both sides have exact payload access
- deterministic fallback parity when exact payload access or materialization is unavailable

## When Exact Payload Can Be Missing

Exact payload lookup is not guaranteed.

Common cases:

- tests or helper code directly create `Value::from_string_hash_len()` / `Value::from_num_list_hash_len()` / `Value::from_string_list_hash_len()` without calling `payload::make_*()`
- random constant generation creates container tokens directly
- registry state was cleared with `payload::clear()`
- GPU exact materialization exceeds bounded per-thread scratch and returns `FallbackToken`

Because of this, callers must not assume every `String`, `NumList`, or `StringList` value has a recoverable payload behind it.

## `index` Return Type Behavior

`index` is type-stable on exact typed-list payloads.

On exact payload lookup:

- `index(string, i)` returns `String`
- `index(NumList, i)` returns `Int` or `Float`
- `index(StringList, i)` returns `String`

On fallback:

- string and typed-list indexing return `FallbackToken`

This means the exact result type is predictable from the typed-list tag, while missing payloads still produce the deterministic fallback token.

## Typed-List Hashing Is Shallow

`payload::make_num_list_value()` and `payload::make_string_list_value()` hash list elements with a shallow helper.

Nested lists are not part of the public typed-list contract. If helper or compatibility code ever constructs container-valued elements, the hash includes the nested container's packed token rather than a recursive deep re-hash of the full nested payload.

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
- fallback results are opaque and intended to compare as mismatches in fitness/equality paths
- typed-list hashing is shallow

## Files To Read Together

For the full runtime picture, read these in order:

1. [cpp/include/g3pvm/core/value.hpp](../cpp/include/g3pvm/core/value.hpp)
2. [cpp/include/g3pvm/runtime/payload/payload.hpp](../cpp/include/g3pvm/runtime/payload/payload.hpp)
3. [cpp/src/runtime/payload/payload.cpp](../cpp/src/runtime/payload/payload.cpp)
4. [cpp/src/runtime/cpu/builtins_cpu.cpp](../cpp/src/runtime/cpu/builtins_cpu.cpp)
5. GPU mirrors under `cpp/src/runtime/gpu/`
