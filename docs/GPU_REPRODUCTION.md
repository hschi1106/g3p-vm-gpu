# GPU Reproduction Backend

## Purpose

The GPU reproduction backend moves one full generation of selection and variation off the CPU hot path while preserving the public GP contract:

- public program representation remains prefix `AstProgram`
- public selection remains tournament-based and controlled by `selection_pressure`
- public crossover remains `typed_subtree`
- public mutation remains the single public mutation path controlled by `mutation_subtree_prob`

The backend is designed to reduce one-generation benchmark cost and to support overlap with GPU fitness evaluation. It is a performance implementation detail, not a new public GP dialect.

Canonical timing names and CLI/JSON output mapping are documented in [TIMING.md](TIMING.md).

## Public Controls

The public control plane is:

- `--repro-backend {cpu|gpu}`
- `--repro-overlap {on|off}`

The benchmark script exposes four formal modes built from those controls:

- `cpu`
- `gpu_eval`
- `gpu_repro`
- `gpu_repro_overlap`

`gpu_repro_overlap` overlaps reproduction input preparation with GPU fitness evaluation. The overlap changes timing visibility, not the public reproduction API.

## Pipeline

One `gpu` reproduction pass is split into the following stages.

### 1. Prepare Inputs

Source files:

- [gpu.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu.cpp)
- [prep.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/prep.cpp)
- [types.hpp](/home/hschi1106/g3p-vm-gpu/cpp/include/g3pvm/evolution/repro/types.hpp)

The host extracts:

- the current population
- scored fitness values
- backend configuration limits

It then computes the preprocessing data needed by device kernels:

- subtree end positions
- typed crossover candidate ranges
- typed donor pool entries, bucketed by result type for subtree mutation

If `--grammar-config PATH` is active, this preprocessing stage filters typed candidate ranges to grammar-allowed subtrees and builds the donor pool with the selected `grammar-config-v1`. Runtime execution remains the full public grammar superset; the config only controls search-space generation.

This stage is reported as:

- `repro_prepare_inputs_ms`
- `repro_preprocess_ms`

### 2. Pack

Source files:

- [pack.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/pack.cpp)
- [pack.hpp](/home/hschi1106/g3p-vm-gpu/cpp/include/g3pvm/evolution/repro/pack.hpp)

The host flattens the typed AST population into bounded GPU-friendly arrays:

- plain node buffers
- per-program metadata
- candidate tables
- compact name id tables
- compact constant tables
- donor pool buffers

This stage is reported as `repro_pack_ms`.

### 3. Upload

Source files:

- [arena.cu](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/arena.cu)
- [launch.cu](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/launch.cu)

Packed host buffers are uploaded into a reusable device arena. The same arena is retained inside the process and grown only when capacity is insufficient.

This stage is reported as `repro_upload_ms`.

### 4. Device Selection And Variation

Source files:

- [selection_kernels.cuh](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/device/selection_kernels.cuh)
- [variation_kernels.cuh](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/device/variation_kernels.cuh)

The device executes two kernel families:

- tournament selection kernel
- variation kernel

Selection preserves the same high-level tournament semantics as the CPU path:

- round-based selection
- chunk size controlled by `selection_pressure`
- without-replacement within each round
- chunk winner chosen by best fitness inside that chunk

The GPU kernel still emits one mating pair per thread, but its per-round permutation is an internal device implementation detail rather than a shared host/device plan. CPU and GPU are not required to use identical RNG streams or identical within-round permutations as long as they preserve the same public tournament contract.

Selection also chooses a typed crossover site pair for each mating pair by scanning the bounded candidate tables for parent A and parent B, finding a common result type, and picking one candidate of that type from each parent.

Variation then applies the same high-level order as the CPU backend:

- every pair first attempts `typed_subtree` crossover
- each resulting child independently samples mutation from `mutation_rate`
- if a child mutates, `mutation_subtree_prob` chooses subtree mutation vs constant perturbation

GPU subtree mutation uses a type-bucketed donor pool keyed by the selected crossover-site type. Constant perturbation is applied directly to the packed child constant table after crossover.

Variation produces packed child buffers plus child metadata such as:

- node count
- max depth
- builtin usage marker
- validity bit

These kernels are reported as:

- `repro_kernel_ms`
- `repro_selection_kernel_ms`
- `repro_variation_kernel_ms`

### 5. Copyback

Source files:

- [copyback.cu](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/copyback.cu)

The backend copies back only live child regions rather than fixed-capacity slabs. Host-side pinned staging is reused across generations to keep D2H cost stable.

This stage is reported as `repro_copyback_ms`.

### 6. Decode

Source files:

- [pack.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/pack.cpp)

The host rebuilds `ProgramGenome` children from copied-back packed buffers. This includes:

- name id lookup
- AST node reconstruction
- constant reconstruction
- program key regeneration
- fallback to the selected parent if the child is marked invalid or decode fails

This stage is reported as `repro_decode_ms`.

## Overlap Model

Source files:

- [evolve.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/evolve.cpp)
- [evolve_cli.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/cli/evolve_cli.cpp)

When `repro_overlap` is enabled and `--engine gpu --repro-backend gpu` is active, the implementation starts reproduction preparation in a background task while GPU fitness evaluation is running.

The overlapped portion is:

- `repro_prepare_inputs_ms`
- `repro_preprocess_ms`
- `repro_pack_ms`

The non-overlapped tail remains:

- upload
- device kernels
- copyback
- decode

Because of this, overlap is only valuable when the hidden preparation work is large enough and the CPU-side preparation does not noticeably slow concurrent GPU evaluation.

## Arena Reuse

Source files:

- [gpu.cpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu.cpp)
- [internal.hpp](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/internal.hpp)
- [arena.cu](/home/hschi1106/g3p-vm-gpu/cpp/src/evolution/repro/gpu/arena.cu)

The backend keeps a process-local runtime cache containing:

- device arena buffers
- pinned host staging buffers

This changes the timing profile:

- `repro_setup_ms` is usually a first-use or growth cost
- `repro_teardown_ms` can remain zero in steady state

This reuse is required for overlap to be worthwhile; otherwise repeated `cudaMalloc` / `cudaFree` and host allocation churn dominate short runs.

## Correctness Model

The GPU reproduction backend is not required to reproduce the exact same child sequence as the CPU backend. It is required to preserve the public GP contract:

- children must decode into valid `ProgramGenome` objects or fall back deterministically
- compiled children must remain legal under the same runtime/compiler rules
- reproduction still follows the same public high-level order: select -> typed crossover on each pair -> child-level mutation
- public CLI semantics and benchmark accounting remain stable

Fitness parity remains a CPU/GPU requirement for evaluation. Reproduction identity is not a parity contract.

## Current Bottlenecks

The backend no longer spends meaningful time inside its device kernels on normal-size runs. The remaining cost is mostly host-side:

- `repro_decode_ms`: packed child reconstruction and program key rebuild
- `repro_preprocess_ms`: subtree/candidate/donor preprocessing
- `repro_pack_ms`: flattening host ASTs into bounded upload buffers

`repro_copyback_ms` and arena lifecycle cost are much smaller after live-region copyback and reusable staging were added.

In overlap mode, improvement is limited when:

- `repro_decode_ms` dominates the post-eval tail
- background preparation contends with CPU resources needed by evaluation orchestration

## How To Read Timings

For fixed-population benchmarks built on `g3pvm_evolve_cli --generations 1 --skip-final-eval on`:

- compare `total_ms` first
- then inspect `eval_ms`
- then inspect reproduction subphases

For overlap mode:

- do not add all reproduction subphases and expect them to equal `repro_ms`
- `repro_ms` is the residual wall-clock cost after any hidden work
- steady-state generation timings are more meaningful than cold-start single-shot runs

## Related Documents

- [DEVELOPMENT.md](/home/hschi1106/g3p-vm-gpu/docs/DEVELOPMENT.md)
- [ARCHITECTURE.md](/home/hschi1106/g3p-vm-gpu/docs/ARCHITECTURE.md)
- [CPP_RUNTIME_PAYLOAD.md](/home/hschi1106/g3p-vm-gpu/docs/CPP_RUNTIME_PAYLOAD.md)
