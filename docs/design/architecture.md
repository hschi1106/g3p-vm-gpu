# Architecture

The maintained product is a native C++/CUDA prefix-AST genetic programming
system. This document owns component boundaries and dependency direction. The
end-to-end sequence is in [dataflow.md](dataflow.md); language and wire behavior
is owned by the [specifications](../../spec/README.md).

## Dependency direction

```text
core value/bytecode contracts
      |             |
      v             v
CPU/GPU runtime   AST/verifier/compiler/operators
      |             |
      +------v------+
       evolution engine
              |
              v
      CLI support and commands
```

Operational Python tools consume files and invoke the native CLI. They do not
implement AST typing, bytecode execution, fitness, or reproduction semantics.

## Product boundaries

### Core and runtime

`cpp/include/g3pvm/core/` owns shared values, errors, opcodes, bytecode, and
bytecode-verification contracts. `cpp/src/runtime/cpu/` executes and scores
bytecode on the host. `cpp/src/runtime/gpu/` packs programs/cases and executes
fitness on CUDA. `cpp/src/runtime/payload/` owns host payload registration and
snapshot lookup for strings and typed lists.

Runtime semantics are defined in `spec/`; implementation details of container
transport are explained in [payload.md](payload.md).

### AST, compiler, and verification

`cpp/include/g3pvm/evolution/` exposes prefix ASTs, node descriptors, verified
AST annotations, grammar search configuration, genome operations, evolution,
and timing. Its implementation is split by responsibility under
`cpp/src/evolution/`.

The host node descriptor is the common source for serialized names, categories,
arity, index fields, builtin mapping, grammar switches, typing rule IDs, and
side-table ownership. Structural/type verification produces subtree, type, and
scope annotations used by typed variation. External AST JSON is verified before
genome construction; external bytecode JSON is verified before execution;
compiler output is verified in debug/test builds.

Verification is a trust-boundary and test invariant. Release evolution does not
perform an additional heap-heavy full-AST verification pass for every individual
after every generation.

### Evolution engine

The engine composes focused owners:

| Owner | Responsibility |
| --- | --- |
| `CaseSet` | Canonical names, input types/bindings, expected values, and return-type inference |
| `PopulationInitialization` | Generated population versus fixed replay boundary |
| compiler/cache | Verified AST-to-bytecode lowering and reuse |
| evaluator adapters | CPU/GPU fitness vectors with one timing shape |
| selection | Fitness canonicalization, ranking, and scored-reference materialization |
| reproduction backends | Selection inputs, typed crossover, mutation, and decoded-child acceptance |
| `PayloadLifetimeManager` | Live payload roots and registry pruning |
| lifecycle overlap | GPU reproduction preparation scheduled around evaluation |
| timing model | Nested evaluation, reproduction, generation, and run aggregates |

CPU and GPU evaluation converge on one fitness-vector boundary before ranking.
CPU and GPU reproduction share the public operator contract but do not promise
child-for-child RNG identity. Detailed reproduction scheduling is in
[gpu-reproduction.md](gpu-reproduction.md); timing names are in
[`../reference/timing.md`](../reference/timing.md).

### CLI

`g3pvm_cli_support` owns the JSON parser, codecs, complete option parser, input
loading, command workflows, and output adaptation. `evolve_cli.cpp` is only the
process-level parse/dispatch/error boundary. No C++ implementation file is
included textually.

Parser flags/defaults are checked against
[`../reference/cli.md`](../reference/cli.md). CLI JSON retains its stable flat
keys even though timing storage inside `EvolutionResult` is nested.

### Operational tools

`tools/g3pvm_tools/` is an independently installable, standard-library Python
package organized into dataset, experiment, report, and shared-format modules.
Historical top-level scripts are compatibility wrappers. The command pipeline
and artifact policy are owned by [`../../tools/README.md`](../../tools/README.md),
and every auxiliary command/binary is classified in
[`../reference/tooling.md`](../reference/tooling.md).

## Stable performance invariants

- GPU fitness uses one production mixed kernel per accepted population.
- GPU reproduction overlap may hide host preprocessing behind evaluation, but
  selection still consumes the completed fitness vector.
- Evolution ranks with lightweight scored references and materializes owned
  scored genomes only for retained public results.
- Final-population retention is opt-in at the CLI boundary.
- Payload roots are retained across active cases, populations, history, best,
  and optional final results.

These invariants are locked by native contract/property/parity tests and the
fixed-population benchmark gate; they are not alternate semantic definitions.

## Test ownership

- `cpp/tests/runtime/`: runtime, codec, CLI, payload, and bytecode contracts
- `cpp/tests/evolution/`: descriptor, verifier, compiler, property, operator,
  and orchestration contracts
- `cpp/tests/fixtures/runtime/`: intent-labelled semantic corpus
- `cpp/tests/gpu/` and `cpp/tests/parity/`: GPU execution and CPU/GPU agreement
- `cpp/tests/fuzz/`: bounded malformed-input smoke and opt-in libFuzzer targets
- `tools/tests/`: operational tool contracts
- `tests/repository/`: docs, CLI-reference, spec-freeze, and layout contracts

Named build/test configurations and focused commands are in
[`../guides/development.md`](../guides/development.md).

## Change ownership

- AST, typing, control flow, or lowering: update the owning grammar/ISA spec and
  verifier/compiler tests.
- Builtin or payload behavior: update the owning builtin spec, payload design
  when transport changes, and CPU/GPU parity coverage.
- Fitness behavior: update `spec/fitness.md`, fitness contracts, and benchmark
  interpretation.
- CLI flags/defaults/output: update the parser, CLI reference, and command
  contract together.
- Repository moves: update `docs/README.md`, the checked repository layout, and
  the external repository skill references.

