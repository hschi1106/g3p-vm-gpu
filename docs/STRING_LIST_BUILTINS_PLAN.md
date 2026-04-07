# String/List Builtins And Typed List Plan

This note records the completed `NumList` / `StringList` sequence-support plan for `g3p-vm-gpu`.

It supersedes the earlier idea of keeping a public generic `List` while adding hidden internal subtypes such as `ListNum`.

## Implementation Status

This plan has been implemented in the current worktree:

- the public generic `List` contract was replaced by `NumList` and `StringList`
- PSB2 conversion now infers list types column-wise and emits `num_list` / `string_list`
- multi-output PSB rows are no longer encoded as list values
- Python and C++ runtime values use explicit typed-list representations
- CPU and GPU payload paths keep `NumList` and `StringList` payloads as distinct tags
- Python and C++ evolution typing/generation now target `NumList` and `StringList`
- first-wave builtins were added: `append`, `reverse`, `find`, and `contains`

Remaining follow-up work is outside this plan:

- runtime-level multi-output support
- stronger string builtins such as `split`, `join`, `replace`, and case conversion
- numeric aggregate builtins such as `sum` and `avg`
- wider PSB1 conversion tooling beyond the already mirrored dataset fetch path

## Decision

The preferred direction is:

- keep public `String`
- replace public generic `List` with two public fixed list types:
  - `NumList`
  - `StringList`
- do not introduce hidden internal-only list subtypes in the first design pass

This is a cleaner contract than:

- keeping one public `List` but secretly splitting it during generation
- or jumping directly to a full generic `List[T]` system

## Why This Is Cleaner

Before this migration, the repo had a mismatch between runtime values and search typing:

- Python typed generation used one synthetic `CONTAINER` bucket in [python/src/g3p_vm_gpu/evolution/random_tree.py](../python/src/g3p_vm_gpu/evolution/random_tree.py)
- C++ evolution typing used a generic `List` tag in [cpp/include/g3pvm/evolution/ast_program.hpp](../cpp/include/g3pvm/evolution/ast_program.hpp)
- C++ typed analysis overclaimed `index(List, i) -> Num` in [cpp/src/evolution/typed_expr_analysis.cpp](../cpp/src/evolution/typed_expr_analysis.cpp)

If the public language keeps a general `List`, then the engine must choose between two bad options:

- keep typing conservative and lose useful structure for vector tasks
- or add hidden internal list subtypes and risk semantic drift between public runtime behavior and internal search behavior

Public `NumList` and `StringList` avoid that split-brain design.

## Dataset Reality

The local datasets already justify fixed list types:

- PSB1 has numeric list tasks such as:
  - [data/psb1_datasets/vectors-summed/vectors-summed-edge.json](../data/psb1_datasets/vectors-summed/vectors-summed-edge.json)
  - [data/psb1_datasets/vector-average/vector-average-edge.json](../data/psb1_datasets/vector-average/vector-average-edge.json)
  - [data/psb1_datasets/last-index-of-zero/last-index-of-zero-edge.json](../data/psb1_datasets/last-index-of-zero/last-index-of-zero-edge.json)
- PSB1 also has string-list tasks such as:
  - [data/psb1_datasets/string-lengths-backwards/string-lengths-backwards-edge.json](../data/psb1_datasets/string-lengths-backwards/string-lengths-backwards-edge.json)
- PSB2 currently appears to use numeric lists, strings, and scalars, but also has multi-output tasks according to [logs/psb2_audit_current.json](../logs/psb2_audit_current.json)

So `NumList` alone is not enough for PSB1 coverage. The smallest practical public typed-list set is:

- `NumList`
- `StringList`

## Non-Goals

- no new public list syntax in this phase
- no hidden internal `ListNum` / `ListString` layer while the public contract still says `List`
- no immediate move to `List[T]` generics
- no support for nested lists
- no support for heterogeneous lists

## Cross-Cutting Requirement: Docs And Skills Must Move Together

Every phase in this plan should update the matching documentation and local repo skill snapshot in the same change.

At minimum, review and update as needed:

- repo docs under `docs/`
- normative specs under `spec/`
- repo-local guidance in `AGENTS.md` when runbooks or contributor expectations change
- repo skill files under `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/`

For the current repo skill, the main files to keep in sync are:

- `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/SKILL.md`
- `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/references/architecture.md`

The intent is simple:

- if the public contract changes, docs and skills both move
- if runtime or evolution invariants change, docs and skills both move
- if workflow or testing expectations change, docs and skills both move

## Public Value Contract

The intended public value domain becomes:

- `Int`
- `Float`
- `Bool`
- `None`
- `String`
- `NumList`
- `StringList`

### `NumList`

- homogeneous sequence of numeric scalar values
- elements may be `Int` or `Float`
- `Bool` is not numeric and is not allowed inside `NumList`
- nested containers are not allowed

### `StringList`

- homogeneous sequence of `String`
- nested containers are not allowed

## Public Builtin Type Rules

The fixed-type list plan implies these rules:

- `len(String | NumList | StringList) -> Int`
- `concat(String, String) -> String`
- `concat(NumList, NumList) -> NumList`
- `concat(StringList, StringList) -> StringList`
- `slice(String, lo, hi) -> String`
- `slice(NumList, lo, hi) -> NumList`
- `slice(StringList, lo, hi) -> StringList`
- `index(String, i) -> String`
- `index(NumList, i) -> Num`
- `index(StringList, i) -> String`

Planned first-wave extensions:

- `append(NumList, Num) -> NumList`
- `append(StringList, String) -> StringList`
- `reverse(String) -> String`
- `reverse(NumList) -> NumList`
- `reverse(StringList) -> StringList`
- `find(String, String) -> Int`
- `contains(String, String) -> Bool`

Deferred builtins:

- `sum`
- `avg`
- `split`
- `join`
- `replace`
- `map`
- `filter`

## Critical Migration Constraint: Empty Lists Need Schema

If the public contract moves to `NumList` and `StringList`, then `[]` is no longer self-describing.

Examples:

- [data/psb1_datasets/vectors-summed/vectors-summed-edge.json](../data/psb1_datasets/vectors-summed/vectors-summed-edge.json) uses empty numeric lists
- [data/psb1_datasets/string-lengths-backwards/string-lengths-backwards-edge.json](../data/psb1_datasets/string-lengths-backwards/string-lengths-backwards-edge.json) uses empty string lists

Because of that, the converter and codec path must infer list type from a whole field schema, not from one value alone.

The rule should be:

- infer the type of each `inputK` or `outputK` column over the whole dataset split
- treat `[]` as that column's declared list type
- reject columns that mix incompatible element kinds

## Critical Migration Constraint: Multi-Output Is Not A List

Before this migration, the PSB2 converter in [tools/convert_psb2_to_fitness_cases.py](../tools/convert_psb2_to_fitness_cases.py) packed multiple outputs into a synthetic `"type": "list"` expected value.

That is acceptable only while `list` means a generic catch-all sequence.

Under `NumList` and `StringList`, this becomes wrong because:

- multi-output records are tuple-like outputs
- they are not the same thing as a sequence value in the source language

So the typed-list migration must also stop using list values as a stand-in for multi-output expectations.

Recommended direction:

- keep multi-output as a separate conversion-layer representation
- do not let multi-output masquerade as `NumList` or `StringList`
- solve runtime multi-output support separately from typed-list support

## Phase 1: Lock The Public Contract

### Objective

Replace the ambiguous public `List` contract with public `NumList` and `StringList`.

### Spec updates

Update:

- [spec/grammar_v1_0.md](../spec/grammar_v1_0.md)
- [spec/builtins_runtime_v1_0.md](../spec/builtins_runtime_v1_0.md)
- [spec/bytecode_format_v1_0.md](../spec/bytecode_format_v1_0.md)
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- repo skill snapshot under `/home/hschi1106/.codex/skills/g3p-vm-gpu-repo/`

Required changes:

- replace `List` in the public value domain
- define `NumList` and `StringList`
- define builtin signatures against the new tags
- update wire-format examples to use explicit typed list names rather than generic `"list"`

### Exit criteria

- the public docs no longer describe one generic `List`
- the wire-format spec no longer leaves empty-list typing implicit

## Phase 2: Fix Converters And Schemas

### Objective

Make dataset ingestion produce stable typed values for `NumList` and `StringList`.

### Main changes

Update:

- [tools/convert_psb2_to_fitness_cases.py](../tools/convert_psb2_to_fitness_cases.py)
- PSB1 conversion tooling when added
- CLI typed-value decoding in [cpp/src/cli/codec.cpp](../cpp/src/cli/codec.cpp)
- any docs and skill text that describes fixture schemas, typed values, or conversion workflow

Required behavior:

- infer field schemas across the dataset, not row-by-row
- emit explicit list tags:
  - `num_list`
  - `string_list`
- reject unsupported mixed-element list columns
- keep multi-output separate from typed list values

### Risks

- empty lists force a schema pass before conversion
- old fixtures and parity harness payloads may need regeneration

### Exit criteria

- empty lists decode unambiguously
- multi-output is no longer encoded as fake list output

## Phase 3: Update Runtime Value Representation

### Objective

Teach Python and C++ runtime layers about `NumList` and `StringList` as real public tags.

### Python

The Python runtime cannot keep using bare `list` as the only list representation, because bare lists do not preserve the distinction between:

- `NumList`
- `StringList`

Update:

- Python value representation and comparison logic in [python/src/g3p_vm_gpu/core/value_semantics.py](../python/src/g3p_vm_gpu/core/value_semantics.py)
- builtin dispatch in [python/src/g3p_vm_gpu/runtime/builtins.py](../python/src/g3p_vm_gpu/runtime/builtins.py)
- interpreter and VM typed value handling
- matching docs and skill notes that describe runtime value semantics

Recommended direction:

- introduce lightweight typed wrappers such as `NumListVal` and `StringListVal`
- update equality and builtin rules to match tags, not raw `isinstance(..., list)` checks

### C++

Update:

- core value tags in [cpp/include/g3pvm/core/value.hpp](../cpp/include/g3pvm/core/value.hpp)
- payload registry in [cpp/src/runtime/payload/payload.cpp](../cpp/src/runtime/payload/payload.cpp)
- CPU builtins in [cpp/src/runtime/cpu/builtins_cpu.cpp](../cpp/src/runtime/cpu/builtins_cpu.cpp)
- GPU builtins in [cpp/src/runtime/gpu/device/builtins_device.cuh](../cpp/src/runtime/gpu/device/builtins_device.cuh)
- host/device packing code
- matching docs and skill notes that describe payload and runtime invariants

Required behavior:

- `NumList` and `StringList` are first-class value tags
- payload registry stores them separately
- builtin exact/fallback behavior stays deterministic

### Exit criteria

- Python and C++ both represent typed lists explicitly
- CPU/GPU parity remains intact for existing string/list builtins

## Phase 4: Update Evolution Typing And Generation

### Objective

Make search and typed-subtree operations use the same typed-list contract as the public runtime.

### Main changes

Update:

- [python/src/g3p_vm_gpu/evolution/random_tree.py](../python/src/g3p_vm_gpu/evolution/random_tree.py)
- [cpp/src/evolution/evolve.cpp](../cpp/src/evolution/evolve.cpp)
- [cpp/src/evolution/typed_expr_analysis.cpp](../cpp/src/evolution/typed_expr_analysis.cpp)
- [cpp/src/evolution/genome_generation.cpp](../cpp/src/evolution/genome_generation.cpp)
- [cpp/src/evolution/subtree_utils.cpp](../cpp/src/evolution/subtree_utils.cpp)
- matching docs and skill notes that describe search typing and generation invariants

Required behavior:

- replace synthetic Python `CONTAINER`
- add `NumList` and `StringList` to generation and typed analysis
- make `index(NumList, i) -> Num`
- make `index(StringList, i) -> String`
- remove the old C++ assumption that generic `List` indexing is numeric

### Exit criteria

- search typing matches runtime typing
- no hidden subtype layer is needed
- compile rate and evolution smoke tests remain acceptable

## Phase 5: Add First-Wave Builtins

### Objective

Add the smallest builtin set that materially improves PSB1/PSB2 usability after the typed-list migration is complete.

### First wave

- `append(NumList, Num) -> NumList`
- `append(StringList, String) -> StringList`
- `reverse(String) -> String`
- `reverse(NumList) -> NumList`
- `reverse(StringList) -> StringList`
- `find(String, String) -> Int`
- `contains(String, String) -> Bool`

### Why not more immediately

`sum`, `avg`, `split`, `join`, and `replace` are all plausible later, but they should come after:

- typed-list migration
- schema-aware conversion
- multi-output cleanup

## Suggested PR Sequence

### PR 1

`refactor: replace public List with NumList and StringList`

- specs
- docs
- skills
- wire-format contract

### PR 2

`refactor: make fixture conversion schema-aware for typed lists`

- converters
- CLI codec
- docs
- skills
- empty-list handling
- stop encoding multi-output as list

### PR 3

`refactor: add NumList and StringList runtime values`

- Python runtime
- C++ value tags
- payload registry
- CPU/GPU builtin paths
- docs
- skills

### PR 4

`refactor: align typed generation with NumList and StringList`

- Python random tree generation
- C++ typed analysis
- C++ genome generation and donor synthesis
- docs
- skills

### PR 5

`feat: add append reverse find contains builtins`

- specs
- Python runtime
- C++ CPU/GPU runtime
- docs
- skills
- parity tests

## Testing Matrix

Every phase should be checked with:

- Python `unittest`
- native `ctest`
- targeted bytecode/typed-value codec tests
- typed generation tests
- interpreter/VM parity
- CPU/GPU parity for affected builtins

Additional required checks:

- fixture conversion tests with empty numeric lists
- fixture conversion tests with empty string lists
- regression tests that multi-output is not decoded as a list value

## Expected Outcome

With `NumList` and `StringList` as the minimum public typed-list set:

- the language contract becomes cleaner
- vector-style numeric tasks become statically tractable
- string-list tasks such as PSB1 `string-lengths-backwards` stay representable
- the repo no longer needs to hide a second list type system inside the search engine

This still will not be sufficient to solve all PSB1 or PSB2 tasks by itself.

The remaining major gaps after this migration will still include:

- multi-output runtime support
- stronger string builtins
- stronger sequence-construction support
