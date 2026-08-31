# Refactor Inventory and Ownership

## Node metadata duplication

`NodeKind` contains 71 values in `cpp/include/g3pvm/evolution/ast_program.hpp`.
Host and device code currently rediscover arity, category, builtin mapping,
typing, metadata, or dependency properties in the following owners:

| File | Duplicated concern | Planned owner |
| --- | --- | --- |
| `cpp/src/evolution/genome.cpp` | expression classification and prefix traversal | host node descriptor |
| `cpp/src/evolution/compiler.cpp` | opcode/builtin mapping, dependency arity, lowering placement | descriptor plus compiler rules |
| `cpp/src/evolution/grammar_config.cpp` | node-to-config switch | descriptor |
| `cpp/src/evolution/typed_expr_analysis.cpp` | categories, arity, typing, binder rules | verifier typed view |
| `cpp/src/evolution/subtree_utils.cpp` | subtree shape/types and metadata remapping | verified subtree boundaries plus remapper |
| `cpp/src/evolution/genome_generation.cpp` | candidate sets and construction | descriptor-filtered generation rules |
| `cpp/src/evolution/repro/pack.cpp` | metadata kinds and packed shape | descriptor-tested packer |
| `cpp/src/evolution/repro/prep.cpp` | constant and candidate classification | descriptor/verified annotations |
| `cpp/src/cli/codec.cpp` | dependency-kind serialization | stable descriptor names |
| `cpp/src/cli/evolve_cli.cpp` | AST JSON kind decoding | CLI codec over descriptor names |
| `cpp/src/evolution/repro/gpu/device/variation_kernels.cuh` | device arity/category tables | minimal device table tested against host descriptors |
| `cpp/src/runtime/{cpu,gpu}/execute_bytecode*` | ASGP dependency dispatch | bytecode/segment verifier plus runtime dispatch |

Tests also construct `NodeKind` directly in the evolution and parity targets;
their builders become verifier boundaries. Mutation and crossover use the
same subtree utilities and must consume verified boundaries when those are
available, without adding release hot-path validation.

Stage 01 centralizes host names, categories, arities, index-field uses, builtin
IDs/arities, side-table requirements, grammar switches, typing rule IDs, and
subtree eligibility. Complex typing and binder semantics remain verifier rules,
not opaque function pointers in the metadata table.

## Python tools

| Script | Class | Inputs | Outputs / consumer | Disposition |
| --- | --- | --- | --- | --- |
| `fetch_psb_datasets.py` | public dataset | suite/problem selection, network or dry run | upstream JSONL for conversion | retain as `psb fetch` |
| `convert_psb_to_fitness_cases.py` | public dataset | PSB JSONL and optional schemas | train/test `fitness-cases` | retain as `psb convert` |
| `materialize_psb_fixtures.py` | public dataset | suite datasets, schema overrides | fixture tree and support manifest | retain as `psb materialize` |
| `grammar_config_profiles.py` | internal experiment support | base grammar and fixture schema | generated grammar profiles | retain shared package API |
| `make_population_seeds.py` | public benchmark | cases, limits, seed range, grammar config | reproducible `population-seeds` | retain as `benchmark population-seeds` |
| `run_psb_regression.py` | public experiment | fixtures, native binary, profiles/seeds | run JSON and summary | retain as `psb run` |
| `compare_psb_baseline.py` | public report/gate | baseline and candidate summaries | comparison JSON and exit gate | retain as `psb compare` |
| `write_psb_manifest.py` | public report | summaries/comparison/exclusions | compact committed PSB evidence | retain as `report psb-manifest` |
| `write_simple_exp_manifest.py` | public report | two native run JSON files | compact speed manifest | retain as `report simple-manifest` |
| `draw/draw.py` | obsolete plotting | legacy timing logs/environment | interactive figures | remove; no replacement |

Current sibling imports in materialization and regression tools are
path-sensitive. Stage 16 will package them and keep the file entry points as
temporary wrappers.

## Native executables and harnesses

| Target/source | Class | Ownership decision |
| --- | --- | --- |
| `g3pvm_evolve_cli` | product CLI | default build; split into reusable CLI support and commands |
| `cpp/tests/runtime/test_vm_cli_harness.cpp` | test harness | register in CMake through the fixture-runner support library |
| `g3pvm_runtime_multi_bench` | benchmark | maintained, but build only with an explicit benchmark option |
| `g3pvm_simple_exp_population_probe` | experiment probe | maintained parity diagnostic; build only with an explicit experiment option |
| `g3pvm_test_*` | tests | normal CTest ownership with contract/property/parity labels |

Stage 11 replaced the CLI's textual inclusion of `json.cpp` and `codec.cpp`
and its second incomplete parser with the linked `g3pvm_cli_support` library.
`options.cpp` and `options.hpp` now own the complete product parser, and focused
contract tests preserve its flags, defaults, and validation. No C++ source
inclusion is an accepted compatibility surface.

## Documentation ownership and known defects

| Current document | Current role | Target owner / action |
| --- | --- | --- |
| `README.md` | entry point plus repeated detail | concise product matrix, five-minute workflow, links |
| `VERSION.md` | release history | retain release/compatibility history only |
| `spec/*.md` | normative contracts | retain; add `spec/README.md` navigation |
| `docs/ARCHITECTURE.md` | architecture plus repeated specs | move to `docs/design/architecture.md`, deduplicate |
| `docs/CPP_RUNTIME_PAYLOAD.md` | payload design | move to `docs/design/payload.md` |
| `docs/GPU_REPRODUCTION.md` | reproduction design | move to `docs/design/gpu-reproduction.md`; replace absolute links |
| `docs/DEVELOPMENT.md` | several workflows and CLI reference | split guides and generated/checked CLI reference |
| `docs/GRAMMAR_CONFIG.md` | operational config explanation | guide that links normative `spec/grammar_config.md` |
| `docs/TIMING.md` | timing reference | move to `docs/reference/timing.md` |
| `docs/FILE_STRUCTURE.md` | hand-maintained map | replace with checked repository-layout reference |
| `experiment/EXPERIMENT_SPEC.md` | ignored experiment plan | move maintained workflow/provenance to guides; do not link an ignored path as stable docs |
| `AGENTS.md` | contributor instructions | retain rules and short commands only |

Known defects captured before refactoring:

- README and `FILE_STRUCTURE.md` link `docs/EXPERIMENT_SPEC.md`, which does not
  exist; the only local copy is ignored under `experiment/`.
- `fitness-cases` format descriptions contain duplicated/collapsed legacy and
  current wording.
- `GPU_REPRODUCTION.md` contains machine-specific absolute file links.
- Stable architecture/config docs still describe the Python reference package
  as a current implementation owner.
- The documented block size 1,024 is not valid for every generated workload on
  the baseline GPU.
- Tool ownership and artifact lifecycle have no single README.

## Compatibility surfaces

The refactor preserves:

- prefix `AstProgram` and `ast-prefix` format identifier
- current `fitness-cases`, bytecode, and grammar-config identifiers in `spec/`
- exact public value tags and `ErrCode` behavior
- tournament selection, `selection_pressure`, `typed_subtree` crossover, and
  crossover-before-mutation ordering
- CPU/GPU fitness parity within documented payload limits
- CLI flags, defaults, JSON keys, and timing meanings recorded in
  [`baseline.md`](baseline.md), except where a separately specified public
  change is tested and documented
- fixed-population seed replay, grammar-config hashes, and benchmark manifest
  comparability

Python module APIs, ad-hoc source compilation, implementation-file inclusion,
the legacy plot script, and default construction of benchmark/probe binaries
are explicitly not compatibility surfaces.
