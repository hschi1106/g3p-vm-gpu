# Tooling Inventory and Ownership

This inventory is the ownership boundary for operational Python commands and
auxiliary native executables. Runtime and grammar semantics do not live in
these tools; `spec/` and the native implementation remain authoritative.

## Operational Python commands

All retained scripts use the Python standard library and are covered by
`tools/tests/`. Stage 16 packages these functions behind one discoverable
command while retaining the file paths as temporary compatibility wrappers.

| Command | Class / owner | Inputs | Outputs | Downstream consumer | Status |
| --- | --- | --- | --- | --- | --- |
| `fetch_psb_datasets.py` | public workflow / datasets | PSB family, problem selection, upstream network | JSONL under `data/psb*_datasets/` | conversion/materialization | maintained |
| `convert_psb_to_fitness_cases.py` | public workflow / datasets | PSB JSONL, split/schema options | `fitness-cases` JSON | native CLI, materializer | maintained |
| `materialize_psb_fixtures.py` | public workflow / datasets | mirrored datasets, exclusions, schema policy | fixtures plus support manifest | PSB experiment runner | maintained |
| `grammar_config_profiles.py` | internal support / experiments | base grammar profile and compatibility choices | generated grammar-config JSON/hash | regression runner | maintained |
| `make_population_seeds.py` | public workflow / experiments | cases path, count, limits, seed base | `population-seeds` JSON | fixed-pop CLI runs | maintained |
| `run_psb_regression.py` | public workflow / experiments | native binary, fixtures, seeds/config, run matrix | per-run JSON and summary | comparison/report commands | maintained |
| `compare_psb_baseline.py` | public workflow / reports | compatible baseline/candidate summaries, tolerance policy | comparison JSON and exit gate | CI/release review | maintained |
| `write_psb_manifest.py` | internal support / reports | PSB comparison/run artifacts | compact PSB evidence manifest | committed `benchmarks/` evidence | maintained |
| `write_simple_exp_manifest.py` | internal support / reports | simple-expression baseline/candidate runs | compact speed manifest | committed `benchmarks/` evidence | maintained |

Generated raw runs belong under ignored `logs/`. Only compact, reviewed evidence
is committed under `benchmarks/`. Dataset mirrors and generated fixtures retain
their existing explicit policies documented in `docs/DEVELOPMENT.md`.

## Native executables and harnesses

| Target | Class / owner | Build policy | Use case |
| --- | --- | --- | --- |
| `g3pvm_evolve_cli` | product CLI / native runtime | default | supported evolution, AST evaluation, and fixed-pop workflow |
| `g3pvm_runtime_multi_bench` | benchmark / performance | `-DG3PVM_BUILD_BENCHMARKS=ON` | low-level runtime throughput experiments |
| `g3pvm_simple_exp_population_probe` | experiment probe / parity research | `-DG3PVM_BUILD_EXPERIMENTS=ON`, CUDA only | diagnostic fixed-pop CPU/GPU fitness comparison; not a product command |
| `g3pvm_test_vm_cli_harness` | test harness / runtime contracts | default, driven by CTest | executes versioned runtime fixtures |
| `g3pvm_test_*` | tests / owning native module | default, registered with CTest | unit, contract, property, integration, GPU, and parity gates |
| `g3pvm_fuzz_*` | fuzz harness / verifier boundaries | `-DG3PVM_BUILD_FUZZERS=ON`, Clang and CPU-only | extended malformed-input campaigns |

The experiment probe source is under `cpp/src/experiments/`, not the CTest
tree. Maintained tests and the fixture harness remain normal CMake/CTest owners.

## Removed utility

`draw/draw.py` and its directory were removed in Stage 15. It consumed legacy
timing logs and required Matplotlib but was not part of a supported workflow.
No replacement is needed: versioned JSON manifests are the reporting boundary,
and visualization belongs in downstream analysis environments.
