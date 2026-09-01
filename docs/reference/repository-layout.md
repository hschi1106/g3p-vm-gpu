# Repository Layout

This reference lists stable, maintained repository paths. The repository test
suite parses the first column and fails when a path disappears, so it replaces
the former hand-maintained tree drawing. Detailed file ownership belongs in the
nearest component documentation, not here.

| Checked path | Role |
| --- | --- |
| `AGENTS.md` | Contributor and coding-agent rules |
| `README.md` | Product entry point and quick start |
| `VERSION.md` | Release and compatibility history |
| `cpp/include/g3pvm/` | Public native headers |
| `cpp/src/runtime/` | CPU/GPU runtime and payload implementation |
| `cpp/src/evolution/` | Compiler, generation, evaluation, selection, and reproduction |
| `cpp/src/cli/` | Native CLI parsing, codecs, commands, and output adaptation |
| `cpp/src/bench/` | Opt-in native benchmarks |
| `cpp/src/experiments/` | Opt-in experiment probes |
| `cpp/tests/` | Native unit, contract, property, parity, and GPU tests |
| `configs/grammar/` | Checked grammar-search presets |
| `configs/psb_schemas/` | PSB fixture schema overrides |
| `configs/psb_tolerances/` | Versioned PSB quality-gate policies |
| `data/fixtures/` | Maintained runtime and benchmark fixtures |
| `docs/design/` | Implementation boundaries and rationale |
| `docs/guides/` | Reproducible development and experiment workflows |
| `docs/reference/` | Mechanically checked interfaces and inventories |
| `docs/refactor/` | Historical refactor evidence |
| `spec/` | Normative behavior contracts |
| `tools/g3pvm_tools/` | Installable operational tool package |
| `tools/tests/` | Operational tool tests |
| `tests/repository/` | Runtime-independent repository contract checks |
| `benchmarks/` | Compact committed validation manifests |

Generated `logs/`, build directories, downloaded datasets, and the historical
ignored `experiment/` working directory are intentionally absent. The
canonical maintained experiment protocol is
[`../guides/experiment-protocol.md`](../guides/experiment-protocol.md).

