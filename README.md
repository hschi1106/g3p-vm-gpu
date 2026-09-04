# GAGP

**GPU-Accelerated Genetic Programming for Program Synthesis.** GAGP is a
native prefix-AST genetic programming system with CPU execution/evolution and
CUDA-accelerated fitness and reproduction backends. The former Python semantic
implementation has been retired; Python remains only in the independent
operational tool package.

## Supported backends

| Area | CPU | CUDA GPU |
| --- | --- | --- |
| Bytecode execution / fitness | Supported | Supported within documented device payload limits |
| Population evaluation | Supported | Supported |
| Reproduction | Supported | Supported, with optional preparation/evaluation overlap |
| AST evaluation command | Supported | CPU command path only |

Public language, bytecode, builtin, fitness, fixture, and grammar-config
contracts are indexed in [`spec/README.md`](spec/README.md). Implementation and
workflow documents must link to those specifications instead of redefining
their semantics.

## Five-minute start

Build and run every configured test:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

Run a small GPU evolution job:

```bash
cpp/build/gagp_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --population-size 64 \
  --generations 2 \
  --out-json logs/simple_exp_1024.run.json
```

The runtime selects the least-used visible CUDA device. Set
`GAGP_CUDA_DEVICE=0` to force a visible-device index.

## Operational tools

Install the standard-library tool package in a virtual environment:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/pip install -e tools
.venv-tools/bin/gagp-tools --help
```

Compatibility wrapper scripts remain available during the command migration.
The complete fetch → convert/materialize → run → compare → manifest flow is in
[`tools/README.md`](tools/README.md).

## Common workflows

- [Development](docs/guides/development.md): builds, named configurations,
  tests, sanitizers/fuzzing, GPU policy, and CLI entry points
- [Native CLI reference](docs/reference/cli.md): mechanically checked flags and
  defaults
- [Benchmarking](docs/guides/benchmarking.md): reproducible fixed-population
  CPU/GPU comparisons
- [PSB workflow](docs/guides/psb-workflow.md): datasets, fixture materialization,
  regression runs, comparisons, and manifests
- [Experiment protocol](docs/guides/experiment-protocol.md): formal performance
  and effectiveness controls
- [Grammar config guide](docs/guides/grammar-config.md): selecting and deriving
  evolution search spaces

Repository and tool checks can also be run directly:

```bash
python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
```

They require no product/runtime Python package and no `PYTHONPATH`.

## Documentation ownership

- [`docs/README.md`](docs/README.md): design, guide, reference, and refactor
  document ownership
- [`spec/README.md`](spec/README.md): normative contracts
- [`benchmarks/README.md`](benchmarks/README.md): committed validation evidence
- [`tools/README.md`](tools/README.md): operational command/artifact lifecycle
- [`AGENTS.md`](AGENTS.md): contributor constraints
- [`VERSION.md`](VERSION.md): release and compatibility history

The checked stable directory map is
[`docs/reference/repository-layout.md`](docs/reference/repository-layout.md).

## Change discipline

Update an owning spec and its conformance tests with semantic changes. Update
the CLI reference and command contracts with parser/output changes. Update the
documentation index, checked repository layout, and external repository skill
references with file moves. Keep compact validation evidence in `benchmarks/`;
keep generated datasets, populations, profiler captures, and raw logs out of
version control.
