# Development

## Build configurations

The default native build includes the product CLI and maintained test targets.
From the repository root:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

`cpp/CMakePresets.json` also defines `debug`, `release`, `cuda-parity`,
`sanitizer`, and `fuzz` configure presets. For example:

```bash
cmake --preset debug -S cpp
cmake --build cpp/build/debug -j
ctest --test-dir cpp/build/debug --output-on-failure
```

Benchmarks and experiment probes are excluded from the default graph. Enable
them intentionally:

```bash
cmake -S cpp -B cpp/build/aux \
  -DGAGP_BUILD_BENCHMARKS=ON \
  -DGAGP_BUILD_EXPERIMENTS=ON
cmake --build cpp/build/aux -j \
  --target gagp_runtime_multi_bench gagp_simple_exp_population_probe
```

The experiment probe requires CUDA. See the
[tooling reference](../reference/tooling.md) for ownership and support status.

## Test layers

Run the complete configured suite:

```bash
ctest --test-dir cpp/build --output-on-failure
```

Focused labels are available for `unit`, `contract`, `property`,
`cpu_gpu_parity`, `gpu`, and `tooling`:

```bash
ctest --test-dir cpp/build -L contract --output-on-failure
ctest --test-dir cpp/build -L property --output-on-failure
ctest --test-dir cpp/build -L cpu_gpu_parity --output-on-failure
```

The runtime semantic corpus is split into independently named scalar,
control-flow, builtin, typed-value, structured-expression, and ASGP contracts.
Fixture data lives under `cpp/tests/fixtures/runtime/`; tests do not compile C++
sources ad hoc.

When Python 3 is available during CMake configuration, CTest also registers the
runtime-independent repository contract suite with the `tooling` label. It can
always be run directly, together with operational tool tests:

```bash
python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
```

Neither suite imports a product/runtime Python package or needs `PYTHONPATH`.

### Sanitizers and fuzzing

The `sanitizer` preset is a CPU-only ASan/UBSan configuration:

```bash
cmake --preset sanitizer -S cpp
cmake --build cpp/build/sanitizer -j
ctest --test-dir cpp/build/sanitizer -L 'unit|contract|property' --output-on-failure
```

The Clang-only `fuzz` preset builds bytecode-JSON and AST-JSON/prefix-stream
libFuzzer targets. Copy the seed corpus before a campaign so minimized inputs do
not modify the source tree:

```bash
cmake --preset fuzz -S cpp
cmake --build cpp/build/fuzz -j \
  --target gagp_fuzz_bytecode_json gagp_fuzz_ast_verify
cp -a cpp/tests/fuzz/corpus/json /tmp/gagp-json-corpus
cpp/build/fuzz/gagp_fuzz_bytecode_json -runs=1000 /tmp/gagp-json-corpus
cpp/build/fuzz/gagp_fuzz_ast_verify -runs=1000 /tmp/gagp-json-corpus
```

Keep any smallest reproducer as a named corpus or deterministic property-test
regression.

## Spec freeze

`benchmarks/spec_freeze.json` records hashes of the normative spec set. A spec
change must update its conformance tests and this manifest in the same change:

```bash
python3 -m unittest tests.repository.test_spec_freeze -v
```

## GPU run policy

GPU-capable native paths select the least-used visible CUDA device. To force a
visible-device index:

```bash
GAGP_CUDA_DEVICE=0 cpp/build/gagp_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --engine gpu --repro-backend gpu --repro-overlap on \
  --population-size 64 --generations 2
```

Use `nsys` only when profiling is needed; do not use `ncu` in this environment.

## Native CLI

[`../reference/cli.md`](../reference/cli.md) is the checked flag/default
reference for `gagp_evolve_cli`. The parser requires `--cases`; `--help` is not
a supported flag and exits through the unknown-argument error contract. Command
stdout and top-level JSON keys are locked by the native CLI contract test.

Grammar configs control generation and reproduction search space, not execution
of materialized AST/bytecode. Operational usage is in
[grammar-config.md](grammar-config.md), while the schema is normative in
[`../../spec/grammar_config.md`](../../spec/grammar_config.md).

## Related workflows

- [Benchmarking](benchmarking.md): fixed-population timing and canonical runs
- [PSB workflow](psb-workflow.md): datasets, regression, comparison, manifests
- [Experiment protocol](experiment-protocol.md): controlled formal studies
- [Operational tools](../../tools/README.md): unified commands and artifacts

