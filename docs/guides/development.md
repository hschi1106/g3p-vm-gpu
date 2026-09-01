# Development

## Build And Test

### Build C++

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

The default graph builds the product CLI and maintained test targets. Auxiliary
executables are opt-in:

```bash
cmake -S cpp -B cpp/build-aux \
  -DG3PVM_BUILD_BENCHMARKS=ON \
  -DG3PVM_BUILD_EXPERIMENTS=ON
cmake --build cpp/build-aux -j \
  --target g3pvm_runtime_multi_bench g3pvm_simple_exp_population_probe
```

The experiment probe requires CUDA. Ownership, support status, and consumers
for all scripts and auxiliary binaries are recorded in
[tooling reference](../reference/tooling.md).

### Tool and repository checks

Operational tools and runtime-independent repository contracts use the Python
standard library only. No product/runtime Python package or `PYTHONPATH` is
required:

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
```

For the unified operational command, install the independent package in a
virtual environment:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/pip install -e tools
.venv-tools/bin/g3pvm-tools --help
```

See [tools/README.md](../../tools/README.md) for the pipeline, every subcommand,
compatibility wrappers, and artifact policy.

### C++ tests

```bash
ctest --test-dir cpp/build --output-on-failure
```

The native semantic corpus is split into independently runnable contracts:

```bash
ctest --test-dir cpp/build -R g3pvm_test_runtime_scalar --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_runtime_control_flow --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_runtime_builtins --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_runtime_typed_values --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_compiler_lowering --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_fixture_codec --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_structured_semantics --output-on-failure
ctest --test-dir cpp/build -R g3pvm_test_asgp_semantics --output-on-failure
```

The JSON corpus lives under `cpp/tests/fixtures/runtime/`. Each scenario names
its semantic intent and declares an exact expected value or error code. The
fixture executable is built normally by CMake; Python tests do not compile C++
sources ad hoc.

### Property, sanitizer, and fuzz gates

The deterministic native property layer covers generation, mutation,
crossover, CPU reproduction, GPU reproduction decode, grammar restrictions,
and payload retention:

```bash
ctest --test-dir cpp/build -L property --output-on-failure
```

The default build includes a bounded 1,000-input malformed-data smoke test.
ASan/UBSan are opt-in and CPU-only so the normal CUDA build is unchanged:

```bash
cmake -S cpp -B /tmp/g3pvm-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DG3PVM_ENABLE_CUDA=OFF \
  -DG3PVM_ENABLE_SANITIZERS=ON
cmake --build /tmp/g3pvm-sanitize -j4
ctest --test-dir /tmp/g3pvm-sanitize \
  -L 'unit|contract|property' --output-on-failure
```

Clang builds additionally expose two libFuzzer targets. Always fuzz a copied
corpus so newly minimized inputs do not appear in the source directory:

```bash
cmake -S cpp -B /tmp/g3pvm-fuzz \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DG3PVM_ENABLE_CUDA=OFF \
  -DG3PVM_BUILD_FUZZERS=ON
cmake --build /tmp/g3pvm-fuzz -j4 \
  --target g3pvm_fuzz_bytecode_json g3pvm_fuzz_ast_verify
cp -a cpp/tests/fuzz/corpus/json /tmp/g3pvm-json-corpus
/tmp/g3pvm-fuzz/g3pvm_fuzz_bytecode_json \
  -runs=1000 /tmp/g3pvm-json-corpus
/tmp/g3pvm-fuzz/g3pvm_fuzz_ast_verify \
  -runs=1000 /tmp/g3pvm-json-corpus
```

If a campaign finds a defect, preserve its smallest reproducer under a
descriptive name in `cpp/tests/fuzz/corpus/` or as a named deterministic seed
regression in the relevant property target.

### Recommended full check

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
python3 -m unittest discover -s tests/repository -p 'test_*.py' -v
cmake --build cpp/build -j4
ctest --test-dir cpp/build --output-on-failure
```

## Spec Freeze Gate

The current breaking-refactor spec freeze is recorded in
`benchmarks/spec_freeze.json`. It hashes the authoritative current
spec set, with `spec/grammar.md` as the latest grammar entrypoint.

When changing any listed current spec, update the manifest in the same semantic
change and run:

```bash
python3 -m unittest tests.repository.test_spec_freeze -v
```

## GPU Run Policy

GPU-capable C++ paths select the least-used visible CUDA device internally.
To force a specific visible-device index for a run, set:

```bash
G3PVM_CUDA_DEVICE=0
```

Current GPU eval runtime behavior:
- one production `Mixed` eval kernel launch per accepted population
- payload flavor labels remain available for offline analysis, but are not used for production eval dispatch

## Public CLI Arguments

This section documents the adjustable arguments that affect supported public workflows.

### `cpp/build/g3pvm_evolve_cli`

The executable and native harnesses link `g3pvm_cli_support`; its option
parser is the single owner of the flags, defaults, and validation below.
`g3pvm_test_cli_options` locks that command-line contract independently of
evolution execution. `g3pvm_test_evolve_cli_contract` locks representative
evolution stdout, top-level JSON sections, and the intentionally unsupported
`--help`/no-argument error behavior.

Core execution args:
- `--cases PATH`: input fitness-cases file; current validation uses
  `fitness-cases`, while baseline comparisons may use
  `fitness-cases`
- `--engine {cpu|gpu}`: evaluation backend
- `--repro-backend {cpu|gpu}`: reproduction backend; `gpu` is the formal GPU reproduction path and does not promise CPU child identity
- `--repro-overlap {on|off}`: when `--engine gpu --repro-backend gpu`, overlap reproduction input prep with GPU evaluation
- `--cpu-repro-ablation {none|gpu_selection|gpu_candidates|gpu_coupled_donor}`: experimental CPU reproduction ablation flag for isolating GPU reproduction behaviors; valid only with `--repro-backend cpu`
- `--blocksize N`: CUDA block size for GPU evaluation; current native CLI default is `1024`
- `--out-json PATH`: write evolution history JSON
- `--eval-ast-json PATH`: evaluate a materialized AST JSON file against
  `--cases` and write an `ast-eval-result`; this is currently a CPU
  evaluation path used by PSB test-set reporting
- `--timing {none|summary|per_gen|all}`: timing verbosity from the native CLI
- `--show-program {none|ast|bytecode|both}`: include final-program dumps in output
- `--population-json PATH`: load a fixed `population-seeds` initial population instead of generating from `--seed`
- `--grammar-config PATH`: load a grammar config JSON file that restricts evolution generation and reproduction donor synthesis; current is the current schema, and checked-in base presets are translated as compatibility input for fair comparisons
- `--skip-final-eval {on|off}`: skip the post-loop final scoring pass; fixed-population timing runs should set this to `on`
- `--retain-final-population {on|off}`: when `off`, keep only `result.best` after the final scoring pass instead of materializing the full final scored population; the native CLI default is `off`

When `--population-json` is not supplied, native evolution infers a homogeneous expected output type from the loaded fitness cases. Under current, generation 0 should seed with that return type for payload return types (`String`, `IntList`, `FloatList`, and `StringList`) when the active grammar allows the inferred type. Other expected output types, mixed or unsupported expected output types, and grammar configs that disable the inferred payload type use generic random generation. A fixed `--population-json` file bypasses this inference and is replayed exactly.

Evolution args:
- `--population-size N`: individuals per generation
- `--generations N`: number of generations to run
- selected parent pairs always attempt `typed_subtree` crossover; mutation is applied afterward per child
- `--mutation-rate F`: probability that a post-crossover child is mutated
- `--mutation-subtree-prob F`: internal mutation operator mix; probability of typed-subtree mutation instead of constant perturbation
- `--selection-pressure N`: tournament size for each round-based without-replacement pass; larger values increase selection pressure; default is `2`
- `--seed N`: RNG seed for deterministic replay

Fitness args:
- `--penalty F`: penalty used for runtime errors and type-mismatch outputs that cannot be compared directly; must be `>= 0`
- `--fuel N`: per-program execution budget

Genome-shape args:
- `--max-expr-depth N`: maximum generated expression depth; default is `7`
- `--max-stmts-per-block N`: maximum statements per block
- `--max-total-nodes N`: maximum total AST nodes in one genome
- `--max-for-k N`: maximum integer constant used when the random generator seeds `ForRange(x, e, ...)` bounds with `Const(K)`; it is a generator limit, not a general static bound on every loop expression
- `--max-call-args N`: maximum allowed builtin call arity during generation/compilation

## Grammar Configs

Checked-in presets live under `configs/grammar/`:
- `all.json`: all public grammar constructs enabled; this is the default when `--grammar-config` is omitted
- `scalar.json`: scalar numeric / boolean search space; sequence values and container builtins disabled
- `string.json`: scalar plus `String` and string-compatible builtins
- `num_list.json`: legacy numeric-list profile; translated to current `IntList` and `FloatList` in current generation
- `string_list.json`: scalar plus `String` / `StringList` and string-list builtins
- `sequence.json`: broad sequence profile, currently equivalent to all first-wave sequence support

The config is a search-space control only. It does not reject bytecode execution or loading of already-materialized ASTs that contain disabled constructs.
The full config schema and replay rules are defined in [grammar-config.md](grammar-config.md).

Current backend support:
- CPU reproduction respects non-default grammar configs.
- GPU reproduction respects non-default grammar configs during host-side preprocess by filtering typed candidates and building config-aware donor buckets.

Example:

```bash
cpp/build/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --grammar-config configs/grammar/scalar.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --population-size 64 \
  --generations 5 \
  --out-json logs/simple_exp_1024.scalar.json
```


## Related workflows

- [Benchmarking](benchmarking.md): fixed-population timing and canonical run commands
- [PSB workflow](psb-workflow.md): dataset materialization, regression runs, comparison, and manifests
- [Experiment protocol](experiment-protocol.md): controlled performance and effectiveness study design

