# Native Refactor Release Audit

This record closes the native-only refactor after Stage 20 at `445c358`. It is
descriptive evidence, not a semantic specification. Normative behavior remains
owned by [`../../spec/`](../../spec/).

## Result

The release audit passed. The maintained product has one C++/CUDA semantic
implementation, native AST and bytecode verification, focused native
conformance/property/parity tests, independently packaged operational Python
tools, explicit native library ownership, and checked documentation ownership.

The former Python runtime/evolution package and `draw/draw.py` are absent.
Historical implementation access is through Git; no archived duplicate source
tree is retained.

## Toolchain and configuration gates

The final audit used CMake 3.22.1, GCC 11.4.0, Clang 14.0.0 for fuzz targets,
CUDA 12.2.140, driver 595.84, and two 24,564 MiB RTX 4090 devices.

All named configure presets resolved: `debug`, `release`, `cuda-parity`,
`sanitizer`, and `fuzz`. The two opt-in libFuzzer binaries built, and both the
AST JSON/verifier and bytecode JSON/verifier targets completed sequential,
fixed-seed 1,000-run smoke campaigns without a crash. Generated fuzz
discoveries were removed after the run; only the bounded committed corpus
remains.

## Correctness gates

| Gate | Result |
| --- | ---: |
| Debug CUDA CTest | 33/33 passed |
| CPU ASan/UBSan CTest | 30/30 passed |
| CPU/GPU parity label | 2/2 passed |
| Operational tool tests | 30/30 passed |
| Repository/spec/docs/tool-ownership checks | 16/16 passed |
| Current PSB runner smoke (`count-odds`, seed 0) | 1/1 run succeeded |

Every non-success `VerifyCode` is named by a deterministic native assertion.
Grammar forms are covered across typed verifier, compiler, focused runtime,
structured/ASGP, property, and CPU/GPU parity owners. The retired 161 Python
semantic/tool tests have a module- and responsibility-level disposition in
[`python-migration.md`](python-migration.md).

## Fixed-population performance gate

The audit replayed `logs/refactor_baseline.seeds.json` with the same 1,024
cases, 1,024 seeds, Debug build, one generation, block size 256, skipped final
evaluation, and all timing fields enabled. Raw JSON stays local under `logs/`;
the hashes below identify the exact audit captures.

| Mode | Baseline wall ms | Audit wall ms | Ratio | Baseline generation ms | Audit generation ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU eval / CPU repro | 21,620.104 | 20,363.759 | 0.942 | 21,581.430 | 20,314.663 |
| GPU eval / CPU repro | 700.130 | 728.659 | 1.041 | 232.244 | 358.109 |
| GPU eval / GPU repro | 756.936 | 796.344 | 1.052 | 308.534 | 456.745 |
| GPU eval / GPU repro overlap | 744.613 | 811.935 | 1.090 | 288.243 | 429.670 |

All wall-clock ratios remain inside the 1.15 release threshold. The larger GPU
generation-phase ratios are explained by two auditable changes rather than a
hidden kernel regression: the seed manifest replays generator seeds rather
than frozen AST bytes, so the current generator produces a different workload,
and current timing assigns roughly 200 ms of host compilation to generation
where the old capture reported roughly 20 ms. GPU evaluation kernel time
improved substantially (for example, overlap mode 154.334 ms to 27.186 ms).

The Stage 19 isolated comparison against its direct parent used identical ASTs,
fitness, and program key: wall time was 649.685 versus 712.512 ms (0.912x) and
generation time was 304.384 versus 351.889 ms (0.865x). After the library-only
Stage 20 split, an additional overlap smoke retained identical fitness/key and
measured 801.155 ms versus the immediately preceding 811.935 ms capture.

| Local audit artifact | SHA-256 |
| --- | --- |
| `logs/final-audit.cpu.json` | `fb4b49898d5a7cc00ad30d10d8f9efa696cfa6afc8845ced847a39bfd74ed63d` |
| `logs/final-audit.gpu-eval.json` | `1722213c8c3a60a7e67b2c73e6e8c5eb514cb2914c68ae8b325cc7b132e9ae02` |
| `logs/final-audit.gpu-repro.json` | `01c589179f5fba5dc89528a56c75802ecf8dd11023847d27d3d913908dd6a593` |
| `logs/final-audit.gpu-overlap.json` | `1b934d9e576f780a624257c3cf18c856d2fb6b5184a3d6b5c589a6404832b6cb` |
| `logs/final-audit-psb-smoke/summary.json` | `abd5bf67443bb044cd039a27c6fabca49470bfa5cb574d2bdb5be48f61c11a54` |

## PSB quality evidence

The representative committed quality gate remains
[`../../benchmarks/psb1_supported_compact_5seed_pop512.json`](../../benchmarks/psb1_supported_compact_5seed_pop512.json).
It is internally consistent and passed: 28/28 supported problems, 140/140
candidate runs successful, zero failed problems, and no train/test quality
failure categories. It is retained as historical comparison evidence; its
recorded grammar-config hash identifies the config used for that campaign and
is not rewritten to claim a fresh 140-run campaign. The current native binary
also completed the documented one-problem PSB runner smoke successfully.

## Ownership and obsolete-path audit

- CMake exposes `gagp_core`, `gagp_runtime_cpu`, `gagp_evolution`,
  `gagp_cli_support`, and `gagp_gpu`; `gagp_cpu` is interface-only
  compatibility.
- No C++ implementation file includes another `.cpp` file.
- Python files are limited to operational tools, their tests, and repository
  checks; none implements runtime, compiler, grammar typing, fitness, or
  evolution semantics.
- Benchmark/probe targets are opt-in and every top-level tool is classified.
- The external repository skill contains no retired documentation, Python
  package, plot-script, or uncommitted-path reference.
- Goals remain local through `.git/info/exclude`; no `goals/` file is tracked.

No unexplained correctness, performance, quality, ownership, or obsolete-path
finding remains open.
