# Python Migration Matrix

This matrix assigns every tracked Python module and test module to a native or
tooling owner before deletion. Every `test_*` method in a row inherits the
row's disposition unless named in the exception column. No Python semantic
test may be deleted until its target stage is green.

## Source modules

| Current module(s) | Responsibility | Target owner | Removal stage |
| --- | --- | --- | --- |
| `core/ast.py` | prefix AST classes, evaluation helpers | native AST descriptor/verifier plus corpus | 10 |
| `core/errors.py` | semantic error codes | `g3pvm/core/errors.hpp` contract tests | 10 |
| `core/value_semantics.py` | exact public value typing/equality | native value/runtime contract tests | 10 |
| `runtime/builtins.py` | scalar, char, and sequence builtin oracle | native runtime corpus and CPU/GPU parity | 10 |
| `runtime/interp.py` | direct AST execution oracle | native AST verifier and semantic corpus | 10 |
| `runtime/compiler.py` | AST-to-bytecode oracle | compiler-lowering and bytecode-verifier tests | 10 |
| `runtime/vm.py` | Python bytecode VM | native runtime corpus and CPU/GPU parity | 10 |
| `evolution/genome.py`, `stmt_codec.py` | Python genome/statement representation | native AST codec and verifier properties | 9 |
| `evolution/random_program.py`, `random_tree.py`, `random_genome.py` | generation | native generation/verify/compile properties | 9 |
| `evolution/mutation.py`, `crossover.py` | variation | native mutation/crossover properties | 9 |
| `evolution/grammar_config.py` | search-space config | native grammar-config contract/properties | 9 |
| `evolution/evolve.py` | scoring, selection, evolution loop | native evolution and CPU/GPU parity tests | 9 |
| `demo.py` | Python demo entry point | native CLI examples | 10 |
| package `__init__.py` files | exports for the duplicate implementation | none after package removal | 9/10 |

## Test modules

| Current test module | Methods | Disposition and native/tool target | Target stage |
| --- | ---: | --- | ---: |
| `test_cpp_vm_equiv.py` | 13 | Move its fixture-harness format rejection, typed-value, builtin, ASGP segment, and compiler round-trip cases into the normal CMake fixture harness; stop ad-hoc source compilation. | 4–6 |
| `test_eval.py` | 28 | Port scalar/control flow, builtins, exact-list behavior, errors, fuel, and random-program execution to verifier/runtime contract fixtures. | 2–7 |
| `test_vm_equiv.py` | 8 | Replace interpreter-vs-Python-VM comparison with compiler-output verification and native CPU execution corpus; fuzz equivalence becomes generation/compile property coverage. | 4–7 |
| `test_grammar_values.py` | 9 | Port format identifier, node inventory, exact value tags/equality, and ASGP declaration/lowering assertions to native AST/value/compiler contracts. | 1–6 |
| `test_grammar_builtins.py` | 8 | Port protected arithmetic, char/string, direct list, singleton, and float-format outcomes to native runtime corpus. | 5 |
| `test_grammar_vm.py` | 30 | Port structured list, binder capture/isolation, evaluation order, fuel/errors, protected ops, and char/list bytecode behavior to structured conformance fixtures. | 5–7 |
| `test_grammar_asgp.py` | 14 | Port DC/DP1D/DP2D typing, phase visibility, dependency, memoization, boundary, nested-form, fuel, and execution cases to verifier and ASGP corpus. | 3–7 |
| `test_evolution_ops.py` | 14 | Replace grammar-config/generation/mutation/crossover compile-rate assertions with native contract and deterministic property tests. | 1–7, 9 |
| `test_evolution_loop.py` | 5 | Replace option validation, scoring, selection, and loop behavior with native evolution tests. | 5–7, 9 |
| `test_simple_evo_fixtures.py` | 3 | Retain the JSON fixtures; execute affine, square, and x+1 through native evolution tests. | 5–6 |
| `test_docs_contract.py` | 2 | Move to runtime-independent repository documentation checks. | 8, 18 |
| `test_spec_freeze.py` | 1 | Move unchanged responsibility to runtime-independent repository checks callable from CTest. | 8, 18 |
| `test_psb1_tools.py` | 3 | Move under `tools/tests`; keep converter schema coverage. | 8 |
| `test_psb2_tools.py` | 5 | Move under `tools/tests`; keep converter schema/error coverage. | 8 |
| `test_psb_fetch_tools.py` | 4 | Move under `tools/tests`; keep dry-run selection/error coverage. | 8 |
| `test_psb_regression_tools.py` | 14 | Move under `tools/tests`; keep materialization, seed, run, comparison, and manifest coverage. | 8 |

The method counts total 161 discovered `def test_*` declarations, matching the
161 tests executed by the baseline runner. Before Stage 10, generate a
runner-derived fully qualified test list and require every executed identifier
to match one of these rows and an existing native/tool test.

## Removal invariants

- Tool tests must import tool modules without `g3p_vm_gpu` or `PYTHONPATH`.
- Golden results are reviewed against `spec/`; they are not copied blindly
  from the Python implementation.
- The C++/Python differential harness disappears only after its fixture corpus
  runs from a normal native target.
- Git history, rather than an archived duplicate directory, preserves the old
  implementation.
