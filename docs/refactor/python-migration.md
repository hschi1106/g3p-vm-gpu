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
| `test_cpp_vm_equiv.py` (removed in Stage 05) | 13 | Replaced by the normal CMake fixture harness, focused semantic corpus, compiler-lowering contracts, and Stage 04 verifier/ASGP boundary tests. | 4–5 |
| `test_eval.py` | 28 | Port scalar/control flow, builtins, exact-list behavior, errors, fuel, and random-program execution to verifier/runtime contract fixtures. | 2–7 |
| `test_vm_equiv.py` (removed in Stage 07) | 8 | Interpreter/VM outcomes are owned by native semantic contracts; compiler-output verification and deterministic generation/variation properties replace its random equivalence loop. | 4–7 |
| `test_grammar_values.py` | 9 | Port format identifier, node inventory, exact value tags/equality, and ASGP declaration/lowering assertions to native AST/value/compiler contracts. | 1–6 |
| `test_grammar_builtins.py` | 8 | Port protected arithmetic, char/string, direct list, singleton, and float-format outcomes to native runtime corpus. | 5 |
| `test_grammar_vm.py` (removed in Stage 06) | 30 | Scalar/value/builtin cases are owned by Stage 05 fixture targets; structured and ASGP cases are owned by focused Stage 06 contracts and verifier/parity targets. | 5–6 |
| `test_grammar_asgp.py` (removed in Stage 06) | 14 | DC/DP1D/DP2D execution, boundary, memoization, phase visibility, dependency/result typing, nested forms, and fuel now have focused native runtime/verifier owners. | 3–6 |
| `test_evolution_ops.py` | 14 | Replace grammar-config/generation/mutation/crossover compile-rate assertions with native contract and deterministic property tests. | 1–7, 9 |
| `test_evolution_loop.py` | 5 | Replace option validation, scoring, selection, and loop behavior with native evolution tests. | 5–7, 9 |
| `test_simple_evo_fixtures.py` | 3 | Retain the JSON fixtures; execute affine, square, and x+1 through native evolution tests. | 5–6 |
| `test_docs_contract.py` | 2 | Move to runtime-independent repository documentation checks. | 8, 18 |
| `test_spec_freeze.py` | 1 | Move unchanged responsibility to runtime-independent repository checks callable from CTest. | 8, 18 |
| `test_psb1_tools.py` (moved to `tools/tests` in Stage 08) | 3 | Tool-owned converter schema coverage; no runtime-package import or `PYTHONPATH`. | 8 |
| `test_psb2_tools.py` (moved to `tools/tests` in Stage 08) | 5 | Tool-owned converter schema/error coverage; no runtime-package import or `PYTHONPATH`. | 8 |
| `test_psb_fetch_tools.py` (moved to `tools/tests` in Stage 08) | 4 | Tool-owned dry-run selection/error coverage; no runtime-package import or `PYTHONPATH`. | 8 |
| `test_psb_regression_tools.py` (moved to `tools/tests` in Stage 08) | 14 | Tool-owned materialization, seed, run, comparison, and manifest coverage; no runtime-package import or `PYTHONPATH`. | 8 |

The method counts total 161 discovered `def test_*` declarations, matching the
161 tests executed by the baseline runner. Before Stage 10, generate a
runner-derived fully qualified test list and require every executed identifier
to match one of these rows and an existing native/tool test.

### Retired cross-language harness mapping

Stage 05 removes `test_cpp_vm_equiv.py` and its per-test temporary `g++`
build. Its 13 methods are owned as follows:

| Retired responsibility | Native owner |
| --- | --- |
| reject old bytecode and fixture format identifiers | `g3pvm_test_fixture_codec` |
| execute current fixtures and report mismatches | `g3pvm_test_fixture_codec` plus the four `g3pvm_test_runtime_*` targets |
| decode and execute ASGP-DC, DP1D, and DP2D segments | `g3pvm_test_bytecode_verify`, `g3pvm_test_cli_json`, and `g3pvm_test_ast_json_boundary` |
| accept compiler-generated ASGP segment layouts | compiler-output property cases in `g3pvm_test_bytecode_verify` and native ASGP AST boundary fixtures |
| `ForRange` lowering and evaluate-once bounds | `g3pvm_test_compiler_lowering` and `g3pvm_test_runtime_control_flow` |
| protected integer builtins | `g3pvm_test_runtime_builtins` |
| `Char` equality and exact `Char`/`String` distinction | `g3pvm_test_runtime_typed_values` |
| string indexing returns `Char` | `g3pvm_test_runtime_typed_values` and `g3pvm_test_runtime_builtins` |
| direct typed-list builtins return exact scalars | `g3pvm_test_runtime_typed_values` and `g3pvm_test_runtime_builtins` |

The baseline remains 161 tests; after this retirement the Python runner owns
148 declarations and the 13 removed responsibilities are native CTest owners.

### Retired structured and ASGP reference mapping

Stage 06 removes the 30-method `test_grammar_vm.py` and 14-method
`test_grammar_asgp.py` suites after reviewing their expected outcomes against
`spec/grammar.md`, `spec/bytecode_isa.md`, `spec/builtins_base.md`, and
`spec/builtins_runtime.md`.

| Retired responsibility | Native owner |
| --- | --- |
| scalar, exact value, char, protected integer, and direct-list behavior | Stage 05 `g3pvm_test_runtime_{scalar,builtins,typed_values}` |
| `MapList` / `FilterList` ordering, evaluate-once source, typed empty result, error, and fuel | `g3pvm_test_structured_semantics` |
| `LinearRec` empty/singleton branch selection, right-to-left step, binder metadata, error, and fuel | `g3pvm_test_structured_semantics`, `g3pvm_test_ast_verify_binders`, and reproduction metadata tests |
| nested binder capture and ordinary-local isolation | `g3pvm_test_structured_semantics` and `g3pvm_test_ast_verify_binders` |
| ASGP-DC success, string traversal, clamped split, result typing, phase visibility, and fuel | `g3pvm_test_asgp_semantics` plus `g3pvm_test_ast_verify_asgp` |
| ASGP-DP1D/DP2D recurrence, boundary, memoization, dependency/result typing, phase visibility, and fuel | `g3pvm_test_asgp_semantics` plus `g3pvm_test_ast_verify_asgp` |
| nested ASGP forms, dependency arity, and malformed side tables | `g3pvm_test_ast_verify_asgp`, `g3pvm_test_bytecode_verify`, and `g3pvm_test_cli_json` |
| interpreter-versus-VM comparison | obsolete once reviewed outcomes execute in the native compiler/runtime contracts; CPU/GPU behavior remains checked by the canonical parity targets |

After Stages 05–06, the remaining Python runner owns 104 declarations. The 57
retired methods have explicit native owners above.

### Retired Python VM equivalence mapping

Stage 07 removes `test_vm_equiv.py` after its eight methods have native owners:

| Retired responsibility | Native owner |
| --- | --- |
| manual program, dynamic loop bound, short-circuit, invalid loop bounds, and missing return | Stage 05 scalar/control-flow corpus and compiler-lowering contracts |
| current builtin and exact `Char`/`String` behavior | Stage 05 builtin and typed-value corpus |
| interpreter-versus-VM random equivalence | `g3pvm_test_genome_properties`: deterministic generation → verify → compile → execute; mutation/crossover/reproduction verify → compile properties |

After Stage 07 the Python runner owned 96 declarations. Stage 08 moves the 26
operational methods to `tools/tests`, leaving 70 transitional semantic/docs
methods under `python/tests`. Evolution and runtime modules remain until their
later roadmap stages.

## Removal invariants

- Tool tests must import tool modules without `g3p_vm_gpu` or `PYTHONPATH`.
- Golden results are reviewed against `spec/`; they are not copied blindly
  from the Python implementation.
- The C++/Python differential harness disappears only after its fixture corpus
  runs from a normal native target.
- Git history, rather than an archived duplicate directory, preserves the old
  implementation.
