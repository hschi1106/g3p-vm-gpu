# Version

Current release: 1.0.0

This release defines the current grammar, AST, bytecode, builtin, fitness, and
PSB fixture contracts in `spec/`.

Public format identifiers are intentionally unversioned in source files:

- `ast-prefix`
- `bytecode-json`
- `bytecode-fixture`
- `fitness-cases`
- `grammar-config`
- `population-seeds`

Version changes are recorded here only. When a future breaking contract change
is made, update this file in the same change as the affected specs, docs, tests,
and benchmark manifests.
