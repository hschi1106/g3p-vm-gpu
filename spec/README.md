# Specifications

Files in this directory are the normative behavioral source of truth. Design
documents and guides explain implementation or use, but they do not override
these contracts.

| Specification | Contract |
| --- | --- |
| [`grammar.md`](grammar.md) | AST grammar, types, binding, control flow, and evaluation order |
| [`bytecode_isa.md`](bytecode_isa.md) | VM instructions and execution semantics |
| [`bytecode_format.md`](bytecode_format.md) | Bytecode and value JSON formats |
| [`builtins_base.md`](builtins_base.md) | Scalar and character builtins |
| [`builtins_runtime.md`](builtins_runtime.md) | Container builtins and runtime payload behavior |
| [`fitness.md`](fitness.md) | Fitness calculation and solved criteria |
| [`fitness_cases.md`](fitness_cases.md) | Fitness-case fixture schema |
| [`grammar_config.md`](grammar_config.md) | Evolution search-space configuration |

Public format identifiers are defined by the specification that owns the
format. Compatibility and release history belong in [`../VERSION.md`](../VERSION.md),
not in historical copies of specifications. A semantic change must update its
owning spec, conformance tests, and `benchmarks/spec_freeze.json` together.

