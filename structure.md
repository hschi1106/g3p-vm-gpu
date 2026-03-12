# Repository Structure

```text
g3p-vm-gpu/
├── python/
│   ├── src/g3p_vm_gpu/
│   │   ├── core/
│   │   ├── runtime/
│   │   ├── evolution/
│   │   ├── __init__.py
│   │   └── demo.py
│   └── tests/
├── cpp/
│   ├── include/g3pvm/
│   │   ├── core/
│   │   ├── runtime/
│   │   │   ├── cpu/
│   │   │   ├── gpu/
│   │   │   └── payload/
│   │   ├── evolution/
│   │   └── cli/
│   ├── src/
│   │   ├── runtime/
│   │   │   ├── cpu/
│   │   │   ├── gpu/
│   │   │   │   └── device/
│   │   │   └── payload/
│   │   ├── evolution/
│   │   ├── cli/
│   │   └── bench/
│   ├── tests/
│   └── CMakeLists.txt
├── spec/
├── docs/
├── tools/
├── scripts/
├── data/
└── logs/
```

## Directory Roles

- `python/`: reference semantics and Python-side tests
- `cpp/`: native runtime, GPU fitness backend, evolution engine, native tests
- `spec/`: normative behavior contracts
- `docs/`: operational and architectural documentation
- `tools/`: dataset fetch, conversion, and audit utilities
- `scripts/`: direct execution wrappers used by humans and agents
- `data/`: fixtures and datasets
- `logs/`: generated artifacts

## Spec Roles

- `spec/grammar_v1_0.md`: language grammar and evaluation rules
- `spec/bytecode_isa_v1_0.md`: VM instruction contract
- `spec/bytecode_format_v1_0.md`: JSON wire format
- `spec/builtins_base_v1_0.md`: scalar builtins
- `spec/builtins_runtime_v1_0.md`: container builtins and payload rules
- `spec/fitness_v1_0.md`: scoring formulas and solved criteria
