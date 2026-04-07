# Repository Structure

```text
g3p-vm-gpu/
├── AGENTS.md
├── README.md
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
│   │   │   └── repro/
│   │   └── cli/
│   ├── src/
│   │   ├── runtime/
│   │   │   ├── cpu/
│   │   │   ├── gpu/
│   │   │   │   └── device/
│   │   │   └── payload/
│   │   ├── evolution/
│   │   │   └── repro/
│   │   ├── cli/
│   │   └── bench/
│   ├── tests/
│   │   ├── evolution/
│   │   ├── gpu/
│   │   ├── parity/
│   │   └── runtime/
│   └── CMakeLists.txt
├── configs/
│   └── grammar/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CPP_RUNTIME_PAYLOAD.md
│   ├── DEVELOPMENT.md
│   ├── GRAMMAR_CONFIG.md
│   ├── GPU_REPRODUCTION.md
│   ├── TIMING.md
│   └── FILE_STRUCTURE.md
├── spec/
├── data/
│   ├── fixtures/
│   ├── psb1_datasets/
│   └── psb2_datasets/
├── meeting/
├── tools/
└── logs/
```

## Directory Roles

- `AGENTS.md`: repo-local working conventions for coding agents
- `README.md`: entrypoint and quick workflow
- `python/`: reference semantics and Python-side tests
- `cpp/`: native runtime, GPU fitness backend, evolution engine, CLIs, and native tests
- `configs/grammar/`: checked-in `grammar-config-v1` presets for evolution search-space control
- `spec/`: normative behavior contracts
- `docs/`: operational, architectural, and payload-model documentation
- `docs/TIMING.md`: canonical timing metric names, scopes, and CLI/JSON mappings
- `docs/GRAMMAR_CONFIG.md`: external config format for evolution grammar search-space controls
- `docs/GPU_REPRODUCTION.md`: GPU reproduction backend data flow, overlap model, and performance notes
- `data/fixtures/`: canonical benchmark and evolution fixtures
- `data/psb1_datasets/`: mirrored PSB1 source datasets
- `data/psb2_datasets/`: mirrored PSB2 source datasets
- `meeting/`: meeting notes and discussion artifacts
- `tools/`: dataset fetch, conversion, and audit utilities
- `logs/`: generated artifacts, benchmark reports, and run outputs

## Spec Roles

- `spec/grammar_v1_0.md`: language grammar and evaluation rules
- `spec/bytecode_isa_v1_0.md`: VM instruction contract
- `spec/bytecode_format_v1_0.md`: JSON wire format
- `spec/builtins_base_v1_0.md`: scalar builtins
- `spec/builtins_runtime_v1_0.md`: container builtins and payload rules
- `spec/fitness_v1_0.md`: scoring formulas and solved criteria
