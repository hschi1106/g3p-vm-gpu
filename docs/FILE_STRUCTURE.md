# Repository Structure

```text
g3p-vm-gpu/
├── AGENTS.md
├── README.md
├── python/
│   ├── src/g3p_vm_gpu/
│   │   ├── core/
│   │   ├── runtime/
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
│   │   ├── fixtures/runtime/
│   │   ├── gpu/
│   │   ├── parity/
│   │   └── runtime/
│   └── CMakeLists.txt
├── configs/
│   ├── grammar/
│   └── psb_tolerances/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CPP_RUNTIME_PAYLOAD.md
│   ├── DEVELOPMENT.md
│   ├── EXPERIMENT_SPEC.md
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
- `python/`: transitional runtime-reference semantics and tests; evolution is
  native-only
- `cpp/`: native runtime, GPU fitness backend, evolution engine, CLIs, and native tests; reusable semantic fixtures live under `cpp/tests/fixtures/runtime/`
- `configs/grammar/`: checked-in legacy grammar presets; the native loader
  translates them for compatibility comparisons
- `configs/psb_tolerances/`: versioned problem-specific PSB quality tolerance policies used by comparison gates
- `spec/`: normative behavior contracts
- `docs/`: operational, architectural, and payload-model documentation
- `docs/EXPERIMENT_SPEC.md`: pre-registered speedup, scaling, dataset, and evolutionary-effectiveness protocol
- `docs/TIMING.md`: canonical timing metric names, scopes, and CLI/JSON mappings
- `docs/GRAMMAR_CONFIG.md`: external config format for evolution grammar search-space controls
- `docs/GPU_REPRODUCTION.md`: GPU reproduction backend data flow, overlap model, and performance notes
- `data/fixtures/`: canonical benchmark and evolution fixtures, including generated PSB smoke fixtures under `data/fixtures/psb1/`
- `data/psb1_datasets/`: mirrored PSB1 source datasets
- `data/psb2_datasets/`: mirrored PSB2 source datasets
- `meeting/`: meeting notes and discussion artifacts
- `tools/`: dataset, fixture, grammar-profile, fixed-population, and PSB
  regression utilities
- `logs/`: generated artifacts, benchmark reports, and run outputs

## Spec Roles

- `spec/grammar.md`: current grammar and evaluation rules
- `spec/bytecode_isa.md`: current VM instruction contract
- `spec/bytecode_format.md`: current JSON wire format
- `spec/builtins_base.md`: current scalar and char builtins
- `spec/builtins_runtime.md`: current container builtins and payload rules
- `spec/fitness.md`: current scoring formulas and solved criteria
- `spec/fitness_cases.md`: current fixture schema
- `spec/grammar_config.md`: current search-space config contract
- Historical spec files are intentionally not kept in-tree. Release details are
  recorded only in `VERSION.md`.
