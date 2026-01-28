# g3p-vm-gpu

g3p-vm-gpu/
  README.md
  LICENSE
  .gitignore

  spec/
    subset_v0_1.md              # 你那套 Python-like subset 規格（語法/限制/錯誤）
    bytecode_isa.md             # opcode/operand/stack effects/錯誤規則（單一真理來源）
    builtins.md                 # abs/min/max/clip 的定義（含型別/錯誤）
    test_contract.md            # 一致性契約：interp == vm_py == vm_gpu 的比較規則

  python/
    pyproject.toml              # 或 setup.cfg/requirements.txt，選一個
    src/
      g3p_vm_gpu/
        __init__.py
        ast.py                  # AST nodes
        errors.py               # ErrorKind/EvalError
        builtins.py             # builtin_apply
        interp.py               # reference interpreter (big-step + fuel)
        compiler.py             # AST -> bytecode（label patching）
        vm.py                   # CPU bytecode VM（驗證用，不追求快）
        fuzz.py                 # grammar/AST fuzz generator
        demo.py                 # 最小可跑 demo
        benches/
          microbench.py         # instruction mix / throughput（CPU）
    tests/
      test_interp.py
      test_vm_equiv.py          # fuzz: interp == vm_py(compile)
      test_builtins.py

  cpp/
    CMakeLists.txt
    include/
      g3pvm/
        bytecode.hpp            # BytecodeProgram struct, encoding helpers
        value.hpp               # Value representation（tagged union）
        errors.hpp              # error codes aligned with spec
        builtins.hpp            # builtin ids + semantics
        vm_cpu.hpp              # (可選) C++ CPU VM baseline
        vm_gpu.hpp              # GPU runner interface
    src/
      bytecode.cpp
      builtins.cpp
      vm_cpu.cpp                # 可選
      vm_gpu.cu                 # CUDA kernel + launcher
      api.cpp                   # 對外 C API / Python binding glue
    tests/
      test_vm_smoke.cpp
      test_equiv_small.cpp      # 小規模對照（跑固定 bytecode case）

  bindings/
    python/
      pybind/
        CMakeLists.txt
        module.cpp              # pybind11: run_gpu(bytecode, inputs) -> results

  tools/
    gen_random_programs.py      # 產生固定 seed 的測試集合（跨語言都用）
    dump_bytecode.py            # human-readable disasm
    compare_results.py          # 比較 interp/vm_py/vm_gpu 的結果差異

  data/
    benchmarks/
      expr_regression/          # 你自己的小 benchmark 任務/資料
      boolean_tasks/
    fixtures/
      bytecode_cases.json       # 固定測試向量（確保回歸）

  scripts/
    run_tests.sh
    run_bench_cpu.sh
    run_bench_gpu.sh
    format.sh

  .github/
    workflows/
      ci.yml                    # Python tests + (可選) C++ build


PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v