cmake_minimum_required(VERSION 3.16)
project(g3p_vm_gpu_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(G3PVM_HAS_CUDA ON)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
else()
  set(G3PVM_HAS_CUDA OFF)
endif()

add_library(g3pvm_cpu
  src/runtime/payload/payload.cpp
  src/runtime/cpu/builtins_cpu.cpp
  src/runtime/cpu/execute_bytecode_cpu.cpp
  src/runtime/cpu/fitness_cpu.cpp
  src/evolution/ast_program.cpp
  src/evolution/genome.cpp
  src/evolution/subtree_utils.cpp
  src/evolution/typed_expr_analysis.cpp
  src/evolution/compiler.cpp
  src/evolution/genome_generation.cpp
  src/evolution/mutation.cpp
  src/evolution/crossover.cpp
  src/evolution/selection.cpp
  src/evolution/evolve.cpp
)
target_include_directories(g3pvm_cpu PUBLIC include)

add_executable(g3pvm_runtime_multi_bench
  src/bench/runtime_multi_bench.cpp
)
target_link_libraries(g3pvm_runtime_multi_bench PRIVATE g3pvm_cpu)

add_executable(g3pvm_evolve_cli
  src/cli/evolve_cli.cpp
)
target_link_libraries(g3pvm_evolve_cli PRIVATE g3pvm_cpu)

add_executable(g3pvm_population_bench_cli
  src/cli/population_bench_cli.cpp
)
target_link_libraries(g3pvm_population_bench_cli PRIVATE g3pvm_cpu)
if(G3PVM_HAS_CUDA)
  add_library(g3pvm_gpu
    src/runtime/gpu/fitness_gpu.cu
    src/runtime/gpu/opcode_map_gpu.cpp
    src/runtime/gpu/host_pack_gpu.cu
    src/runtime/payload/payload.cpp
  )
  target_include_directories(g3pvm_gpu PUBLIC include)
  target_compile_definitions(g3pvm_cpu PUBLIC G3PVM_HAS_CUDA=1)
  target_link_libraries(g3pvm_cpu PUBLIC g3pvm_gpu)
  target_link_libraries(g3pvm_population_bench_cli PRIVATE g3pvm_gpu)
  target_compile_definitions(g3pvm_population_bench_cli PRIVATE G3PVM_HAS_CUDA=1)

  add_executable(g3pvm_test_vm_gpu_smoke
    tests/gpu/test_vm_gpu_smoke.cpp
  )
  target_link_libraries(g3pvm_test_vm_gpu_smoke PRIVATE g3pvm_gpu)
  add_test(NAME g3pvm_test_vm_gpu_smoke COMMAND g3pvm_test_vm_gpu_smoke)

  add_executable(g3pvm_test_fitness_cpu_gpu_parity
    tests/parity/test_fitness_cpu_gpu_parity.cpp
  )
  target_link_libraries(g3pvm_test_fitness_cpu_gpu_parity PRIVATE g3pvm_cpu g3pvm_gpu)
  add_test(NAME g3pvm_test_fitness_cpu_gpu_parity COMMAND g3pvm_test_fitness_cpu_gpu_parity)

  add_executable(g3pvm_test_evolution_cpu_gpu_parity
    tests/parity/test_evolution_cpu_gpu_parity.cpp
  )
  target_link_libraries(g3pvm_test_evolution_cpu_gpu_parity PRIVATE g3pvm_cpu g3pvm_gpu)
  add_test(NAME g3pvm_test_evolution_cpu_gpu_parity COMMAND g3pvm_test_evolution_cpu_gpu_parity)

  add_executable(g3pvm_simple_exp_population_probe
    tests/parity/simple_exp_population_probe.cpp
  )
  target_link_libraries(g3pvm_simple_exp_population_probe PRIVATE g3pvm_cpu g3pvm_gpu)

endif()

add_executable(g3pvm_test_vm_smoke
  tests/runtime/test_vm_smoke.cpp
)
target_link_libraries(g3pvm_test_vm_smoke PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_vm_smoke COMMAND g3pvm_test_vm_smoke)

add_executable(g3pvm_test_vm_edges
  tests/runtime/test_vm_edges.cpp
)
target_link_libraries(g3pvm_test_vm_edges PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_vm_edges COMMAND g3pvm_test_vm_edges)

add_executable(g3pvm_test_genome
  tests/evolution/test_genome.cpp
)
target_link_libraries(g3pvm_test_genome PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_genome COMMAND g3pvm_test_genome)

add_executable(g3pvm_test_evolve
  tests/evolution/test_evolve.cpp
)
target_link_libraries(g3pvm_test_evolve PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_evolve COMMAND g3pvm_test_evolve)
