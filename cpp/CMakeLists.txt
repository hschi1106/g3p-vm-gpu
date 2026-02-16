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
  src/builtins.cpp
  src/vm_cpu.cpp
  src/evo_ast.cpp
  src/evolve.cpp
)
target_include_directories(g3pvm_cpu PUBLIC include)

add_executable(g3pvm_vm_cpu_cli
  src/vm_cpu_cli.cpp
)
target_link_libraries(g3pvm_vm_cpu_cli PRIVATE g3pvm_cpu)

add_executable(g3pvm_vm_cpu_multi_bench
  src/vm_cpu_multi_bench.cpp
)
target_link_libraries(g3pvm_vm_cpu_multi_bench PRIVATE g3pvm_cpu)

add_executable(g3pvm_evolve_cli
  src/evolve_cli.cpp
)
target_link_libraries(g3pvm_evolve_cli PRIVATE g3pvm_cpu)
if(G3PVM_HAS_CUDA)
  add_library(g3pvm_gpu
    src/vm_gpu.cu
    src/vm_gpu/opcode_map.cpp
    src/vm_gpu/host_pack.cu
  )
  target_include_directories(g3pvm_gpu PUBLIC include)
  target_link_libraries(g3pvm_vm_cpu_cli PRIVATE g3pvm_gpu)
  target_compile_definitions(g3pvm_vm_cpu_cli PRIVATE G3PVM_HAS_CUDA=1)

  add_executable(g3pvm_vm_gpu_multi_bench
    src/vm_gpu_multi_bench.cpp
  )
  target_link_libraries(g3pvm_vm_gpu_multi_bench PRIVATE g3pvm_gpu)

  add_executable(g3pvm_test_vm_gpu_smoke
    tests/test_vm_gpu_smoke.cpp
  )
  target_link_libraries(g3pvm_test_vm_gpu_smoke PRIVATE g3pvm_gpu)
  add_test(NAME g3pvm_test_vm_gpu_smoke COMMAND g3pvm_test_vm_gpu_smoke)

  add_executable(g3pvm_test_vm_gpu_multi_batch
    tests/test_vm_gpu_multi_batch.cpp
  )
  target_link_libraries(g3pvm_test_vm_gpu_multi_batch PRIVATE g3pvm_gpu)
  add_test(NAME g3pvm_test_vm_gpu_multi_batch COMMAND g3pvm_test_vm_gpu_multi_batch)

endif()

add_executable(g3pvm_test_vm_smoke
  tests/test_vm_smoke.cpp
)
target_link_libraries(g3pvm_test_vm_smoke PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_vm_smoke COMMAND g3pvm_test_vm_smoke)

add_executable(g3pvm_test_vm_edges
  tests/test_vm_edges.cpp
)
target_link_libraries(g3pvm_test_vm_edges PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_vm_edges COMMAND g3pvm_test_vm_edges)

add_executable(g3pvm_test_evo_ast
  tests/test_evo_ast.cpp
)
target_link_libraries(g3pvm_test_evo_ast PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_evo_ast COMMAND g3pvm_test_evo_ast)

add_executable(g3pvm_test_evolve
  tests/test_evolve.cpp
)
target_link_libraries(g3pvm_test_evolve PRIVATE g3pvm_cpu)
add_test(NAME g3pvm_test_evolve COMMAND g3pvm_test_evolve)
