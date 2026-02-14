cmake_minimum_required(VERSION 3.16)
project(g3p_vm_gpu_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(g3pvm_cpu
  src/builtins.cpp
  src/vm_cpu.cpp
)
target_include_directories(g3pvm_cpu PUBLIC include)

add_executable(g3pvm_vm_cpu_cli
  src/vm_cpu_cli.cpp
)
target_link_libraries(g3pvm_vm_cpu_cli PRIVATE g3pvm_cpu)
