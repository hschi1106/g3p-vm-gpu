#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/vm_cpu.hpp"
#include "types.hpp"

namespace g3pvm::gpu_detail {

struct PackResult {
  std::vector<DProgramMeta> metas;
  std::vector<DInstr> all_code;
  std::vector<Value> all_consts;
  std::vector<Value> packed_case_local_vals;
  std::vector<unsigned char> packed_case_local_set;
  std::size_t total_cases = 0;
  std::size_t max_code_len = 0;
};

PackResult pack_programs_and_shared_cases(const std::vector<BytecodeProgram>& programs,
                                          const std::vector<InputCase>& shared_cases);
PackResult pack_programs_with_shared_case_count(const std::vector<BytecodeProgram>& programs,
                                                int shared_case_count);
void pack_shared_cases_only(const std::vector<InputCase>& shared_cases,
                            std::vector<Value>* packed_case_local_vals,
                            std::vector<unsigned char>* packed_case_local_set);

struct DeviceArena {
  Value* d_consts = nullptr;
  DInstr* d_code = nullptr;
  DProgramMeta* d_metas = nullptr;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  DResult* d_out = nullptr;
  Value* d_expected = nullptr;
  int* d_fitness = nullptr;

  DeviceArena() = default;
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  ~DeviceArena();
};

template <typename T>
bool cuda_alloc_and_copy_in(const std::vector<T>& host, T** dev) {
  if (host.empty()) {
    *dev = nullptr;
    return true;
  }
  if (cudaMalloc(reinterpret_cast<void**>(dev), sizeof(T) * host.size()) != cudaSuccess) {
    return false;
  }
  if (cudaMemcpy(*dev, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*dev);
    *dev = nullptr;
    return false;
  }
  return true;
}

}  // namespace g3pvm::gpu_detail
