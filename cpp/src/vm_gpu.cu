#include "g3pvm/vm_gpu.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "vm_gpu/constants.hpp"
#include "vm_gpu/kernels.cuh"
#include "vm_gpu/opcode_map.hpp"
#include "vm_gpu/types.hpp"

namespace g3pvm {

namespace {

using gpu_detail::DeviceErrCode;
using gpu_detail::DInstr;
using gpu_detail::DProgramMeta;
using gpu_detail::DResult;
using gpu_detail::DERR_TYPE;
using gpu_detail::DERR_VALUE;
using gpu_detail::MAX_LOCALS;
using gpu_detail::DINSTR_HAS_A;
using gpu_detail::DINSTR_HAS_B;

template <typename T>
bool cuda_alloc_and_copy_in(const std::vector<T>& host, T** dev) {
  if (host.empty()) {
    *dev = nullptr;
    return true;
  }
  if (cudaMalloc(reinterpret_cast<void**>(dev), sizeof(T) * host.size()) != cudaSuccess) return false;
  if (cudaMemcpy(*dev, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*dev);
    *dev = nullptr;
    return false;
  }
  return true;
}

std::vector<std::vector<VMResult>> multi_single_error(ErrCode code, const std::string& message) {
  return std::vector<std::vector<VMResult>>{
      std::vector<VMResult>{VMResult{true, Value::none(), Err{code, message}}}};
}

bool select_least_used_device(cudaDeviceProp& props_out, std::string& message_out) {
  int device_count = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess || device_count <= 0) {
    message_out = "cuda device unavailable";
    if (count_err != cudaSuccess) {
      message_out += " err=";
      message_out += cudaGetErrorString(count_err);
    }
    return false;
  }

  int best_dev = -1;
  double best_used_ratio = 2.0;
  std::size_t best_used_bytes = std::numeric_limits<std::size_t>::max();

  for (int dev = 0; dev < device_count; ++dev) {
    if (cudaSetDevice(dev) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total_bytes == 0) {
      continue;
    }
    const std::size_t used_bytes = total_bytes - free_bytes;
    const double used_ratio = static_cast<double>(used_bytes) / static_cast<double>(total_bytes);

    if (best_dev < 0 || used_ratio < best_used_ratio ||
        (std::fabs(used_ratio - best_used_ratio) <= 1e-12 && used_bytes < best_used_bytes)) {
      best_dev = dev;
      best_used_ratio = used_ratio;
      best_used_bytes = used_bytes;
    }
  }

  if (best_dev < 0) {
    message_out = "cuda device unavailable err=no usable device found";
    return false;
  }

  const cudaError_t set_err = cudaSetDevice(best_dev);
  if (set_err != cudaSuccess) {
    message_out = "cuda device select failure err=";
    message_out += cudaGetErrorString(set_err);
    return false;
  }

  const cudaError_t prop_err = cudaGetDeviceProperties(&props_out, best_dev);
  if (prop_err != cudaSuccess) {
    message_out = "cuda device query failure err=";
    message_out += cudaGetErrorString(prop_err);
    return false;
  }

  return true;
}

void cuda_cleanup_multi(Value* d_consts, DInstr* d_code, DProgramMeta* d_metas,
                        Value* d_case_local_vals, unsigned char* d_case_local_set, DResult* d_out) {
  if (d_consts) cudaFree(d_consts);
  if (d_code) cudaFree(d_code);
  if (d_metas) cudaFree(d_metas);
  if (d_case_local_vals) cudaFree(d_case_local_vals);
  if (d_case_local_set) cudaFree(d_case_local_set);
  if (d_out) cudaFree(d_out);
}

void cuda_cleanup_fitness(Value* d_consts, DInstr* d_code, DProgramMeta* d_metas,
                          Value* d_case_local_vals, unsigned char* d_case_local_set,
                          Value* d_expected, int* d_fitness) {
  if (d_consts) cudaFree(d_consts);
  if (d_code) cudaFree(d_code);
  if (d_metas) cudaFree(d_metas);
  if (d_case_local_vals) cudaFree(d_case_local_vals);
  if (d_case_local_set) cudaFree(d_case_local_set);
  if (d_expected) cudaFree(d_expected);
  if (d_fitness) cudaFree(d_fitness);
}

}  // namespace

std::vector<std::vector<VMResult>> run_bytecode_gpu_multi_batch(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    int fuel,
    int blocksize) {
  if (programs.empty() || shared_cases.empty()) {
    return multi_single_error(ErrCode::Value, "programs must not be empty");
  }
  if (blocksize <= 0) {
    return multi_single_error(ErrCode::Value, "invalid gpu blocksize");
  }

  cudaDeviceProp props;
  std::string select_msg;
  if (!select_least_used_device(props, select_msg)) {
    return multi_single_error(ErrCode::Value, select_msg);
  }

  if (blocksize > props.maxThreadsPerBlock) {
    return multi_single_error(ErrCode::Value, "gpu blocksize exceeds maxThreadsPerBlock");
  }

  std::vector<std::vector<VMResult>> result_by_program;
  result_by_program.resize(programs.size());

  std::vector<DProgramMeta> metas(programs.size());
  std::vector<DInstr> all_code;
  std::vector<Value> all_consts;
  std::size_t total_cases = 0;
  std::size_t max_code_len = 0;
  const int shared_case_count = static_cast<int>(shared_cases.size());

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const BytecodeProgram& prog = programs[p];

    DProgramMeta meta;
    meta.code_offset = static_cast<int>(all_code.size());
    meta.const_offset = static_cast<int>(all_consts.size());
    meta.n_locals = prog.n_locals;
    meta.case_offset = static_cast<int>(total_cases);
    meta.case_count = shared_case_count;
    meta.case_local_offset = 0;
    meta.is_valid = 1;
    meta.err_code = DERR_VALUE;

    if (prog.n_locals > MAX_LOCALS) {
      meta.is_valid = 0;
      meta.err_code = DERR_VALUE;
    }

    all_consts.insert(all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      const int op = gpu_detail::host_opcode(ins.op);
      if (op < 0) {
        meta.is_valid = 0;
        meta.err_code = DERR_TYPE;
        continue;
      }
      DInstr di;
      di.op = static_cast<std::uint8_t>(op);
      di.flags = static_cast<std::uint8_t>((ins.has_a ? DINSTR_HAS_A : 0) | (ins.has_b ? DINSTR_HAS_B : 0));
      di.a = static_cast<std::int32_t>(ins.a);
      di.b = static_cast<std::int32_t>(ins.b);
      all_code.push_back(di);
    }
    meta.code_len = static_cast<int>(all_code.size()) - meta.code_offset;
    if (static_cast<std::size_t>(meta.code_len) > max_code_len) {
      max_code_len = static_cast<std::size_t>(meta.code_len);
    }

    if (prog.n_locals < 0 || prog.n_locals > MAX_LOCALS) {
      meta.is_valid = 0;
      meta.err_code = DERR_VALUE;
    }
    total_cases += static_cast<std::size_t>(shared_case_count);
    metas[p] = meta;
  }

  if (total_cases == 0) {
    return multi_single_error(ErrCode::Value, "cases must not be empty");
  }

  const std::size_t shared_bytes = max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    return multi_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }

  std::vector<Value> packed_case_local_vals(shared_cases.size() * MAX_LOCALS, Value::none());
  std::vector<unsigned char> packed_case_local_set(shared_cases.size() * MAX_LOCALS, 0);
  for (std::size_t case_idx = 0; case_idx < shared_cases.size(); ++case_idx) {
    const std::size_t base = case_idx * MAX_LOCALS;
    for (const LocalBinding& binding : shared_cases[case_idx]) {
      if (binding.idx >= 0 && binding.idx < MAX_LOCALS) {
        packed_case_local_vals[base + static_cast<std::size_t>(binding.idx)] = binding.value;
        packed_case_local_set[base + static_cast<std::size_t>(binding.idx)] = 1;
      }
    }
  }

  Value* d_consts = nullptr;
  DInstr* d_code = nullptr;
  DProgramMeta* d_metas = nullptr;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  DResult* d_out = nullptr;

  if (!cuda_alloc_and_copy_in(all_consts, &d_consts) || !cuda_alloc_and_copy_in(all_code, &d_code) ||
      !cuda_alloc_and_copy_in(metas, &d_metas) ||
      !cuda_alloc_and_copy_in(packed_case_local_vals, &d_shared_case_local_vals) ||
      !cuda_alloc_and_copy_in(packed_case_local_set, &d_shared_case_local_set)) {
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, "cuda allocation failure");
  }

  if (cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(DResult) * total_cases) != cudaSuccess) {
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, "cuda allocation failure");
  }

  gpu_detail::vm_multi_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), blocksize, shared_bytes>>>(
      d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
      static_cast<int>(programs.size()), fuel, d_out);

  const cudaError_t launch_err = cudaGetLastError();
  const cudaError_t sync_err = cudaDeviceSynchronize();
  if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
    std::string msg = "cuda kernel execution failure";
    if (launch_err != cudaSuccess) {
      msg += " launch=";
      msg += cudaGetErrorString(launch_err);
    }
    if (sync_err != cudaSuccess) {
      msg += " sync=";
      msg += cudaGetErrorString(sync_err);
    }
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, msg);
  }

  std::vector<DResult> host_out(total_cases);
  if (cudaMemcpy(host_out.data(), d_out, sizeof(DResult) * total_cases, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, "cuda copy-back failure");
  }

  cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const DProgramMeta& meta = metas[p];
    std::vector<VMResult>& outp = result_by_program[p];
    outp.reserve(static_cast<std::size_t>(meta.case_count));
    for (int local_case = 0; local_case < meta.case_count; ++local_case) {
      const DResult& r = host_out[static_cast<std::size_t>(meta.case_offset + local_case)];
      if (r.is_error) {
        outp.push_back(VMResult{true, Value::none(),
                                Err{gpu_detail::from_device_err(r.err_code),
                                    gpu_detail::device_err_message(r.err_code)}});
      } else {
        outp.push_back(VMResult{false, r.value, Err{ErrCode::Value, ""}});
      }
    }
  }

  return result_by_program;
}

std::vector<int> run_bytecode_gpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    int blocksize) {
  if (programs.empty() || shared_cases.empty()) {
    return {};
  }
  if (shared_answer.size() != shared_cases.size()) {
    return {};
  }
  if (blocksize <= 0) {
    return {};
  }

  cudaDeviceProp props;
  std::string select_msg;
  if (!select_least_used_device(props, select_msg)) {
    return {};
  }
  if (blocksize > props.maxThreadsPerBlock) {
    return {};
  }

  const int shared_case_count = static_cast<int>(shared_cases.size());
  std::vector<DProgramMeta> metas(programs.size());
  std::vector<DInstr> all_code;
  std::vector<Value> all_consts;
  std::size_t total_cases = 0;
  std::size_t max_code_len = 0;

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const BytecodeProgram& prog = programs[p];

    DProgramMeta meta;
    meta.code_offset = static_cast<int>(all_code.size());
    meta.const_offset = static_cast<int>(all_consts.size());
    meta.n_locals = prog.n_locals;
    meta.case_offset = static_cast<int>(total_cases);
    meta.case_count = shared_case_count;
    meta.case_local_offset = 0;
    meta.is_valid = 1;
    meta.err_code = DERR_VALUE;

    if (prog.n_locals > MAX_LOCALS || prog.n_locals < 0) {
      meta.is_valid = 0;
      meta.err_code = DERR_VALUE;
    }

    all_consts.insert(all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      const int op = gpu_detail::host_opcode(ins.op);
      if (op < 0) {
        meta.is_valid = 0;
        meta.err_code = DERR_TYPE;
        continue;
      }
      DInstr di;
      di.op = static_cast<std::uint8_t>(op);
      di.flags = static_cast<std::uint8_t>((ins.has_a ? DINSTR_HAS_A : 0) | (ins.has_b ? DINSTR_HAS_B : 0));
      di.a = static_cast<std::int32_t>(ins.a);
      di.b = static_cast<std::int32_t>(ins.b);
      all_code.push_back(di);
    }
    meta.code_len = static_cast<int>(all_code.size()) - meta.code_offset;
    if (static_cast<std::size_t>(meta.code_len) > max_code_len) {
      max_code_len = static_cast<std::size_t>(meta.code_len);
    }

    total_cases += static_cast<std::size_t>(shared_case_count);
    metas[p] = meta;
  }

  if (total_cases == 0) {
    return {};
  }

  const std::size_t shared_bytes = max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    return {};
  }

  std::vector<Value> packed_case_local_vals(shared_cases.size() * MAX_LOCALS, Value::none());
  std::vector<unsigned char> packed_case_local_set(shared_cases.size() * MAX_LOCALS, 0);
  for (std::size_t case_idx = 0; case_idx < shared_cases.size(); ++case_idx) {
    const std::size_t base = case_idx * MAX_LOCALS;
    for (const LocalBinding& binding : shared_cases[case_idx]) {
      if (binding.idx >= 0 && binding.idx < MAX_LOCALS) {
        packed_case_local_vals[base + static_cast<std::size_t>(binding.idx)] = binding.value;
        packed_case_local_set[base + static_cast<std::size_t>(binding.idx)] = 1;
      }
    }
  }

  Value* d_consts = nullptr;
  DInstr* d_code = nullptr;
  DProgramMeta* d_metas = nullptr;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  Value* d_expected = nullptr;
  int* d_fitness = nullptr;

  if (!cuda_alloc_and_copy_in(all_consts, &d_consts) || !cuda_alloc_and_copy_in(all_code, &d_code) ||
      !cuda_alloc_and_copy_in(metas, &d_metas) ||
      !cuda_alloc_and_copy_in(packed_case_local_vals, &d_shared_case_local_vals) ||
      !cuda_alloc_and_copy_in(packed_case_local_set, &d_shared_case_local_set) ||
      !cuda_alloc_and_copy_in(shared_answer, &d_expected)) {
    cuda_cleanup_fitness(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
                         d_expected, d_fitness);
    return {};
  }

  if (cudaMalloc(reinterpret_cast<void**>(&d_fitness), sizeof(int) * programs.size()) != cudaSuccess) {
    cuda_cleanup_fitness(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
                         d_expected, d_fitness);
    return {};
  }

  if (cudaMemset(d_fitness, 0, sizeof(int) * programs.size()) != cudaSuccess) {
    cuda_cleanup_fitness(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
                         d_expected, d_fitness);
    return {};
  }

  gpu_detail::vm_multi_fitness_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), blocksize,
                                                     shared_bytes>>>(
      d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_expected,
      static_cast<int>(programs.size()), fuel, d_fitness);

  const cudaError_t launch_err = cudaGetLastError();
  const cudaError_t sync_err = cudaDeviceSynchronize();
  if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
    cuda_cleanup_fitness(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
                         d_expected, d_fitness);
    return {};
  }

  std::vector<int> host_fitness(programs.size(), 0);
  if (cudaMemcpy(host_fitness.data(), d_fitness, sizeof(int) * programs.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cuda_cleanup_fitness(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
                         d_expected, d_fitness);
    return {};
  }

  cuda_cleanup_fitness(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
                       d_expected, d_fitness);
  return host_fitness;
}

}  // namespace g3pvm
