#include "g3pvm/vm_gpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vm_gpu/constants.hpp"
#include "vm_gpu/host_pack.hpp"
#include "vm_gpu/kernels.cuh"
#include "vm_gpu/opcode_map.hpp"
#include "vm_gpu/types.hpp"

namespace g3pvm {

namespace {

using gpu_detail::DInstr;
using gpu_detail::DProgramMeta;
using gpu_detail::DResult;

std::vector<std::vector<VMResult>> multi_single_error(ErrCode code, const std::string& message) {
  return std::vector<std::vector<VMResult>>{
      std::vector<VMResult>{VMResult{true, Value::none(), Err{code, message}}}};
}

GPUFitnessEvalResult fitness_single_error(ErrCode code, const std::string& message) {
  GPUFitnessEvalResult out;
  out.ok = false;
  out.err = Err{code, message};
  return out;
}

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
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

  const gpu_detail::PackResult packed = gpu_detail::pack_programs_and_shared_cases(programs, shared_cases);
  if (packed.total_cases == 0) {
    return multi_single_error(ErrCode::Value, "cases must not be empty");
  }

  const std::size_t shared_bytes = packed.max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    return multi_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }

  gpu_detail::DeviceArena dev;
  if (!gpu_detail::cuda_alloc_and_copy_in(packed.all_consts, &dev.d_consts) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.all_code, &dev.d_code) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.metas, &dev.d_metas) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.packed_case_local_vals, &dev.d_shared_case_local_vals) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.packed_case_local_set, &dev.d_shared_case_local_set)) {
    return multi_single_error(ErrCode::Value, "cuda allocation failure");
  }

  if (cudaMalloc(reinterpret_cast<void**>(&dev.d_out), sizeof(DResult) * packed.total_cases) != cudaSuccess) {
    return multi_single_error(ErrCode::Value, "cuda allocation failure");
  }

  gpu_detail::vm_multi_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), blocksize, shared_bytes>>>(
      dev.d_consts, dev.d_code, dev.d_metas, dev.d_shared_case_local_vals, dev.d_shared_case_local_set,
      static_cast<int>(programs.size()), fuel, dev.d_out);

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
    return multi_single_error(ErrCode::Value, msg);
  }

  std::vector<DResult> host_out(packed.total_cases);
  if (cudaMemcpy(host_out.data(), dev.d_out, sizeof(DResult) * packed.total_cases, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return multi_single_error(ErrCode::Value, "cuda copy-back failure");
  }

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const DProgramMeta& meta = packed.metas[p];
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

GPUFitnessEvalResult run_bytecode_gpu_multi_fitness_shared_cases_debug(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    int blocksize) {
  if (programs.empty() || shared_cases.empty()) {
    return fitness_single_error(ErrCode::Value, "programs/shared_cases must not be empty");
  }
  if (shared_answer.size() != shared_cases.size()) {
    return fitness_single_error(ErrCode::Value, "shared_answer size mismatch");
  }
  if (blocksize <= 0) {
    return fitness_single_error(ErrCode::Value, "invalid gpu blocksize");
  }

  cudaDeviceProp props;
  std::string select_msg;
  if (!select_least_used_device(props, select_msg)) {
    return fitness_single_error(ErrCode::Value, select_msg);
  }
  if (blocksize > props.maxThreadsPerBlock) {
    return fitness_single_error(ErrCode::Value, "gpu blocksize exceeds maxThreadsPerBlock");
  }

  const auto all_t0 = std::chrono::steady_clock::now();
  const auto pack_t0 = std::chrono::steady_clock::now();
  const gpu_detail::PackResult packed = gpu_detail::pack_programs_and_shared_cases(programs, shared_cases);
  const auto pack_t1 = std::chrono::steady_clock::now();
  if (packed.total_cases == 0) {
    return fitness_single_error(ErrCode::Value, "cases must not be empty");
  }

  const std::size_t shared_bytes = packed.max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    return fitness_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }

  gpu_detail::DeviceArena dev;
  const auto upload_t0 = std::chrono::steady_clock::now();
  if (!gpu_detail::cuda_alloc_and_copy_in(packed.all_consts, &dev.d_consts) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.all_code, &dev.d_code) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.metas, &dev.d_metas) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.packed_case_local_vals, &dev.d_shared_case_local_vals) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.packed_case_local_set, &dev.d_shared_case_local_set) ||
      !gpu_detail::cuda_alloc_and_copy_in(shared_answer, &dev.d_expected)) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }

  if (cudaMalloc(reinterpret_cast<void**>(&dev.d_fitness), sizeof(int) * programs.size()) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }

  if (cudaMemset(dev.d_fitness, 0, sizeof(int) * programs.size()) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda memset failure");
  }
  const auto upload_t1 = std::chrono::steady_clock::now();

  const auto kernel_t0 = std::chrono::steady_clock::now();
  gpu_detail::vm_multi_fitness_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), blocksize,
                                                     shared_bytes>>>(
      dev.d_consts, dev.d_code, dev.d_metas, dev.d_shared_case_local_vals, dev.d_shared_case_local_set,
      dev.d_expected, static_cast<int>(programs.size()), fuel, dev.d_fitness);

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
    return fitness_single_error(ErrCode::Value, msg);
  }
  const auto kernel_t1 = std::chrono::steady_clock::now();

  std::vector<int> host_fitness(programs.size(), 0);
  const auto copy_t0 = std::chrono::steady_clock::now();
  if (cudaMemcpy(host_fitness.data(), dev.d_fitness, sizeof(int) * programs.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda copy-back failure");
  }
  const auto copy_t1 = std::chrono::steady_clock::now();

  GPUFitnessEvalResult out;
  out.ok = true;
  out.fitness = std::move(host_fitness);
  out.pack_programs_ms = ms_between(pack_t0, pack_t1);
  out.upload_programs_ms = ms_between(upload_t0, upload_t1);
  out.kernel_exec_ms = ms_between(kernel_t0, kernel_t1);
  out.copyback_ms = ms_between(copy_t0, copy_t1);
  out.total_ms = ms_between(all_t0, copy_t1);
  out.err = Err{ErrCode::Value, ""};
  return out;
}

std::vector<int> run_bytecode_gpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    int blocksize) {
  const GPUFitnessEvalResult out = run_bytecode_gpu_multi_fitness_shared_cases_debug(
      programs, shared_cases, shared_answer, fuel, blocksize);
  if (!out.ok) {
    return {};
  }
  return out.fitness;
}

struct GPUFitnessSession::Impl {
  bool ready = false;
  int device_id = -1;
  int fuel = 10000;
  int blocksize = 256;
  int shared_case_count = 0;
  std::size_t shared_bytes = 0;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  Value* d_expected = nullptr;
  cudaDeviceProp props{};
  Err last_err{ErrCode::Value, ""};

  ~Impl() {
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
  }
};

GPUFitnessSession::GPUFitnessSession() : impl_(std::make_unique<Impl>()) {}
GPUFitnessSession::~GPUFitnessSession() = default;
GPUFitnessSession::GPUFitnessSession(GPUFitnessSession&&) noexcept = default;
GPUFitnessSession& GPUFitnessSession::operator=(GPUFitnessSession&&) noexcept = default;

bool GPUFitnessSession::is_ready() const { return impl_ && impl_->ready; }

GPUFitnessEvalResult GPUFitnessSession::init(const std::vector<InputCase>& shared_cases,
                                             const std::vector<Value>& shared_answer,
                                             int fuel,
                                             int blocksize) {
  if (shared_cases.empty()) {
    return fitness_single_error(ErrCode::Value, "shared_cases must not be empty");
  }
  if (shared_answer.size() != shared_cases.size()) {
    return fitness_single_error(ErrCode::Value, "shared_answer size mismatch");
  }
  if (blocksize <= 0) {
    return fitness_single_error(ErrCode::Value, "invalid gpu blocksize");
  }

  std::string select_msg;
  if (!select_least_used_device(impl_->props, select_msg)) {
    return fitness_single_error(ErrCode::Value, select_msg);
  }
  if (blocksize > impl_->props.maxThreadsPerBlock) {
    return fitness_single_error(ErrCode::Value, "gpu blocksize exceeds maxThreadsPerBlock");
  }

  int device_id = -1;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda device query failure");
  }

  std::vector<Value> packed_case_local_vals;
  std::vector<unsigned char> packed_case_local_set;
  gpu_detail::pack_shared_cases_only(shared_cases, &packed_case_local_vals, &packed_case_local_set);

  if (!gpu_detail::cuda_alloc_and_copy_in(packed_case_local_vals, &impl_->d_shared_case_local_vals) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed_case_local_set, &impl_->d_shared_case_local_set) ||
      !gpu_detail::cuda_alloc_and_copy_in(shared_answer, &impl_->d_expected)) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }

  impl_->ready = true;
  impl_->device_id = device_id;
  impl_->fuel = fuel;
  impl_->blocksize = blocksize;
  impl_->shared_case_count = static_cast<int>(shared_cases.size());
  impl_->shared_bytes = 0;
  impl_->last_err = Err{ErrCode::Value, ""};
  return GPUFitnessEvalResult{true, {}, 0.0, 0.0, 0.0, 0.0, 0.0, Err{ErrCode::Value, ""}};
}

GPUFitnessEvalResult GPUFitnessSession::eval_programs(const std::vector<BytecodeProgram>& programs) const {
  if (!impl_ || !impl_->ready) {
    return fitness_single_error(ErrCode::Value, "gpu fitness session is not initialized");
  }
  if (programs.empty()) {
    return fitness_single_error(ErrCode::Value, "programs must not be empty");
  }
  if (cudaSetDevice(impl_->device_id) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda set device failure");
  }

  const auto all_t0 = std::chrono::steady_clock::now();
  const auto pack_t0 = std::chrono::steady_clock::now();
  const gpu_detail::PackResult packed =
      gpu_detail::pack_programs_with_shared_case_count(programs, impl_->shared_case_count);
  const auto pack_t1 = std::chrono::steady_clock::now();
  if (packed.total_cases == 0) {
    return fitness_single_error(ErrCode::Value, "cases must not be empty");
  }

  const std::size_t shared_bytes = packed.max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(impl_->props.sharedMemPerBlock)) {
    return fitness_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }

  gpu_detail::DeviceArena dev;
  const auto upload_t0 = std::chrono::steady_clock::now();
  if (!gpu_detail::cuda_alloc_and_copy_in(packed.all_consts, &dev.d_consts) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.all_code, &dev.d_code) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.metas, &dev.d_metas)) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }
  if (cudaMalloc(reinterpret_cast<void**>(&dev.d_fitness), sizeof(int) * programs.size()) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }
  if (cudaMemset(dev.d_fitness, 0, sizeof(int) * programs.size()) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda memset failure");
  }
  const auto upload_t1 = std::chrono::steady_clock::now();

  const auto kernel_t0 = std::chrono::steady_clock::now();
  gpu_detail::vm_multi_fitness_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), impl_->blocksize,
                                                     shared_bytes>>>(
      dev.d_consts, dev.d_code, dev.d_metas, impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set,
      impl_->d_expected, static_cast<int>(programs.size()), impl_->fuel, dev.d_fitness);

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
    return fitness_single_error(ErrCode::Value, msg);
  }
  const auto kernel_t1 = std::chrono::steady_clock::now();

  std::vector<int> host_fitness(programs.size(), 0);
  const auto copy_t0 = std::chrono::steady_clock::now();
  if (cudaMemcpy(host_fitness.data(), dev.d_fitness, sizeof(int) * programs.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda copy-back failure");
  }
  const auto copy_t1 = std::chrono::steady_clock::now();

  GPUFitnessEvalResult out;
  out.ok = true;
  out.fitness = std::move(host_fitness);
  out.pack_programs_ms = ms_between(pack_t0, pack_t1);
  out.upload_programs_ms = ms_between(upload_t0, upload_t1);
  out.kernel_exec_ms = ms_between(kernel_t0, kernel_t1);
  out.copyback_ms = ms_between(copy_t0, copy_t1);
  out.total_ms = ms_between(all_t0, copy_t1);
  out.err = Err{ErrCode::Value, ""};
  return out;
}

}  // namespace g3pvm
