#include "g3pvm/runtime/gpu/fitness_gpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/runtime/payload/payload.hpp"
#include "g3pvm/runtime/gpu/constants_gpu.hpp"
#include "g3pvm/runtime/gpu/host_pack_gpu.hpp"
#include "device/kernels.cuh"
#include "opcode_map_gpu.hpp"
#include "g3pvm/runtime/gpu/device_types_gpu.hpp"

namespace g3pvm {

namespace {

using gpu_detail::DInstr;
using gpu_detail::DListPayloadEntry;
using gpu_detail::DProgramMeta;
using gpu_detail::DResult;
using gpu_detail::DStringPayloadEntry;

struct HostPayloadPack {
  std::vector<DStringPayloadEntry> string_entries;
  std::vector<char> string_bytes;
  std::vector<DListPayloadEntry> list_entries;
  std::vector<Value> list_values;
};

HostPayloadPack build_payload_pack() {
  HostPayloadPack out;
  const auto strings = payload::snapshot_strings();
  const auto lists = payload::snapshot_lists();

  out.string_entries.reserve(strings.size());
  for (const auto& s : strings) {
    DStringPayloadEntry e;
    e.packed = s.key.i;
    e.offset = static_cast<int>(out.string_bytes.size());
    e.len = static_cast<int>(s.data.size());
    out.string_entries.push_back(e);
    out.string_bytes.insert(out.string_bytes.end(), s.data.begin(), s.data.end());
  }

  out.list_entries.reserve(lists.size());
  for (const auto& l : lists) {
    DListPayloadEntry e;
    e.packed = l.key.i;
    e.offset = static_cast<int>(out.list_values.size());
    e.len = static_cast<int>(l.elems.size());
    out.list_entries.push_back(e);
    out.list_values.insert(out.list_values.end(), l.elems.begin(), l.elems.end());
  }
  return out;
}

FitnessEvalResult fitness_single_error(ErrCode code, const std::string& message) {
  FitnessEvalResult out;
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

struct FitnessSessionGpu::Impl {
  bool ready = false;
  int device_id = -1;
  int fuel = 10000;
  int blocksize = 256;
  double penalty = 1.0;
  int shared_case_count = 0;
  std::size_t shared_bytes = 0;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  Value* d_expected = nullptr;
  DStringPayloadEntry* d_string_payload_entries = nullptr;
  char* d_string_payload_bytes = nullptr;
  DListPayloadEntry* d_list_payload_entries = nullptr;
  Value* d_list_payload_values = nullptr;
  int string_payload_entry_count = 0;
  int list_payload_entry_count = 0;
  cudaDeviceProp props{};
  Err last_err{ErrCode::Value, ""};

  ~Impl() {
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
    if (d_string_payload_entries) cudaFree(d_string_payload_entries);
    if (d_string_payload_bytes) cudaFree(d_string_payload_bytes);
    if (d_list_payload_entries) cudaFree(d_list_payload_entries);
    if (d_list_payload_values) cudaFree(d_list_payload_values);
  }
};

FitnessSessionGpu::FitnessSessionGpu() : impl_(std::make_unique<Impl>()) {}
FitnessSessionGpu::~FitnessSessionGpu() = default;
FitnessSessionGpu::FitnessSessionGpu(FitnessSessionGpu&&) noexcept = default;
FitnessSessionGpu& FitnessSessionGpu::operator=(FitnessSessionGpu&&) noexcept = default;

bool FitnessSessionGpu::is_ready() const { return impl_ && impl_->ready; }

FitnessEvalResult FitnessSessionGpu::init(const std::vector<CaseInputs>& shared_cases,
                                             const std::vector<Value>& shared_answer,
                                             int fuel,
                                             int blocksize,
                                             double penalty) {
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
  const HostPayloadPack payload_pack = build_payload_pack();

  if (!gpu_detail::cuda_alloc_and_copy_in(packed_case_local_vals, &impl_->d_shared_case_local_vals) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed_case_local_set, &impl_->d_shared_case_local_set) ||
      !gpu_detail::cuda_alloc_and_copy_in(shared_answer, &impl_->d_expected) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.string_entries, &impl_->d_string_payload_entries) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.string_bytes, &impl_->d_string_payload_bytes) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.list_entries, &impl_->d_list_payload_entries) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.list_values, &impl_->d_list_payload_values)) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }

  impl_->ready = true;
  impl_->device_id = device_id;
  impl_->fuel = fuel;
  impl_->blocksize = blocksize;
  impl_->penalty = penalty;
  impl_->shared_case_count = static_cast<int>(shared_cases.size());
  impl_->string_payload_entry_count = static_cast<int>(payload_pack.string_entries.size());
  impl_->list_payload_entry_count = static_cast<int>(payload_pack.list_entries.size());
  impl_->shared_bytes = 0;
  impl_->last_err = Err{ErrCode::Value, ""};
  return FitnessEvalResult{true, {}, 0.0, 0.0, 0.0, 0.0, 0.0, Err{ErrCode::Value, ""}};
}

FitnessEvalResult FitnessSessionGpu::eval_programs(const std::vector<BytecodeProgram>& programs) const {
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

  const std::size_t shared_bytes =
      packed.max_code_len * sizeof(DInstr) + (alignof(double) - 1u) +
      static_cast<std::size_t>(impl_->blocksize) * sizeof(double);
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
  if (cudaMalloc(reinterpret_cast<void**>(&dev.d_fitness), sizeof(double) * programs.size()) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }
  if (cudaMemset(dev.d_fitness, 0, sizeof(double) * programs.size()) != cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda memset failure");
  }
  const auto upload_t1 = std::chrono::steady_clock::now();

  const auto kernel_t0 = std::chrono::steady_clock::now();
  gpu_detail::evaluate_fitness_device<<<static_cast<unsigned int>(programs.size()), impl_->blocksize,
                                        shared_bytes>>>(
      dev.d_consts, dev.d_code, dev.d_metas, impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set,
      impl_->d_expected,
      impl_->d_string_payload_entries, impl_->string_payload_entry_count, impl_->d_string_payload_bytes,
      impl_->d_list_payload_entries, impl_->list_payload_entry_count, impl_->d_list_payload_values,
      static_cast<int>(programs.size()), impl_->fuel, impl_->penalty, dev.d_fitness);

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

  std::vector<double> host_fitness(programs.size(), 0.0);
  const auto copy_t0 = std::chrono::steady_clock::now();
  if (cudaMemcpy(host_fitness.data(), dev.d_fitness, sizeof(double) * programs.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return fitness_single_error(ErrCode::Value, "cuda copy-back failure");
  }
  const auto copy_t1 = std::chrono::steady_clock::now();

  FitnessEvalResult out;
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
