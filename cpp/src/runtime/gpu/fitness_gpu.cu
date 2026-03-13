#include "g3pvm/runtime/gpu/fitness_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
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

using HostStringPayloadLookup = std::unordered_map<std::int64_t, std::string>;
using HostListPayloadLookup = std::unordered_map<std::int64_t, std::vector<Value>>;

constexpr unsigned kPayloadMaskString = 1U << 0;
constexpr unsigned kPayloadMaskList = 1U << 1;

unsigned value_payload_mask(const Value& v) {
  if (v.tag == ValueTag::String) return kPayloadMaskString;
  if (v.tag == ValueTag::List) return kPayloadMaskList;
  return 0U;
}

void sort_and_unique_tokens(std::vector<std::int64_t>* tokens) {
  std::sort(tokens->begin(), tokens->end());
  tokens->erase(std::unique(tokens->begin(), tokens->end()), tokens->end());
}

void append_payload_tokens_from_values(const std::vector<Value>& values,
                                       std::vector<std::int64_t>* string_tokens,
                                       std::vector<std::int64_t>* list_tokens) {
  for (const Value& v : values) {
    if (v.tag == ValueTag::String) {
      string_tokens->push_back(v.i);
    } else if (v.tag == ValueTag::List) {
      list_tokens->push_back(v.i);
    }
  }
}

void append_payload_tokens_from_programs(const std::vector<BytecodeProgram>& programs,
                                         const std::vector<int>& indices,
                                         std::vector<std::int64_t>* string_tokens,
                                         std::vector<std::int64_t>* list_tokens) {
  for (int idx : indices) {
    if (idx < 0 || idx >= static_cast<int>(programs.size())) {
      continue;
    }
    append_payload_tokens_from_values(programs[static_cast<std::size_t>(idx)].consts, string_tokens, list_tokens);
  }
}

HostPayloadPack build_payload_pack(const HostStringPayloadLookup& string_lookup,
                                   const HostListPayloadLookup& list_lookup,
                                   std::vector<std::int64_t> string_tokens,
                                   std::vector<std::int64_t> list_tokens) {
  HostPayloadPack out;
  sort_and_unique_tokens(&string_tokens);
  sort_and_unique_tokens(&list_tokens);

  out.string_entries.reserve(string_tokens.size());
  for (const std::int64_t packed : string_tokens) {
    const auto it = string_lookup.find(packed);
    if (it == string_lookup.end()) {
      continue;
    }
    DStringPayloadEntry e;
    e.packed = packed;
    e.offset = static_cast<int>(out.string_bytes.size());
    e.len = static_cast<int>(it->second.size());
    out.string_entries.push_back(e);
    out.string_bytes.insert(out.string_bytes.end(), it->second.begin(), it->second.end());
  }

  out.list_entries.reserve(list_tokens.size());
  for (const std::int64_t packed : list_tokens) {
    const auto it = list_lookup.find(packed);
    if (it == list_lookup.end()) {
      continue;
    }
    DListPayloadEntry e;
    e.packed = packed;
    e.offset = static_cast<int>(out.list_values.size());
    e.len = static_cast<int>(it->second.size());
    out.list_entries.push_back(e);
    out.list_values.insert(out.list_values.end(), it->second.begin(), it->second.end());
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

bool query_device(int dev, cudaDeviceProp& props_out, std::string& message_out) {
  const cudaError_t set_err = cudaSetDevice(dev);
  if (set_err != cudaSuccess) {
    message_out = "cuda device select failure err=";
    message_out += cudaGetErrorString(set_err);
    return false;
  }

  const cudaError_t prop_err = cudaGetDeviceProperties(&props_out, dev);
  if (prop_err != cudaSuccess) {
    message_out = "cuda device query failure err=";
    message_out += cudaGetErrorString(prop_err);
    return false;
  }
  return true;
}

bool parse_env_device_override(int* out_dev) {
  const char* raw = std::getenv("G3PVM_CUDA_DEVICE");
  if (raw == nullptr || *raw == '\0') {
    return false;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *out_dev = static_cast<int>(parsed);
  return true;
}

bool select_gpu_device(cudaDeviceProp& props_out, std::string& message_out) {
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

  int override_dev = -1;
  if (parse_env_device_override(&override_dev)) {
    if (override_dev >= device_count) {
      message_out = "cuda device override out of range";
      return false;
    }
    return query_device(override_dev, props_out, message_out);
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

  return query_device(best_dev, props_out, message_out);
}

unsigned shared_cases_payload_mask(const std::vector<CaseBindings>& shared_cases) {
  unsigned mask = 0U;
  for (const CaseBindings& case_bindings : shared_cases) {
    for (const g3pvm::InputBinding& binding : case_bindings) {
      mask |= value_payload_mask(binding.value);
    }
  }
  return mask;
}

std::size_t kernel_shared_bytes(std::size_t max_code_len, int blocksize) {
  return max_code_len * sizeof(DInstr) + (alignof(double) - 1u) +
         static_cast<std::size_t>(blocksize) * sizeof(double);
}

}  // namespace

struct FitnessSessionGpu::Impl {
  bool ready = false;
  int device_id = -1;
  int fuel = 10000;
  int blocksize = 1024;
  double penalty = 1.0;
  int shared_case_count = 0;
  unsigned shared_input_payload_mask = 0U;
  std::size_t shared_bytes = 0;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  Value* d_expected = nullptr;
  HostStringPayloadLookup host_string_payloads;
  HostListPayloadLookup host_list_payloads;
  std::vector<std::int64_t> shared_string_tokens;
  std::vector<std::int64_t> shared_list_tokens;
  cudaDeviceProp props{};
  Err last_err{ErrCode::Value, ""};

  ~Impl() {
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
  }
};

FitnessSessionGpu::FitnessSessionGpu() : impl_(std::make_unique<Impl>()) {}
FitnessSessionGpu::~FitnessSessionGpu() = default;
FitnessSessionGpu::FitnessSessionGpu(FitnessSessionGpu&&) noexcept = default;
FitnessSessionGpu& FitnessSessionGpu::operator=(FitnessSessionGpu&&) noexcept = default;

bool FitnessSessionGpu::is_ready() const { return impl_ && impl_->ready; }

FitnessEvalResult FitnessSessionGpu::init(const std::vector<CaseBindings>& shared_cases,
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
  if (!select_gpu_device(impl_->props, select_msg)) {
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
  const auto string_snapshots = payload::snapshot_strings();
  const auto list_snapshots = payload::snapshot_lists();
  impl_->host_string_payloads.clear();
  impl_->host_list_payloads.clear();
  impl_->host_string_payloads.reserve(string_snapshots.size());
  impl_->host_list_payloads.reserve(list_snapshots.size());
  for (const auto& s : string_snapshots) {
    impl_->host_string_payloads[s.key.i] = s.data;
  }
  for (const auto& l : list_snapshots) {
    impl_->host_list_payloads[l.key.i] = l.elems;
  }
  impl_->shared_string_tokens.clear();
  impl_->shared_list_tokens.clear();
  append_payload_tokens_from_values(packed_case_local_vals, &impl_->shared_string_tokens, &impl_->shared_list_tokens);
  sort_and_unique_tokens(&impl_->shared_string_tokens);
  sort_and_unique_tokens(&impl_->shared_list_tokens);

  if (!gpu_detail::cuda_alloc_and_copy_in(packed_case_local_vals, &impl_->d_shared_case_local_vals) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed_case_local_set, &impl_->d_shared_case_local_set) ||
      !gpu_detail::cuda_alloc_and_copy_in(shared_answer, &impl_->d_expected)) {
    return fitness_single_error(ErrCode::Value, "cuda allocation failure");
  }

  impl_->ready = true;
  impl_->device_id = device_id;
  impl_->fuel = fuel;
  impl_->blocksize = blocksize;
  impl_->penalty = penalty;
  impl_->shared_case_count = static_cast<int>(shared_cases.size());
  impl_->shared_input_payload_mask = shared_cases_payload_mask(shared_cases);
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
      gpu_detail::pack_programs_with_shared_case_count(
          programs, impl_->shared_case_count, impl_->shared_input_payload_mask);
  if (packed.total_cases == 0) {
    return fitness_single_error(ErrCode::Value, "cases must not be empty");
  }

  std::vector<std::int64_t> needed_string_tokens = impl_->shared_string_tokens;
  std::vector<std::int64_t> needed_list_tokens = impl_->shared_list_tokens;
  append_payload_tokens_from_programs(programs, packed.string_payload_program_indices, &needed_string_tokens, &needed_list_tokens);
  append_payload_tokens_from_programs(programs, packed.list_payload_program_indices, &needed_string_tokens, &needed_list_tokens);
  append_payload_tokens_from_programs(programs, packed.mixed_payload_program_indices, &needed_string_tokens, &needed_list_tokens);
  const HostPayloadPack payload_pack =
      build_payload_pack(impl_->host_string_payloads, impl_->host_list_payloads, std::move(needed_string_tokens),
                         std::move(needed_list_tokens));
  const auto pack_t1 = std::chrono::steady_clock::now();

  const std::size_t shared_bytes =
      kernel_shared_bytes(packed.max_code_len, impl_->blocksize);
  const std::size_t shared_bytes_no_payload =
      kernel_shared_bytes(packed.max_code_len_no_payload, impl_->blocksize);
  const std::size_t shared_bytes_string_payload =
      kernel_shared_bytes(packed.max_code_len_string_payload, impl_->blocksize);
  const std::size_t shared_bytes_list_payload =
      kernel_shared_bytes(packed.max_code_len_list_payload, impl_->blocksize);
  const std::size_t shared_bytes_mixed_payload =
      kernel_shared_bytes(packed.max_code_len_mixed_payload, impl_->blocksize);
  if (shared_bytes > static_cast<std::size_t>(impl_->props.sharedMemPerBlock) ||
      shared_bytes_no_payload > static_cast<std::size_t>(impl_->props.sharedMemPerBlock) ||
      shared_bytes_string_payload > static_cast<std::size_t>(impl_->props.sharedMemPerBlock) ||
      shared_bytes_list_payload > static_cast<std::size_t>(impl_->props.sharedMemPerBlock) ||
      shared_bytes_mixed_payload > static_cast<std::size_t>(impl_->props.sharedMemPerBlock)) {
    return fitness_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }

  gpu_detail::DeviceArena dev;
  const auto upload_t0 = std::chrono::steady_clock::now();
  if (!gpu_detail::cuda_alloc_and_copy_in(packed.all_consts, &dev.d_consts) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.all_code, &dev.d_code) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.metas, &dev.d_metas) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.no_payload_program_indices, &dev.d_no_payload_program_indices) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.string_payload_program_indices, &dev.d_string_payload_program_indices) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.list_payload_program_indices, &dev.d_list_payload_program_indices) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed.mixed_payload_program_indices, &dev.d_mixed_payload_program_indices) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.string_entries, &dev.d_string_payload_entries) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.string_bytes, &dev.d_string_payload_bytes) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.list_entries, &dev.d_list_payload_entries) ||
      !gpu_detail::cuda_alloc_and_copy_in(payload_pack.list_values, &dev.d_list_payload_values)) {
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
  cudaError_t launch_err = cudaSuccess;
  if (!packed.no_payload_program_indices.empty()) {
    gpu_detail::evaluate_fitness_subset<gpu_detail::DPayloadFlavor::None>
        <<<static_cast<unsigned int>(packed.no_payload_program_indices.size()), impl_->blocksize,
           shared_bytes_no_payload>>>(
            dev.d_no_payload_program_indices, static_cast<int>(packed.no_payload_program_indices.size()),
            dev.d_consts, dev.d_code, dev.d_metas,
            impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set, impl_->d_expected,
            dev.d_string_payload_entries, static_cast<int>(payload_pack.string_entries.size()), dev.d_string_payload_bytes,
            dev.d_list_payload_entries, static_cast<int>(payload_pack.list_entries.size()), dev.d_list_payload_values,
            impl_->fuel, impl_->penalty, dev.d_fitness);
    launch_err = cudaGetLastError();
  }
  if (launch_err == cudaSuccess && !packed.string_payload_program_indices.empty()) {
    gpu_detail::evaluate_fitness_subset<gpu_detail::DPayloadFlavor::StringOnly>
        <<<static_cast<unsigned int>(packed.string_payload_program_indices.size()), impl_->blocksize,
           shared_bytes_string_payload>>>(
            dev.d_string_payload_program_indices, static_cast<int>(packed.string_payload_program_indices.size()),
            dev.d_consts, dev.d_code, dev.d_metas,
            impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set, impl_->d_expected,
            dev.d_string_payload_entries, static_cast<int>(payload_pack.string_entries.size()), dev.d_string_payload_bytes,
            dev.d_list_payload_entries, static_cast<int>(payload_pack.list_entries.size()), dev.d_list_payload_values,
            impl_->fuel, impl_->penalty, dev.d_fitness);
    launch_err = cudaGetLastError();
  }
  if (launch_err == cudaSuccess && !packed.list_payload_program_indices.empty()) {
    gpu_detail::evaluate_fitness_subset<gpu_detail::DPayloadFlavor::ListOnly>
        <<<static_cast<unsigned int>(packed.list_payload_program_indices.size()), impl_->blocksize,
           shared_bytes_list_payload>>>(
            dev.d_list_payload_program_indices, static_cast<int>(packed.list_payload_program_indices.size()),
            dev.d_consts, dev.d_code, dev.d_metas,
            impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set, impl_->d_expected,
            dev.d_string_payload_entries, static_cast<int>(payload_pack.string_entries.size()), dev.d_string_payload_bytes,
            dev.d_list_payload_entries, static_cast<int>(payload_pack.list_entries.size()), dev.d_list_payload_values,
            impl_->fuel, impl_->penalty, dev.d_fitness);
    launch_err = cudaGetLastError();
  }
  if (launch_err == cudaSuccess && !packed.mixed_payload_program_indices.empty()) {
    gpu_detail::evaluate_fitness_subset<gpu_detail::DPayloadFlavor::Mixed>
        <<<static_cast<unsigned int>(packed.mixed_payload_program_indices.size()), impl_->blocksize,
           shared_bytes_mixed_payload>>>(
            dev.d_mixed_payload_program_indices, static_cast<int>(packed.mixed_payload_program_indices.size()),
            dev.d_consts, dev.d_code, dev.d_metas,
            impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set, impl_->d_expected,
            dev.d_string_payload_entries, static_cast<int>(payload_pack.string_entries.size()), dev.d_string_payload_bytes,
            dev.d_list_payload_entries, static_cast<int>(payload_pack.list_entries.size()), dev.d_list_payload_values,
            impl_->fuel, impl_->penalty, dev.d_fitness);
    launch_err = cudaGetLastError();
  }
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
