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
struct HostListPayloadKey {
  ValueTag tag = ValueTag::None;
  std::int64_t packed = 0;

  bool operator==(const HostListPayloadKey& other) const {
    return tag == other.tag && packed == other.packed;
  }
};

struct HostListPayloadKeyHash {
  std::size_t operator()(const HostListPayloadKey& key) const {
    const std::uint64_t a = static_cast<std::uint64_t>(key.packed);
    return static_cast<std::size_t>((a * 11400714819323198485ULL) ^
                                    static_cast<std::uint64_t>(key.tag));
  }
};

using HostListPayloadLookup = std::unordered_map<HostListPayloadKey, std::vector<Value>, HostListPayloadKeyHash>;

constexpr unsigned kPayloadMaskString = 1U << 0;
constexpr unsigned kPayloadMaskList = 1U << 1;

unsigned value_payload_mask(const Value& v) {
  if (v.tag == ValueTag::String) return kPayloadMaskString;
  if (v.tag == ValueTag::NumList || v.tag == ValueTag::StringList) return kPayloadMaskList;
  return 0U;
}

void sort_and_unique_tokens(std::vector<std::int64_t>* tokens) {
  std::sort(tokens->begin(), tokens->end());
  tokens->erase(std::unique(tokens->begin(), tokens->end()), tokens->end());
}

void sort_and_unique_list_tokens(std::vector<Value>* tokens) {
  std::sort(tokens->begin(), tokens->end(), [](const Value& a, const Value& b) {
    if (a.tag != b.tag) return static_cast<int>(a.tag) < static_cast<int>(b.tag);
    return a.i < b.i;
  });
  tokens->erase(std::unique(tokens->begin(), tokens->end(), [](const Value& a, const Value& b) {
                  return a.tag == b.tag && a.i == b.i;
                }),
                tokens->end());
}

void append_payload_tokens_from_values(const std::vector<Value>& values,
                                       std::vector<std::int64_t>* string_tokens,
                                       std::vector<Value>* list_tokens) {
  for (const Value& v : values) {
    if (v.tag == ValueTag::String) {
      string_tokens->push_back(v.i);
    } else if (v.tag == ValueTag::NumList || v.tag == ValueTag::StringList) {
      list_tokens->push_back(v);
    }
  }
}

HostPayloadPack build_payload_pack(const HostStringPayloadLookup& string_lookup,
                                   const HostListPayloadLookup& list_lookup,
                                   std::vector<std::int64_t> string_tokens,
                                   std::vector<Value> list_tokens) {
  HostPayloadPack out;
  sort_and_unique_tokens(&string_tokens);
  sort_and_unique_list_tokens(&list_tokens);

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
  for (const Value& list_value : list_tokens) {
    const auto it = list_lookup.find(HostListPayloadKey{list_value.tag, list_value.i});
    if (it == list_lookup.end()) {
      continue;
    }
    DListPayloadEntry e;
    e.tag = list_value.tag;
    e.packed = list_value.i;
    e.offset = static_cast<int>(out.list_values.size());
    e.len = static_cast<int>(it->second.size());
    out.list_entries.push_back(e);
    out.list_values.insert(out.list_values.end(), it->second.begin(), it->second.end());
  }
  return out;
}

void populate_payload_cache(const std::vector<std::int64_t>& string_tokens,
                            const std::vector<Value>& list_tokens,
                            HostStringPayloadLookup* string_lookup,
                            HostListPayloadLookup* list_lookup) {
  for (const std::int64_t packed : string_tokens) {
    if (string_lookup->find(packed) != string_lookup->end()) {
      continue;
    }
    std::string payload_value;
    if (!payload::lookup_string_packed(packed, &payload_value)) {
      continue;
    }
    (*string_lookup)[packed] = std::move(payload_value);
  }
  for (const Value& list_value : list_tokens) {
    const HostListPayloadKey key{list_value.tag, list_value.i};
    if (list_lookup->find(key) != list_lookup->end()) {
      continue;
    }
    std::vector<Value> payload_value;
    if (!payload::lookup_list_packed(list_value.tag, list_value.i, &payload_value)) {
      continue;
    }
    (*list_lookup)[key] = std::move(payload_value);
  }
}

FitnessSessionInitResult fitness_init_single_error(ErrCode code, const std::string& message) {
  FitnessSessionInitResult out;
  out.ok = false;
  out.err = Err{code, message};
  return out;
}

FitnessEvalResult fitness_eval_single_error(ErrCode code, const std::string& message) {
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
  mutable HostStringPayloadLookup host_string_payloads;
  mutable HostListPayloadLookup host_list_payloads;
  std::vector<std::int64_t> shared_string_tokens;
  std::vector<Value> shared_list_tokens;
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

FitnessSessionInitResult FitnessSessionGpu::init(const std::vector<CaseBindings>& shared_cases,
                                                 const std::vector<Value>& shared_answer,
                                                 int fuel,
                                                 int blocksize,
                                                 double penalty) {
  if (shared_cases.empty()) {
    return fitness_init_single_error(ErrCode::Value, "shared_cases must not be empty");
  }
  if (shared_answer.size() != shared_cases.size()) {
    return fitness_init_single_error(ErrCode::Value, "shared_answer size mismatch");
  }
  if (blocksize <= 0) {
    return fitness_init_single_error(ErrCode::Value, "invalid gpu blocksize");
  }

  const auto all_t0 = std::chrono::steady_clock::now();
  const auto device_select_t0 = all_t0;
  std::string select_msg;
  if (!select_gpu_device(impl_->props, select_msg)) {
    return fitness_init_single_error(ErrCode::Value, select_msg);
  }
  if (blocksize > impl_->props.maxThreadsPerBlock) {
    return fitness_init_single_error(ErrCode::Value, "gpu blocksize exceeds maxThreadsPerBlock");
  }

  int device_id = -1;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    return fitness_init_single_error(ErrCode::Value, "cuda device query failure");
  }
  const auto device_select_t1 = std::chrono::steady_clock::now();

  std::vector<Value> packed_case_local_vals;
  std::vector<unsigned char> packed_case_local_set;
  const auto shared_case_pack_t0 = std::chrono::steady_clock::now();
  gpu_detail::pack_shared_cases_only(shared_cases, &packed_case_local_vals, &packed_case_local_set);
  const auto shared_case_pack_t1 = std::chrono::steady_clock::now();

  const auto payload_cache_t0 = std::chrono::steady_clock::now();
  impl_->host_string_payloads.clear();
  impl_->host_list_payloads.clear();
  impl_->shared_string_tokens.clear();
  impl_->shared_list_tokens.clear();
  append_payload_tokens_from_values(packed_case_local_vals, &impl_->shared_string_tokens, &impl_->shared_list_tokens);
  sort_and_unique_tokens(&impl_->shared_string_tokens);
  sort_and_unique_list_tokens(&impl_->shared_list_tokens);
  populate_payload_cache(impl_->shared_string_tokens, impl_->shared_list_tokens, &impl_->host_string_payloads,
                         &impl_->host_list_payloads);
  const auto payload_cache_t1 = std::chrono::steady_clock::now();

  const auto upload_t0 = std::chrono::steady_clock::now();
  if (!gpu_detail::cuda_alloc_and_copy_in(packed_case_local_vals, &impl_->d_shared_case_local_vals) ||
      !gpu_detail::cuda_alloc_and_copy_in(packed_case_local_set, &impl_->d_shared_case_local_set) ||
      !gpu_detail::cuda_alloc_and_copy_in(shared_answer, &impl_->d_expected)) {
    return fitness_init_single_error(ErrCode::Value, "cuda allocation failure");
  }
  const auto upload_t1 = std::chrono::steady_clock::now();

  impl_->ready = true;
  impl_->device_id = device_id;
  impl_->fuel = fuel;
  impl_->blocksize = blocksize;
  impl_->penalty = penalty;
  impl_->shared_case_count = static_cast<int>(shared_cases.size());
  impl_->shared_input_payload_mask = shared_cases_payload_mask(shared_cases);
  impl_->shared_bytes = 0;
  impl_->last_err = Err{ErrCode::Value, ""};
  FitnessSessionInitResult out;
  out.ok = true;
  out.timing.device_select_ms = ms_between(device_select_t0, device_select_t1);
  out.timing.shared_case_pack_ms = ms_between(shared_case_pack_t0, shared_case_pack_t1);
  out.timing.payload_cache_warm_ms = ms_between(payload_cache_t0, payload_cache_t1);
  out.timing.upload_ms = ms_between(upload_t0, upload_t1);
  out.timing.total_ms = ms_between(all_t0, upload_t1);
  out.err = Err{ErrCode::Value, ""};
  return out;
}

FitnessEvalResult FitnessSessionGpu::eval_programs(const std::vector<BytecodeProgram>& programs) const {
  if (!impl_ || !impl_->ready) {
    return fitness_eval_single_error(ErrCode::Value, "gpu fitness session is not initialized");
  }
  if (programs.empty()) {
    return fitness_eval_single_error(ErrCode::Value, "programs must not be empty");
  }
  if (cudaSetDevice(impl_->device_id) != cudaSuccess) {
    return fitness_eval_single_error(ErrCode::Value, "cuda set device failure");
  }

  const auto all_t0 = std::chrono::steady_clock::now();
  const auto pack_t0 = std::chrono::steady_clock::now();
  const gpu_detail::PackResult packed =
      gpu_detail::pack_programs_with_shared_case_count(
          programs, impl_->shared_case_count, impl_->shared_input_payload_mask);
  if (packed.total_cases == 0) {
    return fitness_eval_single_error(ErrCode::Value, "cases must not be empty");
  }

  std::vector<std::int64_t> needed_string_tokens = impl_->shared_string_tokens;
  std::vector<Value> needed_list_tokens = impl_->shared_list_tokens;
  for (const BytecodeProgram& program : programs) {
    append_payload_tokens_from_values(program.consts, &needed_string_tokens, &needed_list_tokens);
  }
  sort_and_unique_tokens(&needed_string_tokens);
  sort_and_unique_list_tokens(&needed_list_tokens);
  populate_payload_cache(needed_string_tokens, needed_list_tokens, &impl_->host_string_payloads, &impl_->host_list_payloads);
  const HostPayloadPack payload_pack =
      build_payload_pack(impl_->host_string_payloads, impl_->host_list_payloads, std::move(needed_string_tokens),
                         std::move(needed_list_tokens));
  const auto pack_t1 = std::chrono::steady_clock::now();

  const auto launch_prep_t0 = std::chrono::steady_clock::now();
  const std::size_t shared_bytes = kernel_shared_bytes(packed.max_code_len, impl_->blocksize);
  if (shared_bytes > static_cast<std::size_t>(impl_->props.sharedMemPerBlock)) {
    return fitness_eval_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }
  const auto launch_prep_t1 = std::chrono::steady_clock::now();

  FitnessEvalResult out;
  std::chrono::steady_clock::time_point teardown_t0;
  {
    gpu_detail::DeviceArena dev;
    const auto upload_t0 = std::chrono::steady_clock::now();
    if (!gpu_detail::cuda_alloc_and_copy_in(packed.all_consts, &dev.d_consts) ||
        !gpu_detail::cuda_alloc_and_copy_in(packed.all_code, &dev.d_code) ||
        !gpu_detail::cuda_alloc_and_copy_in(packed.metas, &dev.d_metas) ||
        !gpu_detail::cuda_alloc_and_copy_in(payload_pack.string_entries, &dev.d_string_payload_entries) ||
        !gpu_detail::cuda_alloc_and_copy_in(payload_pack.string_bytes, &dev.d_string_payload_bytes) ||
        !gpu_detail::cuda_alloc_and_copy_in(payload_pack.list_entries, &dev.d_list_payload_entries) ||
        !gpu_detail::cuda_alloc_and_copy_in(payload_pack.list_values, &dev.d_list_payload_values)) {
      return fitness_eval_single_error(ErrCode::Value, "cuda allocation failure");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&dev.d_fitness), sizeof(double) * programs.size()) != cudaSuccess) {
      return fitness_eval_single_error(ErrCode::Value, "cuda allocation failure");
    }
    if (cudaMemset(dev.d_fitness, 0, sizeof(double) * programs.size()) != cudaSuccess) {
      return fitness_eval_single_error(ErrCode::Value, "cuda memset failure");
    }
    const auto upload_t1 = std::chrono::steady_clock::now();

    const auto kernel_t0 = std::chrono::steady_clock::now();
    gpu_detail::evaluate_fitness_programs<gpu_detail::DPayloadFlavor::Mixed>
        <<<static_cast<unsigned int>(programs.size()), impl_->blocksize, shared_bytes>>>(
            static_cast<int>(programs.size()), dev.d_consts, dev.d_code, dev.d_metas,
            impl_->d_shared_case_local_vals, impl_->d_shared_case_local_set, impl_->d_expected,
            dev.d_string_payload_entries, static_cast<int>(payload_pack.string_entries.size()), dev.d_string_payload_bytes,
            dev.d_list_payload_entries, static_cast<int>(payload_pack.list_entries.size()), dev.d_list_payload_values,
            impl_->fuel, impl_->penalty, dev.d_fitness);
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
      return fitness_eval_single_error(ErrCode::Value, msg);
    }
    const auto kernel_t1 = std::chrono::steady_clock::now();

    const auto copy_t0 = std::chrono::steady_clock::now();
    std::vector<double> host_fitness(programs.size(), 0.0);
    if (cudaMemcpy(host_fitness.data(), dev.d_fitness, sizeof(double) * programs.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return fitness_eval_single_error(ErrCode::Value, "cuda copy-back failure");
    }
    const auto copy_t1 = std::chrono::steady_clock::now();

    out.ok = true;
    out.fitness = std::move(host_fitness);
    out.timing.pack_ms = ms_between(pack_t0, pack_t1);
    out.timing.launch_prep_ms = ms_between(launch_prep_t0, launch_prep_t1);
    out.timing.upload_ms = ms_between(upload_t0, upload_t1);
    out.timing.kernel_ms = ms_between(kernel_t0, kernel_t1);
    out.timing.copyback_ms = ms_between(copy_t0, copy_t1);
    out.err = Err{ErrCode::Value, ""};
    teardown_t0 = std::chrono::steady_clock::now();
  }
  const auto teardown_t1 = std::chrono::steady_clock::now();
  out.timing.teardown_ms = ms_between(teardown_t0, teardown_t1);
  out.timing.total_ms = ms_between(all_t0, teardown_t1);
  return out;
}

}  // namespace g3pvm
