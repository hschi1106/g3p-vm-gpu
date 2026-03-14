#include "internal.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "device/selection_kernels.cuh"
#include "device/variation_kernels.cuh"

namespace g3pvm::evo::repro {

namespace {

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

bool ensure_cuda(cudaError_t code, const char* what, std::string* message_out) {
  if (code == cudaSuccess) {
    return true;
  }
  if (message_out != nullptr) {
    std::ostringstream oss;
    oss << what << ": " << cudaGetErrorString(code);
    *message_out = oss.str();
  }
  return false;
}

}  // namespace

bool upload_gpu_repro_inputs(const PackedHostData& packed,
                             GpuReproArena* arena,
                             ReproductionStats* stats,
                             std::string* message_out) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!ensure_cuda(cudaSetDevice(arena->device_id), "cudaSetDevice", message_out)) {
    return false;
  }
  if (!ensure_cuda(cudaMemcpy(arena->d_program_nodes, packed.program_nodes.data(),
                              sizeof(PlainNode) * packed.program_nodes.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy program_nodes", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_metas, packed.metas.data(),
                              sizeof(PackedProgramMeta) * packed.metas.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy metas", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_candidates, packed.candidates.data(),
                              sizeof(CandidateRange) * packed.candidates.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy candidates", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_program_name_ids, packed.program_name_ids.data(),
                              sizeof(std::uint64_t) * packed.program_name_ids.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy program_name_ids", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_program_consts, packed.program_consts.data(),
                              sizeof(Value) * packed.program_consts.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy program_consts", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_donor_nodes, packed.donor_nodes.data(),
                              sizeof(PlainNode) * packed.donor_nodes.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy donor_nodes", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_donor_lens, packed.donor_lens.data(),
                              sizeof(int) * packed.donor_lens.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy donor_lens", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_donor_name_ids, packed.donor_name_ids.data(),
                              sizeof(std::uint64_t) * packed.donor_name_ids.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy donor_name_ids", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_donor_name_counts, packed.donor_name_counts.data(),
                              sizeof(int) * packed.donor_name_counts.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy donor_name_counts", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_donor_consts, packed.donor_consts.data(),
                              sizeof(Value) * packed.donor_consts.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy donor_consts", message_out) ||
      !ensure_cuda(cudaMemcpy(arena->d_donor_const_counts, packed.donor_const_counts.data(),
                              sizeof(int) * packed.donor_const_counts.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy donor_const_counts", message_out) ||
      !ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize upload", message_out)) {
    return false;
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->upload_ms += ms_between(t0, t1);
  }
  return true;
}

bool launch_gpu_repro_kernels(GpuReproArena* arena,
                              const GpuReproConfig& config,
                              const std::vector<double>& fitness,
                              ReproductionStats* stats,
                              std::string* message_out) {
  const auto upload_t0 = std::chrono::steady_clock::now();
  if (!ensure_cuda(cudaSetDevice(arena->device_id), "cudaSetDevice", message_out)) {
    return false;
  }
  if (!ensure_cuda(cudaMemcpy(arena->d_fitness, fitness.data(),
                              sizeof(double) * fitness.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy fitness", message_out)) {
    return false;
  }
  const auto upload_t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->upload_ms += ms_between(upload_t0, upload_t1);
  }

  const int select_threads = 256;
  const int select_blocks = (config.pair_count + select_threads - 1) / select_threads;
  const auto select_t0 = std::chrono::steady_clock::now();
  tournament_select_kernel<<<select_blocks, select_threads>>>(
      arena->d_fitness, config.population_size, config.pair_count, config.candidates_per_program,
      config.donor_pool_size, config.tournament_k, config.mutation_ratio, config.seed, arena->d_parent_a,
      arena->d_parent_b, arena->d_cand_a, arena->d_cand_b, arena->d_donor_idx, arena->d_is_mutation);
  if (!ensure_cuda(cudaGetLastError(), "tournament_select_kernel", message_out) ||
      !ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize selection", message_out)) {
    return false;
  }
  const auto select_t1 = std::chrono::steady_clock::now();

  const auto variation_t0 = std::chrono::steady_clock::now();
  variation_kernel<<<config.pair_count, 128>>>(
      arena->d_program_nodes, arena->d_metas, arena->d_candidates, arena->d_program_name_ids,
      arena->d_program_consts, arena->d_donor_nodes, arena->d_donor_lens, arena->d_donor_name_ids,
      arena->d_donor_name_counts, arena->d_donor_consts, arena->d_donor_const_counts,
      config.max_nodes, config.candidates_per_program, config.max_donor_nodes, config.max_names, config.max_consts,
      config.max_expr_depth, config.max_for_k,
      config.pair_count, arena->d_parent_a, arena->d_parent_b, arena->d_cand_a, arena->d_cand_b,
      arena->d_donor_idx, arena->d_is_mutation, arena->d_child_nodes, arena->d_child_used_len,
      arena->d_child_name_ids, arena->d_child_name_counts, arena->d_child_consts,
      arena->d_child_const_counts, arena->d_child_meta);
  if (!ensure_cuda(cudaGetLastError(), "variation_kernel", message_out) ||
      !ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize variation", message_out)) {
    return false;
  }
  const auto variation_t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->selection_kernel_ms += ms_between(select_t0, select_t1);
    stats->variation_kernel_ms += ms_between(variation_t0, variation_t1);
    stats->kernel_ms += stats->selection_kernel_ms + stats->variation_kernel_ms;
  }
  return true;
}

}  // namespace g3pvm::evo::repro
