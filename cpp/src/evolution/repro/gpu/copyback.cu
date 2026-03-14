#include "internal.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <sstream>
#include <string>

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

__global__ void compute_live_offsets_kernel(const int* child_used_len,
                                            const int* child_name_counts,
                                            const int* child_const_counts,
                                            int child_count,
                                            int* child_node_offsets,
                                            int* child_name_offsets,
                                            int* child_const_offsets) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  child_node_offsets[0] = 0;
  child_name_offsets[0] = 0;
  child_const_offsets[0] = 0;
  for (int i = 0; i < child_count; ++i) {
    child_node_offsets[i + 1] = child_node_offsets[i] + child_used_len[i];
    child_name_offsets[i + 1] = child_name_offsets[i] + child_name_counts[i];
    child_const_offsets[i + 1] = child_const_offsets[i] + child_const_counts[i];
  }
}

__global__ void pack_live_children_kernel(const PlainNode* child_nodes,
                                          const std::uint64_t* child_name_ids,
                                          const Value* child_consts,
                                          const int* child_used_len,
                                          const int* child_name_counts,
                                          const int* child_const_counts,
                                          int child_count,
                                          int max_nodes,
                                          int max_names,
                                          int max_consts,
                                          const int* child_node_offsets,
                                          const int* child_name_offsets,
                                          const int* child_const_offsets,
                                          PlainNode* live_child_nodes,
                                          std::uint64_t* live_child_name_ids,
                                          Value* live_child_consts) {
  const int child_idx = static_cast<int>(blockIdx.x);
  if (child_idx >= child_count) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int src_node_base = child_idx * max_nodes;
  const int src_name_base = child_idx * max_names;
  const int src_const_base = child_idx * max_consts;
  const int dst_node_base = child_node_offsets[child_idx];
  const int dst_name_base = child_name_offsets[child_idx];
  const int dst_const_base = child_const_offsets[child_idx];
  for (int i = tid; i < child_used_len[child_idx]; i += blockDim.x) {
    live_child_nodes[dst_node_base + i] = child_nodes[src_node_base + i];
  }
  for (int i = tid; i < child_name_counts[child_idx]; i += blockDim.x) {
    live_child_name_ids[dst_name_base + i] = child_name_ids[src_name_base + i];
  }
  for (int i = tid; i < child_const_counts[child_idx]; i += blockDim.x) {
    live_child_consts[dst_const_base + i] = child_consts[src_const_base + i];
  }
}

bool copyback_gpu_repro_children(const GpuReproArena& arena,
                                 const GpuReproConfig& config,
                                 GpuReproHostStaging* staging,
                                 GpuReproChildView* out,
                                 ReproductionStats* stats,
                                 std::string* message_out) {
  if (staging == nullptr || out == nullptr) {
    if (message_out != nullptr) {
      *message_out = "gpu reproduction copyback output is null";
    }
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (!ensure_cuda(cudaSetDevice(arena.device_id), "cudaSetDevice", message_out)) {
    return false;
  }

  const int child_count = config.pair_count * 2;
  compute_live_offsets_kernel<<<1, 1>>>(arena.d_child_used_len, arena.d_child_name_counts,
                                        arena.d_child_const_counts, child_count,
                                        arena.d_child_node_offsets, arena.d_child_name_offsets,
                                        arena.d_child_const_offsets);
  if (!ensure_cuda(cudaGetLastError(), "compute_live_offsets_kernel", message_out)) {
    return false;
  }
  pack_live_children_kernel<<<child_count, 128>>>(
      arena.d_child_nodes, arena.d_child_name_ids, arena.d_child_consts, arena.d_child_used_len,
      arena.d_child_name_counts, arena.d_child_const_counts, child_count, config.max_nodes,
      config.max_names, config.max_consts, arena.d_child_node_offsets, arena.d_child_name_offsets,
      arena.d_child_const_offsets, arena.d_live_child_nodes, arena.d_live_child_name_ids,
      arena.d_live_child_consts);
  if (!ensure_cuda(cudaGetLastError(), "pack_live_children_kernel", message_out)) {
    return false;
  }

  if (!ensure_cuda(cudaMemcpyAsync(staging->parent_a, arena.d_parent_a,
                                   sizeof(int) * static_cast<std::size_t>(config.pair_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync parent_a", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->parent_b, arena.d_parent_b,
                                   sizeof(int) * static_cast<std::size_t>(config.pair_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync parent_b", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->is_mutation, arena.d_is_mutation,
                                   sizeof(unsigned char) * static_cast<std::size_t>(config.pair_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync is_mutation", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_used_len, arena.d_child_used_len,
                                   sizeof(int) * static_cast<std::size_t>(child_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_used_len", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_name_counts, arena.d_child_name_counts,
                                   sizeof(int) * static_cast<std::size_t>(child_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_name_counts", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_const_counts, arena.d_child_const_counts,
                                   sizeof(int) * static_cast<std::size_t>(child_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_const_counts", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_meta, arena.d_child_meta,
                                   sizeof(PackedChildMeta) * static_cast<std::size_t>(child_count),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_meta", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_node_offsets, arena.d_child_node_offsets,
                                   sizeof(int) * static_cast<std::size_t>(child_count + 1),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_node_offsets", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_name_offsets, arena.d_child_name_offsets,
                                   sizeof(int) * static_cast<std::size_t>(child_count + 1),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_name_offsets", message_out) ||
      !ensure_cuda(cudaMemcpyAsync(staging->child_const_offsets, arena.d_child_const_offsets,
                                   sizeof(int) * static_cast<std::size_t>(child_count + 1),
                                   cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync child_const_offsets", message_out) ||
      !ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize copyback_meta", message_out)) {
    return false;
  }

  const std::size_t total_nodes =
      static_cast<std::size_t>(staging->child_node_offsets[child_count]);
  const std::size_t total_names =
      static_cast<std::size_t>(staging->child_name_offsets[child_count]);
  const std::size_t total_consts =
      static_cast<std::size_t>(staging->child_const_offsets[child_count]);

  if ((total_nodes > 0 &&
       !ensure_cuda(cudaMemcpyAsync(staging->child_nodes, arena.d_live_child_nodes,
                                    sizeof(PlainNode) * total_nodes, cudaMemcpyDeviceToHost),
                    "cudaMemcpyAsync live_child_nodes", message_out)) ||
      (total_names > 0 &&
       !ensure_cuda(cudaMemcpyAsync(staging->child_name_ids, arena.d_live_child_name_ids,
                                    sizeof(std::uint64_t) * total_names, cudaMemcpyDeviceToHost),
                    "cudaMemcpyAsync live_child_name_ids", message_out)) ||
      (total_consts > 0 &&
       !ensure_cuda(cudaMemcpyAsync(staging->child_consts, arena.d_live_child_consts,
                                    sizeof(Value) * total_consts, cudaMemcpyDeviceToHost),
                    "cudaMemcpyAsync live_child_consts", message_out)) ||
      !ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize copyback_live", message_out)) {
    return false;
  }

  const auto t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->copyback_ms += ms_between(t0, t1);
  }
  out->config = config;
  out->parent_a = staging->parent_a;
  out->parent_b = staging->parent_b;
  out->is_mutation = staging->is_mutation;
  out->child_nodes = staging->child_nodes;
  out->child_node_offsets = staging->child_node_offsets;
  out->child_name_ids = staging->child_name_ids;
  out->child_name_offsets = staging->child_name_offsets;
  out->child_consts = staging->child_consts;
  out->child_const_offsets = staging->child_const_offsets;
  out->child_used_len = staging->child_used_len;
  out->child_name_counts = staging->child_name_counts;
  out->child_const_counts = staging->child_const_counts;
  out->child_meta = staging->child_meta;
  return true;
}

}  // namespace g3pvm::evo::repro
