#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "g3pvm/evolution/repro/stats.hpp"
#include "g3pvm/evolution/repro/types.hpp"

namespace g3pvm::evo::repro {

struct GpuReproArena {
  int device_id = -1;
  GpuReproConfig capacity;
  PlainNode* d_program_nodes = nullptr;
  PackedProgramMeta* d_metas = nullptr;
  CandidateRange* d_candidates = nullptr;
  std::uint64_t* d_program_name_ids = nullptr;
  Value* d_program_consts = nullptr;
  PlainNode* d_donor_nodes = nullptr;
  int* d_donor_lens = nullptr;
  std::uint64_t* d_donor_name_ids = nullptr;
  int* d_donor_name_counts = nullptr;
  Value* d_donor_consts = nullptr;
  int* d_donor_const_counts = nullptr;
  double* d_fitness = nullptr;
  int* d_parent_a = nullptr;
  int* d_parent_b = nullptr;
  int* d_cand_a = nullptr;
  int* d_cand_b = nullptr;
  PlainNode* d_child_nodes = nullptr;
  int* d_child_used_len = nullptr;
  std::uint64_t* d_child_name_ids = nullptr;
  int* d_child_name_counts = nullptr;
  Value* d_child_consts = nullptr;
  int* d_child_const_counts = nullptr;
  PackedChildMeta* d_child_meta = nullptr;
  int* d_child_node_offsets = nullptr;
  int* d_child_name_offsets = nullptr;
  int* d_child_const_offsets = nullptr;
  PlainNode* d_live_child_nodes = nullptr;
  std::uint64_t* d_live_child_name_ids = nullptr;
  Value* d_live_child_consts = nullptr;
};

struct GpuReproHostStaging {
  GpuReproConfig capacity;
  int* parent_a = nullptr;
  int* parent_b = nullptr;
  int* child_used_len = nullptr;
  int* child_name_counts = nullptr;
  int* child_const_counts = nullptr;
  PackedChildMeta* child_meta = nullptr;
  int* child_node_offsets = nullptr;
  int* child_name_offsets = nullptr;
  int* child_const_offsets = nullptr;
  PlainNode* child_nodes = nullptr;
  std::uint64_t* child_name_ids = nullptr;
  Value* child_consts = nullptr;
};

bool gpu_repro_config_fits_capacity(const GpuReproConfig& need, const GpuReproConfig& have);
bool select_gpu_repro_device(int* device_id, std::string* message_out);
void destroy_gpu_repro_arena(GpuReproArena* arena);
bool ensure_gpu_repro_arena_capacity(GpuReproArena* arena,
                                     const GpuReproConfig& config,
                                     std::string* message_out);
void destroy_gpu_repro_host_staging(GpuReproHostStaging* staging);
bool ensure_gpu_repro_host_staging_capacity(GpuReproHostStaging* staging,
                                            const GpuReproConfig& config,
                                            std::string* message_out);
bool upload_gpu_repro_inputs(const PackedHostData& packed,
                             GpuReproArena* arena,
                             ReproductionStats* stats,
                             std::string* message_out);
bool launch_gpu_repro_kernels(GpuReproArena* arena,
                              const GpuReproConfig& config,
                              const std::vector<double>& fitness,
                              ReproductionStats* stats,
                              std::string* message_out);
bool copyback_gpu_repro_children(const GpuReproArena& arena,
                                 const GpuReproConfig& config,
                                 GpuReproHostStaging* staging,
                                 GpuReproChildView* out,
                                 ReproductionStats* stats,
                                 std::string* message_out);

}  // namespace g3pvm::evo::repro
