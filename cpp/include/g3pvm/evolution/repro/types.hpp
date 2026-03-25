#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/core/value.hpp"
#include "g3pvm/evolution/ast_program.hpp"

namespace g3pvm::evo::repro {

constexpr int kGpuReproMaxNames = 64;
constexpr int kGpuReproMaxConsts = 64;
constexpr int kGpuReproKernelMaxNodes = 512;
constexpr int kGpuReproDonorTypeCount = 6;

enum class CandidateTag {
  Expr = 0,
};

struct CandidateRange {
  int start = -1;
  int stop = -1;
  int tag = static_cast<int>(CandidateTag::Expr);
  int aux = static_cast<int>(RType::Invalid);
};

struct PlainNode {
  int kind = 0;
  int i0 = 0;
  int i1 = 0;
};

struct PackedProgramMeta {
  int used_len = 0;
  int name_count = 0;
  int const_count = 0;
};

struct DonorProgram {
  AstProgram ast;
  RType type = RType::Invalid;
};

struct PreprocessOutput {
  std::vector<std::vector<std::size_t>> subtree_ends;
  std::vector<std::vector<CandidateRange>> candidates;
  std::vector<DonorProgram> donor_pool;
};

struct GpuReproConfig {
  int population_size = 0;
  int pair_count = 0;
  int candidates_per_program = 16;
  int donor_pool_size_per_type = 64;
  int max_nodes = 80;
  int max_donor_nodes = 24;
  int max_names = kGpuReproMaxNames;
  int max_consts = kGpuReproMaxConsts;
  int tournament_k = 3;
  int max_expr_depth = 0;
  int max_for_k = 0;
  double mutation_ratio = 0.5;
  double mutation_subtree_ratio = 0.8;
  std::uint64_t seed = 0;
};

struct PackedHostData {
  GpuReproConfig config;
  std::vector<PlainNode> program_nodes;
  std::vector<PackedProgramMeta> metas;
  std::vector<CandidateRange> candidates;
  std::vector<std::uint64_t> program_name_ids;
  std::vector<Value> program_consts;
  std::vector<PlainNode> donor_nodes;
  std::vector<int> donor_lens;
  std::vector<std::uint64_t> donor_name_ids;
  std::vector<int> donor_name_counts;
  std::vector<Value> donor_consts;
  std::vector<int> donor_const_counts;
  std::unordered_map<std::uint64_t, std::string> name_lookup;
};

struct GpuReproSelectionPlan {
  std::vector<int> parent_a;
  std::vector<int> parent_b;
  std::vector<int> cand_a;
  std::vector<int> cand_b;
};

struct PackedChildMeta {
  int node_count = 0;
  int max_depth = 0;
  unsigned char uses_builtins = 0;
  unsigned char valid = 0;
};

struct GpuReproChildData {
  GpuReproConfig config;
  GpuReproSelectionPlan selection;
  std::vector<PlainNode> child_nodes;
  std::vector<int> child_used_len;
  std::vector<std::uint64_t> child_name_ids;
  std::vector<int> child_name_counts;
  std::vector<Value> child_consts;
  std::vector<int> child_const_counts;
  std::vector<PackedChildMeta> child_meta;
};

struct GpuReproChildView {
  GpuReproConfig config;
  const int* parent_a = nullptr;
  const int* parent_b = nullptr;
  const PlainNode* child_nodes = nullptr;
  const int* child_node_offsets = nullptr;
  const std::uint64_t* child_name_ids = nullptr;
  const int* child_name_offsets = nullptr;
  const Value* child_consts = nullptr;
  const int* child_const_offsets = nullptr;
  const int* child_used_len = nullptr;
  const int* child_name_counts = nullptr;
  const int* child_const_counts = nullptr;
  const PackedChildMeta* child_meta = nullptr;
};

}  // namespace g3pvm::evo::repro
