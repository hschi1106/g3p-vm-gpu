#pragma once

#include <cstdint>

#include "g3pvm/core/value.hpp"
#include "g3pvm/evolution/repro/types.hpp"
#include "pack_types.cuh"
#include "random_device.cuh"

namespace g3pvm::evo::repro {

__device__ inline int dmin_int(int a, int b) { return (a < b) ? a : b; }
__device__ inline int dmax_int(int a, int b) { return (a > b) ? a : b; }
__device__ inline int dmax3_int(int a, int b, int c) { return dmax_int(a, dmax_int(b, c)); }

constexpr int kGpuReproMetaStackCapacity = 4 * kGpuReproKernelMaxNodes;

enum class DParseTaskKind : unsigned char {
  Program,
  Block,
  Stmt,
  Expr,
  ReduceUnary,
  ReduceBinary,
  ReduceTernary,
  ReduceBlock,
  ReduceIfStmt,
};

enum class DMutationKind : unsigned char {
  None = 0,
  Subtree = 1,
  Constant = 2,
};

struct DSharedValueSlot {
  alignas(Value) unsigned char bytes[sizeof(Value)];
};

static_assert(sizeof(DSharedValueSlot) == sizeof(Value), "shared Value storage must preserve size");
static_assert(alignof(DSharedValueSlot) == alignof(Value), "shared Value storage must preserve alignment");

__device__ inline bool d_is_builtin_kind(NodeKind kind) {
  return kind == NodeKind::CALL_ABS || kind == NodeKind::CALL_MIN || kind == NodeKind::CALL_MAX ||
         kind == NodeKind::CALL_CLIP || kind == NodeKind::CALL_LEN || kind == NodeKind::CALL_CONCAT ||
         kind == NodeKind::CALL_SLICE || kind == NodeKind::CALL_INDEX || kind == NodeKind::CALL_APPEND ||
         kind == NodeKind::CALL_REVERSE || kind == NodeKind::CALL_FIND || kind == NodeKind::CALL_CONTAINS;
}

__device__ inline bool d_push_parse_task(unsigned char* task_stack, int* task_size, DParseTaskKind task) {
  if (*task_size >= kGpuReproMetaStackCapacity) return false;
  task_stack[*task_size] = static_cast<unsigned char>(task);
  *task_size += 1;
  return true;
}

__device__ inline bool d_push_depth_value(int* depth_stack, int* depth_size, int depth) {
  if (*depth_size >= kGpuReproMetaStackCapacity) return false;
  depth_stack[*depth_size] = depth;
  *depth_size += 1;
  return true;
}

__device__ inline bool d_pop_depth_value(int* depth_stack, int* depth_size, int* out) {
  if (*depth_size <= 0) return false;
  *depth_size -= 1;
  *out = depth_stack[*depth_size];
  return true;
}

__device__ inline PackedChildMeta d_compute_child_meta(const DPlainNode* nodes,
                                                       int used_len,
                                                       int max_expr_depth_limit,
                                                       int max_for_k,
                                                       unsigned char* task_stack,
                                                       int* depth_stack) {
  (void)max_for_k;
  PackedChildMeta out;
  out.node_count = used_len;
  out.max_depth = 0;
  out.uses_builtins = 0;
  out.valid = 0;
  if (used_len <= 0 || used_len > kGpuReproKernelMaxNodes) {
    return out;
  }
  for (int i = 0; i < used_len; ++i) {
    if (d_is_builtin_kind(static_cast<NodeKind>(nodes[i].kind))) {
      out.uses_builtins = 1;
      break;
    }
  }
  int idx = 0;
  int task_size = 0;
  int depth_size = 0;
  if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::Program)) {
    return out;
  }
  while (task_size > 0) {
    const DParseTaskKind task = static_cast<DParseTaskKind>(task_stack[task_size - 1]);
    task_size -= 1;
    int first = 0;
    int second = 0;
    int third = 0;
    switch (task) {
      case DParseTaskKind::Program:
        if (idx >= used_len || static_cast<NodeKind>(nodes[idx].kind) != NodeKind::PROGRAM) {
          return out;
        }
        idx += 1;
        if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::Block)) {
          return out;
        }
        break;
      case DParseTaskKind::Block: {
        if (idx >= used_len) return out;
        const NodeKind kind = static_cast<NodeKind>(nodes[idx].kind);
        if (kind == NodeKind::BLOCK_NIL) {
          idx += 1;
          if (!d_push_depth_value(depth_stack, &depth_size, 0)) {
            return out;
          }
          break;
        }
        if (kind != NodeKind::BLOCK_CONS) {
          return out;
        }
        idx += 1;
        if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::ReduceBlock) ||
            !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Block) ||
            !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Stmt)) {
          return out;
        }
        break;
      }
      case DParseTaskKind::Stmt: {
        if (idx >= used_len) return out;
        const DPlainNode node = nodes[idx];
        const NodeKind kind = static_cast<NodeKind>(node.kind);
        idx += 1;
        if (kind == NodeKind::ASSIGN || kind == NodeKind::RETURN) {
          if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr)) {
            return out;
          }
          break;
        }
        if (kind == NodeKind::IF_STMT) {
          if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::ReduceIfStmt) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Block) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Block) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr)) {
            return out;
          }
          break;
        }
        if (kind == NodeKind::FOR_RANGE) {
          if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::ReduceBlock) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Block) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr)) {
            return out;
          }
          break;
        }
        return out;
      }
      case DParseTaskKind::Expr: {
        if (idx >= used_len) return out;
        const NodeKind kind = static_cast<NodeKind>(nodes[idx].kind);
        idx += 1;
        if (kind == NodeKind::CONST || kind == NodeKind::VAR) {
          if (!d_push_depth_value(depth_stack, &depth_size, 1)) {
            return out;
          }
          break;
        }
        if (kind == NodeKind::NEG || kind == NodeKind::NOT || kind == NodeKind::CALL_ABS ||
            kind == NodeKind::CALL_LEN || kind == NodeKind::CALL_REVERSE) {
          if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::ReduceUnary) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr)) {
            return out;
          }
          break;
        }
        if (kind == NodeKind::ADD || kind == NodeKind::SUB || kind == NodeKind::MUL || kind == NodeKind::DIV ||
            kind == NodeKind::MOD || kind == NodeKind::LT || kind == NodeKind::LE || kind == NodeKind::GT ||
            kind == NodeKind::GE || kind == NodeKind::EQ || kind == NodeKind::NE || kind == NodeKind::AND ||
            kind == NodeKind::OR || kind == NodeKind::CALL_MIN || kind == NodeKind::CALL_MAX ||
            kind == NodeKind::CALL_CONCAT || kind == NodeKind::CALL_INDEX || kind == NodeKind::CALL_APPEND ||
            kind == NodeKind::CALL_FIND || kind == NodeKind::CALL_CONTAINS) {
          if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::ReduceBinary) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr)) {
            return out;
          }
          break;
        }
        if (kind == NodeKind::IF_EXPR || kind == NodeKind::CALL_CLIP || kind == NodeKind::CALL_SLICE) {
          if (!d_push_parse_task(task_stack, &task_size, DParseTaskKind::ReduceTernary) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr) ||
              !d_push_parse_task(task_stack, &task_size, DParseTaskKind::Expr)) {
            return out;
          }
          break;
        }
        return out;
      }
      case DParseTaskKind::ReduceUnary:
        if (!d_pop_depth_value(depth_stack, &depth_size, &first) ||
            !d_push_depth_value(depth_stack, &depth_size, 1 + first)) {
          return out;
        }
        break;
      case DParseTaskKind::ReduceBinary:
        if (!d_pop_depth_value(depth_stack, &depth_size, &second) ||
            !d_pop_depth_value(depth_stack, &depth_size, &first) ||
            !d_push_depth_value(depth_stack, &depth_size, 1 + dmax_int(first, second))) {
          return out;
        }
        break;
      case DParseTaskKind::ReduceTernary:
        if (!d_pop_depth_value(depth_stack, &depth_size, &third) ||
            !d_pop_depth_value(depth_stack, &depth_size, &second) ||
            !d_pop_depth_value(depth_stack, &depth_size, &first) ||
            !d_push_depth_value(depth_stack, &depth_size, 1 + dmax3_int(first, second, third))) {
          return out;
        }
        break;
      case DParseTaskKind::ReduceBlock:
        if (!d_pop_depth_value(depth_stack, &depth_size, &second) ||
            !d_pop_depth_value(depth_stack, &depth_size, &first) ||
            !d_push_depth_value(depth_stack, &depth_size, dmax_int(first, second))) {
          return out;
        }
        break;
      case DParseTaskKind::ReduceIfStmt:
        if (!d_pop_depth_value(depth_stack, &depth_size, &third) ||
            !d_pop_depth_value(depth_stack, &depth_size, &second) ||
            !d_pop_depth_value(depth_stack, &depth_size, &first) ||
            !d_push_depth_value(depth_stack, &depth_size, dmax3_int(first, second, third))) {
          return out;
        }
        break;
    }
  }
  if (idx != used_len || depth_size != 1) {
    return out;
  }
  out.max_depth = depth_stack[0];
  out.valid = static_cast<unsigned char>(out.max_depth <= max_expr_depth_limit ? 1 : 0);
  return out;
}

__host__ __device__ inline bool value_equal_simple(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::None) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int || a.tag == ValueTag::String || a.tag == ValueTag::NumList ||
      a.tag == ValueTag::StringList) return a.i == b.i;
  return a.f == b.f;
}

__device__ inline int remap_name_id(std::uint64_t name_id,
                                    std::uint64_t* child_names,
                                    int* child_name_count,
                                    int max_names) {
  for (int i = 0; i < *child_name_count; ++i) {
    if (child_names[i] == name_id) {
      return i;
    }
  }
  if (*child_name_count < max_names) {
    const int idx = *child_name_count;
    child_names[idx] = name_id;
    *child_name_count += 1;
    return idx;
  }
  return 0;
}

__device__ inline int remap_const_value(const Value& v,
                                        Value* child_consts,
                                        int* child_const_count,
                                        int max_consts) {
  for (int i = 0; i < *child_const_count; ++i) {
    if (value_equal_simple(child_consts[i], v)) {
      return i;
    }
  }
  if (*child_const_count < max_consts) {
    const int idx = *child_const_count;
    child_consts[idx] = v;
    *child_const_count += 1;
    return idx;
  }
  return 0;
}

__device__ inline int find_or_append_const_value(const Value& v,
                                                 Value* child_consts,
                                                 int* child_const_count,
                                                 int max_consts) {
  for (int i = 0; i < *child_const_count; ++i) {
    if (value_equal_simple(child_consts[i], v)) {
      return i;
    }
  }
  if (*child_const_count < max_consts) {
    const int idx = *child_const_count;
    child_consts[idx] = v;
    *child_const_count += 1;
    return idx;
  }
  return -1;
}

__device__ inline DPlainNode remap_node_for_child(const DPlainNode& in,
                                                  const std::uint64_t* source_names,
                                                  const Value* source_consts,
                                                  std::uint64_t* child_names,
                                                  int* child_name_count,
                                                  Value* child_consts,
                                                  int* child_const_count,
                                                  int max_names,
                                                  int max_consts) {
  DPlainNode out = in;
  const NodeKind kind = static_cast<NodeKind>(in.kind);
  if (kind == NodeKind::CONST) {
    out.i0 = remap_const_value(source_consts[in.i0], child_consts, child_const_count, max_consts);
  } else if (kind == NodeKind::VAR || kind == NodeKind::ASSIGN || kind == NodeKind::FOR_RANGE) {
    out.i0 = remap_name_id(source_names[in.i0], child_names, child_name_count, max_names);
  }
  return out;
}

__device__ inline int donor_bucket_for_type(RType type) {
  switch (type) {
    case RType::Num:
      return 0;
    case RType::Bool:
      return 1;
    case RType::NoneType:
      return 2;
    case RType::String:
      return 3;
    case RType::NumList:
      return 4;
    case RType::StringList:
      return 5;
    case RType::Any:
      return 6;
    default:
      return 0;
  }
}

__device__ inline DMutationKind choose_child_mutation_kind(std::uint64_t seed,
                                                           double mutation_ratio,
                                                           double mutation_subtree_ratio) {
  const double mutate_pick = static_cast<double>(seed & 0xffffULL) / 65535.0;
  if (mutate_pick >= mutation_ratio) {
    return DMutationKind::None;
  }
  const double subtree_pick = static_cast<double>((seed >> 16) & 0xffffULL) / 65535.0;
  return subtree_pick < mutation_subtree_ratio ? DMutationKind::Subtree : DMutationKind::Constant;
}

__device__ inline int pick_donor_index_for_type(RType type,
                                                int donor_pool_size_per_type,
                                                std::uint64_t seed) {
  const int bucket = donor_bucket_for_type(type);
  const int slot = donor_pool_size_per_type > 0
                       ? static_cast<int>(hash64(seed ^ 0xe7037ed1a0b428dbULL) %
                                          static_cast<std::uint64_t>(donor_pool_size_per_type))
                       : 0;
  return bucket * donor_pool_size_per_type + slot;
}

__device__ inline void apply_constant_mutation(DPlainNode* child_nodes,
                                               int used_len,
                                               Value* child_consts,
                                               int* child_const_count,
                                               int max_consts,
                                               std::uint64_t seed) {
  int const_node_count = 0;
  for (int i = 0; i < used_len; ++i) {
    if (static_cast<NodeKind>(child_nodes[i].kind) == NodeKind::CONST) {
      ++const_node_count;
    }
  }
  if (const_node_count <= 0) {
    return;
  }

  const int chosen_rank = static_cast<int>(hash64(seed ^ 0x9e3779b97f4a7c15ULL) %
                                           static_cast<std::uint64_t>(const_node_count));
  int seen = 0;
  int node_index = -1;
  for (int i = 0; i < used_len; ++i) {
    if (static_cast<NodeKind>(child_nodes[i].kind) != NodeKind::CONST) {
      continue;
    }
    if (seen == chosen_rank) {
      node_index = i;
      break;
    }
    ++seen;
  }
  if (node_index < 0) {
    return;
  }

  const int const_index = child_nodes[node_index].i0;
  if (const_index < 0 || const_index >= *child_const_count) {
    return;
  }

  Value mutated = child_consts[const_index];
  const std::uint64_t tweak_seed = hash64(seed ^ 0x243f6a8885a308d3ULL);
  if (mutated.tag == ValueTag::Int) {
    mutated.i += static_cast<std::int64_t>(tweak_seed % 5ULL) - 2;
  } else if (mutated.tag == ValueTag::Float) {
    const double frac = static_cast<double>(tweak_seed & 0xffffULL) / 65535.0;
    mutated.f += frac * 2.0 - 1.0;
  } else if (mutated.tag == ValueTag::Bool) {
    mutated.b = !mutated.b;
  }

  const int new_const_index = find_or_append_const_value(mutated, child_consts, child_const_count, max_consts);
  if (new_const_index >= 0) {
    child_nodes[node_index].i0 = new_const_index;
  }
}

__global__ void variation_kernel(const DPlainNode* program_nodes,
                                 const DPackedProgramMeta* metas,
                                 const DCandidateRange* candidates,
                                 const std::uint64_t* program_name_ids,
                                 const Value* program_consts,
                                 const DPlainNode* donor_nodes,
                                 const int* donor_lens,
                                 const std::uint64_t* donor_name_ids,
                                 const int* donor_name_counts,
                                 const Value* donor_consts,
                                 const int* donor_const_counts,
                                 int max_nodes,
                                 int candidates_per_program,
                                 int max_donor_nodes,
                                 int max_names,
                                 int max_consts,
                                 int max_expr_depth,
                                 int max_for_k,
                                 int pair_count,
                                 int donor_pool_size_per_type,
                                 double mutation_ratio,
                                 double mutation_subtree_ratio,
                                 std::uint64_t seed,
                                 const int* parent_a,
                                 const int* parent_b,
                                 const int* cand_a,
                                 const int* cand_b,
                                 DPlainNode* child_nodes,
                                 int* child_used_len_out,
                                 std::uint64_t* child_name_ids_out,
                                 int* child_name_counts_out,
                                 Value* child_consts_out,
                                 int* child_const_counts_out,
                                 PackedChildMeta* child_meta_out) {
  const int pair_idx = static_cast<int>(blockIdx.x);
  if (pair_idx >= pair_count) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int out_idx_a = pair_idx * 2;
  const int out_idx_b = pair_idx * 2 + 1;

  const int pa = parent_a[pair_idx];
  const int pb = parent_b[pair_idx];
  const int base_a = pa * max_nodes;
  const int base_b = pb * max_nodes;
  const DCandidateRange ra = candidates[pa * candidates_per_program + cand_a[pair_idx]];
  const DCandidateRange rb = candidates[pb * candidates_per_program + cand_b[pair_idx]];
  const int len_a = metas[pa].used_len;
  const int len_b = metas[pb].used_len;
  const int name_count_a = metas[pa].name_count;
  const int const_count_a = metas[pa].const_count;
  const int name_count_b = metas[pb].name_count;
  const int const_count_b = metas[pb].const_count;

  const std::uint64_t child_seed_a = hash64(seed ^ (static_cast<std::uint64_t>(out_idx_a + 1) * 0x9e3779b97f4a7c15ULL));
  const std::uint64_t child_seed_b = hash64(seed ^ (static_cast<std::uint64_t>(out_idx_b + 1) * 0x9e3779b97f4a7c15ULL));
  const DMutationKind mutation_kind_a =
      choose_child_mutation_kind(child_seed_a, mutation_ratio, mutation_subtree_ratio);
  const DMutationKind mutation_kind_b =
      choose_child_mutation_kind(child_seed_b, mutation_ratio, mutation_subtree_ratio);

  const bool valid_cross_a = ra.start >= 0 && ra.stop > ra.start && rb.start >= 0 && rb.stop > rb.start;
  const bool valid_cross_b = valid_cross_a;

  int prefix_a = dmax_int(0, ra.start);
  int replace_a = dmax_int(0, ra.stop - ra.start);
  int donor_len_for_a = 0;
  const DPlainNode* donor_ptr_for_a = nullptr;
  const std::uint64_t* donor_names_for_a = nullptr;
  const Value* donor_consts_for_a = nullptr;

  int prefix_b = dmax_int(0, rb.start);
  int replace_b = dmax_int(0, rb.stop - rb.start);
  int donor_len_for_b = 0;
  const DPlainNode* donor_ptr_for_b = nullptr;
  const std::uint64_t* donor_names_for_b = nullptr;
  const Value* donor_consts_for_b = nullptr;

  if (valid_cross_a && mutation_kind_a == DMutationKind::Subtree) {
    const int donor_index_a =
        pick_donor_index_for_type(static_cast<RType>(ra.aux), donor_pool_size_per_type, child_seed_a);
    donor_len_for_a = donor_lens[donor_index_a];
    donor_ptr_for_a = donor_nodes + donor_index_a * max_donor_nodes;
    donor_names_for_a = donor_name_ids + donor_index_a * max_names;
    donor_consts_for_a = donor_consts + donor_index_a * max_consts;
  } else if (valid_cross_a) {
    donor_len_for_a = dmax_int(0, rb.stop - rb.start);
    donor_ptr_for_a = program_nodes + base_b + rb.start;
    donor_names_for_a = program_name_ids + pb * max_names;
    donor_consts_for_a = program_consts + pb * max_consts;
  }

  if (valid_cross_b && mutation_kind_b == DMutationKind::Subtree) {
    const int donor_index_b =
        pick_donor_index_for_type(static_cast<RType>(rb.aux), donor_pool_size_per_type, child_seed_b);
    donor_len_for_b = donor_lens[donor_index_b];
    donor_ptr_for_b = donor_nodes + donor_index_b * max_donor_nodes;
    donor_names_for_b = donor_name_ids + donor_index_b * max_names;
    donor_consts_for_b = donor_consts + donor_index_b * max_consts;
  } else if (valid_cross_b) {
    donor_len_for_b = dmax_int(0, ra.stop - ra.start);
    donor_ptr_for_b = program_nodes + base_a + ra.start;
    donor_names_for_b = program_name_ids + pa * max_names;
    donor_consts_for_b = program_consts + pa * max_consts;
  }

  __shared__ DPlainNode child_a_work[kGpuReproKernelMaxNodes];
  __shared__ DPlainNode child_b_work[kGpuReproKernelMaxNodes];
  __shared__ std::uint64_t child_a_names[kGpuReproMaxNames];
  __shared__ std::uint64_t child_b_names[kGpuReproMaxNames];
  __shared__ DSharedValueSlot child_a_const_storage[kGpuReproMaxConsts];
  __shared__ DSharedValueSlot child_b_const_storage[kGpuReproMaxConsts];
  __shared__ unsigned char meta_task_stack[kGpuReproMetaStackCapacity];
  __shared__ int meta_depth_stack[kGpuReproMetaStackCapacity];
  __shared__ int child_a_name_count;
  __shared__ int child_b_name_count;
  __shared__ int child_a_const_count;
  __shared__ int child_b_const_count;
  __shared__ int child_a_used_len;
  __shared__ int child_b_used_len;
  Value* child_a_consts = reinterpret_cast<Value*>(child_a_const_storage);
  Value* child_b_consts = reinterpret_cast<Value*>(child_b_const_storage);

  if (tid == 0) {
    child_a_name_count = dmin_int(name_count_a, max_names);
    child_a_const_count = dmin_int(const_count_a, max_consts);
    child_b_name_count = dmin_int(name_count_b, max_names);
    child_b_const_count = dmin_int(const_count_b, max_consts);
    for (int i = 0; i < child_a_name_count; ++i) child_a_names[i] = program_name_ids[pa * max_names + i];
    for (int i = 0; i < child_a_const_count; ++i) child_a_consts[i] = program_consts[pa * max_consts + i];
    for (int i = 0; i < child_b_name_count; ++i) child_b_names[i] = program_name_ids[pb * max_names + i];
    for (int i = 0; i < child_b_const_count; ++i) child_b_consts[i] = program_consts[pb * max_consts + i];
  }
  __syncthreads();

  const int suffix_a_start = prefix_a + replace_a;
  const int suffix_a_len = dmax_int(0, len_a - suffix_a_start);
  const int out_a_len = prefix_a + donor_len_for_a + suffix_a_len;
  const bool child_a_fallback =
      !valid_cross_a || out_a_len > max_nodes || prefix_a >= len_a || replace_a <= 0 || donor_ptr_for_a == nullptr;
  if (child_a_fallback) {
    for (int i = tid; i < len_a; i += blockDim.x) {
      child_a_work[i] = program_nodes[base_a + i];
    }
  } else {
    for (int i = tid; i < prefix_a; i += blockDim.x) {
      child_a_work[i] = program_nodes[base_a + i];
    }
    for (int i = tid; i < donor_len_for_a; i += blockDim.x) {
      child_a_work[prefix_a + i] =
          remap_node_for_child(donor_ptr_for_a[i], donor_names_for_a, donor_consts_for_a, child_a_names,
                               &child_a_name_count, child_a_consts, &child_a_const_count, max_names, max_consts);
    }
    for (int i = tid; i < suffix_a_len; i += blockDim.x) {
      child_a_work[prefix_a + donor_len_for_a + i] = program_nodes[base_a + suffix_a_start + i];
    }
  }

  const int suffix_b_start = prefix_b + replace_b;
  const int suffix_b_len = dmax_int(0, len_b - suffix_b_start);
  const int out_b_len = prefix_b + donor_len_for_b + suffix_b_len;
  const bool child_b_fallback =
      !valid_cross_b || out_b_len > max_nodes || prefix_b >= len_b || replace_b <= 0 || donor_ptr_for_b == nullptr;
  if (child_b_fallback) {
    for (int i = tid; i < len_b; i += blockDim.x) {
      child_b_work[i] = program_nodes[base_b + i];
    }
  } else {
    for (int i = tid; i < prefix_b; i += blockDim.x) {
      child_b_work[i] = program_nodes[base_b + i];
    }
    for (int i = tid; i < donor_len_for_b; i += blockDim.x) {
      child_b_work[prefix_b + i] =
          remap_node_for_child(donor_ptr_for_b[i], donor_names_for_b, donor_consts_for_b, child_b_names,
                               &child_b_name_count, child_b_consts, &child_b_const_count, max_names, max_consts);
    }
    for (int i = tid; i < suffix_b_len; i += blockDim.x) {
      child_b_work[prefix_b + donor_len_for_b + i] = program_nodes[base_b + suffix_b_start + i];
    }
  }

  __syncthreads();
  if (tid == 0) {
    child_a_used_len = child_a_fallback ? len_a : out_a_len;
    child_b_used_len = child_b_fallback ? len_b : out_b_len;
    if (mutation_kind_a == DMutationKind::Constant) {
      apply_constant_mutation(child_a_work, child_a_used_len, child_a_consts, &child_a_const_count, max_consts,
                              child_seed_a);
    }
    if (mutation_kind_b == DMutationKind::Constant) {
      apply_constant_mutation(child_b_work, child_b_used_len, child_b_consts, &child_b_const_count, max_consts,
                              child_seed_b);
    }
  }
  __syncthreads();

  const int child_base_ab = out_idx_a * max_nodes;
  const int child_base_ba = out_idx_b * max_nodes;
  for (int i = tid; i < child_a_used_len; i += blockDim.x) {
    child_nodes[child_base_ab + i] = child_a_work[i];
  }
  for (int i = tid; i < child_b_used_len; i += blockDim.x) {
    child_nodes[child_base_ba + i] = child_b_work[i];
  }

  __syncthreads();
  if (tid == 0) {
    child_used_len_out[out_idx_a] = child_a_used_len;
    child_used_len_out[out_idx_b] = child_b_used_len;
    child_name_counts_out[out_idx_a] = child_a_name_count;
    child_const_counts_out[out_idx_a] = child_a_const_count;
    child_name_counts_out[out_idx_b] = child_b_name_count;
    child_const_counts_out[out_idx_b] = child_b_const_count;
    for (int i = 0; i < child_a_name_count; ++i) {
      child_name_ids_out[out_idx_a * max_names + i] = child_a_names[i];
    }
    for (int i = 0; i < child_a_const_count; ++i) {
      child_consts_out[out_idx_a * max_consts + i] = child_a_consts[i];
    }
    for (int i = 0; i < child_b_name_count; ++i) {
      child_name_ids_out[out_idx_b * max_names + i] = child_b_names[i];
    }
    for (int i = 0; i < child_b_const_count; ++i) {
      child_consts_out[out_idx_b * max_consts + i] = child_b_consts[i];
    }
    PackedChildMeta child_a_meta =
        d_compute_child_meta(child_a_work, child_a_used_len, max_expr_depth, max_for_k, meta_task_stack,
                             meta_depth_stack);
    PackedChildMeta child_b_meta =
        d_compute_child_meta(child_b_work, child_b_used_len, max_expr_depth, max_for_k, meta_task_stack,
                             meta_depth_stack);
    if (child_a_fallback) child_a_meta.valid = 0;
    if (child_b_fallback) child_b_meta.valid = 0;
    child_meta_out[out_idx_a] = child_a_meta;
    child_meta_out[out_idx_b] = child_b_meta;
  }
}

}  // namespace g3pvm::evo::repro
