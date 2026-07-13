#include "g3pvm/evolution/repro/pack.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/runtime/payload/payload.hpp"
#include "../subtree_utils.hpp"

namespace g3pvm::evo::repro {

namespace {

std::uint64_t hash_name(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

void register_name(PackedHostData* packed, const std::string& name) {
  const std::uint64_t id = hash_name(name);
  auto it = packed->name_lookup.find(id);
  if (it != packed->name_lookup.end()) {
    if (it->second != name) {
      throw std::runtime_error("gpu reproduction name hash collision");
    }
    return;
  }
  packed->name_lookup.emplace(id, name);
}

std::uint64_t hash64_host(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

bool child_uses_subtree_mutation(const GpuReproConfig& config, int child_index) {
  const std::uint64_t child_seed =
      hash64_host(config.seed ^ (static_cast<std::uint64_t>(child_index + 1) * 0x9e3779b97f4a7c15ULL));
  const double mutate_pick = static_cast<double>(child_seed & 0xffffULL) / 65535.0;
  if (mutate_pick >= config.mutation_ratio) {
    return false;
  }
  const double subtree_pick = static_cast<double>((child_seed >> 16) & 0xffffULL) / 65535.0;
  return subtree_pick < config.mutation_subtree_ratio;
}

std::uint64_t child_seed_for_index(const GpuReproConfig& config, int child_index) {
  return hash64_host(config.seed ^ (static_cast<std::uint64_t>(child_index + 1) * 0x9e3779b97f4a7c15ULL));
}

int donor_bucket_for_type(RType type) {
  switch (type) {
    case RType::Int: return 0;
    case RType::Float: return 1;
    case RType::Bool: return 2;
    case RType::Char: return 3;
    case RType::String: return 4;
    case RType::IntList: return 5;
    case RType::FloatList: return 6;
    case RType::StringList: return 7;
    case RType::Any: return 8;
    default: return 0;
  }
}

int donor_index_for_child(const GpuReproConfig& config, const CandidateRange& target, int child_index) {
  if (config.donor_pool_size_per_type <= 0) {
    return -1;
  }
  const int bucket = donor_bucket_for_type(static_cast<RType>(target.aux));
  const int slot = static_cast<int>(hash64_host(child_seed_for_index(config, child_index) ^
                                                0xe7037ed1a0b428dbULL) %
                                    static_cast<std::uint64_t>(config.donor_pool_size_per_type));
  return bucket * config.donor_pool_size_per_type + slot;
}

bool active_contains(const std::vector<int>& active_binders, int name_index) {
  return std::find(active_binders.begin(), active_binders.end(), name_index) != active_binders.end();
}

const LinearRecBinders* decoded_linear_rec_binders_for_node(const AstProgram& program, std::size_t node_index) {
  for (const LinearRecBinders& binders : program.linear_rec_binders) {
    if (binders.node_index == node_index) {
      return &binders;
    }
  }
  return nullptr;
}

const AsgpDcBinders* decoded_asgp_dc_binders_for_node(const AstProgram& program, std::size_t node_index) {
  for (const AsgpDcBinders& binders : program.asgp_dc_binders) {
    if (binders.node_index == node_index) {
      return &binders;
    }
  }
  return nullptr;
}

const AsgpDp1dSpec* decoded_asgp_dp1d_spec_for_node(const AstProgram& program, std::size_t node_index) {
  for (const AsgpDp1dSpec& spec : program.asgp_dp1d_specs) {
    if (spec.node_index == node_index) {
      return &spec;
    }
  }
  return nullptr;
}

const AsgpDp2dSpec* decoded_asgp_dp2d_spec_for_node(const AstProgram& program, std::size_t node_index) {
  for (const AsgpDp2dSpec& spec : program.asgp_dp2d_specs) {
    if (spec.node_index == node_index) {
      return &spec;
    }
  }
  return nullptr;
}

bool validate_bound_vars_prefix(const AstProgram& program,
                                std::size_t idx,
                                std::vector<int>* active_binders,
                                std::size_t* next_out) {
  if (idx >= program.nodes.size()) {
    return false;
  }
  const AstNode& node = program.nodes[idx];
  if (node.kind == NodeKind::BOUND_VAR) {
    if (node.i0 < 0 || static_cast<std::size_t>(node.i0) >= program.names.size()) {
      return false;
    }
    if (!active_contains(*active_binders, node.i0)) {
      return false;
    }
    *next_out = idx + 1;
    return true;
  }
  if (node.kind == NodeKind::CONST) {
    if (node.i0 < 0 || static_cast<std::size_t>(node.i0) >= program.consts.size()) {
      return false;
    }
    *next_out = idx + 1;
    return true;
  }
  if (node.kind == NodeKind::VAR || node.kind == NodeKind::ASSIGN || node.kind == NodeKind::FOR_RANGE) {
    if (node.i0 < 0 || static_cast<std::size_t>(node.i0) >= program.names.size()) {
      return false;
    }
  }
  if (node.kind == NodeKind::MAP_LIST || node.kind == NodeKind::FILTER_LIST) {
    if (node.i0 < 0 || static_cast<std::size_t>(node.i0) >= program.names.size()) {
      return false;
    }
    std::size_t source_next = 0;
    if (!validate_bound_vars_prefix(program, idx + 1, active_binders, &source_next)) {
      return false;
    }
    active_binders->push_back(node.i0);
    std::size_t body_next = 0;
    const bool ok = validate_bound_vars_prefix(program, source_next, active_binders, &body_next);
    active_binders->pop_back();
    if (!ok) {
      return false;
    }
    *next_out = body_next;
    return true;
  }
  if (node.kind == NodeKind::LINEAR_REC) {
    const LinearRecBinders* binders = decoded_linear_rec_binders_for_node(program, idx);
    if (binders == nullptr) {
      return false;
    }
    if (binders->elem_name < 0 || static_cast<std::size_t>(binders->elem_name) >= program.names.size() ||
        binders->accum_name < 0 || static_cast<std::size_t>(binders->accum_name) >= program.names.size() ||
        binders->index_name < 0 || static_cast<std::size_t>(binders->index_name) >= program.names.size()) {
      return false;
    }
    std::size_t cur = idx + 1;
    for (int child = 0; child < 3; ++child) {
      if (!validate_bound_vars_prefix(program, cur, active_binders, &cur)) {
        return false;
      }
    }
    active_binders->push_back(binders->elem_name);
    active_binders->push_back(binders->accum_name);
    active_binders->push_back(binders->index_name);
    if (!validate_bound_vars_prefix(program, cur, active_binders, &cur)) {
      active_binders->pop_back();
      active_binders->pop_back();
      active_binders->pop_back();
      return false;
    }
    active_binders->pop_back();
    active_binders->pop_back();
    active_binders->pop_back();

    active_binders->push_back(binders->elem_name);
    active_binders->push_back(binders->index_name);
    const bool ok = validate_bound_vars_prefix(program, cur, active_binders, &cur);
    active_binders->pop_back();
    active_binders->pop_back();
    if (!ok) {
      return false;
    }
    *next_out = cur;
    return true;
  }
  if (node.kind == NodeKind::ASGP_DC) {
    const AsgpDcBinders* binders = decoded_asgp_dc_binders_for_node(program, idx);
    if (binders == nullptr) {
      return false;
    }
    if (binders->solve_xs_name < 0 || static_cast<std::size_t>(binders->solve_xs_name) >= program.names.size() ||
        binders->solve_n_name < 0 || static_cast<std::size_t>(binders->solve_n_name) >= program.names.size() ||
        binders->solve_lo_name < 0 || static_cast<std::size_t>(binders->solve_lo_name) >= program.names.size() ||
        binders->divide_n_name < 0 || static_cast<std::size_t>(binders->divide_n_name) >= program.names.size() ||
        binders->combine_left_name < 0 || static_cast<std::size_t>(binders->combine_left_name) >= program.names.size() ||
        binders->combine_right_name < 0 || static_cast<std::size_t>(binders->combine_right_name) >= program.names.size()) {
      return false;
    }
    std::size_t cur = idx + 1;
    if (!validate_bound_vars_prefix(program, cur, active_binders, &cur)) {
      return false;
    }

    std::vector<int> phase_binders = {binders->solve_xs_name, binders->solve_n_name, binders->solve_lo_name};
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    phase_binders = {binders->divide_n_name};
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    phase_binders = {binders->combine_left_name, binders->combine_right_name};
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    *next_out = cur;
    return true;
  }
  if (node.kind == NodeKind::ASGP_DP1D) {
    const AsgpDp1dSpec* spec = decoded_asgp_dp1d_spec_for_node(program, idx);
    if (spec == nullptr) {
      return false;
    }
    if (spec->boundary_const < 0 || static_cast<std::size_t>(spec->boundary_const) >= program.consts.size() ||
        spec->solve_state_name < 0 || static_cast<std::size_t>(spec->solve_state_name) >= program.names.size() ||
        spec->transition_state_name < 0 ||
        static_cast<std::size_t>(spec->transition_state_name) >= program.names.size()) {
      return false;
    }
    for (int name : spec->transition_dep_names) {
      if (name < 0 || static_cast<std::size_t>(name) >= program.names.size()) {
        return false;
      }
    }
    std::size_t cur = idx + 1;
    if (!validate_bound_vars_prefix(program, cur, active_binders, &cur)) {
      return false;
    }
    std::vector<int> phase_binders = {spec->solve_state_name};
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    phase_binders = {spec->transition_state_name};
    phase_binders.insert(phase_binders.end(), spec->transition_dep_names.begin(), spec->transition_dep_names.end());
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    *next_out = cur;
    return true;
  }
  if (node.kind == NodeKind::ASGP_DP2D) {
    const AsgpDp2dSpec* spec = decoded_asgp_dp2d_spec_for_node(program, idx);
    if (spec == nullptr) {
      return false;
    }
    if (spec->boundary_const < 0 || static_cast<std::size_t>(spec->boundary_const) >= program.consts.size() ||
        spec->solve_i_name < 0 || static_cast<std::size_t>(spec->solve_i_name) >= program.names.size() ||
        spec->solve_j_name < 0 || static_cast<std::size_t>(spec->solve_j_name) >= program.names.size() ||
        spec->transition_i_name < 0 || static_cast<std::size_t>(spec->transition_i_name) >= program.names.size() ||
        spec->transition_j_name < 0 || static_cast<std::size_t>(spec->transition_j_name) >= program.names.size()) {
      return false;
    }
    for (int name : spec->transition_dep_names) {
      if (name < 0 || static_cast<std::size_t>(name) >= program.names.size()) {
        return false;
      }
    }
    std::size_t cur = idx + 1;
    if (!validate_bound_vars_prefix(program, cur, active_binders, &cur) ||
        !validate_bound_vars_prefix(program, cur, active_binders, &cur)) {
      return false;
    }
    std::vector<int> phase_binders = {spec->solve_i_name, spec->solve_j_name};
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    phase_binders = {spec->transition_i_name, spec->transition_j_name};
    phase_binders.insert(phase_binders.end(), spec->transition_dep_names.begin(), spec->transition_dep_names.end());
    if (!validate_bound_vars_prefix(program, cur, &phase_binders, &cur)) {
      return false;
    }
    *next_out = cur;
    return true;
  }

  std::size_t cur = idx + 1;
  for (int child = 0; child < subtree::node_arity(node.kind); ++child) {
    if (!validate_bound_vars_prefix(program, cur, active_binders, &cur)) {
      return false;
    }
  }
  *next_out = cur;
  return true;
}

bool decoded_child_has_valid_binders(const ProgramGenome& genome) {
  if (genome.ast.nodes.empty()) {
    return false;
  }
  bool needs_binder_validation = false;
  for (const AstNode& node : genome.ast.nodes) {
    if (node.kind == NodeKind::BOUND_VAR || node.kind == NodeKind::MAP_LIST ||
        node.kind == NodeKind::FILTER_LIST || node.kind == NodeKind::LINEAR_REC ||
        node.kind == NodeKind::ASGP_DC || node.kind == NodeKind::ASGP_DP1D ||
        node.kind == NodeKind::ASGP_DP2D) {
      needs_binder_validation = true;
      break;
    }
  }
  if (!needs_binder_validation) {
    return true;
  }
  std::vector<int> active_binders;
  std::size_t next = 0;
  return validate_bound_vars_prefix(genome.ast, 0, &active_binders, &next) &&
         next == genome.ast.nodes.size() && active_binders.empty();
}

bool payload_available_for_value(const Value& value) {
  if (value.tag == ValueTag::String) {
    std::string ignored;
    return g3pvm::payload::lookup_string(value, &ignored);
  }
  if (value.tag != ValueTag::IntList && value.tag != ValueTag::FloatList &&
      value.tag != ValueTag::StringList) {
    return true;
  }
  std::vector<Value> elems;
  if (!g3pvm::payload::lookup_list(value, &elems)) {
    return false;
  }
  for (const Value& elem : elems) {
    if (!payload_available_for_value(elem)) {
      return false;
    }
  }
  return true;
}

bool payloads_available_for_consts(const std::vector<Value>& consts) {
  for (const Value& value : consts) {
    if (!payload_available_for_value(value)) {
      return false;
    }
  }
  return true;
}

int remap_source_name_to_child(const PackedHostData& packed,
                               const GpuReproChildView& copyback,
                               int child_index,
                               const std::vector<std::uint64_t>& source_names,
                               int source_name_index,
                               AstProgram* out) {
  if (source_name_index < 0 || static_cast<std::size_t>(source_name_index) >= source_names.size()) {
    return -1;
  }
  const std::uint64_t source_id = source_names[static_cast<std::size_t>(source_name_index)];
  const int name_count = copyback.child_name_counts[static_cast<std::size_t>(child_index)];
  const std::size_t name_base = static_cast<std::size_t>(copyback.child_name_offsets[child_index]);
  for (int i = 0; i < name_count; ++i) {
    if (copyback.child_name_ids[name_base + static_cast<std::size_t>(i)] == source_id) {
      return i;
    }
  }
  auto it = packed.name_lookup.find(source_id);
  if (it == packed.name_lookup.end()) {
    return -1;
  }
  out->names.push_back(it->second);
  return static_cast<int>(out->names.size() - 1);
}

int remap_source_name_to_child(const GpuReproChildView& copyback,
                               int child_index,
                               const std::vector<std::uint64_t>& source_names,
                               int source_name_index) {
  if (source_name_index < 0 || static_cast<std::size_t>(source_name_index) >= source_names.size()) {
    return -1;
  }
  const std::uint64_t source_id = source_names[static_cast<std::size_t>(source_name_index)];
  const int name_count = copyback.child_name_counts[static_cast<std::size_t>(child_index)];
  const std::size_t name_base = static_cast<std::size_t>(copyback.child_name_offsets[child_index]);
  for (int i = 0; i < name_count; ++i) {
    if (copyback.child_name_ids[name_base + static_cast<std::size_t>(i)] == source_id) {
      return i;
    }
  }
  return -1;
}

bool packed_value_equal(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::Invalid) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Float) return a.f == b.f;
  return a.i == b.i;
}

int remap_source_const_to_child(const std::vector<Value>& source_consts,
                                int source_const_index,
                                AstProgram* out) {
  if (source_const_index < 0 || static_cast<std::size_t>(source_const_index) >= source_consts.size()) {
    return -1;
  }
  const Value& source_value = source_consts[static_cast<std::size_t>(source_const_index)];
  for (std::size_t i = 0; i < out->consts.size(); ++i) {
    if (packed_value_equal(out->consts[i], source_value)) {
      return static_cast<int>(i);
    }
  }
  out->consts.push_back(source_value);
  return static_cast<int>(out->consts.size() - 1);
}

std::vector<int> decode_plain_ints(const int* raw, int count, int cap) {
  std::vector<int> out;
  const int used = std::max(0, std::min(count, cap));
  out.reserve(static_cast<std::size_t>(used));
  for (int i = 0; i < used; ++i) {
    out.push_back(raw[i]);
  }
  return out;
}

void append_remapped_donor_binder(const PackedHostData& packed,
                                  const GpuReproChildView& copyback,
                                  int child_index,
                                  const std::vector<std::uint64_t>& donor_names,
                                  const PlainLinearRecBinders& binders,
                                  int target_start,
                                  int donor_start,
                                  AstProgram* out) {
  const int elem_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.elem_name, out);
  const int accum_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.accum_name, out);
  const int index_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.index_name, out);
  if (elem_name < 0 || accum_name < 0 || index_name < 0) {
    return;
  }
  out->linear_rec_binders.push_back(LinearRecBinders{
      static_cast<std::size_t>(target_start + (binders.node_index - donor_start)),
      elem_name,
      accum_name,
      index_name,
  });
}

void append_shifted_base_asgp_dc(const PlainAsgpDcBinders& binders,
                                 int removed,
                                 int inserted,
                                 const CandidateRange& base_range,
                                 AstProgram* out) {
  if (binders.node_index >= base_range.start && binders.node_index < base_range.stop) {
    return;
  }
  AsgpDcBinders shifted{
      static_cast<std::size_t>(binders.node_index),
      binders.solve_xs_name,
      binders.solve_n_name,
      binders.solve_lo_name,
      binders.divide_n_name,
      binders.combine_left_name,
      binders.combine_right_name,
  };
  if (binders.node_index >= base_range.stop) {
    shifted.node_index = static_cast<std::size_t>(binders.node_index - removed + inserted);
  }
  out->asgp_dc_binders.push_back(shifted);
}

void append_shifted_base_asgp_dp1d(const PlainAsgpDp1dSpec& spec,
                                   int removed,
                                   int inserted,
                                   const CandidateRange& base_range,
                                   AstProgram* out) {
  if (spec.node_index >= base_range.start && spec.node_index < base_range.stop) {
    return;
  }
  AsgpDp1dSpec shifted{
      static_cast<std::size_t>(spec.node_index),
      spec.lo,
      spec.hi,
      spec.base_state,
      spec.boundary_const,
      static_cast<NodeKind>(spec.dep_kind),
      decode_plain_ints(spec.dep_offsets, spec.dep_offset_count, 3),
      spec.solve_state_name,
      spec.transition_state_name,
      decode_plain_ints(spec.transition_dep_names, spec.transition_dep_count, 3),
  };
  if (spec.node_index >= base_range.stop) {
    shifted.node_index = static_cast<std::size_t>(spec.node_index - removed + inserted);
  }
  out->asgp_dp1d_specs.push_back(std::move(shifted));
}

void append_shifted_base_asgp_dp2d(const PlainAsgpDp2dSpec& spec,
                                   int removed,
                                   int inserted,
                                   const CandidateRange& base_range,
                                   AstProgram* out) {
  if (spec.node_index >= base_range.start && spec.node_index < base_range.stop) {
    return;
  }
  AsgpDp2dSpec shifted{
      static_cast<std::size_t>(spec.node_index),
      spec.i_lo,
      spec.i_hi,
      spec.j_lo,
      spec.j_hi,
      spec.base_i,
      spec.base_j,
      spec.boundary_const,
      static_cast<NodeKind>(spec.dep_kind),
      spec.solve_i_name,
      spec.solve_j_name,
      spec.transition_i_name,
      spec.transition_j_name,
      decode_plain_ints(spec.transition_dep_names, spec.transition_dep_count, 4),
  };
  if (spec.node_index >= base_range.stop) {
    shifted.node_index = static_cast<std::size_t>(spec.node_index - removed + inserted);
  }
  out->asgp_dp2d_specs.push_back(std::move(shifted));
}

void append_remapped_donor_asgp_dc(const PackedHostData& packed,
                                   const GpuReproChildView& copyback,
                                   int child_index,
                                   const std::vector<std::uint64_t>& donor_names,
                                   const PlainAsgpDcBinders& binders,
                                   int target_start,
                                   int donor_start,
                                   AstProgram* out) {
  const int solve_xs_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.solve_xs_name, out);
  const int solve_n_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.solve_n_name, out);
  const int solve_lo_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.solve_lo_name, out);
  const int divide_n_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.divide_n_name, out);
  const int combine_left_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.combine_left_name, out);
  const int combine_right_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, binders.combine_right_name, out);
  if (solve_xs_name < 0 || solve_n_name < 0 || solve_lo_name < 0 || divide_n_name < 0 ||
      combine_left_name < 0 || combine_right_name < 0) {
    return;
  }
  out->asgp_dc_binders.push_back(AsgpDcBinders{
      static_cast<std::size_t>(target_start + (binders.node_index - donor_start)),
      solve_xs_name,
      solve_n_name,
      solve_lo_name,
      divide_n_name,
      combine_left_name,
      combine_right_name,
  });
}

void append_remapped_donor_asgp_dp1d(const PackedHostData& packed,
                                     const GpuReproChildView& copyback,
                                     int child_index,
                                     const std::vector<std::uint64_t>& donor_names,
                                     const std::vector<Value>& donor_consts,
                                     const PlainAsgpDp1dSpec& spec,
                                     int target_start,
                                     int donor_start,
                                     AstProgram* out) {
  const int boundary_const = remap_source_const_to_child(donor_consts, spec.boundary_const, out);
  const int solve_state_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, spec.solve_state_name, out);
  const int transition_state_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, spec.transition_state_name, out);
  std::vector<int> transition_dep_names;
  for (int raw_name : decode_plain_ints(spec.transition_dep_names, spec.transition_dep_count, 3)) {
    const int mapped = remap_source_name_to_child(packed, copyback, child_index, donor_names, raw_name, out);
    if (mapped < 0) return;
    transition_dep_names.push_back(mapped);
  }
  if (boundary_const < 0 || solve_state_name < 0 || transition_state_name < 0) {
    return;
  }
  out->asgp_dp1d_specs.push_back(AsgpDp1dSpec{
      static_cast<std::size_t>(target_start + (spec.node_index - donor_start)),
      spec.lo,
      spec.hi,
      spec.base_state,
      boundary_const,
      static_cast<NodeKind>(spec.dep_kind),
      decode_plain_ints(spec.dep_offsets, spec.dep_offset_count, 3),
      solve_state_name,
      transition_state_name,
      std::move(transition_dep_names),
  });
}

void append_remapped_donor_asgp_dp2d(const PackedHostData& packed,
                                     const GpuReproChildView& copyback,
                                     int child_index,
                                     const std::vector<std::uint64_t>& donor_names,
                                     const std::vector<Value>& donor_consts,
                                     const PlainAsgpDp2dSpec& spec,
                                     int target_start,
                                     int donor_start,
                                     AstProgram* out) {
  const int boundary_const = remap_source_const_to_child(donor_consts, spec.boundary_const, out);
  const int solve_i_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, spec.solve_i_name, out);
  const int solve_j_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, spec.solve_j_name, out);
  const int transition_i_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, spec.transition_i_name, out);
  const int transition_j_name = remap_source_name_to_child(
      packed, copyback, child_index, donor_names, spec.transition_j_name, out);
  std::vector<int> transition_dep_names;
  for (int raw_name : decode_plain_ints(spec.transition_dep_names, spec.transition_dep_count, 4)) {
    const int mapped = remap_source_name_to_child(packed, copyback, child_index, donor_names, raw_name, out);
    if (mapped < 0) return;
    transition_dep_names.push_back(mapped);
  }
  if (boundary_const < 0 || solve_i_name < 0 || solve_j_name < 0 ||
      transition_i_name < 0 || transition_j_name < 0) {
    return;
  }
  out->asgp_dp2d_specs.push_back(AsgpDp2dSpec{
      static_cast<std::size_t>(target_start + (spec.node_index - donor_start)),
      spec.i_lo,
      spec.i_hi,
      spec.j_lo,
      spec.j_hi,
      spec.base_i,
      spec.base_j,
      boundary_const,
      static_cast<NodeKind>(spec.dep_kind),
      solve_i_name,
      solve_j_name,
      transition_i_name,
      transition_j_name,
      std::move(transition_dep_names),
  });
}

void rebuild_child_linear_rec_binders(const PackedHostData& packed,
                                      const GpuReproChildView& copyback,
                                      int child_index,
                                      AstProgram* out) {
  const int pair_index = child_index / 2;
  const bool child_is_a = (child_index & 1) == 0;
  const int base_parent = child_is_a ? copyback.parent_a[pair_index] : copyback.parent_b[pair_index];
  const int donor_parent = child_is_a ? copyback.parent_b[pair_index] : copyback.parent_a[pair_index];
  const int base_cand_index = child_is_a ? copyback.cand_a[pair_index] : copyback.cand_b[pair_index];
  const int donor_cand_index = child_is_a ? copyback.cand_b[pair_index] : copyback.cand_a[pair_index];
  if (base_parent < 0 || donor_parent < 0 || base_cand_index < 0 || donor_cand_index < 0) {
    return;
  }
  const CandidateRange base_range =
      packed.candidates[static_cast<std::size_t>(base_parent * copyback.config.candidates_per_program +
                                                 base_cand_index)];
  const CandidateRange donor_range =
      packed.candidates[static_cast<std::size_t>(donor_parent * copyback.config.candidates_per_program +
                                                 donor_cand_index)];
  if (base_range.start < 0 || base_range.stop <= base_range.start) {
    return;
  }

  const int removed = base_range.stop - base_range.start;
  const bool subtree_mutation = child_uses_subtree_mutation(copyback.config, child_index);
  int donor_pool_index = -1;
  int inserted = std::max(0, donor_range.stop - donor_range.start);
  if (subtree_mutation) {
    donor_pool_index = donor_index_for_child(copyback.config, base_range, child_index);
    if (donor_pool_index < 0 ||
        static_cast<std::size_t>(donor_pool_index) >= packed.donor_lens.size()) {
      return;
    }
    inserted = packed.donor_lens[static_cast<std::size_t>(donor_pool_index)];
  }
  const int base_count = packed.metas[static_cast<std::size_t>(base_parent)].linear_rec_count;
  const std::size_t base_binder_base =
      static_cast<std::size_t>(base_parent * copyback.config.max_linear_rec_binders);

  for (int i = 0; i < base_count; ++i) {
    const PlainLinearRecBinders& binders =
        packed.program_linear_rec_binders[base_binder_base + static_cast<std::size_t>(i)];
    if (binders.node_index >= base_range.start && binders.node_index < base_range.stop) {
      continue;
    }
    LinearRecBinders shifted{
        static_cast<std::size_t>(binders.node_index),
        binders.elem_name,
        binders.accum_name,
        binders.index_name,
    };
    if (binders.node_index >= base_range.stop) {
      shifted.node_index = static_cast<std::size_t>(binders.node_index - removed + inserted);
    }
    out->linear_rec_binders.push_back(shifted);
  }

  if (subtree_mutation) {
    const int donor_count = packed.donor_linear_rec_counts[static_cast<std::size_t>(donor_pool_index)];
    if (donor_count <= 0) {
      return;
    }
    std::vector<std::uint64_t> donor_names;
    const int donor_name_count = packed.donor_name_counts[static_cast<std::size_t>(donor_pool_index)];
    const std::size_t donor_name_base = static_cast<std::size_t>(donor_pool_index * copyback.config.max_names);
    donor_names.reserve(static_cast<std::size_t>(donor_name_count));
    for (int i = 0; i < donor_name_count; ++i) {
      donor_names.push_back(packed.donor_name_ids[donor_name_base + static_cast<std::size_t>(i)]);
    }
    const std::size_t donor_binder_base =
        static_cast<std::size_t>(donor_pool_index * copyback.config.max_linear_rec_binders);
    for (int i = 0; i < donor_count; ++i) {
      const PlainLinearRecBinders& binders =
          packed.donor_linear_rec_binders[donor_binder_base + static_cast<std::size_t>(i)];
      append_remapped_donor_binder(packed, copyback, child_index, donor_names, binders,
                                   base_range.start, 0, out);
    }
    return;
  }

  if (donor_range.start < 0 || donor_range.stop <= donor_range.start) {
    return;
  }

  std::vector<std::uint64_t> donor_names;
  const int donor_name_count = packed.metas[static_cast<std::size_t>(donor_parent)].name_count;
  const std::size_t donor_name_base = static_cast<std::size_t>(donor_parent * copyback.config.max_names);
  donor_names.reserve(static_cast<std::size_t>(donor_name_count));
  for (int i = 0; i < donor_name_count; ++i) {
    donor_names.push_back(packed.program_name_ids[donor_name_base + static_cast<std::size_t>(i)]);
  }

  const int donor_count = packed.metas[static_cast<std::size_t>(donor_parent)].linear_rec_count;
  const std::size_t donor_binder_base =
      static_cast<std::size_t>(donor_parent * copyback.config.max_linear_rec_binders);
  for (int i = 0; i < donor_count; ++i) {
    const PlainLinearRecBinders& binders =
        packed.program_linear_rec_binders[donor_binder_base + static_cast<std::size_t>(i)];
    if (binders.node_index < donor_range.start || binders.node_index >= donor_range.stop) {
      continue;
    }
    append_remapped_donor_binder(packed, copyback, child_index, donor_names, binders,
                                 base_range.start, donor_range.start, out);
  }
}

std::vector<std::uint64_t> program_source_names(const PackedHostData& packed,
                                                const GpuReproChildView& copyback,
                                                int program_index) {
  std::vector<std::uint64_t> out;
  const int name_count = packed.metas[static_cast<std::size_t>(program_index)].name_count;
  const std::size_t name_base = static_cast<std::size_t>(program_index * copyback.config.max_names);
  out.reserve(static_cast<std::size_t>(name_count));
  for (int i = 0; i < name_count; ++i) {
    const std::size_t idx = name_base + static_cast<std::size_t>(i);
    if (idx >= packed.program_name_ids.size()) break;
    out.push_back(packed.program_name_ids[idx]);
  }
  return out;
}

std::vector<Value> program_source_consts(const PackedHostData& packed,
                                         const GpuReproChildView& copyback,
                                         int program_index) {
  std::vector<Value> out;
  const int const_count = packed.metas[static_cast<std::size_t>(program_index)].const_count;
  const std::size_t const_base = static_cast<std::size_t>(program_index * copyback.config.max_consts);
  out.reserve(static_cast<std::size_t>(const_count));
  for (int i = 0; i < const_count; ++i) {
    const std::size_t idx = const_base + static_cast<std::size_t>(i);
    if (idx >= packed.program_consts.size()) break;
    out.push_back(packed.program_consts[idx]);
  }
  return out;
}

std::vector<std::uint64_t> donor_source_names(const PackedHostData& packed,
                                              const GpuReproChildView& copyback,
                                              int donor_index) {
  std::vector<std::uint64_t> out;
  const int name_count = packed.donor_name_counts[static_cast<std::size_t>(donor_index)];
  const std::size_t name_base = static_cast<std::size_t>(donor_index * copyback.config.max_names);
  out.reserve(static_cast<std::size_t>(name_count));
  for (int i = 0; i < name_count; ++i) {
    const std::size_t idx = name_base + static_cast<std::size_t>(i);
    if (idx >= packed.donor_name_ids.size()) break;
    out.push_back(packed.donor_name_ids[idx]);
  }
  return out;
}

std::vector<Value> donor_source_consts(const PackedHostData& packed,
                                       const GpuReproChildView& copyback,
                                       int donor_index) {
  std::vector<Value> out;
  const int const_count = packed.donor_const_counts[static_cast<std::size_t>(donor_index)];
  const std::size_t const_base = static_cast<std::size_t>(donor_index * copyback.config.max_consts);
  out.reserve(static_cast<std::size_t>(const_count));
  for (int i = 0; i < const_count; ++i) {
    const std::size_t idx = const_base + static_cast<std::size_t>(i);
    if (idx >= packed.donor_consts.size()) break;
    out.push_back(packed.donor_consts[idx]);
  }
  return out;
}

void rebuild_child_asgp_metadata(const PackedHostData& packed,
                                 const GpuReproChildView& copyback,
                                 int child_index,
                                 AstProgram* out) {
  const int pair_index = child_index / 2;
  const bool child_is_a = (child_index & 1) == 0;
  const int base_parent = child_is_a ? copyback.parent_a[pair_index] : copyback.parent_b[pair_index];
  const int donor_parent = child_is_a ? copyback.parent_b[pair_index] : copyback.parent_a[pair_index];
  const int base_cand_index = child_is_a ? copyback.cand_a[pair_index] : copyback.cand_b[pair_index];
  const int donor_cand_index = child_is_a ? copyback.cand_b[pair_index] : copyback.cand_a[pair_index];
  if (base_parent < 0 || donor_parent < 0 || base_cand_index < 0 || donor_cand_index < 0) {
    return;
  }
  const CandidateRange base_range =
      packed.candidates[static_cast<std::size_t>(base_parent * copyback.config.candidates_per_program +
                                                 base_cand_index)];
  const CandidateRange donor_range =
      packed.candidates[static_cast<std::size_t>(donor_parent * copyback.config.candidates_per_program +
                                                 donor_cand_index)];
  if (base_range.start < 0 || base_range.stop <= base_range.start) {
    return;
  }

  const int removed = base_range.stop - base_range.start;
  const bool subtree_mutation = child_uses_subtree_mutation(copyback.config, child_index);
  int donor_pool_index = -1;
  int inserted = std::max(0, donor_range.stop - donor_range.start);
  if (subtree_mutation) {
    donor_pool_index = donor_index_for_child(copyback.config, base_range, child_index);
    if (donor_pool_index < 0 ||
        static_cast<std::size_t>(donor_pool_index) >= packed.donor_lens.size()) {
      return;
    }
    inserted = packed.donor_lens[static_cast<std::size_t>(donor_pool_index)];
  }

  const PackedProgramMeta& base_meta = packed.metas[static_cast<std::size_t>(base_parent)];
  const std::size_t base_dc_base =
      static_cast<std::size_t>(base_parent * copyback.config.max_asgp_dc_binders);
  for (int i = 0; i < base_meta.asgp_dc_count; ++i) {
    append_shifted_base_asgp_dc(
        packed.program_asgp_dc_binders[base_dc_base + static_cast<std::size_t>(i)],
        removed, inserted, base_range, out);
  }
  const std::size_t base_dp1_base =
      static_cast<std::size_t>(base_parent * copyback.config.max_asgp_dp1d_specs);
  for (int i = 0; i < base_meta.asgp_dp1d_count; ++i) {
    append_shifted_base_asgp_dp1d(
        packed.program_asgp_dp1d_specs[base_dp1_base + static_cast<std::size_t>(i)],
        removed, inserted, base_range, out);
  }
  const std::size_t base_dp2_base =
      static_cast<std::size_t>(base_parent * copyback.config.max_asgp_dp2d_specs);
  for (int i = 0; i < base_meta.asgp_dp2d_count; ++i) {
    append_shifted_base_asgp_dp2d(
        packed.program_asgp_dp2d_specs[base_dp2_base + static_cast<std::size_t>(i)],
        removed, inserted, base_range, out);
  }

  if (subtree_mutation) {
    const std::vector<std::uint64_t> donor_names = donor_source_names(packed, copyback, donor_pool_index);
    const std::vector<Value> donor_consts = donor_source_consts(packed, copyback, donor_pool_index);
    const std::size_t donor_dc_base =
        static_cast<std::size_t>(donor_pool_index * copyback.config.max_asgp_dc_binders);
    for (int i = 0; i < packed.donor_asgp_dc_counts[static_cast<std::size_t>(donor_pool_index)]; ++i) {
      const PlainAsgpDcBinders& binders =
          packed.donor_asgp_dc_binders[donor_dc_base + static_cast<std::size_t>(i)];
      append_remapped_donor_asgp_dc(
          packed, copyback, child_index, donor_names, binders, base_range.start, 0, out);
    }
    const std::size_t donor_dp1_base =
        static_cast<std::size_t>(donor_pool_index * copyback.config.max_asgp_dp1d_specs);
    for (int i = 0; i < packed.donor_asgp_dp1d_counts[static_cast<std::size_t>(donor_pool_index)]; ++i) {
      const PlainAsgpDp1dSpec& spec =
          packed.donor_asgp_dp1d_specs[donor_dp1_base + static_cast<std::size_t>(i)];
      append_remapped_donor_asgp_dp1d(
          packed, copyback, child_index, donor_names, donor_consts, spec, base_range.start, 0, out);
    }
    const std::size_t donor_dp2_base =
        static_cast<std::size_t>(donor_pool_index * copyback.config.max_asgp_dp2d_specs);
    for (int i = 0; i < packed.donor_asgp_dp2d_counts[static_cast<std::size_t>(donor_pool_index)]; ++i) {
      const PlainAsgpDp2dSpec& spec =
          packed.donor_asgp_dp2d_specs[donor_dp2_base + static_cast<std::size_t>(i)];
      append_remapped_donor_asgp_dp2d(
          packed, copyback, child_index, donor_names, donor_consts, spec, base_range.start, 0, out);
    }
    return;
  }

  if (donor_range.start < 0 || donor_range.stop <= donor_range.start) {
    return;
  }
  const std::vector<std::uint64_t> donor_names = program_source_names(packed, copyback, donor_parent);
  const std::vector<Value> donor_consts = program_source_consts(packed, copyback, donor_parent);
  const PackedProgramMeta& donor_meta = packed.metas[static_cast<std::size_t>(donor_parent)];
  const std::size_t donor_dc_base =
      static_cast<std::size_t>(donor_parent * copyback.config.max_asgp_dc_binders);
  for (int i = 0; i < donor_meta.asgp_dc_count; ++i) {
    const PlainAsgpDcBinders& binders =
        packed.program_asgp_dc_binders[donor_dc_base + static_cast<std::size_t>(i)];
    if (binders.node_index < donor_range.start || binders.node_index >= donor_range.stop) {
      continue;
    }
    append_remapped_donor_asgp_dc(
        packed, copyback, child_index, donor_names, binders, base_range.start, donor_range.start, out);
  }
  const std::size_t donor_dp1_base =
      static_cast<std::size_t>(donor_parent * copyback.config.max_asgp_dp1d_specs);
  for (int i = 0; i < donor_meta.asgp_dp1d_count; ++i) {
    const PlainAsgpDp1dSpec& spec =
        packed.program_asgp_dp1d_specs[donor_dp1_base + static_cast<std::size_t>(i)];
    if (spec.node_index < donor_range.start || spec.node_index >= donor_range.stop) {
      continue;
    }
    append_remapped_donor_asgp_dp1d(
        packed, copyback, child_index, donor_names, donor_consts, spec, base_range.start, donor_range.start, out);
  }
  const std::size_t donor_dp2_base =
      static_cast<std::size_t>(donor_parent * copyback.config.max_asgp_dp2d_specs);
  for (int i = 0; i < donor_meta.asgp_dp2d_count; ++i) {
    const PlainAsgpDp2dSpec& spec =
        packed.program_asgp_dp2d_specs[donor_dp2_base + static_cast<std::size_t>(i)];
    if (spec.node_index < donor_range.start || spec.node_index >= donor_range.stop) {
      continue;
    }
    append_remapped_donor_asgp_dp2d(
        packed, copyback, child_index, donor_names, donor_consts, spec, base_range.start, donor_range.start, out);
  }
}

ProgramGenome decode_one_child(const PackedHostData& packed,
                               const GpuReproChildView& copyback,
                               int child_index) {
  ProgramGenome out;
  out.ast.version = k_ast_prefix_version_current;
  const int used_len = copyback.child_used_len[static_cast<std::size_t>(child_index)];
  const int name_count = copyback.child_name_counts[static_cast<std::size_t>(child_index)];
  const int const_count = copyback.child_const_counts[static_cast<std::size_t>(child_index)];
  if (used_len <= 0 || used_len > copyback.config.max_nodes) {
    return out;
  }
  if (name_count < 0 || name_count > copyback.config.max_names || const_count < 0 ||
      const_count > copyback.config.max_consts) {
    return out;
  }

  out.ast.nodes.reserve(static_cast<std::size_t>(used_len));
  out.ast.names.reserve(static_cast<std::size_t>(name_count));
  out.ast.consts.reserve(static_cast<std::size_t>(const_count));

  const std::size_t node_base = static_cast<std::size_t>(copyback.child_node_offsets[child_index]);
  const std::size_t name_base = static_cast<std::size_t>(copyback.child_name_offsets[child_index]);
  const std::size_t const_base = static_cast<std::size_t>(copyback.child_const_offsets[child_index]);

  for (int i = 0; i < name_count; ++i) {
    const std::uint64_t id = copyback.child_name_ids[name_base + static_cast<std::size_t>(i)];
    auto it = packed.name_lookup.find(id);
    if (it == packed.name_lookup.end()) {
      out.ast.nodes.clear();
      return out;
    }
    out.ast.names.push_back(it->second);
  }
  for (int i = 0; i < const_count; ++i) {
    out.ast.consts.push_back(copyback.child_consts[const_base + static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < used_len; ++i) {
    const PlainNode& n = copyback.child_nodes[node_base + static_cast<std::size_t>(i)];
    out.ast.nodes.push_back(AstNode{static_cast<NodeKind>(n.kind), n.i0, n.i1});
  }
  rebuild_child_linear_rec_binders(packed, copyback, child_index, &out.ast);
  rebuild_child_asgp_metadata(packed, copyback, child_index, &out.ast);
  if (!payloads_available_for_consts(out.ast.consts)) {
    out.ast.nodes.clear();
    return out;
  }
  const PackedChildMeta& meta = copyback.child_meta[static_cast<std::size_t>(child_index)];
  out.meta.node_count = meta.node_count;
  out.meta.max_depth = meta.max_depth;
  out.meta.uses_builtins = meta.uses_builtins != 0;
  out.meta.program_key = ast_cache_key(out.ast);
  return out;
}

const ProgramGenome& fallback_parent_for_child(const std::vector<ScoredGenome>& scored,
                                               const GpuReproChildView& selection,
                                               int child_index) {
  const int pair_index = child_index / 2;
  if ((child_index & 1) == 0) {
    return scored[static_cast<std::size_t>(selection.parent_a[static_cast<std::size_t>(pair_index)])].genome;
  }
  return scored[static_cast<std::size_t>(selection.parent_b[static_cast<std::size_t>(pair_index)])].genome;
}

const ProgramGenome& fallback_parent_for_child(const std::vector<ScoredGenomeRef>& scored,
                                               const GpuReproChildView& selection,
                                               int child_index) {
  const int pair_index = child_index / 2;
  if ((child_index & 1) == 0) {
    return *scored[static_cast<std::size_t>(selection.parent_a[static_cast<std::size_t>(pair_index)])].genome;
  }
  return *scored[static_cast<std::size_t>(selection.parent_b[static_cast<std::size_t>(pair_index)])].genome;
}

}  // namespace

ProgramGenome compact_genome_tables(const ProgramGenome& genome) {
  ProgramGenome out;
  out.ast.version = genome.ast.version;
  out.ast.nodes = genome.ast.nodes;

  std::vector<int> name_map(genome.ast.names.size(), -1);
  std::vector<int> const_map(genome.ast.consts.size(), -1);

  auto map_name = [&](int old_index, int* new_index) -> bool {
    if (old_index < 0 || static_cast<std::size_t>(old_index) >= genome.ast.names.size()) {
      return false;
    }
    int& mapped = name_map[static_cast<std::size_t>(old_index)];
    if (mapped < 0) {
      mapped = static_cast<int>(out.ast.names.size());
      out.ast.names.push_back(genome.ast.names[static_cast<std::size_t>(old_index)]);
    }
    *new_index = mapped;
    return true;
  };

  auto map_const = [&](int old_index, int* new_index) -> bool {
    if (old_index < 0 || static_cast<std::size_t>(old_index) >= genome.ast.consts.size()) {
      return false;
    }
    int& mapped = const_map[static_cast<std::size_t>(old_index)];
    if (mapped < 0) {
      mapped = static_cast<int>(out.ast.consts.size());
      out.ast.consts.push_back(genome.ast.consts[static_cast<std::size_t>(old_index)]);
    }
    *new_index = mapped;
    return true;
  };

  for (AstNode& node : out.ast.nodes) {
    if (node.kind == NodeKind::CONST) {
      if (!map_const(node.i0, &node.i0)) {
        ProgramGenome rebuilt = genome;
        rebuilt.meta = build_genome_meta(rebuilt.ast);
        return rebuilt;
      }
    } else if (node.kind == NodeKind::VAR || node.kind == NodeKind::BOUND_VAR ||
               node.kind == NodeKind::ASSIGN || node.kind == NodeKind::FOR_RANGE ||
               node.kind == NodeKind::MAP_LIST || node.kind == NodeKind::FILTER_LIST) {
      if (!map_name(node.i0, &node.i0)) {
        ProgramGenome rebuilt = genome;
        rebuilt.meta = build_genome_meta(rebuilt.ast);
        return rebuilt;
      }
    }
  }
  for (const LinearRecBinders& binders : genome.ast.linear_rec_binders) {
    int elem_name = 0;
    int accum_name = 0;
    int index_name = 0;
    if (!map_name(binders.elem_name, &elem_name) ||
        !map_name(binders.accum_name, &accum_name) ||
        !map_name(binders.index_name, &index_name)) {
      ProgramGenome rebuilt = genome;
      rebuilt.meta = build_genome_meta(rebuilt.ast);
      return rebuilt;
    }
    out.ast.linear_rec_binders.push_back(
        LinearRecBinders{binders.node_index, elem_name, accum_name, index_name});
  }
  for (const AsgpDcBinders& binders : genome.ast.asgp_dc_binders) {
    int solve_xs_name = 0;
    int solve_n_name = 0;
    int solve_lo_name = 0;
    int divide_n_name = 0;
    int combine_left_name = 0;
    int combine_right_name = 0;
    if (!map_name(binders.solve_xs_name, &solve_xs_name) ||
        !map_name(binders.solve_n_name, &solve_n_name) ||
        !map_name(binders.solve_lo_name, &solve_lo_name) ||
        !map_name(binders.divide_n_name, &divide_n_name) ||
        !map_name(binders.combine_left_name, &combine_left_name) ||
        !map_name(binders.combine_right_name, &combine_right_name)) {
      ProgramGenome rebuilt = genome;
      rebuilt.meta = build_genome_meta(rebuilt.ast);
      return rebuilt;
    }
    out.ast.asgp_dc_binders.push_back(AsgpDcBinders{
        binders.node_index,
        solve_xs_name,
        solve_n_name,
        solve_lo_name,
        divide_n_name,
        combine_left_name,
        combine_right_name,
    });
  }
  for (const AsgpDp1dSpec& spec : genome.ast.asgp_dp1d_specs) {
    int boundary_const = 0;
    int solve_state_name = 0;
    int transition_state_name = 0;
    if (!map_const(spec.boundary_const, &boundary_const) ||
        !map_name(spec.solve_state_name, &solve_state_name) ||
        !map_name(spec.transition_state_name, &transition_state_name)) {
      ProgramGenome rebuilt = genome;
      rebuilt.meta = build_genome_meta(rebuilt.ast);
      return rebuilt;
    }
    std::vector<int> transition_dep_names;
    transition_dep_names.reserve(spec.transition_dep_names.size());
    for (int name : spec.transition_dep_names) {
      int mapped = 0;
      if (!map_name(name, &mapped)) {
        ProgramGenome rebuilt = genome;
        rebuilt.meta = build_genome_meta(rebuilt.ast);
        return rebuilt;
      }
      transition_dep_names.push_back(mapped);
    }
    out.ast.asgp_dp1d_specs.push_back(AsgpDp1dSpec{
        spec.node_index,
        spec.lo,
        spec.hi,
        spec.base_state,
        boundary_const,
        spec.dep_kind,
        spec.dep_offsets,
        solve_state_name,
        transition_state_name,
        std::move(transition_dep_names),
    });
  }
  for (const AsgpDp2dSpec& spec : genome.ast.asgp_dp2d_specs) {
    int boundary_const = 0;
    int solve_i_name = 0;
    int solve_j_name = 0;
    int transition_i_name = 0;
    int transition_j_name = 0;
    if (!map_const(spec.boundary_const, &boundary_const) ||
        !map_name(spec.solve_i_name, &solve_i_name) ||
        !map_name(spec.solve_j_name, &solve_j_name) ||
        !map_name(spec.transition_i_name, &transition_i_name) ||
        !map_name(spec.transition_j_name, &transition_j_name)) {
      ProgramGenome rebuilt = genome;
      rebuilt.meta = build_genome_meta(rebuilt.ast);
      return rebuilt;
    }
    std::vector<int> transition_dep_names;
    transition_dep_names.reserve(spec.transition_dep_names.size());
    for (int name : spec.transition_dep_names) {
      int mapped = 0;
      if (!map_name(name, &mapped)) {
        ProgramGenome rebuilt = genome;
        rebuilt.meta = build_genome_meta(rebuilt.ast);
        return rebuilt;
      }
      transition_dep_names.push_back(mapped);
    }
    out.ast.asgp_dp2d_specs.push_back(AsgpDp2dSpec{
        spec.node_index,
        spec.i_lo,
        spec.i_hi,
        spec.j_lo,
        spec.j_hi,
        spec.base_i,
        spec.base_j,
        boundary_const,
        spec.dep_kind,
        solve_i_name,
        solve_j_name,
        transition_i_name,
        transition_j_name,
        std::move(transition_dep_names),
    });
  }

  out.meta = build_genome_meta(out.ast);
  return out;
}

std::vector<ProgramGenome> compact_population_tables(const std::vector<ProgramGenome>& population) {
  std::vector<ProgramGenome> out;
  out.reserve(population.size());
  for (const ProgramGenome& genome : population) {
    out.push_back(compact_genome_tables(genome));
  }
  return out;
}

PlainAsgpDcBinders encode_asgp_dc_binders(const AsgpDcBinders& binders) {
  return PlainAsgpDcBinders{
      static_cast<int>(binders.node_index),
      binders.solve_xs_name,
      binders.solve_n_name,
      binders.solve_lo_name,
      binders.divide_n_name,
      binders.combine_left_name,
      binders.combine_right_name,
  };
}

PlainAsgpDp1dSpec encode_asgp_dp1d_spec(const AsgpDp1dSpec& spec) {
  PlainAsgpDp1dSpec out;
  out.node_index = static_cast<int>(spec.node_index);
  out.lo = spec.lo;
  out.hi = spec.hi;
  out.base_state = spec.base_state;
  out.boundary_const = spec.boundary_const;
  out.dep_kind = static_cast<int>(spec.dep_kind);
  out.dep_offset_count = std::min<int>(static_cast<int>(spec.dep_offsets.size()), 3);
  for (int i = 0; i < out.dep_offset_count; ++i) {
    out.dep_offsets[i] = spec.dep_offsets[static_cast<std::size_t>(i)];
  }
  out.solve_state_name = spec.solve_state_name;
  out.transition_state_name = spec.transition_state_name;
  out.transition_dep_count = std::min<int>(static_cast<int>(spec.transition_dep_names.size()), 3);
  for (int i = 0; i < out.transition_dep_count; ++i) {
    out.transition_dep_names[i] = spec.transition_dep_names[static_cast<std::size_t>(i)];
  }
  return out;
}

PlainAsgpDp2dSpec encode_asgp_dp2d_spec(const AsgpDp2dSpec& spec) {
  PlainAsgpDp2dSpec out;
  out.node_index = static_cast<int>(spec.node_index);
  out.i_lo = spec.i_lo;
  out.i_hi = spec.i_hi;
  out.j_lo = spec.j_lo;
  out.j_hi = spec.j_hi;
  out.base_i = spec.base_i;
  out.base_j = spec.base_j;
  out.boundary_const = spec.boundary_const;
  out.dep_kind = static_cast<int>(spec.dep_kind);
  out.solve_i_name = spec.solve_i_name;
  out.solve_j_name = spec.solve_j_name;
  out.transition_i_name = spec.transition_i_name;
  out.transition_j_name = spec.transition_j_name;
  out.transition_dep_count = std::min<int>(static_cast<int>(spec.transition_dep_names.size()), 4);
  for (int i = 0; i < out.transition_dep_count; ++i) {
    out.transition_dep_names[i] = spec.transition_dep_names[static_cast<std::size_t>(i)];
  }
  return out;
}

PackedHostData pack_population(const std::vector<ProgramGenome>& population,
                               const PreprocessOutput& prep,
                               const GpuReproConfig& config) {
  PackedHostData out;
  out.config = config;
  const std::size_t total_donor_count = prep.donor_pool.size();
  out.program_nodes.resize(static_cast<std::size_t>(config.population_size * config.max_nodes));
  out.metas.resize(static_cast<std::size_t>(config.population_size));
  out.candidates.resize(static_cast<std::size_t>(config.population_size * config.candidates_per_program));
  out.program_name_ids.resize(static_cast<std::size_t>(config.population_size * config.max_names), 0ULL);
  out.program_consts.resize(static_cast<std::size_t>(config.population_size * config.max_consts), Value::invalid());
  out.program_linear_rec_binders.resize(
      static_cast<std::size_t>(config.population_size * config.max_linear_rec_binders));
  out.program_asgp_dc_binders.resize(
      static_cast<std::size_t>(config.population_size * config.max_asgp_dc_binders));
  out.program_asgp_dp1d_specs.resize(
      static_cast<std::size_t>(config.population_size * config.max_asgp_dp1d_specs));
  out.program_asgp_dp2d_specs.resize(
      static_cast<std::size_t>(config.population_size * config.max_asgp_dp2d_specs));
  out.donor_nodes.resize(total_donor_count * static_cast<std::size_t>(config.max_donor_nodes));
  out.donor_lens.resize(total_donor_count, 0);
  out.donor_name_ids.resize(total_donor_count * static_cast<std::size_t>(config.max_names), 0ULL);
  out.donor_name_counts.resize(total_donor_count, 0);
  out.donor_consts.resize(total_donor_count * static_cast<std::size_t>(config.max_consts), Value::invalid());
  out.donor_const_counts.resize(total_donor_count, 0);
  out.donor_linear_rec_binders.resize(
      total_donor_count * static_cast<std::size_t>(config.max_linear_rec_binders));
  out.donor_linear_rec_counts.resize(total_donor_count, 0);
  out.donor_asgp_dc_binders.resize(
      total_donor_count * static_cast<std::size_t>(config.max_asgp_dc_binders));
  out.donor_asgp_dc_counts.resize(total_donor_count, 0);
  out.donor_asgp_dp1d_specs.resize(
      total_donor_count * static_cast<std::size_t>(config.max_asgp_dp1d_specs));
  out.donor_asgp_dp1d_counts.resize(total_donor_count, 0);
  out.donor_asgp_dp2d_specs.resize(
      total_donor_count * static_cast<std::size_t>(config.max_asgp_dp2d_specs));
  out.donor_asgp_dp2d_counts.resize(total_donor_count, 0);

  for (int p = 0; p < config.population_size; ++p) {
    const ProgramGenome& genome = population[static_cast<std::size_t>(p)];
    const int used_len = std::min<int>(static_cast<int>(genome.ast.nodes.size()), config.max_nodes);
    const int name_count = std::min<int>(static_cast<int>(genome.ast.names.size()), config.max_names);
    const int const_count = std::min<int>(static_cast<int>(genome.ast.consts.size()), config.max_consts);
    const int linear_rec_count =
        std::min<int>(static_cast<int>(genome.ast.linear_rec_binders.size()), config.max_linear_rec_binders);
    const int asgp_dc_count =
        std::min<int>(static_cast<int>(genome.ast.asgp_dc_binders.size()), config.max_asgp_dc_binders);
    const int asgp_dp1d_count =
        std::min<int>(static_cast<int>(genome.ast.asgp_dp1d_specs.size()), config.max_asgp_dp1d_specs);
    const int asgp_dp2d_count =
        std::min<int>(static_cast<int>(genome.ast.asgp_dp2d_specs.size()), config.max_asgp_dp2d_specs);
    out.metas[static_cast<std::size_t>(p)] =
        PackedProgramMeta{used_len, name_count, const_count, linear_rec_count,
                          asgp_dc_count, asgp_dp1d_count, asgp_dp2d_count};
    const std::size_t node_base = static_cast<std::size_t>(p * config.max_nodes);
    const std::size_t name_base = static_cast<std::size_t>(p * config.max_names);
    const std::size_t const_base = static_cast<std::size_t>(p * config.max_consts);
    for (int i = 0; i < used_len; ++i) {
      const AstNode& n = genome.ast.nodes[static_cast<std::size_t>(i)];
      out.program_nodes[node_base + static_cast<std::size_t>(i)] =
          PlainNode{static_cast<int>(n.kind), n.i0, n.i1};
    }
    for (int i = 0; i < name_count; ++i) {
      register_name(&out, genome.ast.names[static_cast<std::size_t>(i)]);
      out.program_name_ids[name_base + static_cast<std::size_t>(i)] =
          hash_name(genome.ast.names[static_cast<std::size_t>(i)]);
    }
    for (int i = 0; i < const_count; ++i) {
      out.program_consts[const_base + static_cast<std::size_t>(i)] =
          genome.ast.consts[static_cast<std::size_t>(i)];
    }
    const std::size_t binder_base = static_cast<std::size_t>(p * config.max_linear_rec_binders);
    for (int i = 0; i < linear_rec_count; ++i) {
      const LinearRecBinders& binders = genome.ast.linear_rec_binders[static_cast<std::size_t>(i)];
      out.program_linear_rec_binders[binder_base + static_cast<std::size_t>(i)] =
          PlainLinearRecBinders{
              static_cast<int>(binders.node_index),
              binders.elem_name,
              binders.accum_name,
              binders.index_name,
          };
    }
    const std::size_t dc_base = static_cast<std::size_t>(p * config.max_asgp_dc_binders);
    for (int i = 0; i < asgp_dc_count; ++i) {
      out.program_asgp_dc_binders[dc_base + static_cast<std::size_t>(i)] =
          encode_asgp_dc_binders(genome.ast.asgp_dc_binders[static_cast<std::size_t>(i)]);
    }
    const std::size_t dp1_base = static_cast<std::size_t>(p * config.max_asgp_dp1d_specs);
    for (int i = 0; i < asgp_dp1d_count; ++i) {
      out.program_asgp_dp1d_specs[dp1_base + static_cast<std::size_t>(i)] =
          encode_asgp_dp1d_spec(genome.ast.asgp_dp1d_specs[static_cast<std::size_t>(i)]);
    }
    const std::size_t dp2_base = static_cast<std::size_t>(p * config.max_asgp_dp2d_specs);
    for (int i = 0; i < asgp_dp2d_count; ++i) {
      out.program_asgp_dp2d_specs[dp2_base + static_cast<std::size_t>(i)] =
          encode_asgp_dp2d_spec(genome.ast.asgp_dp2d_specs[static_cast<std::size_t>(i)]);
    }
  }

  for (int p = 0; p < config.population_size; ++p) {
    const std::vector<CandidateRange>& candidates = prep.candidates[static_cast<std::size_t>(p)];
    const CandidateRange fallback =
        candidates.empty() ? CandidateRange{0, 0, static_cast<int>(CandidateTag::Expr), static_cast<int>(RType::Invalid)}
                           : candidates.front();
    const std::size_t base = static_cast<std::size_t>(p * config.candidates_per_program);
    for (int i = 0; i < config.candidates_per_program; ++i) {
      out.candidates[base + static_cast<std::size_t>(i)] =
          candidates.empty() ? fallback : candidates[static_cast<std::size_t>(i % static_cast<int>(candidates.size()))];
    }
  }

  for (std::size_t i = 0; i < total_donor_count; ++i) {
    const DonorProgram& donor = prep.donor_pool[i];
    const int used_len = std::min<int>(static_cast<int>(donor.ast.nodes.size()), config.max_donor_nodes);
    const int name_count = std::min<int>(static_cast<int>(donor.ast.names.size()), config.max_names);
    const int const_count = std::min<int>(static_cast<int>(donor.ast.consts.size()), config.max_consts);
    out.donor_lens[i] = used_len;
    out.donor_name_counts[i] = name_count;
    out.donor_const_counts[i] = const_count;
    out.donor_linear_rec_counts[i] =
        std::min<int>(static_cast<int>(donor.ast.linear_rec_binders.size()), config.max_linear_rec_binders);
    out.donor_asgp_dc_counts[i] =
        std::min<int>(static_cast<int>(donor.ast.asgp_dc_binders.size()), config.max_asgp_dc_binders);
    out.donor_asgp_dp1d_counts[i] =
        std::min<int>(static_cast<int>(donor.ast.asgp_dp1d_specs.size()), config.max_asgp_dp1d_specs);
    out.donor_asgp_dp2d_counts[i] =
        std::min<int>(static_cast<int>(donor.ast.asgp_dp2d_specs.size()), config.max_asgp_dp2d_specs);
    const std::size_t node_base = i * static_cast<std::size_t>(config.max_donor_nodes);
    const std::size_t name_base = i * static_cast<std::size_t>(config.max_names);
    const std::size_t const_base = i * static_cast<std::size_t>(config.max_consts);
    for (int j = 0; j < used_len; ++j) {
      const AstNode& n = donor.ast.nodes[static_cast<std::size_t>(j)];
      out.donor_nodes[node_base + static_cast<std::size_t>(j)] =
          PlainNode{static_cast<int>(n.kind), n.i0, n.i1};
    }
    for (int j = 0; j < name_count; ++j) {
      register_name(&out, donor.ast.names[static_cast<std::size_t>(j)]);
      out.donor_name_ids[name_base + static_cast<std::size_t>(j)] =
          hash_name(donor.ast.names[static_cast<std::size_t>(j)]);
    }
    for (int j = 0; j < const_count; ++j) {
      out.donor_consts[const_base + static_cast<std::size_t>(j)] =
          donor.ast.consts[static_cast<std::size_t>(j)];
    }
    const std::size_t binder_base = i * static_cast<std::size_t>(config.max_linear_rec_binders);
    for (int j = 0; j < out.donor_linear_rec_counts[i]; ++j) {
      const LinearRecBinders& binders = donor.ast.linear_rec_binders[static_cast<std::size_t>(j)];
      out.donor_linear_rec_binders[binder_base + static_cast<std::size_t>(j)] =
          PlainLinearRecBinders{
              static_cast<int>(binders.node_index),
              binders.elem_name,
              binders.accum_name,
              binders.index_name,
          };
    }
    const std::size_t dc_base = i * static_cast<std::size_t>(config.max_asgp_dc_binders);
    for (int j = 0; j < out.donor_asgp_dc_counts[i]; ++j) {
      out.donor_asgp_dc_binders[dc_base + static_cast<std::size_t>(j)] =
          encode_asgp_dc_binders(donor.ast.asgp_dc_binders[static_cast<std::size_t>(j)]);
    }
    const std::size_t dp1_base = i * static_cast<std::size_t>(config.max_asgp_dp1d_specs);
    for (int j = 0; j < out.donor_asgp_dp1d_counts[i]; ++j) {
      out.donor_asgp_dp1d_specs[dp1_base + static_cast<std::size_t>(j)] =
          encode_asgp_dp1d_spec(donor.ast.asgp_dp1d_specs[static_cast<std::size_t>(j)]);
    }
    const std::size_t dp2_base = i * static_cast<std::size_t>(config.max_asgp_dp2d_specs);
    for (int j = 0; j < out.donor_asgp_dp2d_counts[i]; ++j) {
      out.donor_asgp_dp2d_specs[dp2_base + static_cast<std::size_t>(j)] =
          encode_asgp_dp2d_spec(donor.ast.asgp_dp2d_specs[static_cast<std::size_t>(j)]);
    }
  }
  return out;
}

std::vector<ProgramGenome> decode_gpu_repro_children(const PackedHostData& packed,
                                                     const GpuReproChildView& copyback,
                                                     const std::vector<ScoredGenome>& scored,
                                                     const EvolutionConfig& cfg) {
  std::vector<ScoredGenomeRef> scored_refs;
  scored_refs.reserve(scored.size());
  for (const ScoredGenome& one : scored) {
    scored_refs.push_back(ScoredGenomeRef{&one.genome, one.fitness});
  }
  return decode_gpu_repro_children(packed, copyback, scored_refs, cfg);
}

std::vector<ProgramGenome> decode_gpu_repro_children(const PackedHostData& packed,
                                                     const GpuReproChildView& copyback,
                                                     const std::vector<ScoredGenomeRef>& scored,
                                                     const EvolutionConfig& cfg) {
  std::vector<ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(cfg.population_size));
  const int total_children = std::min<int>(cfg.population_size,
                                           copyback.config.pair_count * 2);
  for (int child_index = 0; child_index < total_children; ++child_index) {
    ProgramGenome next;
    if (copyback.child_meta[static_cast<std::size_t>(child_index)].valid == 0) {
      next = fallback_parent_for_child(scored, copyback, child_index);
    } else {
      next = decode_one_child(packed, copyback, child_index);
      if (next.ast.nodes.empty()) {
        next = fallback_parent_for_child(scored, copyback, child_index);
      } else {
        ProgramGenome compacted = compact_genome_tables(next);
        if (decoded_child_has_valid_binders(compacted)) {
          out.push_back(std::move(compacted));
          continue;
        }
        next = fallback_parent_for_child(scored, copyback, child_index);
      }
    }
    out.push_back(compact_genome_tables(next));
  }
  while (static_cast<int>(out.size()) < cfg.population_size) {
    out.push_back(compact_genome_tables(*scored.front().genome));
  }
  return out;
}

}  // namespace g3pvm::evo::repro
