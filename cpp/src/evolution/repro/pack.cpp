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

ProgramGenome decode_one_child(const PackedHostData& packed,
                               const GpuReproChildView& copyback,
                               int child_index) {
  ProgramGenome out;
  out.ast.version = "ast-prefix-v1";
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

}  // namespace

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
  out.program_consts.resize(static_cast<std::size_t>(config.population_size * config.max_consts), Value::none());
  out.donor_nodes.resize(total_donor_count * static_cast<std::size_t>(config.max_donor_nodes));
  out.donor_lens.resize(total_donor_count, 0);
  out.donor_name_ids.resize(total_donor_count * static_cast<std::size_t>(config.max_names), 0ULL);
  out.donor_name_counts.resize(total_donor_count, 0);
  out.donor_consts.resize(total_donor_count * static_cast<std::size_t>(config.max_consts), Value::none());
  out.donor_const_counts.resize(total_donor_count, 0);

  for (int p = 0; p < config.population_size; ++p) {
    const ProgramGenome& genome = population[static_cast<std::size_t>(p)];
    const int used_len = std::min<int>(static_cast<int>(genome.ast.nodes.size()), config.max_nodes);
    const int name_count = std::min<int>(static_cast<int>(genome.ast.names.size()), config.max_names);
    const int const_count = std::min<int>(static_cast<int>(genome.ast.consts.size()), config.max_consts);
    out.metas[static_cast<std::size_t>(p)] = PackedProgramMeta{used_len, name_count, const_count};
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
  }
  return out;
}

std::vector<ProgramGenome> decode_gpu_repro_children(const PackedHostData& packed,
                                                     const GpuReproChildView& copyback,
                                                     const std::vector<ScoredGenome>& scored,
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
      }
    }
    out.push_back(std::move(next));
  }
  while (static_cast<int>(out.size()) < cfg.population_size) {
    out.push_back(scored.front().genome);
  }
  return out;
}

}  // namespace g3pvm::evo::repro
