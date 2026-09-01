#include "g3pvm/evolution/repro/prep.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "../subtree_utils.hpp"
#include "../typed_expr_analysis.hpp"

namespace g3pvm::evo::repro {

namespace {

std::uint64_t mix_seed(std::uint64_t seed, std::uint64_t salt) {
  std::uint64_t x = seed ^ (salt + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

int clamp_tournament_size(int population_size, int selection_pressure) {
  if (population_size <= 0) {
    return 0;
  }
  return std::max(1, std::min(population_size, selection_pressure));
}

RType donor_type_for_bucket(int bucket) {
  switch (bucket) {
    case 0:
      return RType::Int;
    case 1:
      return RType::Float;
    case 2:
      return RType::Bool;
    case 3:
      return RType::Char;
    case 4:
      return RType::String;
    case 5:
      return RType::IntList;
    case 6:
      return RType::FloatList;
    case 7:
      return RType::StringList;
    case 8:
      return RType::Any;
    default:
      return RType::Int;
  }
}

bool value_allowed_by_grammar(const Value& value, const GrammarConfig& grammar) {
  switch (value.tag) {
    case ValueTag::Int:
      return grammar.value_int;
    case ValueTag::Float:
      return grammar.value_float;
    case ValueTag::Bool:
      return grammar.value_bool;
    case ValueTag::Char:
      return grammar.value_char;
    case ValueTag::Invalid:
      return false;
    case ValueTag::String:
      return grammar.value_string;
    case ValueTag::IntList:
      return grammar.value_int_list;
    case ValueTag::FloatList:
      return grammar.value_float_list;
    case ValueTag::StringList:
      return grammar.value_string_list;
    case ValueTag::FallbackToken:
      return grammar.is_all_enabled();
  }
  return false;
}

bool subtree_allowed_by_grammar(const AstProgram& program,
                                const typed_expr::TypedExprRoot& root,
                                const GrammarConfig& grammar) {
  if (!grammar.allows_type(root.type) || root.start >= root.stop || root.stop > program.nodes.size()) {
    return false;
  }
  for (std::size_t i = root.start; i < root.stop; ++i) {
    const AstNode& node = program.nodes[i];
    if (!grammar.allows_node_kind(node.kind)) {
      return false;
    }
    if (node.kind == NodeKind::CONST) {
      if (node.i0 < 0 || static_cast<std::size_t>(node.i0) >= program.consts.size()) {
        return false;
      }
      if (!value_allowed_by_grammar(program.consts[static_cast<std::size_t>(node.i0)], grammar)) {
        return false;
      }
    }
  }
  return true;
}

CandidateRange candidate_from_typed_root(const typed_expr::TypedExprRoot& root) {
  CandidateRange out;
  out.start = static_cast<int>(root.start);
  out.stop = static_cast<int>(root.stop);
  out.tag = static_cast<int>(CandidateTag::Expr);
  out.aux = static_cast<int>(root.type);
  out.scope_signature = root.scope_signature;
  out.binder_signature = root.binder_signature;
  out.scheme_kind = root.scheme_kind;
  out.phase_name = root.phase_name;
  out.visible_env_signature = root.visible_env_signature;
  out.dp_dependency_arity = root.dp_dependency_arity;
  return out;
}

std::vector<CandidateRange> sample_expr_candidates(const ProgramGenome& genome,
                                                   const std::vector<std::size_t>& subtree_end,
                                                   const VerifiedAst* verified,
                                                   int limit,
                                                   const GrammarConfig& grammar) {
  std::vector<CandidateRange> out;
  if (limit <= 0) {
    return out;
  }
  const std::vector<typed_expr::TypedExprRoot> expr_roots =
      verified != nullptr
          ? typed_expr::collect_typed_expr_roots(genome.ast, *verified)
          : typed_expr::collect_typed_expr_roots(genome.ast, subtree_end);
  std::vector<typed_expr::TypedExprRoot> filtered_roots;
  filtered_roots.reserve(expr_roots.size());
  for (const typed_expr::TypedExprRoot& root : expr_roots) {
    if (!typed_expr::is_asgp_phase_body_root(genome.ast, subtree_end, root) &&
        subtree_allowed_by_grammar(genome.ast, root, grammar)) {
      filtered_roots.push_back(root);
    }
  }
  if (filtered_roots.empty()) {
    out.push_back(CandidateRange{0, 0, static_cast<int>(CandidateTag::Expr), static_cast<int>(RType::Invalid)});
    return out;
  }

  const int used = std::min<int>(static_cast<int>(filtered_roots.size()), limit);
  out.reserve(static_cast<std::size_t>(used));
  const double step = static_cast<double>(filtered_roots.size()) / static_cast<double>(used);
  for (int i = 0; i < used; ++i) {
    std::size_t pick = static_cast<std::size_t>(i * step);
    if (pick >= filtered_roots.size()) {
      pick = filtered_roots.size() - 1;
    }
    const typed_expr::TypedExprRoot& root = filtered_roots[pick];
    out.push_back(candidate_from_typed_root(root));
  }
  return out;
}

DonorProgram make_donor_program(std::uint64_t seed,
                                RType type,
                                const GpuReproConfig& config,
                                const GrammarConfig& grammar) {
  DonorProgram out;
  out.type = grammar.allows_type(type) ? type : RType::Int;
  out.ast.version = k_ast_prefix_version_current;
  std::mt19937_64 rng(seed);
  const int donor_depth = std::max(1, std::min(config.max_nodes, config.max_donor_nodes) / 4);
  out.ast.nodes = subtree::make_random_expr_nodes_for_type(rng, out.ast, out.type, donor_depth, grammar, true);
  if (out.ast.nodes.empty()) {
    out.type = RType::Int;
    out.ast.nodes = subtree::make_random_expr_nodes_for_type(rng, out.ast, RType::Int, donor_depth, grammar, true);
  }
  return out;
}

}  // namespace

GpuReproConfig make_gpu_repro_config(const std::vector<ProgramGenome>& population,
                                     const EvolutionConfig& cfg) {
  GpuReproConfig out;
  out.population_size = static_cast<int>(population.size());
  out.pair_count = (out.population_size + 1) / 2;
  out.candidates_per_program = 16;
  out.donor_pool_size_per_type = std::max(16, std::min(out.population_size, 64));
  out.max_nodes = std::max(1, cfg.limits.max_total_nodes);
  out.max_donor_nodes = std::max(4, std::min(out.max_nodes, cfg.limits.max_expr_depth * 6));
  out.max_names = 1;
  out.max_consts = 1;
  out.max_linear_rec_binders = 1;
  out.max_asgp_dc_binders = 1;
  out.max_asgp_dp1d_specs = 1;
  out.max_asgp_dp2d_specs = 1;
  out.tournament_k = clamp_tournament_size(out.population_size, cfg.selection_pressure);
  out.max_expr_depth = std::max(0, cfg.limits.max_expr_depth);
  out.max_for_k = std::max(0, cfg.limits.max_for_k);
  out.mutation_ratio = std::clamp(cfg.mutation_rate, 0.0, 1.0);
  out.mutation_subtree_ratio = std::clamp(cfg.mutation_subtree_prob, 0.0, 1.0);
  out.seed = cfg.seed;
  for (const ProgramGenome& genome : population) {
    out.max_names = std::max(out.max_names, static_cast<int>(genome.ast.names.size()) + 4);
    out.max_consts = std::max(out.max_consts, static_cast<int>(genome.ast.consts.size()) + 4);
    out.max_linear_rec_binders =
        std::max(out.max_linear_rec_binders, static_cast<int>(genome.ast.linear_rec_binders.size()) + 4);
    out.max_asgp_dc_binders =
        std::max(out.max_asgp_dc_binders, static_cast<int>(genome.ast.asgp_dc_binders.size()) + 4);
    out.max_asgp_dp1d_specs =
        std::max(out.max_asgp_dp1d_specs, static_cast<int>(genome.ast.asgp_dp1d_specs.size()) + 4);
    out.max_asgp_dp2d_specs =
        std::max(out.max_asgp_dp2d_specs, static_cast<int>(genome.ast.asgp_dp2d_specs.size()) + 4);
  }
  return out;
}

PreprocessOutput preprocess_population_impl(const std::vector<ProgramGenome>& population,
                                            const std::vector<VerifiedAst>* verified,
                                            const GpuReproConfig& config,
                                            const GrammarConfig& grammar) {
  grammar.validate();
  if (verified != nullptr && verified->size() != population.size()) {
    throw std::invalid_argument("preprocess VerifiedAst count does not match population");
  }
  PreprocessOutput out;
  out.subtree_ends.resize(population.size());
  out.candidates.resize(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    const VerifiedAst* one_verified =
        verified != nullptr ? &(*verified)[i] : nullptr;
    if (one_verified != nullptr) {
      const std::size_t size = population[i].ast.nodes.size();
      if (one_verified->subtree_end.size() != size ||
          one_verified->expression_types.size() != size ||
          one_verified->expression_scope_signatures.size() != size ||
          one_verified->expression_binder_signatures.size() != size ||
          (size > 0 && one_verified->subtree_end[0] != size)) {
        throw std::invalid_argument("preprocess VerifiedAst does not match program shape");
      }
      out.subtree_ends[i] = one_verified->subtree_end;
    } else {
      out.subtree_ends[i] = subtree::build_subtree_end(population[i].ast);
    }
    out.candidates[i] = sample_expr_candidates(population[i], out.subtree_ends[i],
                                               one_verified, config.candidates_per_program,
                                               grammar);
  }

  out.donor_pool.reserve(
      static_cast<std::size_t>(config.donor_pool_size_per_type * kGpuReproDonorTypeCount));
  for (int bucket = 0; bucket < kGpuReproDonorTypeCount; ++bucket) {
    const RType donor_type = donor_type_for_bucket(bucket);
    for (int i = 0; i < config.donor_pool_size_per_type; ++i) {
      const std::uint64_t donor_seed = mix_seed(
          config.seed, static_cast<std::uint64_t>(bucket * config.donor_pool_size_per_type + i + 1));
      out.donor_pool.push_back(make_donor_program(donor_seed, donor_type, config, grammar));
    }
  }
  return out;
}

PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const GpuReproConfig& config,
                                       const GrammarConfig& grammar) {
  return preprocess_population_impl(population, nullptr, config, grammar);
}

PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const std::vector<VerifiedAst>& verified,
                                       const GpuReproConfig& config,
                                       const GrammarConfig& grammar) {
  return preprocess_population_impl(population, &verified, config, grammar);
}

}  // namespace g3pvm::evo::repro
