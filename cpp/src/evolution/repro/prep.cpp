#include "g3pvm/evolution/repro/prep.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "g3pvm/evolution/evolve.hpp"
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

RType choose_donor_type(const std::vector<CandidateRange>& candidates, std::uint64_t seed) {
  if (candidates.empty()) {
    return RType::Num;
  }
  const std::size_t idx = static_cast<std::size_t>(seed % static_cast<std::uint64_t>(candidates.size()));
  const RType type = static_cast<RType>(candidates[idx].aux);
  if (type == RType::Invalid || type == RType::Any) {
    return RType::Num;
  }
  return type;
}

std::vector<CandidateRange> sample_expr_candidates(const ProgramGenome& genome,
                                                   const std::vector<std::size_t>& subtree_end,
                                                   int limit) {
  std::vector<CandidateRange> out;
  if (limit <= 0) {
    return out;
  }
  const std::vector<typed_expr::TypedExprRoot> expr_roots =
      typed_expr::collect_typed_expr_roots(genome.ast, subtree_end);
  if (expr_roots.empty()) {
    out.push_back(CandidateRange{0, 0, static_cast<int>(CandidateTag::Expr), static_cast<int>(RType::Invalid)});
    return out;
  }

  const int used = std::min<int>(static_cast<int>(expr_roots.size()), limit);
  out.reserve(static_cast<std::size_t>(used));
  const double step = static_cast<double>(expr_roots.size()) / static_cast<double>(used);
  for (int i = 0; i < used; ++i) {
    std::size_t pick = static_cast<std::size_t>(i * step);
    if (pick >= expr_roots.size()) {
      pick = expr_roots.size() - 1;
    }
    const typed_expr::TypedExprRoot& root = expr_roots[pick];
    out.push_back(CandidateRange{static_cast<int>(root.start),
                                 static_cast<int>(root.stop),
                                 static_cast<int>(CandidateTag::Expr),
                                 static_cast<int>(root.type)});
  }
  return out;
}

DonorProgram make_donor_program(std::uint64_t seed, RType type, const GpuReproConfig& config) {
  DonorProgram out;
  out.type = type;
  out.ast.version = "ast-prefix-v1";
  std::mt19937_64 rng(seed);
  const int donor_depth = std::max(1, std::min(config.max_nodes, config.max_donor_nodes) / 4);
  out.ast.nodes = subtree::make_random_expr_nodes_for_type(rng, out.ast, type, donor_depth);
  if (out.ast.nodes.empty()) {
    out.type = RType::Num;
    out.ast.nodes = subtree::make_random_expr_nodes_for_type(rng, out.ast, RType::Num, donor_depth);
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
  out.donor_pool_size = std::max(32, std::min(out.population_size, 256));
  out.max_nodes = std::max(1, cfg.limits.max_total_nodes);
  out.max_donor_nodes = std::max(4, std::min(out.max_nodes, cfg.limits.max_expr_depth * 6));
  out.max_names = 1;
  out.max_consts = 1;
  out.tournament_k = std::max(1, cfg.selection_pressure);
  out.max_expr_depth = std::max(0, cfg.limits.max_expr_depth);
  out.max_for_k = std::max(0, cfg.limits.max_for_k);
  out.mutation_ratio = std::clamp(cfg.mutation_rate, 0.0, 1.0);
  out.seed = cfg.seed;
  for (const ProgramGenome& genome : population) {
    out.max_names = std::max(out.max_names, static_cast<int>(genome.ast.names.size()) + 4);
    out.max_consts = std::max(out.max_consts, static_cast<int>(genome.ast.consts.size()) + 4);
  }
  return out;
}

PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const GpuReproConfig& config) {
  PreprocessOutput out;
  out.subtree_ends.resize(population.size());
  out.candidates.resize(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    out.subtree_ends[i] = subtree::build_subtree_end(population[i].ast);
    out.candidates[i] = sample_expr_candidates(population[i], out.subtree_ends[i], config.candidates_per_program);
  }

  out.donor_pool.reserve(static_cast<std::size_t>(config.donor_pool_size));
  for (int i = 0; i < config.donor_pool_size; ++i) {
    const std::uint64_t donor_seed = mix_seed(config.seed, static_cast<std::uint64_t>(i + 1));
    const std::size_t program_index =
        static_cast<std::size_t>(donor_seed % static_cast<std::uint64_t>(std::max(1, config.population_size)));
    const RType donor_type = choose_donor_type(out.candidates[program_index], donor_seed >> 8);
    out.donor_pool.push_back(make_donor_program(donor_seed, donor_type, config));
  }
  return out;
}

}  // namespace g3pvm::evo::repro
