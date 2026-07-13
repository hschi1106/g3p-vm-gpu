#include "g3pvm/evolution/repro/backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/repro/prep.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "../subtree_utils.hpp"
#include "../typed_expr_analysis.hpp"

namespace g3pvm::evo::repro {

namespace {

std::uint64_t hash64_host(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

std::uint64_t mix_string_seed(std::uint64_t seed, const std::string& value) {
  std::uint64_t out = seed;
  for (unsigned char c : value) {
    out ^= static_cast<std::uint64_t>(c);
    out *= 1099511628211ULL;
  }
  return hash64_host(out);
}

std::uint64_t population_seed(const std::vector<ScoredGenomeRef>& scored, std::uint64_t seed) {
  std::uint64_t out = seed ^ 1469598103934665603ULL;
  for (const ScoredGenomeRef& one : scored) {
    if (one.genome != nullptr) {
      out = mix_string_seed(out, one.genome->meta.program_key);
    }
  }
  return out;
}

int clamp_tournament_size(int population_size, int selection_pressure) {
  if (population_size <= 0) {
    return 0;
  }
  return std::max(1, std::min(population_size, selection_pressure));
}

int gcd_int(int a, int b) {
  while (b != 0) {
    const int next = a % b;
    a = b;
    b = next;
  }
  return a < 0 ? -a : a;
}

int choose_stride(int population_size, std::uint64_t round_seed) {
  if (population_size <= 1) {
    return 1;
  }
  int stride = static_cast<int>(hash64_host(round_seed ^ 0x94d049bb133111ebULL) %
                                static_cast<std::uint64_t>(population_size));
  if (stride <= 0) {
    stride = 1;
  }
  while (gcd_int(stride, population_size) != 1) {
    ++stride;
    if (stride >= population_size) {
      stride = 1;
    }
  }
  return stride;
}

int permuted_index_for_round(int population_size, int logical_index, std::uint64_t round_seed) {
  if (population_size <= 1) {
    return 0;
  }
  const int offset =
      static_cast<int>(hash64_host(round_seed ^ 0xbf58476d1ce4e5b9ULL) %
                       static_cast<std::uint64_t>(population_size));
  const int stride = choose_stride(population_size, round_seed);
  return (offset + logical_index * stride) % population_size;
}

int round_winner_count(int population_size, int tournament_k) {
  return (population_size + tournament_k - 1) / tournament_k;
}

std::size_t gpu_style_best_index_for_parent_slot(const std::vector<ScoredGenomeRef>& scored,
                                                 int tournament_k,
                                                 int slot_index,
                                                 std::uint64_t selection_seed) {
  const int population_size = static_cast<int>(scored.size());
  const int clamped_tournament_k = clamp_tournament_size(population_size, tournament_k);
  const int winners_per_round = round_winner_count(population_size, clamped_tournament_k);
  const int round_index = slot_index / winners_per_round;
  const int slot_in_round = slot_index % winners_per_round;
  const int logical_begin = slot_in_round * clamped_tournament_k;
  const int logical_end = std::min(population_size, logical_begin + clamped_tournament_k);
  const std::uint64_t round_seed =
      hash64_host(selection_seed + static_cast<std::uint64_t>(round_index + 1) * 0x9e3779b97f4a7c15ULL);

  int best = permuted_index_for_round(population_size, logical_begin, round_seed);
  double best_fit = scored[static_cast<std::size_t>(best)].fitness;
  for (int logical = logical_begin + 1; logical < logical_end; ++logical) {
    const int candidate = permuted_index_for_round(population_size, logical, round_seed);
    const double candidate_fit = scored[static_cast<std::size_t>(candidate)].fitness;
    if (candidate_fit > best_fit) {
      best = candidate;
      best_fit = candidate_fit;
    }
  }
  return static_cast<std::size_t>(best);
}

std::vector<std::size_t> gpu_style_selection_indices(const std::vector<ScoredGenomeRef>& scored,
                                                     int selection_pressure,
                                                     int selection_count,
                                                     std::uint64_t selection_seed) {
  if (selection_count < 0) {
    throw std::invalid_argument("selection_count must be >= 0");
  }
  if (scored.empty()) {
    if (selection_count == 0) return {};
    throw std::invalid_argument("scored population is empty");
  }

  std::vector<std::size_t> selected;
  selected.reserve(static_cast<std::size_t>(selection_count));
  const int tournament_k = clamp_tournament_size(static_cast<int>(scored.size()), selection_pressure);
  for (int slot = 0; slot < selection_count; ++slot) {
    selected.push_back(gpu_style_best_index_for_parent_slot(scored, tournament_k, slot, selection_seed));
  }
  return selected;
}

ProgramGenome build_valid_child(const AstProgram& candidate,
                                const ProgramGenome& fallback_parent,
                                const Limits& limits) {
  if (candidate.nodes.empty()) {
    return fallback_parent;
  }

  ProgramGenome out;
  out.ast = candidate;
  out.meta = build_genome_meta(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) {
    return fallback_parent;
  }
  if (out.meta.max_depth > limits.max_expr_depth) {
    return fallback_parent;
  }
  return out;
}

bool candidate_is_valid(const CandidateRange& candidate) {
  return candidate.start >= 0 && candidate.stop > candidate.start &&
         candidate.aux >= static_cast<int>(RType::Int) &&
         candidate.aux <= static_cast<int>(RType::Any);
}

bool candidate_keys_compatible(const CandidateRange& a, const CandidateRange& b) {
  return candidate_is_valid(a) &&
         candidate_is_valid(b) &&
         a.aux == b.aux &&
         a.scope_signature == b.scope_signature &&
         a.binder_signature == b.binder_signature &&
         a.scheme_kind == b.scheme_kind &&
         a.phase_name == b.phase_name &&
         a.visible_env_signature == b.visible_env_signature &&
         a.dp_dependency_arity == b.dp_dependency_arity;
}

unsigned int type_bit(RType type) {
  switch (type) {
    case RType::Int:
      return 1u << 0;
    case RType::Float:
      return 1u << 1;
    case RType::Bool:
      return 1u << 2;
    case RType::Char:
      return 1u << 3;
    case RType::String:
      return 1u << 4;
    case RType::IntList:
      return 1u << 5;
    case RType::FloatList:
      return 1u << 6;
    case RType::StringList:
      return 1u << 7;
    case RType::Any:
      return 1u << 8;
    default:
      return 0u;
  }
}

RType type_from_bit(int bit) {
  switch (bit) {
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
      return RType::Invalid;
  }
}

int pick_candidate_index_for_type(const std::vector<CandidateRange>& candidates,
                                  RType target_type,
                                  std::uint64_t seed) {
  int match_count = 0;
  for (const CandidateRange& candidate : candidates) {
    if (candidate_is_valid(candidate) && static_cast<RType>(candidate.aux) == target_type) {
      ++match_count;
    }
  }
  if (match_count <= 0) {
    return 0;
  }

  const int target_rank = static_cast<int>(seed % static_cast<std::uint64_t>(match_count));
  int seen = 0;
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const CandidateRange& candidate = candidates[static_cast<std::size_t>(i)];
    if (!candidate_is_valid(candidate) || static_cast<RType>(candidate.aux) != target_type) {
      continue;
    }
    if (seen == target_rank) {
      return i;
    }
    ++seen;
  }
  return 0;
}

struct CandidatePair {
  bool valid = false;
  CandidateRange a;
  CandidateRange b;
  RType type = RType::Invalid;
};

CandidatePair choose_gpu_candidate_pair(const std::vector<CandidateRange>& candidates_a,
                                        const std::vector<CandidateRange>& candidates_b,
                                        std::uint64_t seed) {
  int compatible_count = 0;
  for (const CandidateRange& a : candidates_a) {
    for (const CandidateRange& b : candidates_b) {
      if (candidate_keys_compatible(a, b)) {
        ++compatible_count;
      }
    }
  }
  if (compatible_count <= 0) {
    return CandidatePair{};
  }

  const int chosen_rank = static_cast<int>(hash64_host(seed ^ 0x517cc1b727220a95ULL) %
                                           static_cast<std::uint64_t>(compatible_count));
  int seen = 0;
  for (const CandidateRange& a : candidates_a) {
    for (const CandidateRange& b : candidates_b) {
      if (!candidate_keys_compatible(a, b)) {
        continue;
      }
      if (seen == chosen_rank) {
        return CandidatePair{true, a, b, static_cast<RType>(a.aux)};
      }
      ++seen;
    }
  }
  return CandidatePair{};
}

template <typename T>
const T& choose_one(std::mt19937_64& rng, const std::vector<T>& values) {
  const std::size_t last = values.size() - 1;
  const std::size_t idx = std::uniform_int_distribution<std::size_t>(0, last)(rng);
  return values[idx];
}

CandidatePair choose_cpu_candidate_pair(const ProgramGenome& parent_a,
                                        const ProgramGenome& parent_b,
                                        std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<std::size_t> end_a = subtree::build_subtree_end(parent_a.ast);
  const std::vector<std::size_t> end_b = subtree::build_subtree_end(parent_b.ast);
  const std::vector<typed_expr::TypedExprRoot> expr_a =
      typed_expr::collect_typed_expr_roots(parent_a.ast, end_a);
  const std::vector<typed_expr::TypedExprRoot> expr_b =
      typed_expr::collect_typed_expr_roots(parent_b.ast, end_b);
  std::vector<typed_expr::TypedExprRoot> filtered_expr_a;
  std::vector<typed_expr::TypedExprRoot> filtered_expr_b;
  filtered_expr_a.reserve(expr_a.size());
  filtered_expr_b.reserve(expr_b.size());
  for (const typed_expr::TypedExprRoot& root : expr_a) {
    if (!typed_expr::is_asgp_phase_body_root(parent_a.ast, end_a, root)) {
      filtered_expr_a.push_back(root);
    }
  }
  for (const typed_expr::TypedExprRoot& root : expr_b) {
    if (!typed_expr::is_asgp_phase_body_root(parent_b.ast, end_b, root)) {
      filtered_expr_b.push_back(root);
    }
  }
  if (filtered_expr_a.empty() || filtered_expr_b.empty()) {
    return CandidatePair{};
  }

  std::vector<typed_expr::TypedExprRoot> compatible_a;
  compatible_a.reserve(filtered_expr_a.size());
  for (const typed_expr::TypedExprRoot& root_a : filtered_expr_a) {
    const bool in_b = std::any_of(filtered_expr_b.begin(), filtered_expr_b.end(), [&](const typed_expr::TypedExprRoot& root_b) {
      return typed_expr::typed_subtree_keys_compatible(root_a, root_b);
    });
    if (in_b) {
      compatible_a.push_back(root_a);
    }
  }
  if (compatible_a.empty()) {
    return CandidatePair{};
  }

  const typed_expr::TypedExprRoot& chosen_a = choose_one(rng, compatible_a);
  std::vector<typed_expr::TypedExprRoot> roots_a;
  std::vector<typed_expr::TypedExprRoot> roots_b;
  roots_a.reserve(filtered_expr_a.size());
  roots_b.reserve(filtered_expr_b.size());
  for (const typed_expr::TypedExprRoot& root : filtered_expr_a) {
    if (typed_expr::typed_subtree_keys_compatible(chosen_a, root)) {
      roots_a.push_back(root);
    }
  }
  for (const typed_expr::TypedExprRoot& root : filtered_expr_b) {
    if (typed_expr::typed_subtree_keys_compatible(chosen_a, root)) {
      roots_b.push_back(root);
    }
  }
  if (roots_a.empty() || roots_b.empty()) {
    return CandidatePair{};
  }

  const typed_expr::TypedExprRoot& target_a = choose_one(rng, roots_a);
  const typed_expr::TypedExprRoot& target_b = choose_one(rng, roots_b);
  return CandidatePair{true,
                       CandidateRange{static_cast<int>(target_a.start),
                                      static_cast<int>(target_a.stop),
                                      static_cast<int>(CandidateTag::Expr),
                                      static_cast<int>(target_a.type)},
                       CandidateRange{static_cast<int>(target_b.start),
                                      static_cast<int>(target_b.stop),
                                      static_cast<int>(CandidateTag::Expr),
                                      static_cast<int>(target_b.type)},
                       target_a.type};
}

std::pair<ProgramGenome, ProgramGenome> crossover_from_candidate_pair(const ProgramGenome& parent_a,
                                                                      const ProgramGenome& parent_b,
                                                                      const CandidatePair& pair,
                                                                      const Limits& limits) {
  if (!pair.valid) {
    return {parent_a, parent_b};
  }
  const AstProgram child_a_ast = subtree::replace_subtree(
      parent_a.ast, static_cast<std::size_t>(pair.a.start), static_cast<std::size_t>(pair.a.stop),
      parent_b.ast, static_cast<std::size_t>(pair.b.start), static_cast<std::size_t>(pair.b.stop));
  const AstProgram child_b_ast = subtree::replace_subtree(
      parent_b.ast, static_cast<std::size_t>(pair.b.start), static_cast<std::size_t>(pair.b.stop),
      parent_a.ast, static_cast<std::size_t>(pair.a.start), static_cast<std::size_t>(pair.a.stop));
  return {build_valid_child(child_a_ast, parent_a, limits),
          build_valid_child(child_b_ast, parent_b, limits)};
}

std::vector<ProgramGenome> materialize_population(const std::vector<ScoredGenomeRef>& scored) {
  std::vector<ProgramGenome> population;
  population.reserve(scored.size());
  for (const ScoredGenomeRef& one : scored) {
    population.push_back(*one.genome);
  }
  return population;
}

int donor_bucket_for_type(RType type) {
  switch (type) {
    case RType::Int:
      return 0;
    case RType::Float:
      return 1;
    case RType::Bool:
      return 2;
    case RType::Char:
      return 3;
    case RType::String:
      return 4;
    case RType::IntList:
      return 5;
    case RType::FloatList:
      return 6;
    case RType::StringList:
      return 7;
    case RType::Any:
      return 8;
    default:
      return 0;
  }
}

const DonorProgram* pick_donor_for_type(const PreprocessOutput& prep,
                                        RType type,
                                        int donor_pool_size_per_type,
                                        std::uint64_t seed) {
  if (donor_pool_size_per_type <= 0 || prep.donor_pool.empty()) {
    return nullptr;
  }
  const int bucket = donor_bucket_for_type(type);
  const int slot = static_cast<int>(hash64_host(seed ^ 0xe7037ed1a0b428dbULL) %
                                    static_cast<std::uint64_t>(donor_pool_size_per_type));
  const int index = bucket * donor_pool_size_per_type + slot;
  if (index < 0 || static_cast<std::size_t>(index) >= prep.donor_pool.size()) {
    return nullptr;
  }
  return &prep.donor_pool[static_cast<std::size_t>(index)];
}

ProgramGenome donor_replacement_child(const ProgramGenome& parent,
                                      const CandidateRange& target,
                                      RType type,
                                      const PreprocessOutput& prep,
                                      const GpuReproConfig& config,
                                      std::uint64_t seed,
                                      const Limits& limits) {
  if (!candidate_is_valid(target)) {
    return parent;
  }
  const DonorProgram* donor = pick_donor_for_type(prep, type, config.donor_pool_size_per_type, seed);
  if (donor == nullptr || donor->ast.nodes.empty()) {
    return parent;
  }
  const AstProgram child_ast = subtree::replace_subtree(
      parent.ast, static_cast<std::size_t>(target.start), static_cast<std::size_t>(target.stop),
      donor->ast, 0, donor->ast.nodes.size());
  return build_valid_child(child_ast, parent, limits);
}

bool cpu_style_subtree_branch(std::uint64_t seed, double mutation_subtree_prob) {
  std::mt19937_64 rng(seed);
  const double p = std::max(0.0, std::min(1.0, mutation_subtree_prob));
  return std::bernoulli_distribution(p)(rng);
}

ProgramGenome apply_constant_mutation(const ProgramGenome& child,
                                      std::uint64_t seed,
                                      const EvolutionConfig& cfg) {
  return mutate(child, seed, cfg.limits, 0.0, cfg.grammar);
}

struct CpuAblationData {
  GpuReproConfig config;
  PreprocessOutput prep;
};

CpuAblationData prepare_cpu_ablation_data(const std::vector<ScoredGenomeRef>& scored,
                                          const EvolutionConfig& cfg) {
  CpuAblationData out;
  const std::vector<ProgramGenome> population = materialize_population(scored);
  out.config = make_gpu_repro_config(population, cfg);
  out.config.seed = population_seed(scored, cfg.seed);
  out.prep = preprocess_population(population, out.config, cfg.grammar);
  return out;
}

ReproductionResult run_cpu_backend_impl(const std::vector<ScoredGenomeRef>& scored,
                                        const EvolutionConfig& cfg,
                                        std::mt19937_64& rng) {
  ReproductionResult out;
  out.next_population.reserve(static_cast<std::size_t>(cfg.population_size));
  const int pair_count = (cfg.population_size + 1) / 2;
  const int selected_parent_count = pair_count * 2;

  const auto selection_t0 = std::chrono::steady_clock::now();
  std::vector<std::size_t> selected_parent_indices;
  if (cfg.cpu_repro_ablation == CpuReproAblation::GpuSelection) {
    selected_parent_indices =
        gpu_style_selection_indices(scored, cfg.selection_pressure, selected_parent_count, rng());
  } else {
    selected_parent_indices =
        tournament_selection_indices_without_replacement(scored, rng, cfg.selection_pressure, selected_parent_count);
  }
  const auto selection_t1 = std::chrono::steady_clock::now();
  out.stats.selection_ms =
      std::chrono::duration<double, std::milli>(selection_t1 - selection_t0).count();

  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);
  if (cfg.cpu_repro_ablation != CpuReproAblation::GpuSelection && selected_parent_indices.size() > 1) {
    std::shuffle(selected_parent_indices.begin(), selected_parent_indices.end(), rng);
  }

  CpuAblationData ablation_data;
  const bool needs_ablation_data = cfg.cpu_repro_ablation == CpuReproAblation::GpuCandidates ||
                                   cfg.cpu_repro_ablation == CpuReproAblation::GpuCoupledDonor;
  if (needs_ablation_data) {
    const auto prep_t0 = std::chrono::steady_clock::now();
    ablation_data = prepare_cpu_ablation_data(scored, cfg);
    const auto prep_t1 = std::chrono::steady_clock::now();
    out.stats.preprocess_ms += std::chrono::duration<double, std::milli>(prep_t1 - prep_t0).count();
  }

  double crossover_ms = 0.0;
  double mutation_ms = 0.0;
  for (std::size_t i = 0; i + 1 < selected_parent_indices.size() &&
                          static_cast<int>(out.next_population.size()) < cfg.population_size;
       i += 2) {
    const ProgramGenome& parent_a = *scored[selected_parent_indices[i]].genome;
    const ProgramGenome& parent_b = *scored[selected_parent_indices[i + 1]].genome;

    const std::uint64_t crossover_seed = seed_dist(rng);
    const auto crossover_t0 = std::chrono::steady_clock::now();
    std::pair<ProgramGenome, ProgramGenome> children;
    CandidatePair coupled_pair;
    if (cfg.cpu_repro_ablation == CpuReproAblation::GpuCandidates) {
      const CandidatePair pair = choose_gpu_candidate_pair(
          ablation_data.prep.candidates[selected_parent_indices[i]],
          ablation_data.prep.candidates[selected_parent_indices[i + 1]],
          crossover_seed);
      children = crossover_from_candidate_pair(parent_a, parent_b, pair, cfg.limits);
    } else if (cfg.cpu_repro_ablation == CpuReproAblation::GpuCoupledDonor) {
      coupled_pair = choose_cpu_candidate_pair(parent_a, parent_b, crossover_seed);
      children = crossover_from_candidate_pair(parent_a, parent_b, coupled_pair, cfg.limits);
    } else {
      children = crossover(parent_a, parent_b, crossover_seed, cfg.limits);
    }
    const auto crossover_t1 = std::chrono::steady_clock::now();
    crossover_ms += std::chrono::duration<double, std::milli>(crossover_t1 - crossover_t0).count();

    auto maybe_mutate = [&](ProgramGenome& child, const ProgramGenome& parent, const CandidateRange& site) {
      if (prob_dist(rng) < cfg.mutation_rate) {
        const auto mutation_t0 = std::chrono::steady_clock::now();
        const std::uint64_t mutation_seed = seed_dist(rng);
        if (cfg.cpu_repro_ablation == CpuReproAblation::GpuCoupledDonor && !coupled_pair.valid) {
          child = parent;
        } else if (cfg.cpu_repro_ablation == CpuReproAblation::GpuCoupledDonor &&
                   cpu_style_subtree_branch(mutation_seed, cfg.mutation_subtree_prob)) {
          child = donor_replacement_child(parent,
                                          site,
                                          static_cast<RType>(site.aux),
                                          ablation_data.prep,
                                          ablation_data.config,
                                          mutation_seed,
                                          cfg.limits);
        } else if (cfg.cpu_repro_ablation == CpuReproAblation::GpuCoupledDonor) {
          child = apply_constant_mutation(child, mutation_seed, cfg);
        } else {
          child = mutate(child, mutation_seed, cfg.limits, cfg.mutation_subtree_prob, cfg.grammar);
        }
        const auto mutation_t1 = std::chrono::steady_clock::now();
        mutation_ms += std::chrono::duration<double, std::milli>(mutation_t1 - mutation_t0).count();
      }
    };

    maybe_mutate(children.first, parent_a, coupled_pair.a);
    out.next_population.push_back(std::move(children.first));
    if (static_cast<int>(out.next_population.size()) >= cfg.population_size) {
      break;
    }

    maybe_mutate(children.second, parent_b, coupled_pair.b);
    out.next_population.push_back(std::move(children.second));
  }
  out.stats.crossover_ms = crossover_ms;
  out.stats.mutation_ms = mutation_ms;
  return out;
}

ReproductionResult run_cpu_backend(const std::vector<ScoredGenome>& scored,
                                   const EvolutionConfig& cfg,
                                   std::mt19937_64& rng) {
  std::vector<ScoredGenomeRef> scored_refs;
  scored_refs.reserve(scored.size());
  for (const ScoredGenome& one : scored) {
    scored_refs.push_back(ScoredGenomeRef{&one.genome, one.fitness});
  }
  return run_cpu_backend_impl(scored_refs, cfg, rng);
}

ReproductionResult run_cpu_backend(const std::vector<ScoredGenomeRef>& scored,
                                   const EvolutionConfig& cfg,
                                   std::mt19937_64& rng) {
  return run_cpu_backend_impl(scored, cfg, rng);
}

}  // namespace

std::string reproduction_backend_name(ReproductionBackend backend) {
  switch (backend) {
    case ReproductionBackend::Gpu:
      return "gpu";
    case ReproductionBackend::Cpu:
    default:
      return "cpu";
  }
}

ReproductionBackend parse_reproduction_backend_name(const std::string& raw) {
  if (raw == "cpu") {
    return ReproductionBackend::Cpu;
  }
  if (raw == "gpu") {
    return ReproductionBackend::Gpu;
  }
  throw std::invalid_argument("unknown reproduction backend: " + raw);
}

std::string cpu_repro_ablation_name(CpuReproAblation ablation) {
  switch (ablation) {
    case CpuReproAblation::GpuSelection:
      return "gpu_selection";
    case CpuReproAblation::GpuCandidates:
      return "gpu_candidates";
    case CpuReproAblation::GpuCoupledDonor:
      return "gpu_coupled_donor";
    case CpuReproAblation::None:
    default:
      return "none";
  }
}

CpuReproAblation parse_cpu_repro_ablation_name(const std::string& raw) {
  if (raw == "none") {
    return CpuReproAblation::None;
  }
  if (raw == "gpu_selection") {
    return CpuReproAblation::GpuSelection;
  }
  if (raw == "gpu_candidates") {
    return CpuReproAblation::GpuCandidates;
  }
  if (raw == "gpu_coupled_donor") {
    return CpuReproAblation::GpuCoupledDonor;
  }
  throw std::invalid_argument("unknown cpu reproduction ablation: " + raw);
}

ReproductionResult run_reproduction_backend(const std::vector<ScoredGenome>& scored,
                                            const EvolutionConfig& cfg,
                                            std::mt19937_64& rng) {
  if (cfg.reproduction_backend == ReproductionBackend::Gpu) {
    return run_gpu_repro_backend(scored, cfg, rng);
  }
  return run_cpu_backend(scored, cfg, rng);
}

ReproductionResult run_reproduction_backend(const std::vector<ScoredGenomeRef>& scored,
                                            const EvolutionConfig& cfg,
                                            std::mt19937_64& rng) {
  if (cfg.reproduction_backend == ReproductionBackend::Gpu) {
    return run_gpu_repro_backend(scored, cfg, rng);
  }
  return run_cpu_backend(scored, cfg, rng);
}

}  // namespace g3pvm::evo::repro
