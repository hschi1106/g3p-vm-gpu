#include "g3pvm/evolution/repro/backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/selection.hpp"

namespace g3pvm::evo::repro {

namespace {

template <typename TScored>
ReproductionResult run_cpu_backend_impl(const std::vector<TScored>& scored,
                                        const EvolutionConfig& cfg,
                                        std::mt19937_64& rng) {
  ReproductionResult out;
  out.next_population.reserve(static_cast<std::size_t>(cfg.population_size));
  const int pair_count = (cfg.population_size + 1) / 2;
  const int selected_parent_count = pair_count * 2;

  const auto selection_t0 = std::chrono::steady_clock::now();
  std::vector<std::size_t> selected_parent_indices =
      tournament_selection_indices_without_replacement(scored, rng, cfg.selection_pressure, selected_parent_count);
  const auto selection_t1 = std::chrono::steady_clock::now();
  out.stats.selection_ms =
      std::chrono::duration<double, std::milli>(selection_t1 - selection_t0).count();

  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);
  if (selected_parent_indices.size() > 1) {
    std::shuffle(selected_parent_indices.begin(), selected_parent_indices.end(), rng);
  }

  double crossover_ms = 0.0;
  double mutation_ms = 0.0;
  for (std::size_t i = 0; i + 1 < selected_parent_indices.size() &&
                          static_cast<int>(out.next_population.size()) < cfg.population_size;
       i += 2) {
    const ProgramGenome& parent_a = *scored[selected_parent_indices[i]].genome;
    const ProgramGenome& parent_b = *scored[selected_parent_indices[i + 1]].genome;

    const auto crossover_t0 = std::chrono::steady_clock::now();
    auto children = crossover(parent_a, parent_b, seed_dist(rng), cfg.limits);
    const auto crossover_t1 = std::chrono::steady_clock::now();
    crossover_ms += std::chrono::duration<double, std::milli>(crossover_t1 - crossover_t0).count();

    auto maybe_mutate = [&](ProgramGenome& child) {
      if (prob_dist(rng) < cfg.mutation_rate) {
        const auto mutation_t0 = std::chrono::steady_clock::now();
        child = mutate(child, seed_dist(rng), cfg.limits, cfg.mutation_subtree_prob, cfg.grammar);
        const auto mutation_t1 = std::chrono::steady_clock::now();
        mutation_ms += std::chrono::duration<double, std::milli>(mutation_t1 - mutation_t0).count();
      }
    };

    maybe_mutate(children.first);
    out.next_population.push_back(std::move(children.first));
    if (static_cast<int>(out.next_population.size()) >= cfg.population_size) {
      break;
    }

    maybe_mutate(children.second);
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
