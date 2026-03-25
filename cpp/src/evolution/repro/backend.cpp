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

ReproductionResult run_cpu_backend(const std::vector<ScoredGenome>& scored,
                                   const EvolutionConfig& cfg,
                                   std::mt19937_64& rng) {
  ReproductionResult out;
  out.next_population.reserve(static_cast<std::size_t>(cfg.population_size));
  const int pair_count = (cfg.population_size + 1) / 2;
  const int selected_parent_count = pair_count * 2;

  const auto selection_t0 = std::chrono::steady_clock::now();
  std::vector<ProgramGenome> selected_parents =
      tournament_selection_without_replacement(scored, rng, cfg.selection_pressure, selected_parent_count);
  const auto selection_t1 = std::chrono::steady_clock::now();
  out.stats.selection_ms =
      std::chrono::duration<double, std::milli>(selection_t1 - selection_t0).count();

  std::vector<ProgramGenome> offspring = selected_parents;
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);

  const auto crossover_t0 = std::chrono::steady_clock::now();
  if (selected_parents.size() > 1) {
    std::shuffle(offspring.begin(), offspring.end(), rng);
    for (std::size_t i = 0; i + 1 < offspring.size(); i += 2) {
      auto children = crossover(offspring[i], offspring[i + 1], seed_dist(rng), cfg.limits);
      offspring[i] = std::move(children.first);
      offspring[i + 1] = std::move(children.second);
    }
  }
  const auto crossover_t1 = std::chrono::steady_clock::now();
  out.stats.crossover_ms =
      std::chrono::duration<double, std::milli>(crossover_t1 - crossover_t0).count();

  const auto mutation_t0 = std::chrono::steady_clock::now();
  for (ProgramGenome& child : offspring) {
    if (prob_dist(rng) < cfg.mutation_rate) {
      child = mutate(child, seed_dist(rng), cfg.limits, cfg.mutation_subtree_prob);
    }
  }
  const auto mutation_t1 = std::chrono::steady_clock::now();
  out.stats.mutation_ms =
      std::chrono::duration<double, std::milli>(mutation_t1 - mutation_t0).count();

  const std::size_t emitted = std::min<std::size_t>(offspring.size(), static_cast<std::size_t>(cfg.population_size));
  out.next_population.insert(out.next_population.end(),
                             std::make_move_iterator(offspring.begin()),
                             std::make_move_iterator(offspring.begin() + static_cast<std::ptrdiff_t>(emitted)));
  return out;
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

}  // namespace g3pvm::evo::repro
