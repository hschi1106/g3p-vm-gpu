#include "g3pvm/evolution/selection.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace g3pvm::evo {

double canonicalize_fitness_for_ranking(double fitness) {
  if (!std::isfinite(fitness) || fitness == 0.0) {
    return fitness == 0.0 ? 0.0 : fitness;
  }

  int exponent = 0;
  const double mantissa = std::frexp(fitness, &exponent);
  constexpr int kMantissaBits = 40;
  const long long quantized_mantissa = std::llround(std::ldexp(mantissa, kMantissaBits));
  return std::ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

bool scored_genome_sorts_before(const ScoredGenome& a, const ScoredGenome& b) {
  const double fitness_a = canonicalize_fitness_for_ranking(a.fitness);
  const double fitness_b = canonicalize_fitness_for_ranking(b.fitness);
  if (fitness_a != fitness_b) {
    return fitness_a > fitness_b;
  }
  return a.genome.meta.program_key < b.genome.meta.program_key;
}

bool scored_genome_sorts_before(const ScoredGenomeRef& a, const ScoredGenomeRef& b) {
  const double fitness_a = canonicalize_fitness_for_ranking(a.fitness);
  const double fitness_b = canonicalize_fitness_for_ranking(b.fitness);
  if (fitness_a != fitness_b) {
    return fitness_a > fitness_b;
  }
  return a.genome->meta.program_key < b.genome->meta.program_key;
}

std::vector<ScoredGenomeRef> rank_population_refs(
    const std::vector<ProgramGenome>& population,
    const std::vector<double>& fitness,
    bool sort_output) {
  if (fitness.size() != population.size()) {
    throw std::runtime_error("fitness size mismatch");
  }
  std::vector<ScoredGenomeRef> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(ScoredGenomeRef{&population[i], fitness[i]});
  }
  if (sort_output) {
    std::sort(scored.begin(), scored.end(), [](const ScoredGenomeRef& a, const ScoredGenomeRef& b) {
      return scored_genome_sorts_before(a, b);
    });
  }
  return scored;
}

ScoredGenome materialize_scored_genome(const ScoredGenomeRef& scored) {
  if (scored.genome == nullptr) {
    throw std::runtime_error("scored genome ref is null");
  }
  return ScoredGenome{*scored.genome, scored.fitness};
}

std::vector<ScoredGenome> materialize_scored_population(
    const std::vector<ScoredGenomeRef>& scored) {
  std::vector<ScoredGenome> owned;
  owned.reserve(scored.size());
  for (const ScoredGenomeRef& one : scored) {
    owned.push_back(materialize_scored_genome(one));
  }
  return owned;
}

namespace {

int clamp_tournament_size(int population_size, int selection_pressure) {
  if (population_size <= 0) {
    return 0;
  }
  return std::max(1, std::min(population_size, selection_pressure));
}

std::size_t best_index_in_chunk(const std::vector<ScoredGenome>& scored,
                                const std::vector<std::size_t>& shuffled_indices,
                                std::size_t begin,
                                std::size_t end) {
  std::size_t best = shuffled_indices[begin];
  for (std::size_t i = begin + 1; i < end; ++i) {
    const std::size_t candidate = shuffled_indices[i];
    if (scored_genome_sorts_before(scored[candidate], scored[best])) {
      best = candidate;
    }
  }
  return best;
}

std::size_t best_index_in_chunk(const std::vector<ScoredGenomeRef>& scored,
                                const std::vector<std::size_t>& shuffled_indices,
                                std::size_t begin,
                                std::size_t end) {
  std::size_t best = shuffled_indices[begin];
  for (std::size_t i = begin + 1; i < end; ++i) {
    const std::size_t candidate = shuffled_indices[i];
    if (scored_genome_sorts_before(scored[candidate], scored[best])) {
      best = candidate;
    }
  }
  return best;
}

}  // namespace

std::vector<std::size_t> tournament_selection_indices_without_replacement(
    const std::vector<ScoredGenome>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count) {
  if (selection_count < 0) {
    throw std::invalid_argument("selection_count must be >= 0");
  }
  if (selection_count > static_cast<int>(scored.size())) {
    throw std::invalid_argument("selection_count must be <= scored population size");
  }
  if (scored.empty()) {
    if (selection_count == 0) return {};
    throw std::invalid_argument("scored population is empty");
  }

  std::vector<std::size_t> selected;
  selected.reserve(static_cast<std::size_t>(selection_count));
  const int tournament_size = clamp_tournament_size(static_cast<int>(scored.size()), selection_pressure);
  std::vector<std::size_t> shuffled_indices(scored.size());
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), std::size_t{0});

  while (static_cast<int>(selected.size()) < selection_count) {
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);
    for (std::size_t begin = 0;
         begin < shuffled_indices.size() && static_cast<int>(selected.size()) < selection_count;
         begin += static_cast<std::size_t>(tournament_size)) {
      const std::size_t end =
          std::min(shuffled_indices.size(), begin + static_cast<std::size_t>(tournament_size));
      const std::size_t winner_index = best_index_in_chunk(scored, shuffled_indices, begin, end);
      selected.push_back(winner_index);
    }
  }
  return selected;
}

std::vector<std::size_t> tournament_selection_indices_without_replacement(
    const std::vector<ScoredGenomeRef>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count) {
  if (selection_count < 0) {
    throw std::invalid_argument("selection_count must be >= 0");
  }
  if (selection_count > static_cast<int>(scored.size())) {
    throw std::invalid_argument("selection_count must be <= scored population size");
  }
  if (scored.empty()) {
    if (selection_count == 0) return {};
    throw std::invalid_argument("scored population is empty");
  }

  std::vector<std::size_t> selected;
  selected.reserve(static_cast<std::size_t>(selection_count));
  const int tournament_size = clamp_tournament_size(static_cast<int>(scored.size()), selection_pressure);
  std::vector<std::size_t> shuffled_indices(scored.size());
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), std::size_t{0});

  while (static_cast<int>(selected.size()) < selection_count) {
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);
    for (std::size_t begin = 0;
         begin < shuffled_indices.size() && static_cast<int>(selected.size()) < selection_count;
         begin += static_cast<std::size_t>(tournament_size)) {
      const std::size_t end =
          std::min(shuffled_indices.size(), begin + static_cast<std::size_t>(tournament_size));
      const std::size_t winner_index = best_index_in_chunk(scored, shuffled_indices, begin, end);
      selected.push_back(winner_index);
    }
  }
  return selected;
}

std::vector<ProgramGenome> tournament_selection_without_replacement(
    const std::vector<ScoredGenome>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count) {
  const std::vector<std::size_t> selected_indices =
      tournament_selection_indices_without_replacement(scored, rng, selection_pressure, selection_count);
  std::vector<ProgramGenome> selected;
  selected.reserve(selected_indices.size());
  for (std::size_t winner_index : selected_indices) {
    selected.push_back(scored[winner_index].genome);
  }
  return selected;
}

}  // namespace g3pvm::evo
