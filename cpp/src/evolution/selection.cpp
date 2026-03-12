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

namespace {

ProgramGenome tournament_select_once(const std::vector<ScoredGenome>& scored,
                                     std::mt19937_64& rng,
                                     int selection_pressure) {
  if (scored.empty()) {
    throw std::invalid_argument("scored population is empty");
  }

  const int tournament_size = std::max(1, std::min(selection_pressure, static_cast<int>(scored.size())));
  std::vector<std::size_t> candidate_indices(scored.size());
  std::iota(candidate_indices.begin(), candidate_indices.end(), std::size_t{0});
  for (int i = 0; i < tournament_size; ++i) {
    std::uniform_int_distribution<std::size_t> pick(i, candidate_indices.size() - 1);
    const std::size_t chosen = pick(rng);
    std::swap(candidate_indices[static_cast<std::size_t>(i)], candidate_indices[chosen]);
  }

  const ScoredGenome* best = nullptr;
  for (int i = 0; i < tournament_size; ++i) {
    const ScoredGenome& cand = scored[candidate_indices[static_cast<std::size_t>(i)]];
    if (best == nullptr || scored_genome_sorts_before(cand, *best)) {
      best = &cand;
    }
  }
  return best->genome;
}

}  // namespace

std::vector<ProgramGenome> tournament_selection(
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

  std::vector<ProgramGenome> selected;
  selected.reserve(static_cast<std::size_t>(selection_count));
  std::vector<ScoredGenome> available = scored;
  for (int i = 0; i < selection_count; ++i) {
    const ProgramGenome winner = tournament_select_once(available, rng, selection_pressure);
    selected.push_back(winner);
    const auto it = std::find_if(
        available.begin(), available.end(), [&](const ScoredGenome& candidate) {
          return candidate.genome.meta.program_key == winner.meta.program_key;
        });
    if (it == available.end()) {
      throw std::runtime_error("selected winner not found in available pool");
    }
    available.erase(it);
  }
  return selected;
}

}  // namespace g3pvm::evo
