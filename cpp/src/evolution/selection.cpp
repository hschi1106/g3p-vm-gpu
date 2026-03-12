#include "g3pvm/evolution/selection.hpp"

#include <algorithm>
#include <cmath>
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

ProgramGenome tournament_selection(const std::vector<ScoredGenome>& scored,
                                   std::mt19937_64& rng,
                                   int selection_pressure) {
  if (scored.empty()) {
    throw std::invalid_argument("scored population is empty");
  }

  std::uniform_int_distribution<std::size_t> any_pick(0, scored.size() - 1);
  const int tournament_size = std::max(1, std::min(selection_pressure, static_cast<int>(scored.size())));
  const ScoredGenome* best = nullptr;
  for (int i = 0; i < tournament_size; ++i) {
    const ScoredGenome& cand = scored[any_pick(rng)];
    if (best == nullptr || scored_genome_sorts_before(cand, *best)) {
      best = &cand;
    }
  }
  return best->genome;
}

}  // namespace g3pvm::evo
