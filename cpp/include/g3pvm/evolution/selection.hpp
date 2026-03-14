#pragma once

#include <random>
#include <vector>

#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

struct ScoredGenome {
  ProgramGenome genome;
  double fitness = 0.0;
};

double canonicalize_fitness_for_ranking(double fitness);

bool scored_genome_sorts_before(const ScoredGenome& a, const ScoredGenome& b);

std::vector<ProgramGenome> tournament_selection_without_replacement(
    const std::vector<ScoredGenome>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count);

}  // namespace g3pvm::evo
