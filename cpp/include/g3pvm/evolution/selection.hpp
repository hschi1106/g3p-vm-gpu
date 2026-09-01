#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

struct ScoredGenome {
  ProgramGenome genome;
  double fitness = 0.0;
};

struct ScoredGenomeRef {
  const ProgramGenome* genome = nullptr;
  double fitness = 0.0;
};

double canonicalize_fitness_for_ranking(double fitness);

bool scored_genome_sorts_before(const ScoredGenome& a, const ScoredGenome& b);
bool scored_genome_sorts_before(const ScoredGenomeRef& a, const ScoredGenomeRef& b);

std::vector<ScoredGenomeRef> rank_population_refs(
    const std::vector<ProgramGenome>& population,
    const std::vector<double>& fitness,
    bool sort_output = true);
ScoredGenome materialize_scored_genome(const ScoredGenomeRef& scored);
std::vector<ScoredGenome> materialize_scored_population(
    const std::vector<ScoredGenomeRef>& scored);

std::vector<std::size_t> tournament_selection_indices_without_replacement(
    const std::vector<ScoredGenome>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count);

std::vector<std::size_t> tournament_selection_indices_without_replacement(
    const std::vector<ScoredGenomeRef>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count);

std::vector<ProgramGenome> tournament_selection_without_replacement(
    const std::vector<ScoredGenome>& scored,
    std::mt19937_64& rng,
    int selection_pressure,
    int selection_count);

}  // namespace g3pvm::evo
