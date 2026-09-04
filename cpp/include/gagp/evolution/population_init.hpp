#pragma once

#include <vector>

#include "gagp/evolution/case_set.hpp"
#include "gagp/evolution/genome.hpp"

namespace gagp::evo {

struct EvolutionConfig;

struct PopulationInitialization {
  std::vector<ProgramGenome> population;
  bool replayed = false;
};

PopulationInitialization initialize_population(
    const EvolutionConfig& config,
    const CaseSet& case_set,
    const std::vector<ProgramGenome>* replay_population = nullptr);

}  // namespace gagp::evo
