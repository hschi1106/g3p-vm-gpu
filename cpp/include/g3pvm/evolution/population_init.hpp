#pragma once

#include <vector>

#include "g3pvm/evolution/case_set.hpp"
#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

struct EvolutionConfig;

struct PopulationInitialization {
  std::vector<ProgramGenome> population;
  bool replayed = false;
};

PopulationInitialization initialize_population(
    const EvolutionConfig& config,
    const CaseSet& case_set,
    const std::vector<ProgramGenome>* replay_population = nullptr);

}  // namespace g3pvm::evo
