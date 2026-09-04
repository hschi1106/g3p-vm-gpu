#include "gagp/evolution/population_init.hpp"

#include <cstdint>
#include <stdexcept>

#include "gagp/evolution/evolve.hpp"
#include "gagp/evolution/genome_generation.hpp"

namespace gagp::evo {
namespace {

bool should_seed_for_expected_return_type(RType type) {
  return type == RType::String || type == RType::IntList ||
         type == RType::FloatList || type == RType::StringList;
}

}  // namespace

PopulationInitialization initialize_population(
    const EvolutionConfig& config,
    const CaseSet& case_set,
    const std::vector<ProgramGenome>* replay_population) {
  PopulationInitialization out;
  if (replay_population != nullptr) {
    out.population = *replay_population;
    out.replayed = true;
  } else {
    out.population.reserve(static_cast<std::size_t>(config.population_size));
    const bool seed_for_return_type =
        should_seed_for_expected_return_type(case_set.expected_return_type) &&
        config.grammar.allows_type(case_set.expected_return_type);
    for (int i = 0; i < config.population_size; ++i) {
      const std::uint64_t seed = config.seed + static_cast<std::uint64_t>(i);
      if (seed_for_return_type) {
        out.population.push_back(generate_random_genome_for_return_type(
            seed, case_set.expected_return_type, config.limits,
            case_set.input_specs, config.grammar));
      } else {
        out.population.push_back(generate_random_genome(
            seed, config.limits, case_set.input_specs, config.grammar));
      }
    }
  }
  if (static_cast<int>(out.population.size()) != config.population_size) {
    throw std::invalid_argument("initial_population size must match population_size");
  }
  return out;
}

}  // namespace gagp::evo
