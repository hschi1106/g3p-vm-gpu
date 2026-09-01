#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/evolution/case_set.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/population_init.hpp"
#include "g3pvm/evolution/selection.hpp"

namespace {

g3pvm::evo::ProgramGenome keyed_genome(const char* key) {
  g3pvm::evo::ProgramGenome genome;
  genome.meta.program_key = key;
  return genome;
}

}  // namespace

int main() {
  using g3pvm::Value;
  using g3pvm::evo::EvalCase;
  using g3pvm::evo::RType;

  const std::vector<EvalCase> cases = {
      EvalCase{{{"z", Value::from_int(2)}, {"a", Value::from_float(1.5)}},
               Value::from_int(3)},
      EvalCase{{{"z", Value::from_float(4.0)}, {"a", Value::from_float(2.5)}},
               Value::from_int(4)},
  };
  const auto case_set = g3pvm::evo::prepare_case_set(
      cases, g3pvm::evo::GrammarConfig::all_enabled());
  assert((case_set.input_names == std::vector<std::string>{"a", "z"}));
  assert(case_set.input_specs.size() == 2);
  assert(case_set.input_specs[0].type == RType::Float);
  assert(case_set.input_specs[1].type == RType::Any);
  assert(case_set.bindings.size() == 2 && case_set.bindings[0].size() == 2);
  assert(case_set.bindings[0][0].idx == 0 && case_set.bindings[0][1].idx == 1);
  assert(case_set.expected_values.size() == 2);
  assert(case_set.expected_return_type == RType::Int);

  const auto mixed_expected = g3pvm::evo::prepare_case_set(
      {EvalCase{{}, Value::from_int(1)}, EvalCase{{}, Value::from_bool(true)}},
      g3pvm::evo::GrammarConfig::all_enabled());
  assert(mixed_expected.expected_return_type == RType::Invalid);

  std::vector<g3pvm::evo::ProgramGenome> population = {
      keyed_genome("z"), keyed_genome("a"), keyed_genome("m")};
  const auto ranked =
      g3pvm::evo::rank_population_refs(population, {1.0, 1.0, 2.0});
  assert(ranked[0].genome->meta.program_key == "m");
  assert(ranked[1].genome->meta.program_key == "a");
  assert(ranked[2].genome->meta.program_key == "z");
  const auto owned = g3pvm::evo::materialize_scored_population(ranked);
  assert(owned.size() == ranked.size());
  assert(owned[0].genome.meta.program_key == "m" && owned[0].fitness == 2.0);

  g3pvm::evo::EvolutionConfig config;
  config.population_size = static_cast<int>(population.size());
  const auto replay = g3pvm::evo::initialize_population(config, case_set, &population);
  assert(replay.replayed && replay.population.size() == population.size());
  assert(replay.population[1].meta.program_key == "a");

  config.population_size += 1;
  try {
    (void)g3pvm::evo::initialize_population(config, case_set, &population);
    assert(false && "expected replay size mismatch");
  } catch (const std::invalid_argument& error) {
    assert(std::string(error.what()) ==
           "initial_population size must match population_size");
  }
}
