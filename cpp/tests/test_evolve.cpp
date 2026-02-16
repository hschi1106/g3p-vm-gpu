#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/evolve.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

std::vector<g3pvm::evo::FitnessCase> simple_cases() {
  using g3pvm::Value;
  using g3pvm::evo::FitnessCase;
  return {
      FitnessCase{{{"x", Value::from_int(0)}, {"y", Value::from_int(0)}}, Value::from_int(0)},
      FitnessCase{{{"x", Value::from_int(1)}, {"y", Value::from_int(2)}}, Value::from_int(3)},
      FitnessCase{{{"x", Value::from_int(-1)}, {"y", Value::from_int(4)}}, Value::from_int(3)},
      FitnessCase{{{"x", Value::from_int(3)}, {"y", Value::from_int(-2)}}, Value::from_int(1)},
  };
}

bool run_one(g3pvm::evo::SelectionMethod method) {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 24;
  cfg.generations = 8;
  cfg.elitism = 2;
  cfg.mutation_rate = 0.7;
  cfg.crossover_rate = 0.9;
  cfg.crossover_method = g3pvm::evo::CrossoverMethod::Hybrid;
  cfg.selection_method = method;
  cfg.seed = 42;

  const auto result = g3pvm::evo::evolve_population(simple_cases(), cfg);
  if (!check(static_cast<int>(result.history_best_fitness.size()) == cfg.generations,
             "history_best_fitness length mismatch")) {
    return false;
  }
  if (!check(static_cast<int>(result.history_mean_fitness.size()) == cfg.generations,
             "history_mean_fitness length mismatch")) {
    return false;
  }
  if (!check(static_cast<int>(result.final_population.size()) == cfg.population_size,
             "final_population length mismatch")) {
    return false;
  }
  const double min_hist = *std::min_element(result.history_best_fitness.begin(), result.history_best_fitness.end());
  if (!check(result.best.fitness >= min_hist, "best fitness should be >= min history best")) {
    return false;
  }
  return true;
}

bool test_selection_methods() {
  using g3pvm::evo::SelectionMethod;
  for (SelectionMethod m : {SelectionMethod::Tournament,
                            SelectionMethod::Roulette,
                            SelectionMethod::Rank,
                            SelectionMethod::Truncation,
                            SelectionMethod::Random}) {
    if (!run_one(m)) {
      return false;
    }
  }
  return true;
}

bool test_determinism_seed() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 12;
  cfg.generations = 5;
  cfg.seed = 123;
  cfg.selection_method = g3pvm::evo::SelectionMethod::Tournament;

  const auto a = g3pvm::evo::evolve_population(simple_cases(), cfg);
  const auto b = g3pvm::evo::evolve_population(simple_cases(), cfg);

  if (!check(a.history_best_fitness.size() == b.history_best_fitness.size(), "determinism history length mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < a.history_best_fitness.size(); ++i) {
    if (!check(a.history_best_fitness[i] == b.history_best_fitness[i], "determinism best fitness mismatch")) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!test_selection_methods()) return 1;
  if (!test_determinism_seed()) return 1;
  std::cout << "g3pvm_test_evolve: OK\n";
  return 0;
}
