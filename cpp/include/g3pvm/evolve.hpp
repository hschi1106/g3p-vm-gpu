#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/evo_ast.hpp"
#include "g3pvm/value.hpp"

namespace g3pvm::evo {

enum class SelectionMethod {
  Tournament,
  Roulette,
  Rank,
  Truncation,
  Random,
};

using NamedInputs = std::unordered_map<std::string, Value>;

struct FitnessCase {
  NamedInputs inputs;
  Value expected = Value::none();
};

struct EvolutionConfig {
  int population_size = 64;
  int generations = 40;
  int elitism = 2;
  double mutation_rate = 0.5;
  double crossover_rate = 0.9;
  CrossoverMethod crossover_method = CrossoverMethod::Hybrid;
  SelectionMethod selection_method = SelectionMethod::Tournament;
  int tournament_k = 3;
  double truncation_ratio = 0.5;
  std::uint64_t seed = 0;
  int fuel = 20000;
  Limits limits;
  double float_abs_tol = 1e-12;
  double float_rel_tol = 1e-12;
  double reward_match = 1.0;
  double penalty_mismatch = 0.0;
  double penalty_error = -1.0;
};

struct ScoredGenome {
  ProgramGenome genome;
  double fitness = 0.0;
};

struct EvolutionResult {
  ScoredGenome best;
  std::vector<ScoredGenome> history_best;
  std::vector<double> history_best_fitness;
  std::vector<double> history_mean_fitness;
  std::vector<ScoredGenome> final_population;
};

struct EvolutionTiming {
  double init_population_ms = 0.0;
  double final_eval_ms = 0.0;
  double total_ms = 0.0;
  std::vector<double> generation_eval_ms;
  std::vector<double> generation_repro_ms;
  std::vector<double> generation_total_ms;
};

struct EvolutionRun {
  EvolutionResult result;
  EvolutionTiming timing;
};

double evaluate_genome(const ProgramGenome& genome,
                       const std::vector<FitnessCase>& cases,
                       const EvolutionConfig& cfg);
std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<FitnessCase>& cases,
                                              const EvolutionConfig& cfg);
ProgramGenome select_parent(const std::vector<ScoredGenome>& scored,
                            std::mt19937_64& rng,
                            SelectionMethod method,
                            int tournament_k,
                            double truncation_ratio);
EvolutionResult evolve_population(const std::vector<FitnessCase>& cases,
                                  const EvolutionConfig& cfg,
                                  const std::vector<ProgramGenome>* initial_population = nullptr);
EvolutionRun evolve_population_profiled(const std::vector<FitnessCase>& cases,
                                        const EvolutionConfig& cfg,
                                        const std::vector<ProgramGenome>* initial_population = nullptr);

std::string selection_method_name(SelectionMethod method);
std::string crossover_method_name(CrossoverMethod method);

}  // namespace g3pvm::evo
