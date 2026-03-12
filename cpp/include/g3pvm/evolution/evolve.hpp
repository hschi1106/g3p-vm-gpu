#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/core/value.hpp"

namespace g3pvm::evo {

enum class EvalEngine {
  CPU,
  GPU,
};

using NamedInputs = std::unordered_map<std::string, Value>;

struct EvalCase {
  NamedInputs inputs;
  Value expected = Value::none();
};

struct EvolutionConfig {
  int population_size = 64;
  int generations = 40;
  double mutation_rate = 0.5;
  double mutation_subtree_prob = 0.8;
  double crossover_rate = 0.9;
  double penalty = 1.0;
  EvalEngine eval_engine = EvalEngine::CPU;
  int gpu_blocksize = 256;
  int selection_pressure = 3;
  std::uint64_t seed = 0;
  int fuel = 20000;
  Limits limits;
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
  double gpu_session_init_ms = 0.0;
  double final_eval_ms = 0.0;
  double cpu_generations_program_compile_ms_total = 0.0;
  double gpu_generations_program_compile_ms_total = 0.0;
  double gpu_generations_pack_upload_ms_total = 0.0;
  double gpu_generations_kernel_ms_total = 0.0;
  double gpu_generations_copyback_ms_total = 0.0;
  double generations_selection_ms_total = 0.0;
  double generations_crossover_ms_total = 0.0;
  double generations_mutation_ms_total = 0.0;
  double total_ms = 0.0;
  std::vector<double> generation_eval_ms;
  std::vector<double> generation_repro_ms;
  std::vector<double> generation_total_ms;
  std::vector<double> generation_cpu_program_compile_ms;
  std::vector<double> generation_gpu_program_compile_ms;
  std::vector<double> generation_gpu_pack_upload_ms;
  std::vector<double> generation_gpu_kernel_ms;
  std::vector<double> generation_gpu_copyback_ms;
  std::vector<double> generation_selection_ms;
  std::vector<double> generation_crossover_ms;
  std::vector<double> generation_mutation_ms;
};

struct EvolutionRun {
  EvolutionResult result;
  EvolutionTiming timing;
};

std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<EvalCase>& cases,
                                              const EvolutionConfig& cfg);
ProgramGenome tournament_selection(const std::vector<ScoredGenome>& scored,
                                   std::mt19937_64& rng,
                                   int selection_pressure);
EvolutionResult evolve_population(const std::vector<EvalCase>& cases,
                                  const EvolutionConfig& cfg,
                                  const std::vector<ProgramGenome>* initial_population = nullptr);
EvolutionRun evolve_population_profiled(const std::vector<EvalCase>& cases,
                                        const EvolutionConfig& cfg,
                                        const std::vector<ProgramGenome>* initial_population = nullptr);

std::string eval_engine_name(EvalEngine engine);

}  // namespace g3pvm::evo
