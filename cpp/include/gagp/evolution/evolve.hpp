#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "gagp/evolution/case_set.hpp"
#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/grammar_config.hpp"
#include "gagp/evolution/input_spec.hpp"
#include "gagp/evolution/repro/backend.hpp"
#include "gagp/evolution/selection.hpp"
#include "gagp/evolution/timing.hpp"

namespace gagp::evo {

enum class EvalEngine {
  CPU,
  GPU,
};

struct EvolutionConfig {
  int population_size = 64;
  int generations = 40;
  double mutation_rate = 0.5;
  double mutation_subtree_prob = 0.8;
  double penalty = 1.0;
  EvalEngine eval_engine = EvalEngine::CPU;
  repro::ReproductionBackend reproduction_backend = repro::ReproductionBackend::Cpu;
  repro::CpuReproAblation cpu_repro_ablation = repro::CpuReproAblation::None;
  bool repro_overlap = false;
  int gpu_blocksize = 1024;
  int selection_pressure = 2;
  std::uint64_t seed = 0;
  int fuel = 20000;
  Limits limits;
  GrammarConfig grammar;
  bool skip_final_eval = false;
  bool retain_final_population = true;
  // Derived by evolve_population for verifier checks at reproduction boundaries.
  std::vector<InputSpec> verification_inputs;
};

struct EvolutionResult {
  ScoredGenome best;
  std::vector<ScoredGenome> history_best;
  std::vector<double> history_best_fitness;
  std::vector<double> history_mean_fitness;
  std::vector<ScoredGenome> final_population;
  bool final_eval_skipped = false;
  EvolutionTiming timing;
};

std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<EvalCase>& cases,
                                              const EvolutionConfig& cfg);
EvolutionResult evolve_population(const std::vector<EvalCase>& cases,
                                  const EvolutionConfig& cfg,
                                  const std::vector<ProgramGenome>* initial_population = nullptr);

std::string eval_engine_name(EvalEngine engine);

}  // namespace gagp::evo
