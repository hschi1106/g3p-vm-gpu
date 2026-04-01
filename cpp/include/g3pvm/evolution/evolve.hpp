#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/repro/backend.hpp"
#include "g3pvm/evolution/selection.hpp"
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
  double penalty = 1.0;
  EvalEngine eval_engine = EvalEngine::CPU;
  repro::ReproductionBackend reproduction_backend = repro::ReproductionBackend::Cpu;
  bool repro_overlap = false;
  int gpu_blocksize = 1024;
  int selection_pressure = 3;
  std::uint64_t seed = 0;
  int fuel = 20000;
  Limits limits;
  bool skip_final_eval = false;
};

struct EvolutionResult {
  ScoredGenome best;
  std::vector<ScoredGenome> history_best;
  std::vector<double> history_best_fitness;
  std::vector<double> history_mean_fitness;
  std::vector<ScoredGenome> final_population;
  double init_population_ms = 0.0;
  double gpu_eval_init_ms = 0.0;
  double final_eval_ms = 0.0;
  bool final_eval_skipped = false;
  double cpu_compile_ms_total = 0.0;
  double gpu_compile_ms_total = 0.0;
  double gpu_eval_call_ms_total = 0.0;
  double gpu_eval_pack_ms_total = 0.0;
  double gpu_eval_launch_prep_ms_total = 0.0;
  double gpu_eval_upload_ms_total = 0.0;
  double gpu_eval_pack_upload_ms_total = 0.0;
  double gpu_eval_kernel_ms_total = 0.0;
  double gpu_eval_copyback_ms_total = 0.0;
  double gpu_eval_teardown_ms_total = 0.0;
  double generations_selection_ms_total = 0.0;
  double generations_crossover_ms_total = 0.0;
  double generations_mutation_ms_total = 0.0;
  double generations_repro_prepare_inputs_ms_total = 0.0;
  double generations_repro_setup_ms_total = 0.0;
  double generations_repro_preprocess_ms_total = 0.0;
  double generations_repro_pack_ms_total = 0.0;
  double generations_repro_upload_ms_total = 0.0;
  double generations_repro_kernel_ms_total = 0.0;
  double generations_repro_copyback_ms_total = 0.0;
  double generations_repro_decode_ms_total = 0.0;
  double generations_repro_teardown_ms_total = 0.0;
  double generations_repro_selection_kernel_ms_total = 0.0;
  double generations_repro_variation_kernel_ms_total = 0.0;
  double total_ms = 0.0;
  std::vector<double> generation_eval_ms;
  std::vector<double> generation_repro_ms;
  std::vector<double> generation_total_ms;
  std::vector<double> generation_cpu_compile_ms;
  std::vector<double> generation_gpu_compile_ms;
  std::vector<double> generation_gpu_eval_call_ms;
  std::vector<double> generation_gpu_eval_pack_ms;
  std::vector<double> generation_gpu_eval_launch_prep_ms;
  std::vector<double> generation_gpu_eval_upload_ms;
  std::vector<double> generation_gpu_eval_pack_upload_ms;
  std::vector<double> generation_gpu_eval_kernel_ms;
  std::vector<double> generation_gpu_eval_copyback_ms;
  std::vector<double> generation_gpu_eval_teardown_ms;
  std::vector<double> generation_selection_ms;
  std::vector<double> generation_crossover_ms;
  std::vector<double> generation_mutation_ms;
  std::vector<double> generation_repro_prepare_inputs_ms;
  std::vector<double> generation_repro_setup_ms;
  std::vector<double> generation_repro_preprocess_ms;
  std::vector<double> generation_repro_pack_ms;
  std::vector<double> generation_repro_upload_ms;
  std::vector<double> generation_repro_kernel_ms;
  std::vector<double> generation_repro_copyback_ms;
  std::vector<double> generation_repro_decode_ms;
  std::vector<double> generation_repro_teardown_ms;
  std::vector<double> generation_repro_selection_kernel_ms;
  std::vector<double> generation_repro_variation_kernel_ms;
};

std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<EvalCase>& cases,
                                              const EvolutionConfig& cfg);
EvolutionResult evolve_population(const std::vector<EvalCase>& cases,
                                  const EvolutionConfig& cfg,
                                  const std::vector<ProgramGenome>* initial_population = nullptr);

std::string eval_engine_name(EvalEngine engine);

}  // namespace g3pvm::evo
