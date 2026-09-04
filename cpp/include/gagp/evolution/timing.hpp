#pragma once

#include <vector>

namespace gagp::evo {

struct EvaluationTiming {
  double cpu_compile_ms = 0.0;
  double gpu_compile_ms = 0.0;
  double gpu_eval_call_ms = 0.0;
  double gpu_eval_pack_ms = 0.0;
  double gpu_eval_launch_prep_ms = 0.0;
  double gpu_eval_upload_ms = 0.0;
  double gpu_eval_kernel_ms = 0.0;
  double gpu_eval_copyback_ms = 0.0;
  double gpu_eval_teardown_ms = 0.0;

  double gpu_eval_pack_upload_ms() const {
    return gpu_eval_pack_ms + gpu_eval_upload_ms;
  }
};

struct ReproductionTiming {
  double selection_ms = 0.0;
  double crossover_ms = 0.0;
  double mutation_ms = 0.0;
  double prepare_inputs_ms = 0.0;
  double setup_ms = 0.0;
  double preprocess_ms = 0.0;
  double pack_ms = 0.0;
  double upload_ms = 0.0;
  double kernel_ms = 0.0;
  double copyback_ms = 0.0;
  double decode_ms = 0.0;
  double teardown_ms = 0.0;
  double selection_kernel_ms = 0.0;
  double variation_kernel_ms = 0.0;
};

struct GenerationTiming {
  double eval_ms = 0.0;
  double repro_ms = 0.0;
  double total_ms = 0.0;
  EvaluationTiming evaluation;
  ReproductionTiming reproduction;
};

struct EvolutionTiming {
  double init_population_ms = 0.0;
  double gpu_eval_init_ms = 0.0;
  double final_eval_ms = 0.0;
  double total_ms = 0.0;
  EvaluationTiming evaluation_totals;
  ReproductionTiming reproduction_totals;
  std::vector<GenerationTiming> generations;
};

void accumulate_timing(EvaluationTiming* total, const EvaluationTiming& value);
void accumulate_timing(ReproductionTiming* total, const ReproductionTiming& value);

}  // namespace gagp::evo
