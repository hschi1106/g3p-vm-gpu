#pragma once

namespace g3pvm::evo::repro {

struct ReproductionStats {
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

}  // namespace g3pvm::evo::repro
