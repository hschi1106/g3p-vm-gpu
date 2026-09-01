#include "g3pvm/evolution/timing.hpp"

namespace g3pvm::evo {

void accumulate_timing(EvaluationTiming* total, const EvaluationTiming& value) {
  total->cpu_compile_ms += value.cpu_compile_ms;
  total->gpu_compile_ms += value.gpu_compile_ms;
  total->gpu_eval_call_ms += value.gpu_eval_call_ms;
  total->gpu_eval_pack_ms += value.gpu_eval_pack_ms;
  total->gpu_eval_launch_prep_ms += value.gpu_eval_launch_prep_ms;
  total->gpu_eval_upload_ms += value.gpu_eval_upload_ms;
  total->gpu_eval_kernel_ms += value.gpu_eval_kernel_ms;
  total->gpu_eval_copyback_ms += value.gpu_eval_copyback_ms;
  total->gpu_eval_teardown_ms += value.gpu_eval_teardown_ms;
}

void accumulate_timing(ReproductionTiming* total, const ReproductionTiming& value) {
  total->selection_ms += value.selection_ms;
  total->crossover_ms += value.crossover_ms;
  total->mutation_ms += value.mutation_ms;
  total->prepare_inputs_ms += value.prepare_inputs_ms;
  total->setup_ms += value.setup_ms;
  total->preprocess_ms += value.preprocess_ms;
  total->pack_ms += value.pack_ms;
  total->upload_ms += value.upload_ms;
  total->kernel_ms += value.kernel_ms;
  total->copyback_ms += value.copyback_ms;
  total->decode_ms += value.decode_ms;
  total->teardown_ms += value.teardown_ms;
  total->selection_kernel_ms += value.selection_kernel_ms;
  total->variation_kernel_ms += value.variation_kernel_ms;
}

}  // namespace g3pvm::evo
