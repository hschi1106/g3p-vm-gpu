#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/evolution/case_set.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/lifecycle.hpp"
#include "g3pvm/evolution/population_init.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "g3pvm/evolution/timing.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

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

  g3pvm::evo::EvaluationTiming eval_a;
  eval_a.cpu_compile_ms = 2.0;
  eval_a.gpu_eval_pack_ms = 3.0;
  eval_a.gpu_eval_upload_ms = 5.0;
  g3pvm::evo::EvaluationTiming eval_total;
  g3pvm::evo::accumulate_timing(&eval_total, eval_a);
  g3pvm::evo::accumulate_timing(&eval_total, eval_a);
  assert(eval_total.cpu_compile_ms == 4.0);
  assert(eval_total.gpu_eval_pack_upload_ms() == 16.0);
  assert(eval_total.gpu_eval_kernel_ms == 0.0);

  g3pvm::evo::ReproductionTiming repro_a;
  repro_a.prepare_inputs_ms = 1.0;
  repro_a.decode_ms = 4.0;
  g3pvm::evo::ReproductionTiming repro_total;
  g3pvm::evo::accumulate_timing(&repro_total, repro_a);
  assert(repro_total.prepare_inputs_ms == 1.0);
  assert(repro_total.decode_ms == 4.0);
  assert(repro_total.kernel_ms == 0.0);

  g3pvm::evo::EvolutionConfig overlap_config;
  assert(!g3pvm::evo::gpu_reproduction_overlap_enabled(overlap_config));
  overlap_config.eval_engine = g3pvm::evo::EvalEngine::GPU;
  overlap_config.reproduction_backend =
      g3pvm::evo::repro::ReproductionBackend::Gpu;
  overlap_config.repro_overlap = true;
  assert(g3pvm::evo::gpu_reproduction_overlap_enabled(overlap_config));

  g3pvm::payload::clear();
  const Value retained = g3pvm::payload::make_string_value("retained");
  const Value discarded = g3pvm::payload::make_string_value("discarded");
  const g3pvm::evo::PayloadLifetimeManager lifetime(
      {EvalCase{{{"input", retained}}, Value::from_int(0)}});
  lifetime.retain({}, {});
  std::string payload_text;
  assert(g3pvm::payload::lookup_string(retained, &payload_text));
  assert(payload_text == "retained");
  assert(!g3pvm::payload::lookup_string(discarded, &payload_text));
  g3pvm::payload::clear();

  g3pvm::evo::EvolutionConfig timing_config;
  timing_config.population_size = 4;
  timing_config.generations = 2;
  timing_config.seed = 17;
  timing_config.skip_final_eval = true;
  const auto timed = g3pvm::evo::evolve_population(cases, timing_config);
  assert(timed.timing.generations.size() == 2);
  double cpu_compile_sum = 0.0;
  double selection_sum = 0.0;
  for (const auto& generation : timed.timing.generations) {
    cpu_compile_sum += generation.evaluation.cpu_compile_ms;
    selection_sum += generation.reproduction.selection_ms;
    assert(generation.evaluation.gpu_compile_ms == 0.0);
    assert(generation.evaluation.gpu_eval_call_ms == 0.0);
    assert(generation.evaluation.gpu_eval_kernel_ms == 0.0);
  }
  assert(timed.timing.evaluation_totals.cpu_compile_ms == cpu_compile_sum);
  assert(timed.timing.evaluation_totals.gpu_eval_call_ms == 0.0);
  assert(timed.timing.reproduction_totals.selection_ms == selection_sum);
  assert(timed.timing.final_eval_ms == 0.0);
}
