#include "g3pvm/evolution/evolve.hpp"

#include <chrono>
#include <future>
#include <stdexcept>
#include <unordered_map>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/population_init.hpp"
#include "g3pvm/evolution/repro/backend.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#endif

namespace g3pvm::evo {

namespace {

void append_payload_root_if_needed(const Value& value, std::vector<Value>* roots) {
  if (roots == nullptr) {
    return;
  }
  if (value.tag == ValueTag::String || value.tag == ValueTag::IntList ||
      value.tag == ValueTag::FloatList || value.tag == ValueTag::StringList) {
    roots->push_back(value);
  }
}

void append_payload_roots_from_cases(const std::vector<EvalCase>& cases, std::vector<Value>* roots) {
  if (roots == nullptr) {
    return;
  }
  for (const EvalCase& one_case : cases) {
    for (const auto& kv : one_case.inputs) {
      append_payload_root_if_needed(kv.second, roots);
    }
    append_payload_root_if_needed(one_case.expected, roots);
  }
}

void append_payload_roots_from_genome(const ProgramGenome& genome, std::vector<Value>* roots) {
  if (roots == nullptr) {
    return;
  }
  for (const Value& value : genome.ast.consts) {
    append_payload_root_if_needed(value, roots);
  }
}

void append_payload_roots_from_population(const std::vector<ProgramGenome>& population, std::vector<Value>* roots) {
  if (roots == nullptr) {
    return;
  }
  for (const ProgramGenome& genome : population) {
    append_payload_roots_from_genome(genome, roots);
  }
}

void append_payload_roots_from_scored(const std::vector<ScoredGenome>& scored, std::vector<Value>* roots) {
  if (roots == nullptr) {
    return;
  }
  for (const ScoredGenome& one : scored) {
    append_payload_roots_from_genome(one.genome, roots);
  }
}

std::vector<Value> build_live_payload_roots(const std::vector<Value>& case_payload_roots,
                                            const std::vector<ProgramGenome>& population,
                                            const std::vector<ScoredGenome>& history_best,
                                            const ScoredGenome* best,
                                            const std::vector<ScoredGenome>* final_population) {
  std::vector<Value> roots;
  roots.reserve(case_payload_roots.size() + population.size() * 8U + history_best.size() * 8U + 16U);
  roots.insert(roots.end(), case_payload_roots.begin(), case_payload_roots.end());
  append_payload_roots_from_population(population, &roots);
  append_payload_roots_from_scored(history_best, &roots);
  if (best != nullptr) {
    append_payload_roots_from_genome(best->genome, &roots);
  }
  if (final_population != nullptr) {
    append_payload_roots_from_scored(*final_population, &roots);
  }
  return roots;
}

struct CompileCache {
  std::unordered_map<std::string, BytecodeProgram> by_program;
};

struct CompiledPopulation {
  std::vector<BytecodeProgram> programs;
  double compile_ms = 0.0;
};

struct PopulationEvaluation {
  std::vector<double> fitness;
  bool gpu = false;
  double compile_ms = 0.0;
  double eval_call_ms = 0.0;
  double pack_ms = 0.0;
  double launch_prep_ms = 0.0;
  double upload_ms = 0.0;
  double kernel_ms = 0.0;
  double copyback_ms = 0.0;
  double teardown_ms = 0.0;
};

CompiledPopulation compile_population(const std::vector<ProgramGenome>& population,
                                      const std::vector<std::string>& input_names,
                                      CompileCache* compile_cache) {
  CompiledPopulation out;
  out.programs.reserve(population.size());
  CompileCache local_cache;
  CompileCache* cache = (compile_cache != nullptr) ? compile_cache : &local_cache;
  for (const ProgramGenome& genome : population) {
    const std::string& key = genome.meta.program_key;
    if (cache != nullptr) {
      auto it = cache->by_program.find(key);
      if (it != cache->by_program.end()) {
        out.programs.push_back(it->second);
        continue;
      }
    }

    const auto t0 = std::chrono::steady_clock::now();
    BytecodeProgram bc = compile_for_eval(genome, input_names);
    const auto t1 = std::chrono::steady_clock::now();
    out.compile_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (cache != nullptr) {
      cache->by_program.emplace(key, bc);
    }
    out.programs.push_back(std::move(bc));
  }
  return out;
}

std::vector<ScoredGenomeRef> record_and_rank_evaluation(
    const std::vector<ProgramGenome>& population,
    PopulationEvaluation evaluation,
    EvolutionResult* result,
    bool record_per_gen,
    double* fitness_sum_out,
    std::vector<double>* raw_fitness_out,
    bool sort_output) {
  for (double& value : evaluation.fitness) {
    value = canonicalize_fitness_for_ranking(value);
  }
  if (evaluation.fitness.size() != population.size()) {
    throw std::runtime_error("fitness size mismatch");
  }

  if (evaluation.gpu) {
    result->gpu_compile_ms_total += evaluation.compile_ms;
    result->gpu_eval_call_ms_total += evaluation.eval_call_ms;
    result->gpu_eval_pack_ms_total += evaluation.pack_ms;
    result->gpu_eval_launch_prep_ms_total += evaluation.launch_prep_ms;
    result->gpu_eval_upload_ms_total += evaluation.upload_ms;
    result->gpu_eval_pack_upload_ms_total += evaluation.pack_ms + evaluation.upload_ms;
    result->gpu_eval_kernel_ms_total += evaluation.kernel_ms;
    result->gpu_eval_copyback_ms_total += evaluation.copyback_ms;
    result->gpu_eval_teardown_ms_total += evaluation.teardown_ms;
  } else {
    result->cpu_compile_ms_total += evaluation.compile_ms;
  }

  if (record_per_gen) {
    result->generation_cpu_compile_ms.push_back(evaluation.gpu ? 0.0 : evaluation.compile_ms);
    result->generation_gpu_compile_ms.push_back(evaluation.gpu ? evaluation.compile_ms : 0.0);
    result->generation_gpu_eval_call_ms.push_back(evaluation.eval_call_ms);
    result->generation_gpu_eval_pack_ms.push_back(evaluation.pack_ms);
    result->generation_gpu_eval_launch_prep_ms.push_back(evaluation.launch_prep_ms);
    result->generation_gpu_eval_upload_ms.push_back(evaluation.upload_ms);
    result->generation_gpu_eval_pack_upload_ms.push_back(evaluation.pack_ms + evaluation.upload_ms);
    result->generation_gpu_eval_kernel_ms.push_back(evaluation.kernel_ms);
    result->generation_gpu_eval_copyback_ms.push_back(evaluation.copyback_ms);
    result->generation_gpu_eval_teardown_ms.push_back(evaluation.teardown_ms);
  }

  if (fitness_sum_out != nullptr) {
    long double sum = 0.0L;
    for (double one : evaluation.fitness) sum += static_cast<long double>(one);
    *fitness_sum_out = static_cast<double>(sum);
  }
  if (raw_fitness_out != nullptr) *raw_fitness_out = evaluation.fitness;
  return rank_population_refs(population, evaluation.fitness, sort_output);
}

std::vector<ScoredGenomeRef> score_population_cpu_refs(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    const std::vector<CaseBindings>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    double penalty,
    int reduction_lanes,
    CompileCache* compile_cache,
    EvolutionResult* result,
    bool record_per_gen,
    double* fitness_sum_out,
    std::vector<double>* raw_fitness_out,
    bool sort_output) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  PopulationEvaluation evaluation;
  evaluation.compile_ms = compiled.compile_ms;
  evaluation.fitness = eval_fitness_cpu(
      compiled.programs, shared_cases, shared_answer, fuel, penalty, reduction_lanes);
  return record_and_rank_evaluation(
      population, std::move(evaluation), result, record_per_gen,
      fitness_sum_out, raw_fitness_out, sort_output);
}

#ifdef G3PVM_HAS_CUDA
std::vector<ScoredGenomeRef> score_population_gpu_refs(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    FitnessSessionGpu* session,
    CompileCache* compile_cache,
    EvolutionResult* result,
    bool record_per_gen,
    double* fitness_sum_out,
    std::vector<double>* raw_fitness_out,
    bool sort_output) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  FitnessEvalResult fit = session->eval_programs(compiled.programs);
  if (!fit.ok) {
    throw std::runtime_error("gpu fitness evaluation failed: " + fit.err.message);
  }
  PopulationEvaluation evaluation;
  evaluation.fitness = std::move(fit.fitness);
  evaluation.gpu = true;
  evaluation.compile_ms = compiled.compile_ms;
  evaluation.eval_call_ms = fit.timing.total_ms;
  evaluation.pack_ms = fit.timing.pack_ms;
  evaluation.launch_prep_ms = fit.timing.launch_prep_ms;
  evaluation.upload_ms = fit.timing.upload_ms;
  evaluation.kernel_ms = fit.timing.kernel_ms;
  evaluation.copyback_ms = fit.timing.copyback_ms;
  evaluation.teardown_ms = fit.timing.teardown_ms;
  return record_and_rank_evaluation(
      population, std::move(evaluation), result, record_per_gen,
      fitness_sum_out, raw_fitness_out, sort_output);
}
#endif

}  // namespace

std::string eval_engine_name(EvalEngine engine) {
  if (engine == EvalEngine::GPU) return "gpu";
  return "cpu";
}

std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<EvalCase>& cases,
                                              const EvolutionConfig& cfg) {
  const CaseSet case_set = prepare_case_set(cases, cfg.grammar);
  EvolutionResult result;
  return materialize_scored_population(score_population_cpu_refs(
      population, case_set.input_names, case_set.bindings, case_set.expected_values,
      cfg.fuel, cfg.penalty, cfg.gpu_blocksize, nullptr, &result, false,
      nullptr, nullptr, true));
}

EvolutionResult evolve_population(const std::vector<EvalCase>& cases,
                                  const EvolutionConfig& cfg,
                                  const std::vector<ProgramGenome>* initial_population) {
  if (cases.empty()) {
    throw std::invalid_argument("cases must not be empty");
  }
  if (cfg.population_size <= 0) {
    throw std::invalid_argument("population_size must be > 0");
  }
  if (cfg.generations <= 0) {
    throw std::invalid_argument("generations must be > 0");
  }
  if (cfg.reproduction_backend != repro::ReproductionBackend::Cpu &&
      cfg.cpu_repro_ablation != repro::CpuReproAblation::None) {
    throw std::invalid_argument("cpu_repro_ablation requires cpu reproduction backend");
  }
  cfg.grammar.validate();

  const auto all_t0 = std::chrono::steady_clock::now();
  std::mt19937_64 rng(cfg.seed);
  const CaseSet case_set = prepare_case_set(cases, cfg.grammar);
  EvolutionConfig reproduction_cfg = cfg;
  reproduction_cfg.verification_inputs = case_set.input_specs;
  std::vector<Value> case_payload_roots;
  append_payload_roots_from_cases(cases, &case_payload_roots);
  const auto init_t0 = std::chrono::steady_clock::now();
  PopulationInitialization initialization =
      initialize_population(cfg, case_set, initial_population);
  std::vector<ProgramGenome> population = std::move(initialization.population);
  const auto init_t1 = std::chrono::steady_clock::now();

  EvolutionResult result;
  result.init_population_ms = std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
  result.generation_eval_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_total_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_cpu_compile_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_compile_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_call_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_pack_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_launch_prep_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_upload_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_pack_upload_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_kernel_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_copyback_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_eval_teardown_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_selection_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_crossover_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_mutation_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_prepare_inputs_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_setup_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_preprocess_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_pack_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_upload_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_kernel_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_copyback_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_decode_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_teardown_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_selection_kernel_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_variation_kernel_ms.reserve(static_cast<std::size_t>(cfg.generations));

#ifdef G3PVM_HAS_CUDA
  FitnessSessionGpu gpu_session;
#endif
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
    const FitnessSessionInitResult init_result =
        gpu_session.init(case_set.bindings, case_set.expected_values, cfg.fuel,
                         cfg.gpu_blocksize, cfg.penalty);
    if (!init_result.ok) {
      throw std::runtime_error("gpu fitness session init failed: " + init_result.err.message);
    }
    result.gpu_eval_init_ms = init_result.timing.total_ms;
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  }

  g3pvm::payload::retain_only(build_live_payload_roots(case_payload_roots, population, result.history_best,
                                                       nullptr, nullptr));

  for (int gen = 0; gen < cfg.generations; ++gen) {
    const auto gen_t0 = std::chrono::steady_clock::now();
    const auto eval_t0 = std::chrono::steady_clock::now();
    std::vector<ScoredGenomeRef> scored;
    double fitness_sum = 0.0;
    std::vector<double> raw_fitness;
    const bool overlap_gpu =
        cfg.eval_engine == EvalEngine::GPU &&
        cfg.reproduction_backend == repro::ReproductionBackend::Gpu &&
        cfg.repro_overlap;
    struct OverlapPrepared {
      repro::GpuReproPreparedData prepared;
      repro::ReproductionStats stats;
    };
    std::future<OverlapPrepared> overlap_future;
    if (overlap_gpu) {
      const std::uint64_t repro_seed = rng();
      overlap_future = std::async(std::launch::async, [population, reproduction_cfg, repro_seed]() {
        OverlapPrepared out;
        out.prepared = repro::prepare_gpu_repro_backend_inputs(
            population, reproduction_cfg, repro_seed, &out.stats);
        return out;
      });
    }
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
      scored = score_population_gpu_refs(population, case_set.input_names, &gpu_session, nullptr,
                                         &result, true, &fitness_sum, overlap_gpu ? &raw_fitness : nullptr, true);
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      scored = score_population_cpu_refs(
          population, case_set.input_names, case_set.bindings, case_set.expected_values,
          cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
          nullptr, &result, true, &fitness_sum, nullptr, true);
    }
    const auto eval_t1 = std::chrono::steady_clock::now();
    const ScoredGenomeRef& best = scored.front();
    result.history_best.push_back(materialize_scored_genome(best));
    result.history_best_fitness.push_back(best.fitness);

    const double mean = fitness_sum / static_cast<double>(scored.size());
    result.history_mean_fitness.push_back(mean);

    const auto repro_t0 = std::chrono::steady_clock::now();
    repro::ReproductionResult reproduction;
    if (overlap_gpu) {
      OverlapPrepared overlap = overlap_future.get();
      const std::vector<ScoredGenomeRef> repro_scored =
          rank_population_refs(population, raw_fitness, false);
      reproduction = repro::run_gpu_repro_backend_prepared(
          repro_scored, reproduction_cfg, overlap.prepared, &overlap.stats);
    } else {
      reproduction = repro::run_reproduction_backend(scored, reproduction_cfg, rng);
    }
    const auto repro_t1 = std::chrono::steady_clock::now();

    population = std::move(reproduction.next_population);
    g3pvm::payload::retain_only(build_live_payload_roots(case_payload_roots, population, result.history_best,
                                                         nullptr, nullptr));
    const auto gen_t1 = std::chrono::steady_clock::now();

    result.generations_selection_ms_total += reproduction.stats.selection_ms;
    result.generations_crossover_ms_total += reproduction.stats.crossover_ms;
    result.generations_mutation_ms_total += reproduction.stats.mutation_ms;
    result.generations_repro_prepare_inputs_ms_total += reproduction.stats.prepare_inputs_ms;
    result.generations_repro_setup_ms_total += reproduction.stats.setup_ms;
    result.generations_repro_preprocess_ms_total += reproduction.stats.preprocess_ms;
    result.generations_repro_pack_ms_total += reproduction.stats.pack_ms;
    result.generations_repro_upload_ms_total += reproduction.stats.upload_ms;
    result.generations_repro_kernel_ms_total += reproduction.stats.kernel_ms;
    result.generations_repro_copyback_ms_total += reproduction.stats.copyback_ms;
    result.generations_repro_decode_ms_total += reproduction.stats.decode_ms;
    result.generations_repro_teardown_ms_total += reproduction.stats.teardown_ms;
    result.generations_repro_selection_kernel_ms_total += reproduction.stats.selection_kernel_ms;
    result.generations_repro_variation_kernel_ms_total += reproduction.stats.variation_kernel_ms;
    result.generation_selection_ms.push_back(reproduction.stats.selection_ms);
    result.generation_crossover_ms.push_back(reproduction.stats.crossover_ms);
    result.generation_mutation_ms.push_back(reproduction.stats.mutation_ms);
    result.generation_repro_prepare_inputs_ms.push_back(reproduction.stats.prepare_inputs_ms);
    result.generation_repro_setup_ms.push_back(reproduction.stats.setup_ms);
    result.generation_repro_preprocess_ms.push_back(reproduction.stats.preprocess_ms);
    result.generation_repro_pack_ms.push_back(reproduction.stats.pack_ms);
    result.generation_repro_upload_ms.push_back(reproduction.stats.upload_ms);
    result.generation_repro_kernel_ms.push_back(reproduction.stats.kernel_ms);
    result.generation_repro_copyback_ms.push_back(reproduction.stats.copyback_ms);
    result.generation_repro_decode_ms.push_back(reproduction.stats.decode_ms);
    result.generation_repro_teardown_ms.push_back(reproduction.stats.teardown_ms);
    result.generation_repro_selection_kernel_ms.push_back(reproduction.stats.selection_kernel_ms);
    result.generation_repro_variation_kernel_ms.push_back(reproduction.stats.variation_kernel_ms);

    result.generation_eval_ms.push_back(
        std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count());
    result.generation_repro_ms.push_back(
        std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count());
    result.generation_total_ms.push_back(
        std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    (void)gen;
  }

  if (cfg.skip_final_eval) {
    result.final_eval_skipped = true;
    result.final_eval_ms = 0.0;
    const std::vector<ProgramGenome> empty_population;
    g3pvm::payload::retain_only(build_live_payload_roots(case_payload_roots, empty_population, result.history_best,
                                                         nullptr, nullptr));
  } else {
    const auto final_eval_t0 = std::chrono::steady_clock::now();
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
      const std::vector<ScoredGenomeRef> final_scored =
          score_population_gpu_refs(population, case_set.input_names, &gpu_session,
                                    nullptr, &result, false, nullptr, nullptr, true);
      result.best = materialize_scored_genome(final_scored.front());
      result.final_population = cfg.retain_final_population
                                    ? materialize_scored_population(final_scored)
                                    : std::vector<ScoredGenome>{};
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      const std::vector<ScoredGenomeRef> final_scored = score_population_cpu_refs(
          population, case_set.input_names, case_set.bindings,
          case_set.expected_values, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
          nullptr, &result, false, nullptr, nullptr, true);
      result.best = materialize_scored_genome(final_scored.front());
      result.final_population = cfg.retain_final_population
                                    ? materialize_scored_population(final_scored)
                                    : std::vector<ScoredGenome>{};
    }
    const auto final_eval_t1 = std::chrono::steady_clock::now();
    result.final_eval_ms = std::chrono::duration<double, std::milli>(final_eval_t1 - final_eval_t0).count();
    const std::vector<ProgramGenome> empty_population;
    const std::vector<ProgramGenome>& retained_population = cfg.retain_final_population ? population : empty_population;
    g3pvm::payload::retain_only(build_live_payload_roots(case_payload_roots, retained_population, result.history_best,
                                                         &result.best,
                                                         cfg.retain_final_population ? &result.final_population : nullptr));
  }
  result.total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - all_t0).count();
  return result;
}

}  // namespace g3pvm::evo
