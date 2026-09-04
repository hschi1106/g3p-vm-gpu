#include "gagp/evolution/evolve.hpp"

#include <chrono>
#include <stdexcept>
#include <unordered_map>

#include "gagp/evolution/compiler.hpp"
#include "gagp/evolution/lifecycle.hpp"
#include "gagp/evolution/population_init.hpp"
#include "gagp/evolution/repro/backend.hpp"
#include "gagp/evolution/repro/gpu.hpp"
#include "gagp/evolution/selection.hpp"
#include "gagp/runtime/cpu/fitness_cpu.hpp"
#ifdef GAGP_HAS_CUDA
#include "gagp/runtime/gpu/fitness_gpu.hpp"
#endif

namespace gagp::evo {

namespace {

struct CompileCache {
  std::unordered_map<std::string, BytecodeProgram> by_program;
};

struct CompiledPopulation {
  std::vector<BytecodeProgram> programs;
  double compile_ms = 0.0;
};

struct PopulationEvaluation {
  std::vector<double> fitness;
  EvaluationTiming timing;
};

ReproductionTiming reproduction_timing_from_stats(
    const repro::ReproductionStats& stats) {
  ReproductionTiming timing;
  timing.selection_ms = stats.selection_ms;
  timing.crossover_ms = stats.crossover_ms;
  timing.mutation_ms = stats.mutation_ms;
  timing.prepare_inputs_ms = stats.prepare_inputs_ms;
  timing.setup_ms = stats.setup_ms;
  timing.preprocess_ms = stats.preprocess_ms;
  timing.pack_ms = stats.pack_ms;
  timing.upload_ms = stats.upload_ms;
  timing.kernel_ms = stats.kernel_ms;
  timing.copyback_ms = stats.copyback_ms;
  timing.decode_ms = stats.decode_ms;
  timing.teardown_ms = stats.teardown_ms;
  timing.selection_kernel_ms = stats.selection_kernel_ms;
  timing.variation_kernel_ms = stats.variation_kernel_ms;
  return timing;
}

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
    GenerationTiming* generation_timing,
    double* fitness_sum_out,
    std::vector<double>* raw_fitness_out,
    bool sort_output) {
  for (double& value : evaluation.fitness) {
    value = canonicalize_fitness_for_ranking(value);
  }
  if (evaluation.fitness.size() != population.size()) {
    throw std::runtime_error("fitness size mismatch");
  }

  accumulate_timing(&result->timing.evaluation_totals, evaluation.timing);
  if (generation_timing != nullptr) generation_timing->evaluation = evaluation.timing;

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
    GenerationTiming* generation_timing,
    double* fitness_sum_out,
    std::vector<double>* raw_fitness_out,
    bool sort_output) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  PopulationEvaluation evaluation;
  evaluation.timing.cpu_compile_ms = compiled.compile_ms;
  evaluation.fitness = eval_fitness_cpu(
      compiled.programs, shared_cases, shared_answer, fuel, penalty, reduction_lanes);
  return record_and_rank_evaluation(
      population, std::move(evaluation), result, generation_timing,
      fitness_sum_out, raw_fitness_out, sort_output);
}

#ifdef GAGP_HAS_CUDA
std::vector<ScoredGenomeRef> score_population_gpu_refs(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    FitnessSessionGpu* session,
    CompileCache* compile_cache,
    EvolutionResult* result,
    GenerationTiming* generation_timing,
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
  evaluation.timing.gpu_compile_ms = compiled.compile_ms;
  evaluation.timing.gpu_eval_call_ms = fit.timing.total_ms;
  evaluation.timing.gpu_eval_pack_ms = fit.timing.pack_ms;
  evaluation.timing.gpu_eval_launch_prep_ms = fit.timing.launch_prep_ms;
  evaluation.timing.gpu_eval_upload_ms = fit.timing.upload_ms;
  evaluation.timing.gpu_eval_kernel_ms = fit.timing.kernel_ms;
  evaluation.timing.gpu_eval_copyback_ms = fit.timing.copyback_ms;
  evaluation.timing.gpu_eval_teardown_ms = fit.timing.teardown_ms;
  return record_and_rank_evaluation(
      population, std::move(evaluation), result, generation_timing,
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
      cfg.fuel, cfg.penalty, cfg.gpu_blocksize, nullptr, &result, nullptr,
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
  const PayloadLifetimeManager payload_lifetime(cases);
  const auto init_t0 = std::chrono::steady_clock::now();
  PopulationInitialization initialization =
      initialize_population(cfg, case_set, initial_population);
  std::vector<ProgramGenome> population = std::move(initialization.population);
  const auto init_t1 = std::chrono::steady_clock::now();

  EvolutionResult result;
  result.timing.init_population_ms =
      std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
  result.timing.generations.reserve(static_cast<std::size_t>(cfg.generations));

#ifdef GAGP_HAS_CUDA
  FitnessSessionGpu gpu_session;
#endif
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef GAGP_HAS_CUDA
    const FitnessSessionInitResult init_result =
        gpu_session.init(case_set.bindings, case_set.expected_values, cfg.fuel,
                         cfg.gpu_blocksize, cfg.penalty);
    if (!init_result.ok) {
      throw std::runtime_error("gpu fitness session init failed: " + init_result.err.message);
    }
    result.timing.gpu_eval_init_ms = init_result.timing.total_ms;
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  }

  payload_lifetime.retain(population, result.history_best);

  for (int gen = 0; gen < cfg.generations; ++gen) {
    GenerationTiming generation_timing;
    const auto gen_t0 = std::chrono::steady_clock::now();
    const auto eval_t0 = std::chrono::steady_clock::now();
    std::vector<ScoredGenomeRef> scored;
    double fitness_sum = 0.0;
    std::vector<double> raw_fitness;
    const bool overlap_gpu = gpu_reproduction_overlap_enabled(cfg);
    std::future<OverlapPrepared> overlap_future;
    if (overlap_gpu) {
      overlap_future = start_gpu_reproduction_overlap(
          population, reproduction_cfg, rng());
    }
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef GAGP_HAS_CUDA
      scored = score_population_gpu_refs(population, case_set.input_names, &gpu_session, nullptr,
                                         &result, &generation_timing, &fitness_sum,
                                         overlap_gpu ? &raw_fitness : nullptr, true);
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      scored = score_population_cpu_refs(
          population, case_set.input_names, case_set.bindings, case_set.expected_values,
          cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
          nullptr, &result, &generation_timing, &fitness_sum, nullptr, true);
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
      reproduction = finish_gpu_reproduction_overlap(
          &overlap_future, population, raw_fitness, reproduction_cfg);
    } else {
      reproduction = repro::run_reproduction_backend(scored, reproduction_cfg, rng);
    }
    const auto repro_t1 = std::chrono::steady_clock::now();

    population = std::move(reproduction.next_population);
    payload_lifetime.retain(population, result.history_best);
    const auto gen_t1 = std::chrono::steady_clock::now();

    generation_timing.reproduction = reproduction_timing_from_stats(reproduction.stats);
    accumulate_timing(&result.timing.reproduction_totals,
                      generation_timing.reproduction);
    generation_timing.eval_ms =
        std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count();
    generation_timing.repro_ms =
        std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count();
    generation_timing.total_ms =
        std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
    result.timing.generations.push_back(generation_timing);
    (void)gen;
  }

  if (cfg.skip_final_eval) {
    result.final_eval_skipped = true;
    result.timing.final_eval_ms = 0.0;
    const std::vector<ProgramGenome> empty_population;
    payload_lifetime.retain(empty_population, result.history_best);
  } else {
    const auto final_eval_t0 = std::chrono::steady_clock::now();
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef GAGP_HAS_CUDA
      const std::vector<ScoredGenomeRef> final_scored =
          score_population_gpu_refs(population, case_set.input_names, &gpu_session,
                                    nullptr, &result, nullptr, nullptr, nullptr, true);
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
          nullptr, &result, nullptr, nullptr, nullptr, true);
      result.best = materialize_scored_genome(final_scored.front());
      result.final_population = cfg.retain_final_population
                                    ? materialize_scored_population(final_scored)
                                    : std::vector<ScoredGenome>{};
    }
    const auto final_eval_t1 = std::chrono::steady_clock::now();
    result.timing.final_eval_ms =
        std::chrono::duration<double, std::milli>(final_eval_t1 - final_eval_t0).count();
    const std::vector<ProgramGenome> empty_population;
    const std::vector<ProgramGenome>& retained_population = cfg.retain_final_population ? population : empty_population;
    payload_lifetime.retain(
        retained_population, result.history_best, &result.best,
        cfg.retain_final_population ? &result.final_population : nullptr);
  }
  result.timing.total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - all_t0).count();
  return result;
}

}  // namespace gagp::evo
