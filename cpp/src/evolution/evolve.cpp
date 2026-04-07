#include "g3pvm/evolution/evolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/evolution/repro/backend.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#endif

namespace g3pvm::evo {

namespace {

std::vector<std::string> build_canonical_input_names(const std::vector<EvalCase>& cases);

RType infer_input_rtype(const Value& value) {
  if (value.tag == ValueTag::Bool) return RType::Bool;
  if (value.tag == ValueTag::None) return RType::NoneType;
  if (value.tag == ValueTag::String) return RType::String;
  if (value.tag == ValueTag::NumList) return RType::NumList;
  if (value.tag == ValueTag::StringList) return RType::StringList;
  if (value.tag == ValueTag::FallbackToken) return RType::Any;
  return RType::Num;
}

RType merge_input_rtype(RType a, RType b) {
  if (a == RType::Invalid) return b;
  if (b == RType::Invalid || a == b) return a;
  return RType::Any;
}

std::vector<InputSpec> build_canonical_input_specs(const std::vector<EvalCase>& cases) {
  const std::vector<std::string> input_names = build_canonical_input_names(cases);
  std::vector<InputSpec> specs;
  specs.reserve(input_names.size());
  for (const std::string& name : input_names) {
    RType type = RType::Invalid;
    for (const EvalCase& one_case : cases) {
      const auto it = one_case.inputs.find(name);
      if (it == one_case.inputs.end()) {
        continue;
      }
      type = merge_input_rtype(type, infer_input_rtype(it->second));
    }
    specs.push_back(InputSpec{name, type});
  }
  return specs;
}

std::vector<ProgramGenome> init_population(const EvolutionConfig& cfg,
                                           const std::vector<InputSpec>& input_specs) {
  std::vector<ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(cfg.population_size));
  for (int i = 0; i < cfg.population_size; ++i) {
    out.push_back(generate_random_genome(cfg.seed + static_cast<std::uint64_t>(i), cfg.limits, input_specs, cfg.grammar));
  }
  return out;
}

std::vector<std::string> build_canonical_input_names(const std::vector<EvalCase>& cases) {
  std::set<std::string> names;
  for (const EvalCase& one_case : cases) {
    for (const auto& kv : one_case.inputs) {
      names.insert(kv.first);
    }
  }
  return std::vector<std::string>(names.begin(), names.end());
}

std::vector<CaseBindings> build_shared_case_bindings(const std::vector<EvalCase>& cases,
                                                     const std::vector<std::string>& input_names) {
  std::vector<CaseBindings> out;
  out.reserve(cases.size());
  for (const EvalCase& one_case : cases) {
    CaseBindings c;
    c.reserve(input_names.size());
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      auto it = one_case.inputs.find(input_names[i]);
      if (it != one_case.inputs.end()) {
        c.push_back(InputBinding{static_cast<int>(i), it->second});
      }
    }
    out.push_back(std::move(c));
  }
  return out;
}

std::vector<Value> build_expected_values(const std::vector<EvalCase>& cases) {
  std::vector<Value> out;
  out.reserve(cases.size());
  for (const EvalCase& one_case : cases) {
    out.push_back(one_case.expected);
  }
  return out;
}

struct CompileCache {
  std::unordered_map<std::string, BytecodeProgram> by_program;
};

struct CompiledPopulation {
  std::vector<BytecodeProgram> programs;
  double compile_ms = 0.0;
};

std::vector<ScoredGenome> build_scored_population(const std::vector<ProgramGenome>& population,
                                                  const std::vector<double>& fitness,
                                                  bool sort_output) {
  if (fitness.size() != population.size()) {
    throw std::runtime_error("fitness size mismatch");
  }
  std::vector<ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(ScoredGenome{population[i], fitness[i]});
  }
  if (sort_output) {
    std::sort(scored.begin(), scored.end(), scored_genome_sorts_before);
  }
  return scored;
}

CompiledPopulation compile_population(const std::vector<ProgramGenome>& population,
                                      const std::vector<std::string>& input_names,
                                      CompileCache* compile_cache) {
  CompiledPopulation out;
  out.programs.reserve(population.size());
  for (const ProgramGenome& genome : population) {
    const std::string& key = genome.meta.program_key;
    if (compile_cache != nullptr) {
      auto it = compile_cache->by_program.find(key);
      if (it != compile_cache->by_program.end()) {
        out.programs.push_back(it->second);
        continue;
      }
    }

    const auto t0 = std::chrono::steady_clock::now();
    BytecodeProgram bc = compile_for_eval(genome, input_names);
    const auto t1 = std::chrono::steady_clock::now();
    out.compile_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (compile_cache != nullptr) {
      compile_cache->by_program.emplace(key, bc);
    }
    out.programs.push_back(std::move(bc));
  }
  return out;
}

std::vector<ScoredGenome> score_population_cpu(
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
    std::vector<double>* raw_fitness_out) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  std::vector<double> fitness =
      eval_fitness_cpu(compiled.programs, shared_cases, shared_answer, fuel, penalty, reduction_lanes);
  for (double& value : fitness) {
    value = canonicalize_fitness_for_ranking(value);
  }
  if (fitness.size() != population.size()) {
    throw std::runtime_error("cpu fitness size mismatch");
  }

  result->cpu_compile_ms_total += compiled.compile_ms;
  if (record_per_gen) {
    result->generation_cpu_compile_ms.push_back(compiled.compile_ms);
    result->generation_gpu_compile_ms.push_back(0.0);
    result->generation_gpu_eval_call_ms.push_back(0.0);
    result->generation_gpu_eval_pack_ms.push_back(0.0);
    result->generation_gpu_eval_launch_prep_ms.push_back(0.0);
    result->generation_gpu_eval_upload_ms.push_back(0.0);
    result->generation_gpu_eval_pack_upload_ms.push_back(0.0);
    result->generation_gpu_eval_kernel_ms.push_back(0.0);
    result->generation_gpu_eval_copyback_ms.push_back(0.0);
    result->generation_gpu_eval_teardown_ms.push_back(0.0);
  }

  if (fitness_sum_out != nullptr) {
    long double sum = 0.0L;
    for (double one : fitness) {
      sum += static_cast<long double>(one);
    }
    *fitness_sum_out = static_cast<double>(sum);
  }
  if (raw_fitness_out != nullptr) {
    *raw_fitness_out = fitness;
  }
  return build_scored_population(population, fitness, true);
}

#ifdef G3PVM_HAS_CUDA
std::vector<ScoredGenome> score_population_gpu(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    FitnessSessionGpu* session,
    CompileCache* compile_cache,
    EvolutionResult* result,
    bool record_per_gen,
    double* fitness_sum_out,
    std::vector<double>* raw_fitness_out) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  FitnessEvalResult fit = session->eval_programs(compiled.programs);
  if (!fit.ok) {
    throw std::runtime_error("gpu fitness evaluation failed: " + fit.err.message);
  }
  for (double& value : fit.fitness) {
    value = canonicalize_fitness_for_ranking(value);
  }
  if (fit.fitness.size() != population.size()) {
    throw std::runtime_error("gpu fitness size mismatch");
  }

  result->gpu_compile_ms_total += compiled.compile_ms;
  result->gpu_eval_call_ms_total += fit.timing.total_ms;
  result->gpu_eval_pack_ms_total += fit.timing.pack_ms;
  result->gpu_eval_launch_prep_ms_total += fit.timing.launch_prep_ms;
  result->gpu_eval_upload_ms_total += fit.timing.upload_ms;
  result->gpu_eval_pack_upload_ms_total += fit.timing.pack_ms + fit.timing.upload_ms;
  result->gpu_eval_kernel_ms_total += fit.timing.kernel_ms;
  result->gpu_eval_copyback_ms_total += fit.timing.copyback_ms;
  result->gpu_eval_teardown_ms_total += fit.timing.teardown_ms;
  if (record_per_gen) {
    result->generation_cpu_compile_ms.push_back(0.0);
    result->generation_gpu_compile_ms.push_back(compiled.compile_ms);
    result->generation_gpu_eval_call_ms.push_back(fit.timing.total_ms);
    result->generation_gpu_eval_pack_ms.push_back(fit.timing.pack_ms);
    result->generation_gpu_eval_launch_prep_ms.push_back(fit.timing.launch_prep_ms);
    result->generation_gpu_eval_upload_ms.push_back(fit.timing.upload_ms);
    result->generation_gpu_eval_pack_upload_ms.push_back(fit.timing.pack_ms + fit.timing.upload_ms);
    result->generation_gpu_eval_kernel_ms.push_back(fit.timing.kernel_ms);
    result->generation_gpu_eval_copyback_ms.push_back(fit.timing.copyback_ms);
    result->generation_gpu_eval_teardown_ms.push_back(fit.timing.teardown_ms);
  }

  const std::vector<double>& fitness = fit.fitness;
  if (fitness_sum_out != nullptr) {
    long double sum = 0.0L;
    for (double one : fitness) {
      sum += static_cast<long double>(one);
    }
    *fitness_sum_out = static_cast<double>(sum);
  }
  if (raw_fitness_out != nullptr) {
    *raw_fitness_out = fitness;
  }
  return build_scored_population(population, fitness, true);
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
  const std::vector<std::string> input_names = build_canonical_input_names(cases);
  const std::vector<CaseBindings> shared_case_bindings = build_shared_case_bindings(cases, input_names);
  const std::vector<Value> expected_values = build_expected_values(cases);
  CompileCache cache;
  EvolutionResult result;
  return score_population_cpu(population, input_names, shared_case_bindings, expected_values, cfg.fuel,
                              cfg.penalty, cfg.gpu_blocksize, &cache, &result, false, nullptr, nullptr);
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
  cfg.grammar.validate();

  const auto all_t0 = std::chrono::steady_clock::now();
  std::mt19937_64 rng(cfg.seed);
  const std::vector<InputSpec> canonical_input_specs = build_canonical_input_specs(cases);
  const std::vector<std::string> canonical_input_names = build_canonical_input_names(cases);
  const std::vector<CaseBindings> shared_case_bindings = build_shared_case_bindings(cases, canonical_input_names);
  const std::vector<Value> expected_values = build_expected_values(cases);
  const auto init_t0 = std::chrono::steady_clock::now();
  std::vector<ProgramGenome> population;
  if (initial_population == nullptr) {
    population = init_population(cfg, canonical_input_specs);
  } else {
    population = *initial_population;
  }
  const auto init_t1 = std::chrono::steady_clock::now();
  if (static_cast<int>(population.size()) != cfg.population_size) {
    throw std::invalid_argument("initial_population size must match population_size");
  }

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

  CompileCache compile_cache;
#ifdef G3PVM_HAS_CUDA
  FitnessSessionGpu gpu_session;
#endif
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
    const FitnessSessionInitResult init_result =
        gpu_session.init(shared_case_bindings, expected_values, cfg.fuel, cfg.gpu_blocksize, cfg.penalty);
    if (!init_result.ok) {
      throw std::runtime_error("gpu fitness session init failed: " + init_result.err.message);
    }
    result.gpu_eval_init_ms = init_result.timing.total_ms;
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  }

  for (int gen = 0; gen < cfg.generations; ++gen) {
    const auto gen_t0 = std::chrono::steady_clock::now();
    const auto eval_t0 = std::chrono::steady_clock::now();
    std::vector<ScoredGenome> scored;
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
      overlap_future = std::async(std::launch::async, [population, cfg, repro_seed]() {
        OverlapPrepared out;
        out.prepared = repro::prepare_gpu_repro_backend_inputs(population, cfg, repro_seed, &out.stats);
        return out;
      });
    }
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
      scored = score_population_gpu(population, canonical_input_names, &gpu_session, &compile_cache,
                                    &result, true, &fitness_sum, overlap_gpu ? &raw_fitness : nullptr);
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      scored = score_population_cpu(
          population, canonical_input_names, shared_case_bindings, expected_values, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
          &compile_cache, &result, true, &fitness_sum, nullptr);
    }
    const auto eval_t1 = std::chrono::steady_clock::now();
    const ScoredGenome& best = scored.front();
    result.history_best.push_back(best);
    result.history_best_fitness.push_back(best.fitness);

    const double mean = fitness_sum / static_cast<double>(scored.size());
    result.history_mean_fitness.push_back(mean);

    const auto repro_t0 = std::chrono::steady_clock::now();
    repro::ReproductionResult reproduction;
    if (overlap_gpu) {
      OverlapPrepared overlap = overlap_future.get();
      const std::vector<ScoredGenome> repro_scored = build_scored_population(population, raw_fitness, false);
      reproduction = repro::run_gpu_repro_backend_prepared(repro_scored, cfg, overlap.prepared, &overlap.stats);
    } else {
      reproduction = repro::run_reproduction_backend(scored, cfg, rng);
    }
    const auto repro_t1 = std::chrono::steady_clock::now();

    population = std::move(reproduction.next_population);
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
  } else {
    const auto final_eval_t0 = std::chrono::steady_clock::now();
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
      result.final_population =
          score_population_gpu(population, canonical_input_names, &gpu_session, &compile_cache, &result, false,
                               nullptr, nullptr);
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      result.final_population = score_population_cpu(population, canonical_input_names, shared_case_bindings,
                                                     expected_values, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
                                                     &compile_cache, &result, false, nullptr, nullptr);
    }
    const auto final_eval_t1 = std::chrono::steady_clock::now();
    result.best = result.final_population.front();
    result.final_eval_ms = std::chrono::duration<double, std::milli>(final_eval_t1 - final_eval_t0).count();
  }
  result.total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - all_t0).count();
  return result;
}

}  // namespace g3pvm::evo
