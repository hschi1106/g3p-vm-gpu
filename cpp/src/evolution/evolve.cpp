#include "g3pvm/evolution/evolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#endif

namespace g3pvm::evo {

namespace {

std::vector<ProgramGenome> init_population(const EvolutionConfig& cfg) {
  std::vector<ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(cfg.population_size));
  for (int i = 0; i < cfg.population_size; ++i) {
    out.push_back(generate_random_genome(cfg.seed + static_cast<std::uint64_t>(i), cfg.limits));
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

struct ReproductionResult {
  std::vector<ProgramGenome> next_population;
  double selection_ms = 0.0;
  double crossover_ms = 0.0;
  double mutation_ms = 0.0;
};

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
    double* fitness_sum_out) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  std::vector<double> fitness =
      eval_fitness_cpu(compiled.programs, shared_cases, shared_answer, fuel, penalty, reduction_lanes);
  for (double& value : fitness) {
    value = canonicalize_fitness_for_ranking(value);
  }
  if (fitness.size() != population.size()) {
    throw std::runtime_error("cpu fitness size mismatch");
  }

  result->cpu_generations_program_compile_ms_total += compiled.compile_ms;
  if (record_per_gen) {
    result->generation_cpu_program_compile_ms.push_back(compiled.compile_ms);
  }

  if (fitness_sum_out != nullptr) {
    long double sum = 0.0L;
    for (double one : fitness) {
      sum += static_cast<long double>(one);
    }
    *fitness_sum_out = static_cast<double>(sum);
  }

  std::vector<ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(ScoredGenome{population[i], fitness[i]});
  }
  std::sort(scored.begin(), scored.end(), scored_genome_sorts_before);
  return scored;
}

#ifdef G3PVM_HAS_CUDA
std::vector<ScoredGenome> score_population_gpu(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    FitnessSessionGpu* session,
    CompileCache* compile_cache,
    EvolutionResult* result,
    bool record_per_gen,
    double* fitness_sum_out) {
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

  result->gpu_generations_program_compile_ms_total += compiled.compile_ms;
  result->gpu_generations_pack_upload_ms_total += fit.upload_programs_ms + fit.pack_programs_ms;
  result->gpu_generations_kernel_ms_total += fit.kernel_exec_ms;
  result->gpu_generations_copyback_ms_total += fit.copyback_ms;
  if (record_per_gen) {
    result->generation_gpu_program_compile_ms.push_back(compiled.compile_ms);
    result->generation_gpu_pack_upload_ms.push_back(fit.upload_programs_ms + fit.pack_programs_ms);
    result->generation_gpu_kernel_ms.push_back(fit.kernel_exec_ms);
    result->generation_gpu_copyback_ms.push_back(fit.copyback_ms);
  }

  const std::vector<double>& fitness = fit.fitness;
  if (fitness_sum_out != nullptr) {
    long double sum = 0.0L;
    for (double one : fitness) {
      sum += static_cast<long double>(one);
    }
    *fitness_sum_out = static_cast<double>(sum);
  }

  std::vector<ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(ScoredGenome{population[i], fitness[i]});
  }
  std::sort(scored.begin(), scored.end(), scored_genome_sorts_before);
  return scored;
}
#endif

ReproductionResult reproduce_population(const std::vector<ScoredGenome>& scored,
                                        const EvolutionConfig& cfg,
                                        std::mt19937_64& rng) {
  ReproductionResult out;
  out.next_population.reserve(static_cast<std::size_t>(cfg.population_size));
  const int offspring_count = cfg.population_size;

  const auto selection_t0 = std::chrono::steady_clock::now();
  std::vector<ProgramGenome> selected_parents =
      tournament_selection_without_replacement(scored, rng, cfg.selection_pressure, offspring_count);
  const auto selection_t1 = std::chrono::steady_clock::now();
  out.selection_ms = std::chrono::duration<double, std::milli>(selection_t1 - selection_t0).count();

  std::vector<ProgramGenome> offspring = selected_parents;
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);

  const auto crossover_t0 = std::chrono::steady_clock::now();
  if (selected_parents.size() > 1) {
    std::shuffle(offspring.begin(), offspring.end(), rng);
    for (std::size_t i = 0; i + 1 < offspring.size(); i += 2) {
      if (prob_dist(rng) >= cfg.crossover_rate) {
        continue;
      }
      auto children = crossover(offspring[i], offspring[i + 1], seed_dist(rng), cfg.limits);
      offspring[i] = std::move(children.first);
      offspring[i + 1] = std::move(children.second);
    }
  }
  const auto crossover_t1 = std::chrono::steady_clock::now();
  out.crossover_ms = std::chrono::duration<double, std::milli>(crossover_t1 - crossover_t0).count();

  const auto mutation_t0 = std::chrono::steady_clock::now();
  for (ProgramGenome& child : offspring) {
    if (prob_dist(rng) < cfg.mutation_rate) {
      child = mutate(child, seed_dist(rng), cfg.limits, cfg.mutation_subtree_prob);
    }
  }
  const auto mutation_t1 = std::chrono::steady_clock::now();
  out.mutation_ms = std::chrono::duration<double, std::milli>(mutation_t1 - mutation_t0).count();

  out.next_population.insert(out.next_population.end(),
                             std::make_move_iterator(offspring.begin()),
                             std::make_move_iterator(offspring.end()));
  return out;
}

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
                              cfg.penalty, cfg.gpu_blocksize, &cache, &result, false, nullptr);
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

  const auto all_t0 = std::chrono::steady_clock::now();
  std::mt19937_64 rng(cfg.seed);
  const auto init_t0 = std::chrono::steady_clock::now();
  std::vector<ProgramGenome> population;
  if (initial_population == nullptr) {
    population = init_population(cfg);
  } else {
    population = *initial_population;
  }
  const auto init_t1 = std::chrono::steady_clock::now();
  if (static_cast<int>(population.size()) != cfg.population_size) {
    throw std::invalid_argument("initial_population size must match population_size");
  }

  const std::vector<std::string> canonical_input_names = build_canonical_input_names(cases);
  const std::vector<CaseBindings> shared_case_bindings = build_shared_case_bindings(cases, canonical_input_names);
  const std::vector<Value> expected_values = build_expected_values(cases);

  EvolutionResult result;
  result.init_population_ms = std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
  result.generation_eval_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_repro_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_total_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_cpu_program_compile_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_program_compile_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_pack_upload_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_kernel_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_gpu_copyback_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_selection_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_crossover_ms.reserve(static_cast<std::size_t>(cfg.generations));
  result.generation_mutation_ms.reserve(static_cast<std::size_t>(cfg.generations));

  CompileCache compile_cache;
#ifdef G3PVM_HAS_CUDA
  FitnessSessionGpu gpu_session;
#endif
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
    const auto gpu_init_t0 = std::chrono::steady_clock::now();
    const FitnessEvalResult init_result =
        gpu_session.init(shared_case_bindings, expected_values, cfg.fuel, cfg.gpu_blocksize, cfg.penalty);
    if (!init_result.ok) {
      throw std::runtime_error("gpu fitness session init failed: " + init_result.err.message);
    }
    const auto gpu_init_t1 = std::chrono::steady_clock::now();
    result.gpu_session_init_ms =
        std::chrono::duration<double, std::milli>(gpu_init_t1 - gpu_init_t0).count();
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  }

  for (int gen = 0; gen < cfg.generations; ++gen) {
    const auto gen_t0 = std::chrono::steady_clock::now();
    const auto eval_t0 = std::chrono::steady_clock::now();
    std::vector<ScoredGenome> scored;
    double fitness_sum = 0.0;
    if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
      scored = score_population_gpu(population, canonical_input_names, &gpu_session, &compile_cache,
                                    &result, true, &fitness_sum);
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      scored = score_population_cpu(
          population, canonical_input_names, shared_case_bindings, expected_values, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
          &compile_cache, &result, true, &fitness_sum);
    }
    const auto eval_t1 = std::chrono::steady_clock::now();
    const ScoredGenome& best = scored.front();
    result.history_best.push_back(best);
    result.history_best_fitness.push_back(best.fitness);

    const double mean = fitness_sum / static_cast<double>(scored.size());
    result.history_mean_fitness.push_back(mean);

    const auto repro_t0 = std::chrono::steady_clock::now();
    ReproductionResult reproduction = reproduce_population(scored, cfg, rng);
    const auto repro_t1 = std::chrono::steady_clock::now();

    population = std::move(reproduction.next_population);
    const auto gen_t1 = std::chrono::steady_clock::now();

    result.generations_selection_ms_total += reproduction.selection_ms;
    result.generations_crossover_ms_total += reproduction.crossover_ms;
    result.generations_mutation_ms_total += reproduction.mutation_ms;
    result.generation_selection_ms.push_back(reproduction.selection_ms);
    result.generation_crossover_ms.push_back(reproduction.crossover_ms);
    result.generation_mutation_ms.push_back(reproduction.mutation_ms);

    result.generation_eval_ms.push_back(
        std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count());
    result.generation_repro_ms.push_back(
        std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count());
    result.generation_total_ms.push_back(
        std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    (void)gen;
  }

  const auto final_eval_t0 = std::chrono::steady_clock::now();
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
        result.final_population =
        score_population_gpu(population, canonical_input_names, &gpu_session, &compile_cache, &result, false, nullptr);
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  } else {
    result.final_population = score_population_cpu(
        population, canonical_input_names, shared_case_bindings, expected_values, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
        &compile_cache, &result, false, nullptr);
  }
  const auto final_eval_t1 = std::chrono::steady_clock::now();

  result.best = result.final_population.front();
  result.final_eval_ms = std::chrono::duration<double, std::milli>(final_eval_t1 - final_eval_t0).count();
  result.total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - all_t0).count();
  return result;
}

}  // namespace g3pvm::evo
