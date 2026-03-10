#include "g3pvm/evolution/evolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/runtime/exec_cpu.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/runtime/fitness_gpu.hpp"
#endif

namespace g3pvm::evo {

namespace {

double canonicalize_fitness_for_order(double fitness) {
  if (!std::isfinite(fitness) || fitness == 0.0) {
    return fitness == 0.0 ? 0.0 : fitness;
  }

  int exponent = 0;
  const double mantissa = std::frexp(fitness, &exponent);
  constexpr int kMantissaBits = 40;
  const long long quantized_mantissa = std::llround(std::ldexp(mantissa, kMantissaBits));
  return std::ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

void canonicalize_fitness_vector(std::vector<double>* fitness) {
  if (fitness == nullptr) {
    return;
  }
  for (double& value : *fitness) {
    value = canonicalize_fitness_for_order(value);
  }
}

bool scored_genome_better(const ScoredGenome& a, const ScoredGenome& b) {
  const double fitness_a = canonicalize_fitness_for_order(a.fitness);
  const double fitness_b = canonicalize_fitness_for_order(b.fitness);
  if (fitness_a != fitness_b) {
    return fitness_a > fitness_b;
  }
  return a.genome.meta.program_key < b.genome.meta.program_key;
}

std::vector<ProgramGenome> init_population(const EvolutionConfig& cfg) {
  std::vector<ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(cfg.population_size));
  for (int i = 0; i < cfg.population_size; ++i) {
    out.push_back(make_random_genome(cfg.seed + static_cast<std::uint64_t>(i), cfg.limits));
  }
  return out;
}

std::vector<std::string> collect_case_input_names(const std::vector<FitnessCase>& cases) {
  std::set<std::string> names;
  for (const FitnessCase& one_case : cases) {
    for (const auto& kv : one_case.inputs) {
      names.insert(kv.first);
    }
  }
  return std::vector<std::string>(names.begin(), names.end());
}

std::vector<CaseInputs> to_shared_cases(const std::vector<FitnessCase>& cases,
                                       const std::vector<std::string>& input_names) {
  std::vector<CaseInputs> out;
  out.reserve(cases.size());
  for (const FitnessCase& one_case : cases) {
    CaseInputs c;
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

std::vector<Value> to_shared_answer(const std::vector<FitnessCase>& cases) {
  std::vector<Value> out;
  out.reserve(cases.size());
  for (const FitnessCase& one_case : cases) {
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
    BytecodeProgram bc = compile_for_eval_with_preset_locals(genome, input_names);
    const auto t1 = std::chrono::steady_clock::now();
    out.compile_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (compile_cache != nullptr) {
      compile_cache->by_program.emplace(key, bc);
    }
    out.programs.push_back(std::move(bc));
  }
  return out;
}

std::vector<double> evaluate_population_fitness_cpu_shared_cases(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    double penalty,
    int reduction_lanes,
    CompileCache* compile_cache,
    EvolutionTiming* timing,
    bool record_per_gen) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  std::vector<double> fitness =
      eval_fitness_cpu(compiled.programs, shared_cases, shared_answer, fuel, penalty, reduction_lanes);
  canonicalize_fitness_vector(&fitness);
  if (fitness.size() != population.size()) {
    throw std::runtime_error("cpu fitness size mismatch");
  }

  timing->cpu_generations_program_compile_ms_total += compiled.compile_ms;
  if (record_per_gen) {
    timing->generation_cpu_program_compile_ms.push_back(compiled.compile_ms);
  }

  return fitness;
}

std::vector<ScoredGenome> evaluate_population_cpu_shared_cases(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    double penalty,
    int reduction_lanes,
    CompileCache* compile_cache,
    EvolutionTiming* timing,
    bool record_per_gen,
    double* fitness_sum_out) {
  const std::vector<double> fitness = evaluate_population_fitness_cpu_shared_cases(
      population, input_names, shared_cases, shared_answer, fuel, penalty, reduction_lanes,
      compile_cache, timing, record_per_gen);
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
  std::sort(scored.begin(), scored.end(), scored_genome_better);
  return scored;
}

#ifdef G3PVM_HAS_CUDA
std::vector<double> evaluate_population_fitness_gpu(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    FitnessSessionGpu* session,
    CompileCache* compile_cache,
    EvolutionTiming* timing,
    bool record_per_gen) {
  const CompiledPopulation compiled = compile_population(population, input_names, compile_cache);
  FitnessEvalResult fit = session->eval_programs(compiled.programs);
  if (!fit.ok) {
    throw std::runtime_error("gpu fitness evaluation failed: " + fit.err.message);
  }
  canonicalize_fitness_vector(&fit.fitness);
  if (fit.fitness.size() != population.size()) {
    throw std::runtime_error("gpu fitness size mismatch");
  }

  timing->gpu_generations_program_compile_ms_total += compiled.compile_ms;
  timing->gpu_generations_pack_upload_ms_total += fit.upload_programs_ms + fit.pack_programs_ms;
  timing->gpu_generations_kernel_ms_total += fit.kernel_exec_ms;
  timing->gpu_generations_copyback_ms_total += fit.copyback_ms;
  if (record_per_gen) {
    timing->generation_gpu_program_compile_ms.push_back(compiled.compile_ms);
    timing->generation_gpu_pack_upload_ms.push_back(fit.upload_programs_ms + fit.pack_programs_ms);
    timing->generation_gpu_kernel_ms.push_back(fit.kernel_exec_ms);
    timing->generation_gpu_copyback_ms.push_back(fit.copyback_ms);
  }

  return fit.fitness;
}

std::vector<ScoredGenome> evaluate_population_gpu(
    const std::vector<ProgramGenome>& population,
    const std::vector<std::string>& input_names,
    FitnessSessionGpu* session,
    CompileCache* compile_cache,
    EvolutionTiming* timing,
    bool record_per_gen,
    double* fitness_sum_out) {
  const std::vector<double> fitness =
      evaluate_population_fitness_gpu(population, input_names, session, compile_cache, timing, record_per_gen);
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
  std::sort(scored.begin(), scored.end(), scored_genome_better);
  return scored;
}
#endif

}  // namespace

std::string eval_engine_name(EvalEngine engine) {
  if (engine == EvalEngine::GPU) return "gpu";
  return "cpu";
}

double evaluate_genome(const ProgramGenome& genome,
                       const std::vector<FitnessCase>& cases,
                       const EvolutionConfig& cfg) {
  const std::vector<std::string> input_names = collect_case_input_names(cases);
  const std::vector<CaseInputs> shared_cases = to_shared_cases(cases, input_names);
  const std::vector<Value> shared_answer = to_shared_answer(cases);
  std::vector<ProgramGenome> one{genome};
  CompileCache cache;
  EvolutionTiming timing;
  const std::vector<ScoredGenome> scored =
      evaluate_population_cpu_shared_cases(one, input_names, shared_cases, shared_answer, cfg.fuel,
                                           cfg.penalty, cfg.gpu_blocksize, &cache, &timing, false, nullptr);
  return scored.front().fitness;
}

std::vector<double> evaluate_population_fitness(const std::vector<ProgramGenome>& population,
                                                const std::vector<FitnessCase>& cases,
                                                const EvolutionConfig& cfg) {
  const std::vector<std::string> input_names = collect_case_input_names(cases);
  const std::vector<CaseInputs> shared_cases = to_shared_cases(cases, input_names);
  const std::vector<Value> shared_answer = to_shared_answer(cases);
  CompileCache cache;
  EvolutionTiming timing;
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
    FitnessSessionGpu session;
    const FitnessEvalResult init_result =
        session.init(shared_cases, shared_answer, cfg.fuel, cfg.gpu_blocksize, cfg.penalty);
    if (!init_result.ok) {
      throw std::runtime_error("gpu fitness session init failed: " + init_result.err.message);
    }
    return evaluate_population_fitness_gpu(population, input_names, &session, &cache, &timing, false);
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  }
  return evaluate_population_fitness_cpu_shared_cases(population, input_names, shared_cases, shared_answer,
                                                      cfg.fuel, cfg.penalty, cfg.gpu_blocksize, &cache,
                                                      &timing, false);
}

std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<FitnessCase>& cases,
                                              const EvolutionConfig& cfg) {
  const std::vector<std::string> input_names = collect_case_input_names(cases);
  const std::vector<CaseInputs> shared_cases = to_shared_cases(cases, input_names);
  const std::vector<Value> shared_answer = to_shared_answer(cases);
  CompileCache cache;
  EvolutionTiming timing;
  return evaluate_population_cpu_shared_cases(population, input_names, shared_cases, shared_answer, cfg.fuel,
                                              cfg.penalty, cfg.gpu_blocksize, &cache, &timing, false, nullptr);
}

ProgramGenome select_parent_tournament(const std::vector<ScoredGenome>& scored,
                                       std::mt19937_64& rng,
                                       int selection_pressure) {
  if (scored.empty()) {
    throw std::invalid_argument("scored population is empty");
  }

  std::uniform_int_distribution<std::size_t> any_pick(0, scored.size() - 1);
  const int tournament_size = std::max(1, std::min(selection_pressure, static_cast<int>(scored.size())));
  const ScoredGenome* best = nullptr;
  for (int i = 0; i < tournament_size; ++i) {
    const ScoredGenome& cand = scored[any_pick(rng)];
    if (best == nullptr || scored_genome_better(cand, *best)) {
      best = &cand;
    }
  }
  return best->genome;
}

EvolutionResult evolve_population(const std::vector<FitnessCase>& cases,
                                  const EvolutionConfig& cfg,
                                  const std::vector<ProgramGenome>* initial_population) {
  return evolve_population_profiled(cases, cfg, initial_population).result;
}

EvolutionRun evolve_population_profiled(const std::vector<FitnessCase>& cases,
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
  if (cfg.elitism < 0 || cfg.elitism > cfg.population_size) {
    throw std::invalid_argument("elitism must be in [0, population_size]");
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

  const std::vector<std::string> case_input_names = collect_case_input_names(cases);
  const std::vector<CaseInputs> shared_cases = to_shared_cases(cases, case_input_names);
  const std::vector<Value> shared_answer = to_shared_answer(cases);

  EvolutionRun run;
  run.timing.init_population_ms = std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
  EvolutionResult& result = run.result;
  run.timing.generation_eval_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_repro_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_total_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_cpu_program_compile_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_gpu_program_compile_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_gpu_pack_upload_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_gpu_kernel_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_gpu_copyback_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_selection_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_crossover_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_mutation_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_elite_copy_ms.reserve(static_cast<std::size_t>(cfg.generations));

  CompileCache compile_cache;
#ifdef G3PVM_HAS_CUDA
  FitnessSessionGpu gpu_session;
#endif
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
    const auto gpu_init_t0 = std::chrono::steady_clock::now();
    const FitnessEvalResult init_result =
        gpu_session.init(shared_cases, shared_answer, cfg.fuel, cfg.gpu_blocksize, cfg.penalty);
    if (!init_result.ok) {
      throw std::runtime_error("gpu fitness session init failed: " + init_result.err.message);
    }
    const auto gpu_init_t1 = std::chrono::steady_clock::now();
    run.timing.gpu_session_init_ms =
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
      scored = evaluate_population_gpu(population, case_input_names, &gpu_session, &compile_cache,
                                       &run.timing, true, &fitness_sum);
#else
      throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
    } else {
      scored = evaluate_population_cpu_shared_cases(
          population, case_input_names, shared_cases, shared_answer, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
          &compile_cache, &run.timing, true, &fitness_sum);
    }
    const auto eval_t1 = std::chrono::steady_clock::now();
    const ScoredGenome& best = scored.front();
    result.history_best.push_back(best);
    result.history_best_fitness.push_back(best.fitness);

    const double mean = fitness_sum / static_cast<double>(scored.size());
    result.history_mean_fitness.push_back(mean);

    std::vector<ProgramGenome> next_population;
    next_population.reserve(static_cast<std::size_t>(cfg.population_size));

    const auto elite_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.elitism; ++i) {
      next_population.push_back(scored[static_cast<std::size_t>(i)].genome);
    }
    const auto elite_t1 = std::chrono::steady_clock::now();

    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);

    double selection_ms = 0.0;
    double crossover_ms = 0.0;
    double mutation_ms = 0.0;

    const auto repro_t0 = std::chrono::steady_clock::now();
    while (static_cast<int>(next_population.size()) < cfg.population_size) {
      const auto select_t0 = std::chrono::steady_clock::now();
      const ProgramGenome p1 =
          select_parent_tournament(scored, rng, cfg.selection_pressure);
      const auto select_t1 = std::chrono::steady_clock::now();
      selection_ms += std::chrono::duration<double, std::milli>(select_t1 - select_t0).count();

      ProgramGenome child = p1;
      if (prob_dist(rng) < cfg.crossover_rate) {
        const auto select2_t0 = std::chrono::steady_clock::now();
        const ProgramGenome p2 =
            select_parent_tournament(scored, rng, cfg.selection_pressure);
        const auto select2_t1 = std::chrono::steady_clock::now();
        selection_ms += std::chrono::duration<double, std::milli>(select2_t1 - select2_t0).count();

        const auto cross_t0 = std::chrono::steady_clock::now();
        child = crossover(p1, p2, seed_dist(rng), cfg.limits);
        const auto cross_t1 = std::chrono::steady_clock::now();
        crossover_ms += std::chrono::duration<double, std::milli>(cross_t1 - cross_t0).count();
      }

      if (prob_dist(rng) < cfg.mutation_rate) {
        const auto mut_t0 = std::chrono::steady_clock::now();
        child = mutate(child, seed_dist(rng), cfg.limits, cfg.mutation_subtree_prob);
        const auto mut_t1 = std::chrono::steady_clock::now();
        mutation_ms += std::chrono::duration<double, std::milli>(mut_t1 - mut_t0).count();
      }
      next_population.push_back(child);
    }
    const auto repro_t1 = std::chrono::steady_clock::now();

    population = std::move(next_population);
    const auto gen_t1 = std::chrono::steady_clock::now();

    const double elite_ms = std::chrono::duration<double, std::milli>(elite_t1 - elite_t0).count();
    run.timing.generations_selection_ms_total += selection_ms;
    run.timing.generations_crossover_ms_total += crossover_ms;
    run.timing.generations_mutation_ms_total += mutation_ms;
    run.timing.generations_elite_copy_ms_total += elite_ms;
    run.timing.generation_selection_ms.push_back(selection_ms);
    run.timing.generation_crossover_ms.push_back(crossover_ms);
    run.timing.generation_mutation_ms.push_back(mutation_ms);
    run.timing.generation_elite_copy_ms.push_back(elite_ms);

    run.timing.generation_eval_ms.push_back(
        std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count());
    run.timing.generation_repro_ms.push_back(
        std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count());
    run.timing.generation_total_ms.push_back(
        std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    (void)gen;
  }

  const auto final_eval_t0 = std::chrono::steady_clock::now();
  if (cfg.eval_engine == EvalEngine::GPU) {
#ifdef G3PVM_HAS_CUDA
    result.final_population =
        evaluate_population_gpu(population, case_input_names, &gpu_session, &compile_cache, &run.timing, false, nullptr);
#else
    throw std::runtime_error("gpu evaluation requested but CUDA is unavailable in this build");
#endif
  } else {
    result.final_population = evaluate_population_cpu_shared_cases(
        population, case_input_names, shared_cases, shared_answer, cfg.fuel, cfg.penalty, cfg.gpu_blocksize,
        &compile_cache, &run.timing, false, nullptr);
  }
  const auto final_eval_t1 = std::chrono::steady_clock::now();

  result.best = result.final_population.front();
  run.timing.final_eval_ms = std::chrono::duration<double, std::milli>(final_eval_t1 - final_eval_t0).count();
  run.timing.total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - all_t0).count();
  return run;
}

}  // namespace g3pvm::evo
