#include "g3pvm/evolve.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>
#include <stdexcept>

#include "g3pvm/value_semantics.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace g3pvm::evo {

namespace {

bool is_close(const Value& a, const Value& b, double abs_tol, double rel_tol) {
  if (a.tag == ValueTag::Bool || b.tag == ValueTag::Bool) {
    return a.tag == b.tag && a.b == b.b;
  }
  if (a.tag == ValueTag::Float || b.tag == ValueTag::Float) {
    if (!(is_numeric(a) && is_numeric(b))) {
      return false;
    }
    const double af = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
    const double bf = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
    const double diff = std::fabs(af - bf);
    return diff <= std::max(abs_tol, rel_tol * std::max(std::fabs(af), std::fabs(bf)));
  }
  if (a.tag == ValueTag::None || b.tag == ValueTag::None) {
    return a.tag == ValueTag::None && b.tag == ValueTag::None;
  }
  return a.tag == b.tag && a.i == b.i;
}

std::vector<ProgramGenome> init_population(const EvolutionConfig& cfg) {
  std::vector<ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(cfg.population_size));
  for (int i = 0; i < cfg.population_size; ++i) {
    out.push_back(make_random_genome(cfg.seed + static_cast<std::uint64_t>(i), cfg.limits));
  }
  return out;
}

}  // namespace

std::string selection_method_name(SelectionMethod method) {
  if (method == SelectionMethod::Tournament) return "tournament";
  if (method == SelectionMethod::Roulette) return "roulette";
  if (method == SelectionMethod::Rank) return "rank";
  if (method == SelectionMethod::Truncation) return "truncation";
  return "random";
}

std::string crossover_method_name(CrossoverMethod method) {
  if (method == CrossoverMethod::TopLevelSplice) return "top_level_splice";
  if (method == CrossoverMethod::TypedSubtree) return "typed_subtree";
  return "hybrid";
}

double evaluate_genome(const ProgramGenome& genome,
                       const std::vector<FitnessCase>& cases,
                       const EvolutionConfig& cfg) {
  const BytecodeProgram program = compile_for_eval(genome);
  double score = 0.0;

  for (const FitnessCase& one_case : cases) {
    std::vector<std::pair<int, Value>> inputs;
    inputs.reserve(one_case.inputs.size());
    for (const auto& kv : one_case.inputs) {
      auto it = program.var2idx.find(kv.first);
      if (it != program.var2idx.end()) {
        inputs.push_back({it->second, kv.second});
      }
    }

    const VMResult out = run_bytecode(program, inputs, cfg.fuel);
    if (out.is_error) {
      score += cfg.penalty_error;
    } else if (is_close(out.value, one_case.expected, cfg.float_abs_tol, cfg.float_rel_tol)) {
      score += cfg.reward_match;
    } else {
      score += cfg.penalty_mismatch;
    }
  }

  return score;
}

std::vector<ScoredGenome> evaluate_population(const std::vector<ProgramGenome>& population,
                                              const std::vector<FitnessCase>& cases,
                                              const EvolutionConfig& cfg) {
  std::vector<ScoredGenome> scored;
  scored.reserve(population.size());
  for (const ProgramGenome& g : population) {
    scored.push_back(ScoredGenome{g, evaluate_genome(g, cases, cfg)});
  }
  std::sort(scored.begin(), scored.end(), [](const ScoredGenome& a, const ScoredGenome& b) {
    return a.fitness > b.fitness;
  });
  return scored;
}

ProgramGenome select_parent(const std::vector<ScoredGenome>& scored,
                            std::mt19937_64& rng,
                            SelectionMethod method,
                            int tournament_k,
                            double truncation_ratio) {
  if (scored.empty()) {
    throw std::invalid_argument("scored population is empty");
  }

  std::uniform_int_distribution<std::size_t> any_pick(0, scored.size() - 1);

  if (method == SelectionMethod::Random) {
    return scored[any_pick(rng)].genome;
  }

  if (method == SelectionMethod::Tournament) {
    const int k = std::max(1, std::min(tournament_k, static_cast<int>(scored.size())));
    const ScoredGenome* best = nullptr;
    for (int i = 0; i < k; ++i) {
      const ScoredGenome& cand = scored[any_pick(rng)];
      if (best == nullptr || cand.fitness > best->fitness) {
        best = &cand;
      }
    }
    return best->genome;
  }

  if (method == SelectionMethod::Roulette) {
    double min_fit = scored[0].fitness;
    for (const ScoredGenome& s : scored) {
      min_fit = std::min(min_fit, s.fitness);
    }
    const double shift = (min_fit <= 0.0) ? (-min_fit + 1e-9) : 0.0;
    std::vector<double> weights;
    weights.reserve(scored.size());
    double total = 0.0;
    for (const ScoredGenome& s : scored) {
      const double w = s.fitness + shift;
      weights.push_back(w);
      total += w;
    }
    if (total <= 0.0) {
      return scored[any_pick(rng)].genome;
    }
    std::uniform_real_distribution<double> pick_dist(0.0, total);
    const double pick = pick_dist(rng);
    double acc = 0.0;
    for (std::size_t i = 0; i < scored.size(); ++i) {
      acc += weights[i];
      if (acc >= pick) {
        return scored[i].genome;
      }
    }
    return scored.back().genome;
  }

  if (method == SelectionMethod::Rank) {
    std::vector<ScoredGenome> ranked = scored;
    std::sort(ranked.begin(), ranked.end(), [](const ScoredGenome& a, const ScoredGenome& b) {
      return a.fitness < b.fitness;
    });

    const int n = static_cast<int>(ranked.size());
    const int total = n * (n + 1) / 2;
    std::uniform_int_distribution<int> pick_dist(1, total);
    const int pick = pick_dist(rng);

    int acc = 0;
    for (int i = 0; i < n; ++i) {
      acc += (i + 1);
      if (acc >= pick) {
        return ranked[static_cast<std::size_t>(i)].genome;
      }
    }
    return ranked.back().genome;
  }

  if (method == SelectionMethod::Truncation) {
    const double ratio = std::min(std::max(truncation_ratio, 0.05), 1.0);
    const int keep_n = std::max(1, static_cast<int>(scored.size() * ratio));
    std::uniform_int_distribution<int> pick_dist(0, keep_n - 1);
    return scored[static_cast<std::size_t>(pick_dist(rng))].genome;
  }

  throw std::invalid_argument("unknown selection method");
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

  EvolutionRun run;
  run.timing.init_population_ms = std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
  EvolutionResult& result = run.result;
  run.timing.generation_eval_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_repro_ms.reserve(static_cast<std::size_t>(cfg.generations));
  run.timing.generation_total_ms.reserve(static_cast<std::size_t>(cfg.generations));

  for (int gen = 0; gen < cfg.generations; ++gen) {
    const auto gen_t0 = std::chrono::steady_clock::now();
    const auto eval_t0 = std::chrono::steady_clock::now();
    const std::vector<ScoredGenome> scored = evaluate_population(population, cases, cfg);
    const auto eval_t1 = std::chrono::steady_clock::now();
    const ScoredGenome& best = scored.front();
    result.history_best.push_back(best);
    result.history_best_fitness.push_back(best.fitness);

    double mean = 0.0;
    for (const ScoredGenome& s : scored) {
      mean += s.fitness;
    }
    mean /= static_cast<double>(scored.size());
    result.history_mean_fitness.push_back(mean);

    std::vector<ProgramGenome> next_population;
    next_population.reserve(static_cast<std::size_t>(cfg.population_size));
    for (int i = 0; i < cfg.elitism; ++i) {
      next_population.push_back(scored[static_cast<std::size_t>(i)].genome);
    }

    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);

    const auto repro_t0 = std::chrono::steady_clock::now();
    while (static_cast<int>(next_population.size()) < cfg.population_size) {
      const ProgramGenome p1 =
          select_parent(scored, rng, cfg.selection_method, cfg.tournament_k, cfg.truncation_ratio);

      ProgramGenome child = p1;
      if (prob_dist(rng) < cfg.crossover_rate) {
        const ProgramGenome p2 =
            select_parent(scored, rng, cfg.selection_method, cfg.tournament_k, cfg.truncation_ratio);
        child = crossover(p1, p2, seed_dist(rng), cfg.crossover_method, cfg.limits);
      }
      if (prob_dist(rng) < cfg.mutation_rate) {
        child = mutate(child, seed_dist(rng), cfg.limits);
      }
      next_population.push_back(child);
    }
    const auto repro_t1 = std::chrono::steady_clock::now();

    population = std::move(next_population);
    const auto gen_t1 = std::chrono::steady_clock::now();
    run.timing.generation_eval_ms.push_back(
        std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count());
    run.timing.generation_repro_ms.push_back(
        std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count());
    run.timing.generation_total_ms.push_back(
        std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    (void)gen;
  }

  const auto final_eval_t0 = std::chrono::steady_clock::now();
  result.final_population = evaluate_population(population, cases, cfg);
  const auto final_eval_t1 = std::chrono::steady_clock::now();
  result.best = result.final_population.front();
  run.timing.final_eval_ms = std::chrono::duration<double, std::milli>(final_eval_t1 - final_eval_t0).count();
  run.timing.total_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - all_t0)
                            .count();
  return run;
}

}  // namespace g3pvm::evo
