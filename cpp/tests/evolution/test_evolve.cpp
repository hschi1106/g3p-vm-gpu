#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/selection.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

std::vector<g3pvm::evo::EvalCase> simple_cases() {
  using g3pvm::Value;
  using g3pvm::evo::EvalCase;
  return {
      EvalCase{{{"x", Value::from_int(0)}, {"y", Value::from_int(0)}}, Value::from_int(0)},
      EvalCase{{{"x", Value::from_int(1)}, {"y", Value::from_int(2)}}, Value::from_int(3)},
      EvalCase{{{"x", Value::from_int(-1)}, {"y", Value::from_int(4)}}, Value::from_int(3)},
      EvalCase{{{"x", Value::from_int(3)}, {"y", Value::from_int(-2)}}, Value::from_int(1)},
  };
}

bool run_one(int selection_pressure) {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 24;
  cfg.generations = 8;
  cfg.mutation_rate = 0.7;
  cfg.mutation_subtree_prob = 0.8;
  cfg.selection_pressure = selection_pressure;
  cfg.seed = 42;

  const auto result = g3pvm::evo::evolve_population(simple_cases(), cfg);
  if (!check(static_cast<int>(result.history_best_fitness.size()) == cfg.generations,
             "history_best_fitness length mismatch")) {
    return false;
  }
  if (!check(static_cast<int>(result.history_mean_fitness.size()) == cfg.generations,
             "history_mean_fitness length mismatch")) {
    return false;
  }
  if (!check(static_cast<int>(result.final_population.size()) == cfg.population_size,
             "final_population length mismatch")) {
    return false;
  }
  const double min_hist = *std::min_element(result.history_best_fitness.begin(), result.history_best_fitness.end());
  if (!check(result.best.fitness >= min_hist, "best fitness should be >= min history best")) {
    return false;
  }
  return true;
}

bool test_selection_pressure_variants() {
  for (int selection_pressure : {1, 3, 5}) {
    if (!run_one(selection_pressure)) {
      return false;
    }
  }
  return true;
}

bool test_determinism_seed() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 12;
  cfg.generations = 5;
  cfg.seed = 123;
  cfg.selection_pressure = 3;

  const auto a = g3pvm::evo::evolve_population(simple_cases(), cfg);
  const auto b = g3pvm::evo::evolve_population(simple_cases(), cfg);

  if (!check(a.history_best_fitness.size() == b.history_best_fitness.size(),
             "determinism history length mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < a.history_best_fitness.size(); ++i) {
    if (!check(a.history_best_fitness[i] == b.history_best_fitness[i],
               "determinism best fitness mismatch")) {
      return false;
    }
    if (!check(a.history_best[i].genome.meta.program_key ==
                   b.history_best[i].genome.meta.program_key,
               "determinism best program key mismatch")) {
      return false;
    }
  }
  return true;
}

g3pvm::evo::ProgramGenome make_dummy_genome(const std::string& key) {
  g3pvm::evo::ProgramGenome genome;
  genome.meta.program_key = key;
  return genome;
}

bool test_round_based_tournament_selection_without_replacement_repeats_winners() {
  using g3pvm::evo::ScoredGenome;

  std::vector<ScoredGenome> scored;
  scored.push_back(ScoredGenome{make_dummy_genome("best"), 10.0});
  scored.push_back(ScoredGenome{make_dummy_genome("mid_a"), 7.0});
  scored.push_back(ScoredGenome{make_dummy_genome("mid_b"), 5.0});
  scored.push_back(ScoredGenome{make_dummy_genome("worst"), 1.0});

  std::mt19937_64 rng(123);
  const std::vector<g3pvm::evo::ProgramGenome> selected =
      g3pvm::evo::tournament_selection_without_replacement(scored, rng, 4, 4);

  if (!check(selected.size() == 4, "selection_count size mismatch")) {
    return false;
  }
  for (const auto& genome : selected) {
    if (!check(genome.meta.program_key == "best",
               "full-pressure round-based tournament should repeatedly select the best genome")) {
      return false;
    }
  }
  return true;
}

bool test_round_based_tournament_selection_without_replacement_visits_each_genome_once_when_k_is_one() {
  using g3pvm::evo::ScoredGenome;

  std::vector<ScoredGenome> scored;
  scored.push_back(ScoredGenome{make_dummy_genome("a"), 1.0});
  scored.push_back(ScoredGenome{make_dummy_genome("b"), 3.0});
  scored.push_back(ScoredGenome{make_dummy_genome("c"), 2.0});
  scored.push_back(ScoredGenome{make_dummy_genome("d"), 5.0});
  scored.push_back(ScoredGenome{make_dummy_genome("e"), 4.0});
  scored.push_back(ScoredGenome{make_dummy_genome("f"), 6.0});

  std::mt19937_64 rng(321);
  const std::vector<g3pvm::evo::ProgramGenome> selected =
      g3pvm::evo::tournament_selection_without_replacement(scored, rng, 1, 6);
  if (!check(selected.size() == 6, "selection_count mismatch at k=1")) {
    return false;
  }

  std::vector<std::string> keys;
  keys.reserve(selected.size());
  for (const auto& genome : selected) {
    keys.push_back(genome.meta.program_key);
  }
  std::sort(keys.begin(), keys.end());
  const std::vector<std::string> expected = {"a", "b", "c", "d", "e", "f"};
  if (!check(keys == expected, "k=1 round should visit each genome exactly once")) {
    return false;
  }
  for (std::size_t i = 1; i < keys.size(); ++i) {
    if (!check(keys[i - 1] != keys[i], "k=1 round should not repeat winners within the round")) {
      return false;
    }
  }
  return true;
}

bool test_gpu_backend_smoke() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.generations = 2;
  cfg.seed = 7;
  cfg.eval_engine = g3pvm::evo::EvalEngine::CPU;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;

  try {
    const auto result = g3pvm::evo::evolve_population(simple_cases(), cfg);
    if (!check(static_cast<int>(result.final_population.size()) == cfg.population_size,
               "gpu reproduction final_population length mismatch")) {
      return false;
    }
    if (!check(result.generations_repro_kernel_ms_total >= 0.0, "gpu reproduction kernel timing missing")) {
      return false;
    }
    if (!check(result.generations_repro_decode_ms_total >= 0.0, "gpu reproduction decode timing missing")) {
      return false;
    }
    if (!check(result.generations_repro_teardown_ms_total >= 0.0, "gpu reproduction teardown timing missing")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_evolve: SKIP gpu (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu reproduction backend failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

bool test_repro_overlap_smoke() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.generations = 2;
  cfg.seed = 9;
  cfg.eval_engine = g3pvm::evo::EvalEngine::GPU;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.repro_overlap = true;

  try {
    const auto result = g3pvm::evo::evolve_population(simple_cases(), cfg);
    if (!check(static_cast<int>(result.final_population.size()) == cfg.population_size,
               "gpu reproduction overlap final_population length mismatch")) {
      return false;
    }
    if (!check(result.generations_repro_prepare_inputs_ms_total >= 0.0,
               "gpu reproduction overlap prepare timings missing")) {
      return false;
    }
    if (!check(result.generations_repro_decode_ms_total >= 0.0,
               "gpu reproduction overlap decode timings missing")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_evolve: SKIP gpu overlap (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu reproduction overlap failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

}  // namespace

int main() {
  if (!test_selection_pressure_variants()) return 1;
  if (!test_determinism_seed()) return 1;
  if (!test_round_based_tournament_selection_without_replacement_repeats_winners()) return 1;
  if (!test_round_based_tournament_selection_without_replacement_visits_each_genome_once_when_k_is_one()) return 1;
  if (!test_gpu_backend_smoke()) return 1;
  if (!test_repro_overlap_smoke()) return 1;
  std::cout << "g3pvm_test_evolve: OK\n";
  return 0;
}
