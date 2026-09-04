#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gagp/evolution/evolve.hpp"
#include "gagp/evolution/genome_generation.hpp"
#include "gagp/evolution/selection.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

gagp::evo::ProgramGenome make_nan_genome() {
  using gagp::Value;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  ProgramGenome genome;
  genome.ast = AstProgram{
      {
          AstNode{NodeKind::PROGRAM, 0, 0},
          AstNode{NodeKind::BLOCK_CONS, 0, 0},
          AstNode{NodeKind::RETURN, 0, 0},
          AstNode{NodeKind::CONST, 0, 0},
          AstNode{NodeKind::BLOCK_NIL, 0, 0},
      },
      {},
      {Value::from_float(std::numeric_limits<double>::quiet_NaN())},
  };
  genome.meta = gagp::evo::build_genome_meta(genome.ast);
  return genome;
}

std::vector<gagp::evo::EvalCase> simple_cases() {
  using gagp::Value;
  using gagp::evo::EvalCase;
  return {
      EvalCase{{{"x", Value::from_int(0)}, {"y", Value::from_int(0)}}, Value::from_int(0)},
      EvalCase{{{"x", Value::from_int(1)}, {"y", Value::from_int(2)}}, Value::from_int(3)},
      EvalCase{{{"x", Value::from_int(-1)}, {"y", Value::from_int(4)}}, Value::from_int(3)},
      EvalCase{{{"x", Value::from_int(3)}, {"y", Value::from_int(-2)}}, Value::from_int(1)},
  };
}

gagp::evo::GrammarConfig const_return_grammar() {
  gagp::evo::GrammarConfig grammar = gagp::evo::GrammarConfig::all_enabled();
  grammar.statement_assign = false;
  grammar.statement_if_stmt = false;
  grammar.statement_for_range = false;
  grammar.expression_var = false;
  grammar.expression_if_expr = false;
  grammar.unary_neg = false;
  grammar.unary_not = false;
  grammar.binary_add = false;
  grammar.binary_sub = false;
  grammar.binary_mul = false;
  grammar.binary_div = false;
  grammar.binary_mod = false;
  grammar.binary_lt = false;
  grammar.binary_le = false;
  grammar.binary_gt = false;
  grammar.binary_ge = false;
  grammar.binary_eq = false;
  grammar.binary_ne = false;
  grammar.binary_and = false;
  grammar.binary_or = false;
  grammar.builtin_abs = false;
  grammar.builtin_min = false;
  grammar.builtin_max = false;
  grammar.builtin_clip = false;
  grammar.builtin_len = false;
  grammar.builtin_concat = false;
  grammar.builtin_slice = false;
  grammar.builtin_index = false;
  grammar.builtin_append = false;
  grammar.builtin_reverse = false;
  grammar.builtin_find = false;
  grammar.builtin_contains = false;
  return grammar;
}

bool expected_type_seed_case_has_no_type_penalty(const gagp::Value& expected,
                                                 const std::string& label) {
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.generations = 1;
  cfg.seed = 555;
  cfg.selection_pressure = 1;
  cfg.skip_final_eval = true;
  cfg.grammar = const_return_grammar();

  const auto result = gagp::evo::evolve_population({gagp::evo::EvalCase{{}, expected}}, cfg);
  if (!check(result.history_mean_fitness.size() == 1, label + " history_mean_fitness length mismatch")) {
    return false;
  }
  if (!check(result.history_mean_fitness[0] >= 0.0,
             label + " initial population should avoid return-type penalties")) {
    return false;
  }
  return true;
}

bool run_one(int selection_pressure) {
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 24;
  cfg.generations = 8;
  cfg.mutation_rate = 0.7;
  cfg.mutation_subtree_prob = 0.8;
  cfg.selection_pressure = selection_pressure;
  cfg.seed = 42;

  const auto result = gagp::evo::evolve_population(simple_cases(), cfg);
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
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 12;
  cfg.generations = 5;
  cfg.seed = 123;
  cfg.selection_pressure = 3;

  const auto a = gagp::evo::evolve_population(simple_cases(), cfg);
  const auto b = gagp::evo::evolve_population(simple_cases(), cfg);

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

bool test_skip_final_eval() {
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 12;
  cfg.generations = 3;
  cfg.seed = 321;
  cfg.selection_pressure = 3;
  cfg.skip_final_eval = true;

  const auto result = gagp::evo::evolve_population(simple_cases(), cfg);
  if (!check(result.final_eval_skipped, "skip_final_eval should mark final eval as skipped")) {
    return false;
  }
  if (!check(result.final_population.empty(), "skip_final_eval should leave final_population empty")) {
    return false;
  }
  if (!check(result.timing.final_eval_ms == 0.0,
             "skip_final_eval should report zero final_eval_ms")) {
    return false;
  }
  if (!check(static_cast<int>(result.history_best_fitness.size()) == cfg.generations,
             "skip_final_eval should still record history")) {
    return false;
  }
  return true;
}

bool test_retain_final_population_off_keeps_best_only() {
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 12;
  cfg.generations = 3;
  cfg.seed = 777;
  cfg.selection_pressure = 3;
  cfg.retain_final_population = false;

  const auto result = gagp::evo::evolve_population(simple_cases(), cfg);
  if (!check(!result.final_eval_skipped, "retain_final_population off should still run final eval")) {
    return false;
  }
  if (!check(result.final_population.empty(),
             "retain_final_population off should not materialize final_population")) {
    return false;
  }
  if (!check(!result.best.genome.meta.program_key.empty(),
             "retain_final_population off should still materialize result.best")) {
    return false;
  }
  return true;
}

bool test_initial_population_override() {
  gagp::evo::EvolutionConfig cfg_a;
  cfg_a.population_size = 6;
  cfg_a.generations = 1;
  cfg_a.seed = 11;
  cfg_a.selection_pressure = 3;

  std::vector<gagp::evo::ProgramGenome> initial_population;
  initial_population.reserve(static_cast<std::size_t>(cfg_a.population_size));
  for (int i = 0; i < cfg_a.population_size; ++i) {
    initial_population.push_back(
        gagp::evo::generate_random_genome(1000 + static_cast<std::uint64_t>(i), cfg_a.limits));
  }

  gagp::evo::EvolutionConfig cfg_b = cfg_a;
  cfg_b.seed = 9999;

  const auto a = gagp::evo::evolve_population(simple_cases(), cfg_a, &initial_population);
  const auto b = gagp::evo::evolve_population(simple_cases(), cfg_b, &initial_population);
  if (!check(a.history_best.size() == 1 && b.history_best.size() == 1,
             "initial_population override history length mismatch")) {
    return false;
  }
  if (!check(a.history_best[0].genome.meta.program_key == b.history_best[0].genome.meta.program_key,
             "initial_population should override cfg.seed for generation-0 population")) {
    return false;
  }
  if (!check(a.history_best_fitness[0] == b.history_best_fitness[0],
             "initial_population should preserve generation-0 fitness")) {
    return false;
  }
  return true;
}

bool test_generated_initial_population_uses_expected_return_type() {
  using gagp::Value;
  if (!expected_type_seed_case_has_no_type_penalty(
          gagp::payload::make_string_value("target"), "string expected")) {
    return false;
  }
  if (!expected_type_seed_case_has_no_type_penalty(
          gagp::payload::make_int_list_value({Value::from_int(42)}), "int list expected")) {
    return false;
  }
  if (!expected_type_seed_case_has_no_type_penalty(
          gagp::payload::make_string_list_value({gagp::payload::make_string_value("x")}),
          "string list expected")) {
    return false;
  }
  return true;
}

bool test_nonfinite_fitness_is_clamped_to_penalty() {
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.generations = 1;
  cfg.penalty = 3.5;
  cfg.skip_final_eval = true;

  std::vector<gagp::evo::ProgramGenome> initial_population(
      static_cast<std::size_t>(cfg.population_size), make_nan_genome());
  const auto result = gagp::evo::evolve_population(simple_cases(), cfg, &initial_population);

  if (!check(result.history_best_fitness.size() == 1, "nonfinite clamp best history length mismatch")) {
    return false;
  }
  if (!check(result.history_mean_fitness.size() == 1, "nonfinite clamp mean history length mismatch")) {
    return false;
  }
  if (!check(result.history_best_fitness[0] == -14.0,
             "nonfinite best fitness should clamp per case to -penalty")) {
    return false;
  }
  if (!check(result.history_mean_fitness[0] == -14.0,
             "nonfinite mean fitness should clamp per case to -penalty")) {
    return false;
  }
  return true;
}

bool test_cpu_repro_ablation_modes_smoke_and_determinism() {
  for (const auto ablation : {gagp::evo::repro::CpuReproAblation::GpuSelection,
                              gagp::evo::repro::CpuReproAblation::GpuCandidates,
                              gagp::evo::repro::CpuReproAblation::GpuCoupledDonor}) {
    gagp::evo::EvolutionConfig cfg;
    cfg.population_size = 16;
    cfg.generations = 4;
    cfg.seed = 2024;
    cfg.selection_pressure = 3;
    cfg.mutation_rate = 0.7;
    cfg.mutation_subtree_prob = 0.6;
    cfg.reproduction_backend = gagp::evo::repro::ReproductionBackend::Cpu;
    cfg.cpu_repro_ablation = ablation;
    cfg.grammar = gagp::evo::GrammarConfig::scalar();

    const auto a = gagp::evo::evolve_population(simple_cases(), cfg);
    const auto b = gagp::evo::evolve_population(simple_cases(), cfg);
    if (!check(static_cast<int>(a.history_best_fitness.size()) == cfg.generations,
               "cpu repro ablation history_best_fitness length mismatch")) {
      return false;
    }
    if (!check(static_cast<int>(a.history_mean_fitness.size()) == cfg.generations,
               "cpu repro ablation history_mean_fitness length mismatch")) {
      return false;
    }
    if (!check(static_cast<int>(a.final_population.size()) == cfg.population_size,
               "cpu repro ablation final_population length mismatch")) {
      return false;
    }
    if (!check(a.history_best_fitness == b.history_best_fitness,
               "cpu repro ablation best history should be deterministic")) {
      return false;
    }
    if (!check(a.history_mean_fitness == b.history_mean_fitness,
               "cpu repro ablation mean history should be deterministic")) {
      return false;
    }
  }
  return true;
}

bool test_cpu_repro_ablation_rejects_gpu_backend() {
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.generations = 2;
  cfg.reproduction_backend = gagp::evo::repro::ReproductionBackend::Gpu;
  cfg.cpu_repro_ablation = gagp::evo::repro::CpuReproAblation::GpuSelection;

  try {
    (void)gagp::evo::evolve_population(simple_cases(), cfg);
  } catch (const std::invalid_argument& err) {
    return std::string(err.what()).find("cpu_repro_ablation") != std::string::npos;
  }
  std::cerr << "FAIL: cpu repro ablation should reject gpu reproduction backend\n";
  return false;
}

bool test_legacy_num_list_input_compat_uses_any_for_generation() {
  using gagp::evo::RType;

  gagp::evo::GrammarConfig grammar = gagp::evo::GrammarConfig::all_enabled();
  if (!check(gagp::evo::generation_input_type_for_grammar(RType::IntList, grammar) == RType::IntList,
             "exact current int list input should stay IntList")) {
    return false;
  }
  if (!check(gagp::evo::generation_input_type_for_grammar(RType::FloatList, grammar) == RType::FloatList,
             "exact current float list input should stay FloatList")) {
    return false;
  }

  grammar.compat_legacy_num_list_inputs_as_any = true;
  if (!check(gagp::evo::generation_input_type_for_grammar(RType::IntList, grammar) == RType::Any,
             "legacy NumList compat should generate from Any for IntList inputs")) {
    return false;
  }
  if (!check(gagp::evo::generation_input_type_for_grammar(RType::FloatList, grammar) == RType::Any,
             "legacy NumList compat should generate from Any for FloatList inputs")) {
    return false;
  }
  if (!check(gagp::evo::generation_input_type_for_grammar(RType::StringList, grammar) == RType::StringList,
             "legacy NumList compat should not change StringList inputs")) {
    return false;
  }
  return true;
}

gagp::evo::ProgramGenome make_dummy_genome(const std::string& key) {
  gagp::evo::ProgramGenome genome;
  genome.meta.program_key = key;
  return genome;
}

bool test_round_based_tournament_selection_without_replacement_repeats_winners() {
  using gagp::evo::ScoredGenome;

  std::vector<ScoredGenome> scored;
  scored.push_back(ScoredGenome{make_dummy_genome("best"), 10.0});
  scored.push_back(ScoredGenome{make_dummy_genome("mid_a"), 7.0});
  scored.push_back(ScoredGenome{make_dummy_genome("mid_b"), 5.0});
  scored.push_back(ScoredGenome{make_dummy_genome("worst"), 1.0});

  std::mt19937_64 rng(123);
  const std::vector<std::size_t> selected =
      gagp::evo::tournament_selection_indices_without_replacement(scored, rng, 4, 4);

  if (!check(selected.size() == 4, "selection_count size mismatch")) {
    return false;
  }
  for (std::size_t winner_index : selected) {
    if (!check(scored[winner_index].genome.meta.program_key == "best",
               "full-pressure round-based tournament should repeatedly select the best genome")) {
      return false;
    }
  }
  return true;
}

bool test_round_based_tournament_selection_without_replacement_visits_each_genome_once_when_k_is_one() {
  using gagp::evo::ScoredGenome;

  std::vector<ScoredGenome> scored;
  scored.push_back(ScoredGenome{make_dummy_genome("a"), 1.0});
  scored.push_back(ScoredGenome{make_dummy_genome("b"), 3.0});
  scored.push_back(ScoredGenome{make_dummy_genome("c"), 2.0});
  scored.push_back(ScoredGenome{make_dummy_genome("d"), 5.0});
  scored.push_back(ScoredGenome{make_dummy_genome("e"), 4.0});
  scored.push_back(ScoredGenome{make_dummy_genome("f"), 6.0});

  std::mt19937_64 rng(321);
  const std::vector<std::size_t> selected =
      gagp::evo::tournament_selection_indices_without_replacement(scored, rng, 1, 6);
  if (!check(selected.size() == 6, "selection_count mismatch at k=1")) {
    return false;
  }

  std::vector<std::string> keys;
  keys.reserve(selected.size());
  for (std::size_t winner_index : selected) {
    keys.push_back(scored[winner_index].genome.meta.program_key);
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
#ifdef GAGP_HAS_CUDA
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.generations = 2;
  cfg.seed = 7;
  cfg.eval_engine = gagp::evo::EvalEngine::CPU;
  cfg.reproduction_backend = gagp::evo::repro::ReproductionBackend::Gpu;

  try {
    const auto result = gagp::evo::evolve_population(simple_cases(), cfg);
    if (!check(static_cast<int>(result.final_population.size()) == cfg.population_size,
               "gpu reproduction final_population length mismatch")) {
      return false;
    }
    if (!check(result.timing.reproduction_totals.kernel_ms >= 0.0,
               "gpu reproduction kernel timing missing")) {
      return false;
    }
    if (!check(result.timing.reproduction_totals.decode_ms >= 0.0,
               "gpu reproduction decode timing missing")) {
      return false;
    }
    if (!check(result.timing.reproduction_totals.teardown_ms >= 0.0,
               "gpu reproduction teardown timing missing")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "gagp_test_evolve: SKIP gpu (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu reproduction backend failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

bool test_repro_overlap_smoke() {
#ifdef GAGP_HAS_CUDA
  gagp::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.generations = 2;
  cfg.seed = 9;
  cfg.eval_engine = gagp::evo::EvalEngine::GPU;
  cfg.reproduction_backend = gagp::evo::repro::ReproductionBackend::Gpu;
  cfg.repro_overlap = true;

  try {
    const auto result = gagp::evo::evolve_population(simple_cases(), cfg);
    if (!check(static_cast<int>(result.final_population.size()) == cfg.population_size,
               "gpu reproduction overlap final_population length mismatch")) {
      return false;
    }
    if (!check(result.timing.reproduction_totals.prepare_inputs_ms >= 0.0,
               "gpu reproduction overlap prepare timings missing")) {
      return false;
    }
    if (!check(result.timing.reproduction_totals.decode_ms >= 0.0,
               "gpu reproduction overlap decode timings missing")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "gagp_test_evolve: SKIP gpu overlap (" << message << ")\n";
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
  if (!test_skip_final_eval()) return 1;
  if (!test_retain_final_population_off_keeps_best_only()) return 1;
  if (!test_initial_population_override()) return 1;
  if (!test_generated_initial_population_uses_expected_return_type()) return 1;
  if (!test_nonfinite_fitness_is_clamped_to_penalty()) return 1;
  if (!test_cpu_repro_ablation_modes_smoke_and_determinism()) return 1;
  if (!test_cpu_repro_ablation_rejects_gpu_backend()) return 1;
  if (!test_legacy_num_list_input_compat_uses_any_for_generation()) return 1;
  if (!test_round_based_tournament_selection_without_replacement_repeats_winners()) return 1;
  if (!test_round_based_tournament_selection_without_replacement_visits_each_genome_once_when_k_is_one()) return 1;
  if (!test_gpu_backend_smoke()) return 1;
  if (!test_repro_overlap_smoke()) return 1;
  std::cout << "gagp_test_evolve: OK\n";
  return 0;
}
