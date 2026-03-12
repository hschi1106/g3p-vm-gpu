#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/core/value.hpp"
#include "g3pvm/evolution/evolve.hpp"

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
      EvalCase{{{"x", Value::from_int(-2)}}, Value::from_int(-1)},
      EvalCase{{{"x", Value::from_int(-1)}}, Value::from_int(0)},
      EvalCase{{{"x", Value::from_int(0)}}, Value::from_int(1)},
      EvalCase{{{"x", Value::from_int(1)}}, Value::from_int(2)},
      EvalCase{{{"x", Value::from_int(2)}}, Value::from_int(3)},
      EvalCase{{{"x", Value::from_int(3)}}, Value::from_int(4)},
  };
}

bool same_history(const g3pvm::evo::EvolutionResult& cpu, const g3pvm::evo::EvolutionResult& gpu) {
  if (!check(cpu.history_best.size() == gpu.history_best.size(), "history_best size mismatch")) {
    return false;
  }
  if (!check(cpu.history_best_fitness.size() == gpu.history_best_fitness.size(),
             "history_best_fitness size mismatch")) {
    return false;
  }
  if (!check(cpu.history_mean_fitness.size() == gpu.history_mean_fitness.size(),
             "history_mean_fitness size mismatch")) {
    return false;
  }

  for (std::size_t i = 0; i < cpu.history_best.size(); ++i) {
    if (!check(cpu.history_best[i].fitness == gpu.history_best[i].fitness,
               "history best fitness mismatch at generation " + std::to_string(i))) {
      return false;
    }
    if (!check(cpu.history_best[i].genome.meta.program_key ==
                   gpu.history_best[i].genome.meta.program_key,
               "history best program key mismatch at generation " + std::to_string(i))) {
      return false;
    }
    if (!check(cpu.history_best_fitness[i] == gpu.history_best_fitness[i],
               "history_best_fitness mismatch at generation " + std::to_string(i))) {
      return false;
    }
    if (!check(cpu.history_mean_fitness[i] == gpu.history_mean_fitness[i],
               "history_mean_fitness mismatch at generation " + std::to_string(i))) {
      return false;
    }
  }

  if (!check(cpu.best.fitness == gpu.best.fitness, "final best fitness mismatch")) {
    return false;
  }
  if (!check(cpu.best.genome.meta.program_key ==
                 gpu.best.genome.meta.program_key,
             "final best program key mismatch")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  g3pvm::evo::EvolutionConfig cpu_cfg;
  cpu_cfg.population_size = 64;
  cpu_cfg.generations = 8;
  cpu_cfg.mutation_rate = 0.7;
  cpu_cfg.mutation_subtree_prob = 0.8;
  cpu_cfg.crossover_rate = 0.9;
  cpu_cfg.selection_pressure = 3;
  cpu_cfg.seed = 42;
  cpu_cfg.eval_engine = g3pvm::evo::EvalEngine::CPU;

  g3pvm::evo::EvolutionConfig gpu_cfg = cpu_cfg;
  gpu_cfg.eval_engine = g3pvm::evo::EvalEngine::GPU;
  gpu_cfg.gpu_blocksize = 128;

  const auto cpu = g3pvm::evo::evolve_population(simple_cases(), cpu_cfg);
  try {
    const auto gpu = g3pvm::evo::evolve_population(simple_cases(), gpu_cfg);
    if (!same_history(cpu, gpu)) {
      return 1;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_evolution_cpu_gpu_parity: SKIP (" << message << ")\n";
      return 0;
    }
    std::cerr << "FAIL: gpu evolution run failed: " << message << "\n";
    return 1;
  }

  std::cout << "g3pvm_test_evolution_cpu_gpu_parity: OK\n";
  return 0;
}
