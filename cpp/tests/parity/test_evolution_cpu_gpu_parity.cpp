#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gagp/core/value.hpp"
#include "gagp/evolution/evolve.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

std::vector<gagp::evo::EvalCase> simple_cases() {
  using gagp::Value;
  using gagp::evo::EvalCase;
  return {
      EvalCase{{{"x", Value::from_int(-2)}}, Value::from_int(-1)},
      EvalCase{{{"x", Value::from_int(-1)}}, Value::from_int(0)},
      EvalCase{{{"x", Value::from_int(0)}}, Value::from_int(1)},
      EvalCase{{{"x", Value::from_int(1)}}, Value::from_int(2)},
      EvalCase{{{"x", Value::from_int(2)}}, Value::from_int(3)},
      EvalCase{{{"x", Value::from_int(3)}}, Value::from_int(4)},
  };
}

bool genome_contains_asgp(const gagp::evo::ProgramGenome& genome) {
  for (const gagp::evo::AstNode& node : genome.ast.nodes) {
    if (node.kind == gagp::evo::NodeKind::ASGP_DC ||
        node.kind == gagp::evo::NodeKind::ASGP_DP1D ||
        node.kind == gagp::evo::NodeKind::ASGP_DP2D) {
      return true;
    }
  }
  return false;
}

void diagnose_generation_population(const std::vector<gagp::evo::EvalCase>& cases,
                                    gagp::evo::EvolutionConfig cpu_cfg,
                                    gagp::evo::EvolutionConfig gpu_cfg,
                                    int generation) {
  cpu_cfg.generations = generation;
  gpu_cfg.generations = generation;
  cpu_cfg.retain_final_population = true;
  gpu_cfg.retain_final_population = true;
  const auto cpu = gagp::evo::evolve_population(cases, cpu_cfg);
  const auto gpu = gagp::evo::evolve_population(cases, gpu_cfg);

  std::unordered_map<std::string, const gagp::evo::ScoredGenome*> gpu_by_key;
  for (const gagp::evo::ScoredGenome& one : gpu.final_population) {
    gpu_by_key[one.genome.meta.program_key] = &one;
  }
  for (const gagp::evo::ScoredGenome& one : cpu.final_population) {
    auto it = gpu_by_key.find(one.genome.meta.program_key);
    if (it == gpu_by_key.end()) {
      std::cerr << "DIAG: missing gpu program key at generation " << generation
                << " key=" << one.genome.meta.program_key
                << " asgp=" << (genome_contains_asgp(one.genome) ? "true" : "false") << "\n";
      return;
    }
    if (one.fitness != it->second->fitness) {
      std::cerr << "DIAG: fitness mismatch at generation " << generation
                << " key=" << one.genome.meta.program_key
                << " cpu=" << one.fitness
                << " gpu=" << it->second->fitness
                << " asgp=" << (genome_contains_asgp(one.genome) ? "true" : "false")
                << " nodes=" << one.genome.meta.node_count << "\n";
      return;
    }
  }
}

bool same_history(const gagp::evo::EvolutionResult& cpu, const gagp::evo::EvolutionResult& gpu) {
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
               "history_mean_fitness mismatch at generation " + std::to_string(i) +
                   " cpu=" + std::to_string(cpu.history_mean_fitness[i]) +
                   " gpu=" + std::to_string(gpu.history_mean_fitness[i]))) {
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
  gagp::evo::EvolutionConfig cpu_cfg;
  cpu_cfg.population_size = 64;
  cpu_cfg.generations = 8;
  cpu_cfg.mutation_rate = 0.7;
  cpu_cfg.mutation_subtree_prob = 0.8;
  cpu_cfg.selection_pressure = 3;
  cpu_cfg.seed = 42;
  cpu_cfg.eval_engine = gagp::evo::EvalEngine::CPU;

  gagp::evo::EvolutionConfig gpu_cfg = cpu_cfg;
  gpu_cfg.eval_engine = gagp::evo::EvalEngine::GPU;
  gpu_cfg.gpu_blocksize = 128;

  const auto cpu = gagp::evo::evolve_population(simple_cases(), cpu_cfg);
  try {
    const auto gpu = gagp::evo::evolve_population(simple_cases(), gpu_cfg);
    if (!same_history(cpu, gpu)) {
      diagnose_generation_population(simple_cases(), cpu_cfg, gpu_cfg, 4);
      return 1;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "gagp_test_evolution_cpu_gpu_parity: SKIP (" << message << ")\n";
      return 0;
    }
    std::cerr << "FAIL: gpu evolution run failed: " << message << "\n";
    return 1;
  }

  std::cout << "gagp_test_evolution_cpu_gpu_parity: OK\n";
  return 0;
}
