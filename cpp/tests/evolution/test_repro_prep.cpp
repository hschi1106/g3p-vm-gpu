#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/repro/pack.hpp"
#include "g3pvm/evolution/repro/prep.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

std::vector<g3pvm::evo::ProgramGenome> make_population() {
  g3pvm::evo::Limits limits;
  limits.max_expr_depth = 5;
  limits.max_stmts_per_block = 6;
  limits.max_total_nodes = 80;
  limits.max_for_k = 16;
  limits.max_call_args = 3;

  std::vector<g3pvm::evo::ProgramGenome> out;
  out.reserve(4);
  for (std::uint64_t seed = 1; seed <= 4; ++seed) {
    out.push_back(g3pvm::evo::generate_random_genome(seed, limits));
  }
  return out;
}

bool test_preprocess_and_pack() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = make_population();
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg);

  if (!check(prep.subtree_ends.size() == population.size(), "subtree_ends size mismatch")) return false;
  if (!check(prep.candidates.size() == population.size(), "candidates size mismatch")) return false;
  if (!check(static_cast<int>(prep.donor_pool.size()) == repro_cfg.donor_pool_size,
             "donor_pool size mismatch")) {
    return false;
  }

  for (std::size_t i = 0; i < population.size(); ++i) {
    if (!check(prep.subtree_ends[i].size() == population[i].ast.nodes.size(),
               "subtree_ends node count mismatch")) {
      return false;
    }
    if (!check(!prep.candidates[i].empty(), "candidate set should not be empty")) {
      return false;
    }
  }

  const g3pvm::evo::repro::PackedHostData packed =
      g3pvm::evo::repro::pack_population(population, prep, repro_cfg);
  if (!check(static_cast<int>(packed.metas.size()) == repro_cfg.population_size, "meta size mismatch")) return false;
  if (!check(static_cast<int>(packed.donor_lens.size()) == repro_cfg.donor_pool_size,
             "donor_lens size mismatch")) {
    return false;
  }
  if (!check(static_cast<int>(packed.candidates.size()) ==
                 repro_cfg.population_size * repro_cfg.candidates_per_program,
             "packed candidates size mismatch")) {
    return false;
  }
  if (!check(!packed.name_lookup.empty(), "name lookup should not be empty")) {
    return false;
  }
  return true;
}

bool test_gpu_prepared_backend_smoke() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = make_population();
  std::vector<g3pvm::evo::ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(g3pvm::evo::ScoredGenome{
        population[i],
        static_cast<double>(population.size() - i),
    });
  }

  g3pvm::evo::repro::ReproductionStats prep_stats;
  try {
    const auto prepared =
        g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, 123, &prep_stats);
    g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
    const auto reproduction =
        g3pvm::evo::repro::run_gpu_repro_backend_prepared(scored, cfg, prepared, &run_stats);
    if (!check(static_cast<int>(reproduction.next_population.size()) == cfg.population_size,
               "prepared gpu reproduction child count mismatch")) {
      return false;
    }
    for (const auto& child : reproduction.next_population) {
      const auto bc = g3pvm::evo::compile_for_eval(child);
      if (!check(!bc.code.empty(), "prepared gpu reproduction child should compile")) {
        return false;
      }
    }
    if (!check(run_stats.copyback_ms >= 0.0, "prepared gpu reproduction copyback stats missing")) {
      return false;
    }
    if (!check(run_stats.decode_ms >= 0.0, "prepared gpu reproduction decode stats missing")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP gpu prepared (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu reproduction prepared backend failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

}  // namespace

int main() {
  if (!test_preprocess_and_pack()) return 1;
  if (!test_gpu_prepared_backend_smoke()) return 1;
  std::cout << "g3pvm_test_repro_prep: OK\n";
  return 0;
}
