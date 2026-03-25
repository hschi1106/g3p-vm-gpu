#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/repro/pack.hpp"
#include "g3pvm/evolution/repro/prep.hpp"
#include "g3pvm/evolution/selection.hpp"

#ifdef G3PVM_HAS_CUDA
#include "../../src/evolution/repro/gpu/internal.hpp"
#endif

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

std::vector<g3pvm::evo::ProgramGenome> make_population(int count = 4) {
  g3pvm::evo::Limits limits;
  limits.max_expr_depth = 5;
  limits.max_stmts_per_block = 6;
  limits.max_total_nodes = 80;
  limits.max_for_k = 16;
  limits.max_call_args = 3;

  std::vector<g3pvm::evo::ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(count));
  for (int seed = 1; seed <= count; ++seed) {
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
  if (!check(static_cast<int>(prep.donor_pool.size()) ==
                 repro_cfg.donor_pool_size_per_type * g3pvm::evo::repro::kGpuReproDonorTypeCount,
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
  if (!check(static_cast<int>(packed.donor_lens.size()) ==
                 repro_cfg.donor_pool_size_per_type * g3pvm::evo::repro::kGpuReproDonorTypeCount,
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

#ifdef G3PVM_HAS_CUDA
bool copyback_gpu_selection_parents(const g3pvm::evo::EvolutionConfig& cfg,
                                    const std::vector<g3pvm::evo::ProgramGenome>& population,
                                    const std::vector<double>& ranked_fitness,
                                    std::uint64_t seed,
                                    std::vector<int>* parent_a_out,
                                    std::vector<int>* parent_b_out,
                                    std::string* message_out) {
  g3pvm::evo::repro::ReproductionStats prep_stats;
  const auto prepared =
      g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, seed, &prep_stats);

  g3pvm::evo::repro::GpuReproArena arena;
  g3pvm::evo::repro::GpuReproHostStaging staging;
  struct Cleanup {
    g3pvm::evo::repro::GpuReproArena* arena = nullptr;
    g3pvm::evo::repro::GpuReproHostStaging* staging = nullptr;
    ~Cleanup() {
      if (staging != nullptr) g3pvm::evo::repro::destroy_gpu_repro_host_staging(staging);
      if (arena != nullptr) g3pvm::evo::repro::destroy_gpu_repro_arena(arena);
    }
  } cleanup{&arena, &staging};

  if (!g3pvm::evo::repro::ensure_gpu_repro_arena_capacity(&arena, prepared.config, message_out) ||
      !g3pvm::evo::repro::ensure_gpu_repro_host_staging_capacity(&staging, prepared.config, message_out)) {
    return false;
  }

  g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
  if (!g3pvm::evo::repro::upload_gpu_repro_inputs(prepared.packed, &arena, &run_stats, message_out) ||
      !g3pvm::evo::repro::launch_gpu_repro_kernels(&arena, prepared.config, ranked_fitness, &run_stats, message_out)) {
    return false;
  }

  g3pvm::evo::repro::GpuReproChildView copyback;
  if (!g3pvm::evo::repro::copyback_gpu_repro_children(
          arena, prepared.config, &staging, &copyback, &run_stats, message_out)) {
    return false;
  }

  parent_a_out->assign(copyback.parent_a, copyback.parent_a + prepared.config.pair_count);
  parent_b_out->assign(copyback.parent_b, copyback.parent_b + prepared.config.pair_count);
  return true;
}
#endif

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

bool test_gpu_selection_preserves_round_based_tournament_invariants() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.seed = 42;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = make_population(cfg.population_size);
  std::vector<double> ranked_fitness;
  ranked_fitness.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    ranked_fitness.push_back(
        g3pvm::evo::canonicalize_fitness_for_ranking(static_cast<double>(i + 1)));
  }

  std::string message;
  std::vector<int> parent_a;
  std::vector<int> parent_b;

  cfg.selection_pressure = cfg.population_size;
  if (!copyback_gpu_selection_parents(cfg, population, ranked_fitness, 777, &parent_a, &parent_b, &message)) {
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP gpu selection invariants (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu selection invariant setup failed: " << message << "\n";
    return false;
  }
  const int best_index = static_cast<int>(population.size()) - 1;
  for (std::size_t i = 0; i < parent_a.size(); ++i) {
    if (!check(parent_a[i] == best_index && parent_b[i] == best_index,
               "full-pressure GPU selection should always pick the best genome")) {
      return false;
    }
  }

  cfg.selection_pressure = 1;
  message.clear();
  if (!copyback_gpu_selection_parents(cfg, population, ranked_fitness, 778, &parent_a, &parent_b, &message)) {
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP gpu selection invariants (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu selection invariant rerun failed: " << message << "\n";
    return false;
  }

  std::vector<bool> seen(population.size(), false);
  for (std::size_t pair = 0; pair < parent_a.size(); ++pair) {
    for (int parent : {parent_a[pair], parent_b[pair]}) {
      if (!check(parent >= 0 && parent < static_cast<int>(population.size()),
                 "gpu selection parent index out of range")) {
        return false;
      }
      if (!check(!seen[static_cast<std::size_t>(parent)],
                 "k=1 GPU selection should not repeat winners within the first round")) {
        return false;
      }
      seen[static_cast<std::size_t>(parent)] = true;
    }
  }
  for (bool one_seen : seen) {
    if (!check(one_seen, "k=1 GPU selection should visit every genome exactly once")) {
      return false;
    }
  }
#endif
  return true;
}

}  // namespace

int main() {
  if (!test_preprocess_and_pack()) return 1;
  if (!test_gpu_prepared_backend_smoke()) return 1;
  if (!test_gpu_selection_preserves_round_based_tournament_invariants()) return 1;
  std::cout << "g3pvm_test_repro_prep: OK\n";
  return 0;
}
