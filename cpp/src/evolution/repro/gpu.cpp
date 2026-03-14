#include "g3pvm/evolution/repro/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/repro/pack.hpp"
#include "g3pvm/evolution/repro/prep.hpp"
#include "gpu/internal.hpp"

namespace g3pvm::evo::repro {

namespace {

struct GpuReproRuntimeCache {
  GpuReproArena arena;
  GpuReproHostStaging staging;

  ~GpuReproRuntimeCache() {
    destroy_gpu_repro_host_staging(&staging);
    destroy_gpu_repro_arena(&arena);
  }
};

GpuReproRuntimeCache& gpu_runtime_cache() {
  static GpuReproRuntimeCache cache;
  return cache;
}

std::vector<double> extract_fitness(const std::vector<ScoredGenome>& scored) {
  std::vector<double> fitness;
  fitness.reserve(scored.size());
  for (const ScoredGenome& one : scored) {
    fitness.push_back(one.fitness);
  }
  return fitness;
}

}  // namespace

GpuReproPreparedData prepare_gpu_repro_backend_inputs(const std::vector<ProgramGenome>& population,
                                                      const EvolutionConfig& cfg,
                                                      std::uint64_t seed,
                                                      ReproductionStats* stats) {
  GpuReproPreparedData out;
  const auto prepare_t0 = std::chrono::steady_clock::now();
  out.config = make_gpu_repro_config(population, cfg);
  out.config.seed = seed;
  if (out.config.max_names > kGpuReproMaxNames || out.config.max_consts > kGpuReproMaxConsts ||
      out.config.max_nodes > kGpuReproKernelMaxNodes) {
    throw std::runtime_error("gpu reproduction unsupported: kernel scratch limits exceeded");
  }
  const auto prepare_t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->prepare_inputs_ms +=
        std::chrono::duration<double, std::milli>(prepare_t1 - prepare_t0).count();
  }

  const auto prep_t0 = std::chrono::steady_clock::now();
  const PreprocessOutput prep = preprocess_population(population, out.config);
  const auto prep_t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->preprocess_ms += std::chrono::duration<double, std::milli>(prep_t1 - prep_t0).count();
  }

  const auto pack_t0 = std::chrono::steady_clock::now();
  out.packed = pack_population(population, prep, out.config);
  const auto pack_t1 = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->pack_ms += std::chrono::duration<double, std::milli>(pack_t1 - pack_t0).count();
  }
  return out;
}

ReproductionResult run_gpu_repro_backend_prepared(const std::vector<ScoredGenome>& scored,
                                                  const EvolutionConfig& cfg,
                                                  const GpuReproPreparedData& prepared,
                                                  ReproductionStats* stats) {
  if (scored.empty()) {
    return ReproductionResult{};
  }
  if (static_cast<int>(scored.size()) != prepared.config.population_size) {
    throw std::runtime_error("gpu reproduction prepared population size mismatch");
  }

  GpuReproRuntimeCache& cache = gpu_runtime_cache();
  std::string message;
  const auto setup_t0 = std::chrono::steady_clock::now();
  if (!ensure_gpu_repro_arena_capacity(&cache.arena, prepared.config, &message) ||
      !ensure_gpu_repro_host_staging_capacity(&cache.staging, prepared.config, &message)) {
    throw std::runtime_error(message);
  }
  const auto setup_t1 = std::chrono::steady_clock::now();

  ReproductionResult out;
  if (stats != nullptr) {
    out.stats = *stats;
  }
  out.stats.setup_ms += std::chrono::duration<double, std::milli>(setup_t1 - setup_t0).count();

  try {
    if (!upload_gpu_repro_inputs(prepared.packed, &cache.arena, &out.stats, &message)) {
      throw std::runtime_error(message);
    }
    if (!launch_gpu_repro_kernels(&cache.arena, prepared.config, extract_fitness(scored), &out.stats, &message)) {
      throw std::runtime_error(message);
    }
    GpuReproChildView copyback;
    if (!copyback_gpu_repro_children(cache.arena, prepared.config, &cache.staging, &copyback, &out.stats, &message)) {
      throw std::runtime_error(message);
    }
    const auto decode_t0 = std::chrono::steady_clock::now();
    out.next_population = decode_gpu_repro_children(prepared.packed, copyback, scored, cfg);
    const auto decode_t1 = std::chrono::steady_clock::now();
    out.stats.decode_ms += std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
  } catch (...) {
    throw;
  }
  return out;
}

ReproductionResult run_gpu_repro_backend(const std::vector<ScoredGenome>& scored,
                                         const EvolutionConfig& cfg,
                                         std::mt19937_64& rng) {
#ifndef G3PVM_HAS_CUDA
  (void)scored;
  (void)cfg;
  (void)rng;
  throw std::runtime_error("gpu reproduction requested but CUDA is unavailable in this build");
#else
  if (scored.empty()) {
    return ReproductionResult{};
  }

  std::vector<ProgramGenome> population;
  population.reserve(scored.size());
  for (const ScoredGenome& one : scored) {
    population.push_back(one.genome);
  }
  ReproductionStats prep_stats;
  const GpuReproPreparedData prepared = prepare_gpu_repro_backend_inputs(population, cfg, rng(), &prep_stats);
  return run_gpu_repro_backend_prepared(scored, cfg, prepared, &prep_stats);
#endif
}

}  // namespace g3pvm::evo::repro
