#pragma once

#include <random>
#include <vector>

#include "gagp/evolution/repro/backend.hpp"
#include "gagp/evolution/repro/types.hpp"

namespace gagp::evo {
struct EvolutionConfig;
}

namespace gagp::evo::repro {

struct GpuReproPreparedData {
  GpuReproConfig config;
  PackedHostData packed;
};

GpuReproPreparedData prepare_gpu_repro_backend_inputs(const std::vector<ProgramGenome>& population,
                                                      const EvolutionConfig& cfg,
                                                      std::uint64_t seed,
                                                      ReproductionStats* stats = nullptr);

ReproductionResult run_gpu_repro_backend_prepared(const std::vector<ScoredGenome>& scored,
                                                  const EvolutionConfig& cfg,
                                                  const GpuReproPreparedData& prepared,
                                                  ReproductionStats* stats = nullptr);
ReproductionResult run_gpu_repro_backend_prepared(const std::vector<ScoredGenomeRef>& scored,
                                                  const EvolutionConfig& cfg,
                                                  const GpuReproPreparedData& prepared,
                                                  ReproductionStats* stats = nullptr);

ReproductionResult run_gpu_repro_backend(const std::vector<ScoredGenome>& scored,
                                         const EvolutionConfig& cfg,
                                         std::mt19937_64& rng);
ReproductionResult run_gpu_repro_backend(const std::vector<ScoredGenomeRef>& scored,
                                         const EvolutionConfig& cfg,
                                         std::mt19937_64& rng);

}  // namespace gagp::evo::repro
