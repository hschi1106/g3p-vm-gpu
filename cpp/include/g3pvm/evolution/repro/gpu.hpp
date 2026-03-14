#pragma once

#include <random>
#include <vector>

#include "g3pvm/evolution/repro/backend.hpp"
#include "g3pvm/evolution/repro/types.hpp"

namespace g3pvm::evo {
struct EvolutionConfig;
}

namespace g3pvm::evo::repro {

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

ReproductionResult run_gpu_repro_backend(const std::vector<ScoredGenome>& scored,
                                         const EvolutionConfig& cfg,
                                         std::mt19937_64& rng);

}  // namespace g3pvm::evo::repro
