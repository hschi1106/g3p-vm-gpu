#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/repro/stats.hpp"
#include "g3pvm/evolution/selection.hpp"

namespace g3pvm::evo {
struct EvolutionConfig;
}

namespace g3pvm::evo::repro {

enum class ReproductionBackend {
  Cpu,
  Gpu,
};

struct ReproductionResult {
  std::vector<ProgramGenome> next_population;
  ReproductionStats stats;
};

std::string reproduction_backend_name(ReproductionBackend backend);
ReproductionBackend parse_reproduction_backend_name(const std::string& raw);

ReproductionResult run_reproduction_backend(const std::vector<ScoredGenome>& scored,
                                            const EvolutionConfig& cfg,
                                            std::mt19937_64& rng);

}  // namespace g3pvm::evo::repro
