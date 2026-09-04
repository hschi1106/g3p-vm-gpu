#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/repro/stats.hpp"
#include "gagp/evolution/selection.hpp"

namespace gagp::evo {
struct EvolutionConfig;
}

namespace gagp::evo::repro {

enum class ReproductionBackend {
  Cpu,
  Gpu,
};

enum class CpuReproAblation {
  None,
  GpuSelection,
  GpuCandidates,
  GpuCoupledDonor,
};

struct ReproductionResult {
  std::vector<ProgramGenome> next_population;
  ReproductionStats stats;
};

std::string reproduction_backend_name(ReproductionBackend backend);
ReproductionBackend parse_reproduction_backend_name(const std::string& raw);
std::string cpu_repro_ablation_name(CpuReproAblation ablation);
CpuReproAblation parse_cpu_repro_ablation_name(const std::string& raw);

ReproductionResult run_reproduction_backend(const std::vector<ScoredGenome>& scored,
                                            const EvolutionConfig& cfg,
                                            std::mt19937_64& rng);
ReproductionResult run_reproduction_backend(const std::vector<ScoredGenomeRef>& scored,
                                            const EvolutionConfig& cfg,
                                            std::mt19937_64& rng);

}  // namespace gagp::evo::repro
