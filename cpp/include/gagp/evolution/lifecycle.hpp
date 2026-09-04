#pragma once

#include <cstdint>
#include <future>
#include <vector>

#include "gagp/core/value.hpp"
#include "gagp/evolution/case_set.hpp"
#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/repro/backend.hpp"
#include "gagp/evolution/repro/gpu.hpp"
#include "gagp/evolution/selection.hpp"

namespace gagp::evo {

struct EvolutionConfig;

class PayloadLifetimeManager {
 public:
  explicit PayloadLifetimeManager(const std::vector<EvalCase>& cases);

  void retain(const std::vector<ProgramGenome>& population,
              const std::vector<ScoredGenome>& history_best,
              const ScoredGenome* best = nullptr,
              const std::vector<ScoredGenome>* final_population = nullptr) const;

 private:
  std::vector<Value> case_roots_;
};

struct OverlapPrepared {
  repro::GpuReproPreparedData prepared;
  repro::ReproductionStats stats;
};

bool gpu_reproduction_overlap_enabled(const EvolutionConfig& config);
std::future<OverlapPrepared> start_gpu_reproduction_overlap(
    const std::vector<ProgramGenome>& population,
    const EvolutionConfig& config,
    std::uint64_t seed);
repro::ReproductionResult finish_gpu_reproduction_overlap(
    std::future<OverlapPrepared>* future,
    const std::vector<ProgramGenome>& population,
    const std::vector<double>& raw_fitness,
    const EvolutionConfig& config);

}  // namespace gagp::evo
