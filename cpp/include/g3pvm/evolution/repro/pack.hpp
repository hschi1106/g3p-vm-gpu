#pragma once

#include <vector>

#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/repro/types.hpp"
#include "g3pvm/evolution/selection.hpp"

namespace g3pvm::evo {
struct EvolutionConfig;
}

namespace g3pvm::evo::repro {

PackedHostData pack_population(const std::vector<ProgramGenome>& population,
                               const PreprocessOutput& prep,
                               const GpuReproConfig& config);

std::vector<ProgramGenome> decode_gpu_repro_children(const PackedHostData& packed,
                                                     const GpuReproChildView& copyback,
                                                     const std::vector<ScoredGenome>& scored,
                                                     const EvolutionConfig& cfg);

}  // namespace g3pvm::evo::repro
