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

ProgramGenome compact_genome_tables(const ProgramGenome& genome);
std::vector<ProgramGenome> compact_population_tables(const std::vector<ProgramGenome>& population);

std::vector<ProgramGenome> decode_gpu_repro_children(const PackedHostData& packed,
                                                     const GpuReproChildView& copyback,
                                                     const std::vector<ScoredGenome>& scored,
                                                     const EvolutionConfig& cfg);
std::vector<ProgramGenome> decode_gpu_repro_children(const PackedHostData& packed,
                                                     const GpuReproChildView& copyback,
                                                     const std::vector<ScoredGenomeRef>& scored,
                                                     const EvolutionConfig& cfg);

}  // namespace g3pvm::evo::repro
