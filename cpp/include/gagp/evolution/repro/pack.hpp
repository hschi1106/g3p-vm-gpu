#pragma once

#include <vector>

#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/repro/types.hpp"
#include "gagp/evolution/selection.hpp"

namespace gagp::evo {
struct EvolutionConfig;
}

namespace gagp::evo::repro {

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

}  // namespace gagp::evo::repro
