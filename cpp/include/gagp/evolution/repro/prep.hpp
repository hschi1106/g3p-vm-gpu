#pragma once

#include <vector>

#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/ast_verify.hpp"
#include "gagp/evolution/grammar_config.hpp"
#include "gagp/evolution/repro/types.hpp"

namespace gagp::evo {
struct EvolutionConfig;
}

namespace gagp::evo::repro {

GpuReproConfig make_gpu_repro_config(const std::vector<ProgramGenome>& population,
                                     const EvolutionConfig& cfg);

PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const GpuReproConfig& config,
                                       const GrammarConfig& grammar = GrammarConfig{});
PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const std::vector<VerifiedAst>& verified,
                                       const GpuReproConfig& config,
                                       const GrammarConfig& grammar = GrammarConfig{});

}  // namespace gagp::evo::repro
