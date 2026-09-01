#pragma once

#include <vector>

#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "g3pvm/evolution/repro/types.hpp"

namespace g3pvm::evo {
struct EvolutionConfig;
}

namespace g3pvm::evo::repro {

GpuReproConfig make_gpu_repro_config(const std::vector<ProgramGenome>& population,
                                     const EvolutionConfig& cfg);

PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const GpuReproConfig& config,
                                       const GrammarConfig& grammar = GrammarConfig{});
PreprocessOutput preprocess_population(const std::vector<ProgramGenome>& population,
                                       const std::vector<VerifiedAst>& verified,
                                       const GpuReproConfig& config,
                                       const GrammarConfig& grammar = GrammarConfig{});

}  // namespace g3pvm::evo::repro
