#pragma once

#include <cstdint>

#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/grammar_config.hpp"

namespace g3pvm::evo {

ProgramGenome mutate(const ProgramGenome& genome,
                     std::uint64_t seed,
                     const Limits& limits = Limits{},
                     double mutation_subtree_prob = 0.8,
                     const GrammarConfig& grammar = GrammarConfig{});

}  // namespace g3pvm::evo
