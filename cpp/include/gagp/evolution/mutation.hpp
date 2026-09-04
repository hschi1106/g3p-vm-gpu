#pragma once

#include <cstdint>

#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/grammar_config.hpp"
#include "gagp/evolution/ast_verify.hpp"

namespace gagp::evo {

ProgramGenome mutate(const ProgramGenome& genome,
                     std::uint64_t seed,
                     const Limits& limits = Limits{},
                     double mutation_subtree_prob = 0.8,
                     const GrammarConfig& grammar = GrammarConfig{});
ProgramGenome mutate(const ProgramGenome& genome,
                     const VerifiedAst& verified,
                     std::uint64_t seed,
                     const Limits& limits = Limits{},
                     double mutation_subtree_prob = 0.8,
                     const GrammarConfig& grammar = GrammarConfig{});

}  // namespace gagp::evo
