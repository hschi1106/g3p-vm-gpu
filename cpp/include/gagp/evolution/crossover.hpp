#pragma once

#include <cstdint>
#include <utility>

#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/ast_verify.hpp"

namespace gagp::evo {

std::pair<ProgramGenome, ProgramGenome> crossover(const ProgramGenome& parent_a,
                                                  const ProgramGenome& parent_b,
                                                  std::uint64_t seed,
                                                  const Limits& limits = Limits{});
std::pair<ProgramGenome, ProgramGenome> crossover(const ProgramGenome& parent_a,
                                                  const VerifiedAst& verified_a,
                                                  const ProgramGenome& parent_b,
                                                  const VerifiedAst& verified_b,
                                                  std::uint64_t seed,
                                                  const Limits& limits = Limits{});

}  // namespace gagp::evo
