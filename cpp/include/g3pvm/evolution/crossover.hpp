#pragma once

#include <cstdint>

#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

ProgramGenome crossover(const ProgramGenome& parent_a,
                        const ProgramGenome& parent_b,
                        std::uint64_t seed,
                        const Limits& limits = Limits{});

}  // namespace g3pvm::evo
