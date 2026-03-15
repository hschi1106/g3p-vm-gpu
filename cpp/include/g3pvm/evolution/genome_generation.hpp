#pragma once

#include <cstdint>

#include "g3pvm/evolution/ast_program.hpp"
#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

ProgramGenome generate_random_genome(std::uint64_t seed, const Limits& limits = Limits{});
ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits = Limits{});

}  // namespace g3pvm::evo
