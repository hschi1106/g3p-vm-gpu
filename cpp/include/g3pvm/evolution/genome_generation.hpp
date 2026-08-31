#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "g3pvm/evolution/ast_program.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "g3pvm/evolution/input_spec.hpp"

namespace g3pvm::evo {

ProgramGenome generate_random_genome(std::uint64_t seed, const Limits& limits = Limits{});
ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const GrammarConfig& grammar);
ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const std::vector<InputSpec>& input_specs);
ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const std::vector<InputSpec>& input_specs,
                                     const GrammarConfig& grammar);
ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits = Limits{});
ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const GrammarConfig& grammar);
ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const std::vector<InputSpec>& input_specs);
ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const std::vector<InputSpec>& input_specs,
                                                     const GrammarConfig& grammar);

}  // namespace g3pvm::evo
