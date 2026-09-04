#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gagp/evolution/ast_program.hpp"
#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/grammar_config.hpp"
#include "gagp/evolution/input_spec.hpp"

namespace gagp::evo {

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

}  // namespace gagp::evo
