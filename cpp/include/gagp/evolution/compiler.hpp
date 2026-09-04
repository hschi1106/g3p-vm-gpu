#pragma once

#include <string>
#include <vector>

#include "gagp/core/bytecode.hpp"
#include "gagp/evolution/ast_verify.hpp"
#include "gagp/evolution/genome.hpp"

namespace gagp::evo {

BytecodeProgram compile_for_eval(const ProgramGenome& genome,
                                 const std::vector<std::string>& preset_locals = {});
BytecodeProgram compile_for_eval(const ProgramGenome& genome,
                                 const VerifiedAst& verified,
                                 const std::vector<std::string>& preset_locals = {});

}  // namespace gagp::evo
