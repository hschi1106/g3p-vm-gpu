#pragma once

#include <string>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

BytecodeProgram compile_for_eval(const ProgramGenome& genome,
                                 const std::vector<std::string>& preset_locals = {});

}  // namespace g3pvm::evo
