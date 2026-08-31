#pragma once

#include <string>

#include "g3pvm/evolution/ast_program.hpp"

namespace g3pvm::evo {

struct InputSpec {
  std::string name;
  RType type = RType::Invalid;
};

}  // namespace g3pvm::evo
