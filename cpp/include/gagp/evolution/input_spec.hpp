#pragma once

#include <string>

#include "gagp/evolution/ast_program.hpp"

namespace gagp::evo {

struct InputSpec {
  std::string name;
  RType type = RType::Invalid;
};

}  // namespace gagp::evo
