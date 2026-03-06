#pragma once

#include <cstddef>
#include <vector>

#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo::typed_expr {

struct TypedExprRoot {
  std::size_t start = 0;
  std::size_t stop = 0;
  RType type = RType::Invalid;
};

std::vector<TypedExprRoot> collect_typed_expr_roots(const AstProgram& program,
                                                    const std::vector<std::size_t>& subtree_end);

}  // namespace g3pvm::evo::typed_expr
