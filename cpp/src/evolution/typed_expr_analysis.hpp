#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gagp/evolution/ast_program.hpp"
#include "gagp/evolution/ast_verify.hpp"

namespace gagp::evo::typed_expr {

struct TypedExprRoot {
  std::size_t start = 0;
  std::size_t stop = 0;
  RType type = RType::Invalid;
  std::uint64_t scope_signature = 0;
  std::uint64_t binder_signature = 0;
  int scheme_kind = 0;
  int phase_name = 0;
  std::uint64_t visible_env_signature = 0;
  int dp_dependency_arity = -1;
};

std::vector<TypedExprRoot> collect_typed_expr_roots(const AstProgram& program,
                                                    const std::vector<std::size_t>& subtree_end);
std::vector<TypedExprRoot> collect_typed_expr_roots(const AstProgram& program,
                                                    const VerifiedAst& verified);
bool is_asgp_phase_body_root(const AstProgram& program,
                             const std::vector<std::size_t>& subtree_end,
                             const TypedExprRoot& root);
bool is_statement_value_root(const AstProgram& program, const TypedExprRoot& root);
bool typed_subtree_keys_compatible(const TypedExprRoot& a, const TypedExprRoot& b);

}  // namespace gagp::evo::typed_expr
