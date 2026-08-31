#include <iostream>
#include <string>

#include "g3pvm/evolution/ast_verify.hpp"

namespace {

using namespace g3pvm::evo;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool expect_code(const AstProgram& ast, VerifyCode code, const std::string& label) {
  const AstVerifyResult result = verify_ast(ast, {});
  return check(!result, label + " should fail") &&
         check(result.diagnostic.code == code,
               label + " expected " + verify_code_name(code) + " but got " +
                   verify_code_name(result.diagnostic.code));
}

AstProgram dc_program() {
  AstProgram ast;
  ast.names = {"xs", "n", "lo", "divide_n", "left", "right", "ordinary"};
  ast.consts = {
      g3pvm::Value::from_int_list_hash_len(1, 3),
      g3pvm::Value::from_int(0),
      g3pvm::Value::from_int(1),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CALL_INDEX, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 4, 0},
      AstNode{NodeKind::BOUND_VAR, 5, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 3, 4, 5}};
  return ast;
}

AstProgram dp1_program() {
  AstProgram ast;
  ast.names = {"solve_s", "transition_s", "dep"};
  ast.consts = {
      g3pvm::Value::from_int(4),
      g3pvm::Value::from_int(1),
      g3pvm::Value::from_int(0),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP1D, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.asgp_dp1d_specs = {AsgpDp1dSpec{
      3, 0, 5, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 1, {2},
  }};
  return ast;
}

AstProgram dp2_program() {
  AstProgram ast;
  ast.names = {"solve_i", "solve_j", "transition_i", "transition_j", "a", "b"};
  ast.consts = {
      g3pvm::Value::from_int(2),
      g3pvm::Value::from_int(1),
      g3pvm::Value::from_int(0),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP2D, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 4, 0},
      AstNode{NodeKind::BOUND_VAR, 5, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.asgp_dp2d_specs = {AsgpDp2dSpec{
      3, 0, 3, 0, 3, 0, 0, 2, NodeKind::DP2_CROSS_BACKWARD,
      0, 1, 2, 3, {4, 5},
  }};
  return ast;
}

}  // namespace

int main() {
  using namespace g3pvm::evo;

  AstProgram ast = dc_program();
  AstVerifyResult result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Int,
             "ASGP-DC phase/result typing")) return 1;

  ast = dc_program();
  ast.nodes[5] = AstNode{NodeKind::VAR, 6, 0};
  ast.nodes.erase(ast.nodes.begin() + 6, ast.nodes.begin() + 8);
  if (!expect_code(ast, VerifyCode::UndefinedLocal,
                   "ASGP phase ordinary-local isolation")) return 1;

  ast = dc_program();
  ast.nodes[6].i0 = 3;
  if (!expect_code(ast, VerifyCode::UndefinedBinder,
                   "ASGP phase binder visibility")) return 1;

  ast = dc_program();
  ast.asgp_dc_binders[0].solve_n_name = ast.asgp_dc_binders[0].solve_xs_name;
  if (!expect_code(ast, VerifyCode::DuplicateBinder, "ASGP-DC duplicate solve role")) return 1;

  ast = dp1_program();
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Int,
             "ASGP-DP1D phase/result typing")) return 1;

  ast.consts[2] = g3pvm::Value::from_float(0.0);
  if (!expect_code(ast, VerifyCode::TypeMismatch, "ASGP-DP1D boundary type")) return 1;

  ast = dp1_program();
  ast.asgp_dp1d_specs[0].transition_dep_names[0] =
      ast.asgp_dp1d_specs[0].transition_state_name;
  if (!expect_code(ast, VerifyCode::DuplicateBinder,
                   "ASGP-DP1D transition binder collision")) return 1;

  ast = dp2_program();
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Int,
             "ASGP-DP2D phase/result typing")) return 1;

  ast.nodes[4] = AstNode{NodeKind::CONST, 1, 0};
  ast.consts[1] = g3pvm::Value::from_float(1.0);
  if (!expect_code(ast, VerifyCode::TypeMismatch, "ASGP-DP2D state type")) return 1;

  ast = dc_program();
  const std::size_t nested_index = 5;
  ast.nodes.erase(ast.nodes.begin() + 5, ast.nodes.begin() + 8);
  ast.nodes.insert(ast.nodes.begin() + 5, {
      AstNode{NodeKind::ASGP_DP1D, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::CONST, 2, 0},
  });
  ast.asgp_dp1d_specs = {AsgpDp1dSpec{
      nested_index, 0, 2, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 1, 3, {4},
  }};
  if (!expect_code(ast, VerifyCode::NestedAsgp, "nested ASGP phase form")) return 1;

  return 0;
}
