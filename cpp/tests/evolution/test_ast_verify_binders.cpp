#include <iostream>
#include <string>
#include <vector>

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

AstProgram map_program() {
  AstProgram ast;
  ast.names = {"u", "outside"};
  ast.consts = {
      g3pvm::Value::from_int_list_hash_len(1, 2),
      g3pvm::Value::from_int(1),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  return ast;
}

AstProgram filter_program() {
  AstProgram ast;
  ast.names = {"u"};
  ast.consts = {
      g3pvm::Value::from_float_list_hash_len(1, 2),
      g3pvm::Value::from_float(0.0),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::FILTER_LIST, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::GT, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  return ast;
}

AstProgram linear_program() {
  AstProgram ast;
  ast.names = {"u", "v", "idx"};
  ast.consts = {
      g3pvm::Value::from_int_list_hash_len(1, 3),
      g3pvm::Value::from_int(0),
      g3pvm::Value::from_int(0),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  return ast;
}

}  // namespace

int main() {
  using namespace g3pvm::evo;

  AstProgram ast = map_program();
  AstVerifyResult result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::IntList,
             "MapList binder and result type")) return 1;

  ast.nodes[6].i0 = 1;
  if (!expect_code(ast, VerifyCode::UndefinedBinder, "MapList invisible binder")) return 1;

  ast = filter_program();
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::FloatList,
             "FilterList binder and predicate")) return 1;

  ast.nodes[5] = AstNode{NodeKind::CONST, 1, 0};
  ast.nodes.erase(ast.nodes.begin() + 6, ast.nodes.begin() + 8);
  if (!expect_code(ast, VerifyCode::TypeMismatch, "FilterList non-Bool predicate")) return 1;

  ast = linear_program();
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Int,
             "LinearRec binder/result rules")) return 1;

  ast.linear_rec_binders[0].accum_name = ast.linear_rec_binders[0].elem_name;
  if (!expect_code(ast, VerifyCode::DuplicateBinder, "LinearRec duplicate role")) return 1;

  ast = map_program();
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  if (!expect_code(ast, VerifyCode::UndefinedBinder, "top-level BoundVar")) return 1;

  ast = map_program();
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::IntList,
             "nested binder shadowing remains capture-safe")) return 1;

  ast = map_program();
  ast.names = {"shared"};
  ast.consts = {
      g3pvm::Value::from_int(10),
      g3pvm::Value::from_int_list_hash_len(1, 2),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::IntList,
             "ordinary locals and binders use separate namespaces")) return 1;

  return 0;
}
