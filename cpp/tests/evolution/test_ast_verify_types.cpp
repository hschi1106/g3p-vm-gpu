#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "typed_expr_analysis.hpp"

namespace {

using namespace g3pvm::evo;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

AstProgram return_program(std::vector<AstNode> expression,
                          std::vector<g3pvm::Value> consts,
                          std::vector<std::string> names = {}) {
  AstProgram ast;
  ast.names = std::move(names);
  ast.consts = std::move(consts);
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
  };
  ast.nodes.insert(ast.nodes.end(), expression.begin(), expression.end());
  ast.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return ast;
}

bool expect_code(const AstProgram& ast, const std::vector<InputSpec>& inputs,
                 VerifyCode code, const std::string& label,
                 const VerifyOptions& options = VerifyOptions{}) {
  const AstVerifyResult result = verify_ast(ast, inputs, options);
  return check(!result, label + " should fail") &&
         check(result.diagnostic.code == code,
               label + " expected " + verify_code_name(code) + " but got " +
                   verify_code_name(result.diagnostic.code));
}

}  // namespace

int main() {
  using namespace g3pvm::evo;

  AstProgram ast = return_program(
      {AstNode{NodeKind::ADD, 0, 0}, AstNode{NodeKind::VAR, 0, 0},
       AstNode{NodeKind::CONST, 0, 0}},
      {g3pvm::Value::from_int(2)}, {"x"});
  AstVerifyResult result = verify_ast(ast, {InputSpec{"x", RType::Int}});
  if (!check(result.ok, "typed input arithmetic") ||
      !check(result.verified.return_type == RType::Int, "inferred Int return") ||
      !check(result.verified.expression_types[3] == RType::Int, "ADD annotation") ||
      !check(result.verified.expression_types[4] == RType::Int, "Var annotation") ||
      !check(result.verified.expression_scope_signatures[3] != 0, "scope annotation")) return 1;

  const auto roots = typed_expr::collect_typed_expr_roots(ast, result.verified);
  if (!check(!roots.empty(), "verified typed-root view") ||
      !check(roots.front().type == RType::Int, "verified root type")) return 1;

  if (!expect_code(ast, {}, VerifyCode::UndefinedLocal, "missing input")) return 1;
  if (!expect_code(ast, {InputSpec{"x", RType::Any}}, VerifyCode::InvalidInputType,
                   "non-exact input")) return 1;
  if (!expect_code(ast, {InputSpec{"x", RType::Int}, InputSpec{"x", RType::Int}},
                   VerifyCode::DuplicateInput, "duplicate input")) return 1;

  ast = return_program({AstNode{NodeKind::ADD, 0, 0}, AstNode{NodeKind::CONST, 0, 0},
                        AstNode{NodeKind::CONST, 1, 0}},
                       {g3pvm::Value::from_int(1), g3pvm::Value::from_float(2.0)});
  if (!expect_code(ast, {}, VerifyCode::TypeMismatch, "mixed numeric add")) return 1;

  ast = return_program({AstNode{NodeKind::IF_EXPR, 0, 0}, AstNode{NodeKind::CONST, 0, 0},
                        AstNode{NodeKind::CONST, 1, 0}, AstNode{NodeKind::CONST, 2, 0}},
                       {g3pvm::Value::from_bool(true), g3pvm::Value::from_int(1),
                        g3pvm::Value::from_float(1.0)});
  if (!expect_code(ast, {}, VerifyCode::TypeMismatch, "IfExpr branch mismatch")) return 1;

  ast = return_program({AstNode{NodeKind::CALL_INDEX, 0, 0}, AstNode{NodeKind::CONST, 0, 0},
                        AstNode{NodeKind::CONST, 1, 0}},
                       {g3pvm::Value::from_string_hash_len(1, 3), g3pvm::Value::from_int(0)});
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Char,
             "String index infers Char")) return 1;

  ast = return_program({AstNode{NodeKind::CALL_SINGLETON, 0, 0},
                        AstNode{NodeKind::CONST, 0, 0}},
                       {g3pvm::Value::from_string_hash_len(2, 1)});
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::StringList,
             "String singleton infers StringList")) return 1;

  ast.names = {"x"};
  ast.names.push_back("x");
  if (!expect_code(ast, {}, VerifyCode::DuplicateName, "duplicate AST name")) return 1;

  ast = return_program({AstNode{NodeKind::CONST, 0, 0}},
                       {g3pvm::Value::from_string_hash_len(3, 1)});
  GrammarConfig scalar = GrammarConfig::scalar();
  VerifyOptions options;
  options.grammar_config = &scalar;
  if (!expect_code(ast, {}, VerifyCode::GrammarConfigDisallowed,
                   "grammar-config eligibility", options)) return 1;

  ast = return_program({AstNode{NodeKind::CONST, 0, 0}},
                       {g3pvm::Value::from_int(1)});
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.names = {"local"};
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Int,
             "Assign defines an ordinary local")) return 1;

  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::IF_STMT, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.consts = {g3pvm::Value::from_bool(true), g3pvm::Value::from_int(1)};
  ast.names.clear();
  result = verify_ast(ast, {});
  if (!check(result.ok && result.verified.return_type == RType::Int,
             "IfStmt branches preserve one program return type")) return 1;
  ast.consts[0] = g3pvm::Value::from_int(1);
  if (!expect_code(ast, {}, VerifyCode::TypeMismatch, "IfStmt non-Bool condition")) return 1;

  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.consts = {g3pvm::Value::from_int(1), g3pvm::Value::from_float(1.0)};
  ast.names.clear();
  if (!expect_code(ast, {}, VerifyCode::InconsistentReturnType,
                   "inconsistent returns")) return 1;

  ast.nodes = {AstNode{NodeKind::PROGRAM, 0, 0}, AstNode{NodeKind::BLOCK_NIL, 0, 0}};
  ast.consts.clear();
  if (!expect_code(ast, {}, VerifyCode::MissingReturn, "missing return")) return 1;

  ast = return_program({AstNode{NodeKind::CONST, 0, 0}},
                       {g3pvm::Value::from_float(2.0)}, {"i"});
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::FOR_RANGE, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  if (!expect_code(ast, {}, VerifyCode::TypeMismatch, "ForRange Float bound")) return 1;

  const std::vector<InputSpec> generated_inputs{{"x", RType::Int}};
  for (std::uint64_t seed = 0; seed < 256; ++seed) {
    const ProgramGenome genome = generate_random_genome(seed, Limits{}, generated_inputs);
    result = verify_ast(genome.ast, generated_inputs);
    if (!check(result.ok,
               "generated seed " + std::to_string(seed) + " should type-check: " +
                   verify_code_name(result.diagnostic.code) + " " + result.diagnostic.message +
                   " program=" + ast_to_string(genome.ast))) return 1;
  }

  return 0;
}
