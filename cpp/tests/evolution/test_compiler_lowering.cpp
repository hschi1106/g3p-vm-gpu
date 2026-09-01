#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/input_spec.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"

namespace {

using g3pvm::Value;
using g3pvm::evo::AstNode;
using g3pvm::evo::AstProgram;
using g3pvm::evo::NodeKind;
using g3pvm::evo::ProgramGenome;
using g3pvm::evo::RType;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

ProgramGenome genome(AstProgram ast) {
  ProgramGenome out;
  out.meta = g3pvm::evo::build_genome_meta(ast);
  out.ast = std::move(ast);
  return out;
}

bool test_for_range_bound_is_evaluated_once() {
  AstProgram ast;
  ast.names = {"n", "count", "i"};
  ast.consts = {Value::from_int(0), Value::from_int(1)};
  ast.nodes = {
      {NodeKind::PROGRAM, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::ASSIGN, 1, 0},
      {NodeKind::CONST, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::FOR_RANGE, 2, 0},
      {NodeKind::VAR, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::ASSIGN, 0, 0},
      {NodeKind::CONST, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::ASSIGN, 1, 0},
      {NodeKind::ADD, 0, 0},
      {NodeKind::VAR, 1, 0},
      {NodeKind::CONST, 1, 0},
      {NodeKind::BLOCK_NIL, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::RETURN, 0, 0},
      {NodeKind::VAR, 1, 0},
      {NodeKind::BLOCK_NIL, 0, 0},
  };

  const std::vector<g3pvm::evo::InputSpec> inputs{{"n", RType::Int}};
  const auto verified = g3pvm::evo::verify_ast(ast, inputs);
  if (!check(verified.ok, "evaluate-once loop AST should verify")) return false;

  const auto bytecode = g3pvm::evo::compile_for_eval(genome(ast), {"n"});
  const auto result = g3pvm::execute_bytecode_cpu(
      bytecode, {{0, Value::from_int(3)}}, 20000);
  return check(!result.is_error && result.value.tag == g3pvm::ValueTag::Int &&
                   result.value.i == 3,
               "loop bound must be cached before the loop body mutates n");
}

AstProgram short_circuit_program(NodeKind logical, bool lhs) {
  AstProgram ast;
  ast.consts = {Value::from_bool(lhs), Value::from_float(1.0), Value::from_float(0.0)};
  ast.nodes = {
      {NodeKind::PROGRAM, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::RETURN, 0, 0},
      {logical, 0, 0},
      {NodeKind::CONST, 0, 0},
      {NodeKind::EQ, 0, 0},
      {NodeKind::DIV, 0, 0},
      {NodeKind::CONST, 1, 0},
      {NodeKind::CONST, 2, 0},
      {NodeKind::CONST, 2, 0},
      {NodeKind::BLOCK_NIL, 0, 0},
  };
  return ast;
}

bool test_logical_operators_short_circuit() {
  for (const auto& item :
       std::vector<std::pair<NodeKind, bool>>{{NodeKind::AND, false},
                                              {NodeKind::OR, true}}) {
    AstProgram ast = short_circuit_program(item.first, item.second);
    const auto verified = g3pvm::evo::verify_ast(ast, {});
    if (!check(verified.ok,
               std::string("short-circuit AST should verify: ") +
                   g3pvm::evo::verify_code_name(verified.diagnostic.code) + " " +
                   verified.diagnostic.message)) return false;
    const auto result = g3pvm::execute_bytecode_cpu(
        g3pvm::evo::compile_for_eval(genome(std::move(ast))), {}, 20000);
    if (!check(!result.is_error && result.value.tag == g3pvm::ValueTag::Bool &&
                   result.value.b == item.second,
               "short-circuit lowering must skip division by zero")) return false;
  }
  return true;
}

bool test_verified_compile_reuses_and_validates_annotations() {
  AstProgram ast = short_circuit_program(NodeKind::AND, false);
  const auto verified = g3pvm::evo::verify_ast(ast, {});
  if (!check(verified.ok, "verified compile fixture should verify")) return false;

  const ProgramGenome program = genome(ast);
  const auto bytecode = g3pvm::evo::compile_for_eval(program, verified.verified);
  const auto result = g3pvm::execute_bytecode_cpu(bytecode, {}, 20000);
  if (!check(!result.is_error && result.value.tag == g3pvm::ValueTag::Bool &&
                 !result.value.b,
             "verified compile should preserve execution semantics")) return false;

  g3pvm::evo::VerifiedAst malformed = verified.verified;
  malformed.subtree_end.pop_back();
  try {
    (void)g3pvm::evo::compile_for_eval(program, malformed);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return check(false, "verified compile should reject annotations for another AST shape");
}

}  // namespace

int main() {
  if (!test_for_range_bound_is_evaluated_once()) return 1;
  if (!test_logical_operators_short_circuit()) return 1;
  if (!test_verified_compile_reuses_and_validates_annotations()) return 1;
  std::cout << "g3pvm_test_compiler_lowering: OK\n";
  return 0;
}
