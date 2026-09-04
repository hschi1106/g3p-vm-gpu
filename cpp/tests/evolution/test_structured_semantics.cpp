#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gagp/core/builtin.hpp"
#include "gagp/core/errors.hpp"
#include "gagp/evolution/ast_verify.hpp"
#include "gagp/evolution/compiler.hpp"
#include "gagp/evolution/genome.hpp"
#include "gagp/runtime/cpu/execute_bytecode_cpu.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace {

using Expr = std::vector<gagp::evo::AstNode>;
using gagp::ErrCode;
using gagp::ExecResult;
using gagp::Value;
using gagp::evo::AstNode;
using gagp::evo::AstProgram;
using gagp::evo::LinearRecBinders;
using gagp::evo::ListTypeTag;
using gagp::evo::NodeKind;
using gagp::evo::ProgramGenome;
using gagp::evo::VerifyCode;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

void append(Expr* target, const Expr& child) {
  target->insert(target->end(), child.begin(), child.end());
}

Expr leaf(NodeKind kind, int i0 = 0, int i1 = 0) {
  return {{kind, i0, i1}};
}

Expr unary(NodeKind kind, const Expr& child) {
  Expr out{{kind, 0, 0}};
  append(&out, child);
  return out;
}

Expr binary(NodeKind kind, const Expr& lhs, const Expr& rhs) {
  Expr out{{kind, 0, 0}};
  append(&out, lhs);
  append(&out, rhs);
  return out;
}

Expr map_list(int binder, ListTypeTag output_type, const Expr& source, const Expr& body) {
  Expr out{{NodeKind::MAP_LIST, binder, static_cast<int>(output_type)}};
  append(&out, source);
  append(&out, body);
  return out;
}

Expr filter_list(int binder, const Expr& source, const Expr& predicate) {
  Expr out{{NodeKind::FILTER_LIST, binder, 0}};
  append(&out, source);
  append(&out, predicate);
  return out;
}

AstProgram return_program(const Expr& expression, std::vector<Value> consts,
                          std::vector<std::string> names = {}) {
  AstProgram ast;
  ast.consts = std::move(consts);
  ast.names = std::move(names);
  ast.nodes = {
      {NodeKind::PROGRAM, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::RETURN, 0, 0},
  };
  append(&ast.nodes, expression);
  ast.nodes.push_back({NodeKind::BLOCK_NIL, 0, 0});
  return ast;
}

ExecResult run(const AstProgram& ast, int fuel = 20000) {
  const auto verified = gagp::evo::verify_ast(ast, {});
  if (!verified.ok) {
    return {true, Value::invalid(),
            {ErrCode::Value,
             std::string("test AST did not verify: ") +
                 gagp::evo::verify_code_name(verified.diagnostic.code) + " " +
                 verified.diagnostic.message}};
  }
  ProgramGenome genome;
  genome.ast = ast;
  genome.meta = gagp::evo::build_genome_meta(ast);
  return gagp::execute_bytecode_cpu(gagp::evo::compile_for_eval(genome), {}, fuel);
}

bool exact_list(const Value& value, gagp::ValueTag tag,
                const std::vector<Value>& expected) {
  if (value.tag != tag) return false;
  std::vector<Value> actual;
  if (!gagp::payload::lookup_list(value, &actual) || actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i].tag != expected[i].tag) return false;
    if (actual[i].tag == gagp::ValueTag::Int && actual[i].i != expected[i].i) return false;
    if (actual[i].tag == gagp::ValueTag::Float && actual[i].f != expected[i].f) return false;
    if (actual[i].tag == gagp::ValueTag::String) {
      std::string lhs;
      std::string rhs;
      if (!gagp::payload::lookup_string(actual[i], &lhs) ||
          !gagp::payload::lookup_string(expected[i], &rhs) || lhs != rhs) return false;
    }
  }
  return true;
}

bool test_map_and_filter_order_and_empty_tags() {
  gagp::payload::clear();
  const Value ints = gagp::payload::make_int_list_value(
      {Value::from_int(1), Value::from_int(2), Value::from_int(3)});
  AstProgram map = return_program(
      map_list(0, ListTypeTag::Int, leaf(NodeKind::CONST, 0),
               binary(NodeKind::MUL, leaf(NodeKind::BOUND_VAR, 0),
                      leaf(NodeKind::CONST, 1))),
      {ints, Value::from_int(2)}, {"x"});
  const ExecResult mapped = run(map);
  if (!check(!mapped.is_error &&
                 exact_list(mapped.value, gagp::ValueTag::IntList,
                            {Value::from_int(2), Value::from_int(4), Value::from_int(6)}),
             "MapList should visit left-to-right and preserve IntList")) return false;

  const Value unordered = gagp::payload::make_int_list_value(
      {Value::from_int(3), Value::from_int(1), Value::from_int(4),
       Value::from_int(1), Value::from_int(5)});
  AstProgram filter = return_program(
      filter_list(0, leaf(NodeKind::CONST, 0),
                  binary(NodeKind::GT, leaf(NodeKind::BOUND_VAR, 0),
                         leaf(NodeKind::CONST, 1))),
      {unordered, Value::from_int(2)}, {"x"});
  const ExecResult filtered = run(filter);
  if (!check(!filtered.is_error &&
                 exact_list(filtered.value, gagp::ValueTag::IntList,
                            {Value::from_int(3), Value::from_int(4), Value::from_int(5)}),
             "FilterList should preserve source order")) return false;

  const Value empty_strings = gagp::payload::make_string_list_value({});
  const Value suffix = gagp::payload::make_string_value("!");
  const ExecResult empty_map = run(return_program(
      map_list(0, ListTypeTag::String, leaf(NodeKind::CONST, 0),
               binary(NodeKind::CALL_CONCAT, leaf(NodeKind::BOUND_VAR, 0),
                      leaf(NodeKind::CONST, 1))),
      {empty_strings, suffix}, {"s"}));
  if (!check(!empty_map.is_error && empty_map.value.tag == gagp::ValueTag::StringList &&
                 gagp::Value::container_len(empty_map.value) == 0,
             "empty MapList should retain StringList tag")) return false;

  const Value floats = gagp::payload::make_float_list_value(
      {Value::from_float(1.0), Value::from_float(2.0)});
  const ExecResult empty_filter = run(return_program(
      filter_list(0, leaf(NodeKind::CONST, 0),
                  binary(NodeKind::GT, leaf(NodeKind::BOUND_VAR, 0),
                         leaf(NodeKind::CONST, 1))),
      {floats, Value::from_float(5.0)}, {"x"}));
  return check(!empty_filter.is_error &&
                   empty_filter.value.tag == gagp::ValueTag::FloatList &&
                   gagp::Value::container_len(empty_filter.value) == 0,
               "empty FilterList should retain FloatList tag");
}

bool test_sources_are_lowered_once() {
  gagp::payload::clear();
  const Value first = gagp::payload::make_int_list_value({Value::from_int(1)});
  const Value second = gagp::payload::make_int_list_value({Value::from_int(2)});
  Expr source = leaf(NodeKind::CONST, 0);
  constexpr int reverse_count = 20;
  for (int i = 0; i < reverse_count; ++i) source = unary(NodeKind::CALL_REVERSE, source);
  source = binary(NodeKind::CALL_CONCAT, source, leaf(NodeKind::CONST, 1));

  AstProgram ast = return_program(
      map_list(0, ListTypeTag::Int, source, leaf(NodeKind::BOUND_VAR, 0)),
      {first, second}, {"x"});
  const auto verified = gagp::evo::verify_ast(ast, {});
  if (!check(verified.ok, "source-evaluate-once AST should verify")) return false;
  ProgramGenome genome;
  genome.ast = ast;
  genome.meta = gagp::evo::build_genome_meta(ast);
  const gagp::BytecodeProgram bytecode = gagp::evo::compile_for_eval(genome);
  const int lowered_reverses = static_cast<int>(std::count_if(
      bytecode.code.begin(), bytecode.code.end(), [](const gagp::Instr& instruction) {
        return instruction.op == gagp::Opcode::CallBuiltin && instruction.has_a &&
               instruction.a == static_cast<int>(gagp::BuiltinId::Reverse);
      }));
  if (!check(lowered_reverses == reverse_count,
             "MapList source expression should appear once in lowered bytecode")) return false;
  const ExecResult result = gagp::execute_bytecode_cpu(bytecode, {}, 20000);
  if (!check(!result.is_error &&
                 exact_list(result.value, gagp::ValueTag::IntList,
                            {Value::from_int(1), Value::from_int(2)}),
             "source-evaluate-once MapList result mismatch")) return false;

  AstProgram filter_ast = return_program(
      filter_list(0, source,
                  binary(NodeKind::GT, leaf(NodeKind::BOUND_VAR, 0),
                         leaf(NodeKind::CONST, 2))),
      {first, second, Value::from_int(0)}, {"x"});
  ProgramGenome filter_genome;
  filter_genome.ast = filter_ast;
  filter_genome.meta = gagp::evo::build_genome_meta(filter_ast);
  const gagp::BytecodeProgram filter_bytecode =
      gagp::evo::compile_for_eval(filter_genome);
  const int filter_reverses = static_cast<int>(std::count_if(
      filter_bytecode.code.begin(), filter_bytecode.code.end(),
      [](const gagp::Instr& instruction) {
        return instruction.op == gagp::Opcode::CallBuiltin && instruction.has_a &&
               instruction.a == static_cast<int>(gagp::BuiltinId::Reverse);
      }));
  const ExecResult filter_result =
      gagp::execute_bytecode_cpu(filter_bytecode, {}, 20000);
  return check(filter_reverses == reverse_count && !filter_result.is_error &&
                   exact_list(filter_result.value, gagp::ValueTag::IntList,
                              {Value::from_int(1), Value::from_int(2)}),
               "FilterList source expression should be evaluated once");
}

AstProgram linear_program(const Value& source, const Value& singleton,
                          const Value& empty, const Expr& step, const Expr& one) {
  Expr expression{{NodeKind::LINEAR_REC, 0, 0}};
  append(&expression, leaf(NodeKind::CONST, 0));
  append(&expression, leaf(NodeKind::CONST, 1));
  append(&expression, leaf(NodeKind::CONST, 2));
  append(&expression, step);
  append(&expression, one);
  AstProgram ast = return_program(expression, {source, singleton, empty}, {"u", "v", "i"});
  ast.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  return ast;
}

bool test_linear_rec_cases_and_order() {
  gagp::payload::clear();
  const Value empty_list = gagp::payload::make_int_list_value({});
  const Expr unused = binary(NodeKind::MOD, leaf(NodeKind::CONST, 3),
                             leaf(NodeKind::CONST, 4));
  AstProgram empty = linear_program(empty_list, Value::from_int(5), Value::from_int(42),
                                    unused, unused);
  empty.consts.push_back(Value::from_int(1));
  empty.consts.push_back(Value::from_int(0));
  const ExecResult empty_out = run(empty);
  if (!check(!empty_out.is_error && empty_out.value.tag == gagp::ValueTag::Int &&
                 empty_out.value.i == 42,
             "LinearRec empty case must not evaluate other branches")) return false;

  const Value singleton_list =
      gagp::payload::make_int_list_value({Value::from_int(7)});
  const Expr one = binary(
      NodeKind::ADD,
      binary(NodeKind::MUL, leaf(NodeKind::BOUND_VAR, 0), leaf(NodeKind::CONST, 3)),
      leaf(NodeKind::BOUND_VAR, 2));
  AstProgram singleton = linear_program(
      singleton_list, Value::from_int(5), Value::from_int(0),
      binary(NodeKind::ADD, leaf(NodeKind::BOUND_VAR, 0),
             leaf(NodeKind::BOUND_VAR, 1)),
      one);
  singleton.consts.push_back(Value::from_int(10));
  const ExecResult singleton_out = run(singleton);
  if (!check(!singleton_out.is_error && singleton_out.value.i == 75,
             "LinearRec singleton case mismatch")) return false;

  const Value list = gagp::payload::make_int_list_value(
      {Value::from_int(1), Value::from_int(2), Value::from_int(3)});
  Expr step = binary(
      NodeKind::ADD,
      binary(NodeKind::MUL, leaf(NodeKind::BOUND_VAR, 1), leaf(NodeKind::CONST, 3)),
      leaf(NodeKind::BOUND_VAR, 0));
  Expr one_case = binary(
      NodeKind::ADD,
      binary(NodeKind::MUL, leaf(NodeKind::BOUND_VAR, 0), leaf(NodeKind::CONST, 4)),
      leaf(NodeKind::BOUND_VAR, 2));
  AstProgram ordered = linear_program(list, Value::from_int(4), Value::from_int(0),
                                      step, one_case);
  ordered.consts.push_back(Value::from_int(10));
  ordered.consts.push_back(Value::from_int(100));
  const ExecResult ordered_out = run(ordered);
  return check(!ordered_out.is_error && ordered_out.value.tag == gagp::ValueTag::Int &&
                   ordered_out.value.i == 30621,
               "LinearRec must apply the step from right to left");
}

bool test_binder_capture_and_ordinary_local_isolation() {
  gagp::payload::clear();
  const Value outer = gagp::payload::make_int_list_value(
      {Value::from_int(1), Value::from_int(2)});
  const Value inner = gagp::payload::make_int_list_value({Value::from_int(10)});
  const Expr nested = map_list(0, ListTypeTag::Int, leaf(NodeKind::CONST, 1),
                               leaf(NodeKind::BOUND_VAR, 0));
  const Expr inner_at_zero = binary(NodeKind::CALL_INDEX, nested,
                                    leaf(NodeKind::CONST, 2));
  const Expr body = binary(NodeKind::ADD, leaf(NodeKind::BOUND_VAR, 0), inner_at_zero);
  const ExecResult nested_out = run(return_program(
      map_list(0, ListTypeTag::Int, leaf(NodeKind::CONST, 0), body),
      {outer, inner, Value::from_int(0)}, {"x"}));
  if (!check(!nested_out.is_error &&
                 exact_list(nested_out.value, gagp::ValueTag::IntList,
                            {Value::from_int(11), Value::from_int(12)}),
             "nested binders should be capture-safe")) return false;

  AstProgram ordinary;
  ordinary.names = {"x"};
  ordinary.consts = {Value::from_int(100),
                     gagp::payload::make_int_list_value({Value::from_int(1)})};
  ordinary.nodes = {
      {NodeKind::PROGRAM, 0, 0}, {NodeKind::BLOCK_CONS, 0, 0},
      {NodeKind::ASSIGN, 0, 0}, {NodeKind::CONST, 0, 0},
      {NodeKind::BLOCK_CONS, 0, 0}, {NodeKind::RETURN, 0, 0},
      {NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      {NodeKind::CONST, 1, 0}, {NodeKind::ADD, 0, 0},
      {NodeKind::BOUND_VAR, 0, 0}, {NodeKind::VAR, 0, 0},
      {NodeKind::BLOCK_NIL, 0, 0},
  };
  const ExecResult ordinary_out = run(ordinary);
  return check(!ordinary_out.is_error &&
                   exact_list(ordinary_out.value, gagp::ValueTag::IntList,
                              {Value::from_int(101)}),
               "BoundVar must remain distinct from an ordinary local of the same name");
}

bool expect_verify_code(const AstProgram& ast, VerifyCode code,
                        const std::string& intent) {
  const auto result = gagp::evo::verify_ast(ast, {});
  return check(!result.ok && result.diagnostic.code == code, intent);
}

bool test_errors_and_fuel() {
  gagp::payload::clear();
  const Value ints = gagp::payload::make_int_list_value({Value::from_int(1)});
  if (!expect_verify_code(
          return_program(map_list(0, ListTypeTag::String, leaf(NodeKind::CONST, 0),
                                  leaf(NodeKind::BOUND_VAR, 0)),
                         {gagp::payload::make_string_value("abc")}, {"c"}),
          VerifyCode::TypeMismatch, "MapList should reject a non-list source")) return false;
  if (!expect_verify_code(
          return_program(filter_list(0, leaf(NodeKind::CONST, 0),
                                     leaf(NodeKind::BOUND_VAR, 0)),
                         {ints}, {"x"}),
          VerifyCode::TypeMismatch, "FilterList predicate must be Bool")) return false;

  const Value zeros = gagp::payload::make_float_list_value(
      {Value::from_float(0.0), Value::from_float(1.0)});
  const ExecResult zero_div = run(return_program(
      map_list(0, ListTypeTag::Float, leaf(NodeKind::CONST, 0),
               binary(NodeKind::DIV, leaf(NodeKind::CONST, 1),
                      leaf(NodeKind::BOUND_VAR, 0))),
      {zeros, Value::from_float(1.0)}, {"x"}));
  if (!check(zero_div.is_error && zero_div.err.code == ErrCode::ZeroDiv,
             "MapList body should preserve ZeroDivisionError")) return false;

  const AstProgram map = return_program(
      map_list(0, ListTypeTag::Int, leaf(NodeKind::CONST, 0),
               leaf(NodeKind::BOUND_VAR, 0)),
      {gagp::payload::make_int_list_value(
          {Value::from_int(1), Value::from_int(2), Value::from_int(3)})},
      {"x"});
  const AstProgram filter = return_program(
      filter_list(0, leaf(NodeKind::CONST, 0),
                  binary(NodeKind::GT, leaf(NodeKind::BOUND_VAR, 0),
                         leaf(NodeKind::CONST, 1))),
      {gagp::payload::make_int_list_value(
           {Value::from_int(1), Value::from_int(2), Value::from_int(3)}),
       Value::from_int(1)}, {"x"});
  const Value list = gagp::payload::make_int_list_value(
      {Value::from_int(1), Value::from_int(2), Value::from_int(3)});
  const AstProgram linear = linear_program(
      list, Value::from_int(0), Value::from_int(0),
      binary(NodeKind::ADD, leaf(NodeKind::BOUND_VAR, 0),
             leaf(NodeKind::BOUND_VAR, 1)),
      leaf(NodeKind::BOUND_VAR, 0));
  for (const AstProgram* ast : {&map, &filter, &linear}) {
    const ExecResult result = run(*ast, 8);
    if (!check(result.is_error && result.err.code == ErrCode::Timeout,
               "structured expression should report Timeout on fuel exhaustion")) return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!test_map_and_filter_order_and_empty_tags()) return 1;
  if (!test_sources_are_lowered_once()) return 1;
  if (!test_linear_rec_cases_and_order()) return 1;
  if (!test_binder_capture_and_ordinary_local_isolation()) return 1;
  if (!test_errors_and_fuel()) return 1;
  std::cout << "gagp_test_structured_semantics: OK\n";
  return 0;
}
