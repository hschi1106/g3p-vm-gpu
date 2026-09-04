#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "gagp/core/bytecode.hpp"
#include "gagp/core/value_semantics.hpp"
#include "gagp/runtime/cpu/fitness_cpu.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace {

using gagp::Value;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool score_is(const Value& actual, const Value& expected, double wanted,
              const std::string& context) {
  double score = 0.0;
  const bool scored =
      gagp::vm_semantics::fitness_score_for_values(actual, expected, 5.0, score);
  return check(scored && std::fabs(score - wanted) < 1e-12, context);
}

gagp::BytecodeProgram returns(const Value& value) {
  gagp::BytecodeProgram program;
  program.consts = {value};
  program.code = {
      {gagp::Opcode::PushConst, 0, 0, true, false},
      {gagp::Opcode::Return, 0, 0, false, false},
  };
  return program;
}

}  // namespace

int main() {
  gagp::payload::clear();
  if (!score_is(Value::from_int(3), Value::from_int(3), 0.0, "numeric exact")) return 1;
  if (!score_is(Value::from_float(7.5), Value::from_int(5), -2.5,
                "cross-numeric difference")) return 1;
  if (!score_is(Value::from_int(20), Value::from_int(0), -5.0,
                "numeric penalty clamp")) return 1;
  if (!score_is(Value::from_bool(true), Value::from_int(1), -5.0,
                "non-numeric actual for numeric expected")) return 1;
  if (!score_is(Value::from_bool(true), Value::from_bool(true), 1.0, "bool exact")) return 1;
  if (!score_is(Value::from_bool(false), Value::from_bool(true), 0.0, "bool mismatch")) return 1;
  if (!score_is(Value::from_char('a'), Value::from_char('a'), 1.0, "char exact")) return 1;

  const Value string_a = gagp::payload::make_string_value("same");
  const Value string_b = gagp::payload::make_string_value("different");
  if (!score_is(string_a, string_a, 1.0, "string exact")) return 1;
  if (!score_is(string_b, string_a, 0.0, "string mismatch")) return 1;
  const Value ints = gagp::payload::make_int_list_value({Value::from_int(1)});
  const Value floats = gagp::payload::make_float_list_value({Value::from_float(1.0)});
  if (!score_is(ints, ints, 1.0, "typed-list exact")) return 1;
  if (!score_is(floats, ints, -5.0, "typed-list type mismatch")) return 1;

  const std::vector<gagp::CaseBindings> cases(2);
  const auto solved = gagp::eval_fitness_cpu(
      {returns(Value::from_bool(true))}, cases,
      {Value::from_bool(true), Value::from_bool(true)}, 100, 5.0);
  if (!check(solved.size() == 1 && solved[0] == 2.0,
             "two exact discrete cases define solved score 2")) return 1;

  gagp::BytecodeProgram runtime_error;
  runtime_error.code = {{gagp::Opcode::PushConst, 99, 0, true, false}};
  const auto failed = gagp::eval_fitness_cpu(
      {runtime_error}, {gagp::CaseBindings{}}, {Value::from_int(0)}, 100, 5.0);
  if (!check(failed.size() == 1 && failed[0] == -5.0,
             "runtime error must score one penalty")) return 1;

  std::cout << "gagp_test_fitness_contract: OK\n";
  return 0;
}
