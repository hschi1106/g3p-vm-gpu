#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/bytecode_verify.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::AsgpDcSegment;
using g3pvm::AsgpDp1dSegment;
using g3pvm::AsgpDp2dSegment;
using g3pvm::BuiltinId;
using g3pvm::BytecodeProgram;
using g3pvm::ErrCode;
using g3pvm::ExecResult;
using g3pvm::Instr;
using g3pvm::Opcode;
using g3pvm::PhaseProgram;
using g3pvm::Value;

Instr ins(Opcode op) { return {op, 0, 0, false, false}; }
Instr ins_a(Opcode op, int a) { return {op, a, 0, true, false}; }
Instr ins_ab(Opcode op, int a, int b) { return {op, a, b, true, true}; }

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

PhaseProgram phase(std::vector<Value> consts, std::vector<Instr> code, int n_locals,
                   std::initializer_list<std::pair<const int, int>> binders) {
  PhaseProgram out;
  out.consts = std::move(consts);
  out.code = std::move(code);
  out.n_locals = n_locals;
  out.binder_locals.insert(binders.begin(), binders.end());
  return out;
}

ExecResult run_verified(const BytecodeProgram& program, int fuel = 20000) {
  const g3pvm::BytecodeVerifyResult verified = g3pvm::verify_bytecode(program);
  if (!verified) {
    return {true, Value::invalid(),
            {ErrCode::Value,
             std::string("test bytecode did not verify: ") +
                 g3pvm::bytecode_verify_code_name(verified.diagnostic.code) + " " +
                 verified.diagnostic.message}};
  }
  return g3pvm::execute_bytecode_cpu(program, {}, fuel);
}

AsgpDcSegment dc_segment(PhaseProgram solve, PhaseProgram divide,
                         PhaseProgram combine) {
  AsgpDcSegment segment;
  segment.solve_xs_name = 10;
  segment.solve_n_name = 11;
  segment.solve_lo_name = 12;
  segment.divide_n_name = 13;
  segment.combine_left_name = 14;
  segment.combine_right_name = 15;
  segment.solve = std::move(solve);
  segment.divide = std::move(divide);
  segment.combine = std::move(combine);
  return segment;
}

BytecodeProgram dc_program(Value source, AsgpDcSegment segment) {
  BytecodeProgram program;
  program.consts = {source};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::AsgpDc, 0),
                  ins(Opcode::Return)};
  program.asgp_dc_segments = {std::move(segment)};
  return program;
}

PhaseProgram dc_index_solve(bool as_string = false) {
  std::vector<Instr> code{
      ins_a(Opcode::Load, 0), ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(BuiltinId::Index), 2),
  };
  if (as_string) {
    code.push_back(ins_ab(Opcode::CallBuiltin,
                          static_cast<int>(BuiltinId::CharToString), 1));
  }
  code.push_back(ins(Opcode::Return));
  return phase({Value::from_int(0)}, std::move(code), 3,
               {{10, 0}, {11, 1}, {12, 2}});
}

PhaseProgram dc_divide(Value split) {
  return phase({split}, {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)}, 1,
               {{13, 0}});
}

PhaseProgram dc_combine(BuiltinId builtin) {
  return phase({},
               {ins_a(Opcode::Load, 0), ins_a(Opcode::Load, 1),
                builtin == BuiltinId::Abs
                    ? ins(Opcode::Add)
                    : ins_ab(Opcode::CallBuiltin, static_cast<int>(builtin), 2),
                ins(Opcode::Return)},
               2, {{14, 0}, {15, 1}});
}

bool test_dc_success_errors_visibility_and_fuel() {
  g3pvm::payload::clear();
  const Value ints = g3pvm::payload::make_int_list_value(
      {Value::from_int(1), Value::from_int(2), Value::from_int(3),
       Value::from_int(4)});
  const BytecodeProgram sum = dc_program(
      ints, dc_segment(dc_index_solve(), dc_divide(Value::from_int(999)),
                       dc_combine(BuiltinId::Abs)));
  const ExecResult sum_out = run_verified(sum);
  if (!check(!sum_out.is_error && sum_out.value.tag == g3pvm::ValueTag::Int &&
                 sum_out.value.i == 10,
             "ASGP-DC should sum IntList with a clamped split")) return false;

  const BytecodeProgram text = dc_program(
      g3pvm::payload::make_string_value("abc"),
      dc_segment(dc_index_solve(true), dc_divide(Value::from_int(1)),
                 dc_combine(BuiltinId::Concat)));
  const ExecResult text_out = run_verified(text);
  std::string exact;
  if (!check(!text_out.is_error &&
                 g3pvm::payload::lookup_string(text_out.value, &exact) && exact == "abc",
             "ASGP-DC should traverse String as a Char sequence")) return false;

  const ExecResult source_error = run_verified(dc_program(
      Value::from_int(1),
      dc_segment(dc_index_solve(), dc_divide(Value::from_int(1)),
                 dc_combine(BuiltinId::Abs))));
  if (!check(source_error.is_error && source_error.err.code == ErrCode::Type,
             "ASGP-DC should reject a non-sequence source")) return false;

  const ExecResult split_error = run_verified(dc_program(
      ints, dc_segment(dc_index_solve(), dc_divide(Value::from_float(1.5)),
                       dc_combine(BuiltinId::Abs))));
  if (!check(split_error.is_error && split_error.err.code == ErrCode::Type,
             "ASGP-DC divide phase must return Int")) return false;

  const PhaseProgram varying_solve = phase(
      {Value::from_int(0), Value::from_int(1), Value::from_bool(true)},
      {ins_a(Opcode::Load, 2), ins_a(Opcode::PushConst, 0), ins(Opcode::Eq),
       ins_a(Opcode::JmpIfFalse, 6), ins_a(Opcode::PushConst, 1),
       ins(Opcode::Return), ins_a(Opcode::PushConst, 2), ins(Opcode::Return)},
      3, {{10, 0}, {11, 1}, {12, 2}});
  const ExecResult recursive_type_error = run_verified(dc_program(
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
      dc_segment(varying_solve, dc_divide(Value::from_int(1)),
                 phase({}, {ins_a(Opcode::Load, 0), ins(Opcode::Return)}, 2,
                       {{14, 0}, {15, 1}}))));
  if (!check(recursive_type_error.is_error &&
                 recursive_type_error.err.code == ErrCode::Type,
             "ASGP-DC recursive results must have matching types")) return false;

  PhaseProgram hidden = phase({}, {ins_a(Opcode::Load, 3), ins(Opcode::Return)},
                              4, {{10, 0}, {11, 1}, {12, 2}});
  const ExecResult hidden_error = run_verified(dc_program(
      g3pvm::payload::make_int_list_value({Value::from_int(1)}),
      dc_segment(std::move(hidden), dc_divide(Value::from_int(1)),
                 dc_combine(BuiltinId::Abs))));
  if (!check(hidden_error.is_error && hidden_error.err.code == ErrCode::Name,
             "ASGP-DC phases must not see ordinary locals")) return false;

  const ExecResult timeout = run_verified(sum, 6);
  return check(timeout.is_error && timeout.err.code == ErrCode::Timeout,
               "ASGP-DC recursion should consume fuel");
}

AsgpDp1dSegment dp1_segment(bool two_dependencies, Value boundary,
                            bool hidden_solve = false, bool bool_transition = false) {
  AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = two_dependencies ? 20 : 5;
  segment.base_state = 0;
  segment.boundary_value = boundary;
  segment.dep_kind = -1;
  segment.dep_offsets = two_dependencies ? std::vector<int>{1, 2}
                                         : std::vector<int>{1};
  segment.solve_state_name = 20;
  segment.transition_state_name = 21;
  segment.transition_dep_names = two_dependencies ? std::vector<int>{22, 23}
                                                  : std::vector<int>{22};
  segment.solve = hidden_solve
      ? phase({}, {ins_a(Opcode::Load, 1), ins(Opcode::Return)}, 2, {{20, 0}})
      : phase({Value::from_int(1)},
              {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)}, 1, {{20, 0}});
  if (bool_transition) {
    segment.transition = phase({Value::from_bool(true)},
                               {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)},
                               two_dependencies ? 3 : 2,
                               two_dependencies
                                   ? std::initializer_list<std::pair<const int, int>>{
                                         {21, 0}, {22, 1}, {23, 2}}
                                   : std::initializer_list<std::pair<const int, int>>{
                                         {21, 0}, {22, 1}});
  } else if (two_dependencies) {
    segment.transition = phase(
        {}, {ins_a(Opcode::Load, 1), ins_a(Opcode::Load, 2), ins(Opcode::Add),
             ins(Opcode::Return)},
        3, {{21, 0}, {22, 1}, {23, 2}});
  } else {
    segment.transition = phase(
        {}, {ins_a(Opcode::Load, 1), ins_a(Opcode::Load, 0), ins(Opcode::Add),
             ins(Opcode::Return)},
        2, {{21, 0}, {22, 1}});
  }
  return segment;
}

BytecodeProgram dp1_program(Value state, AsgpDp1dSegment segment) {
  BytecodeProgram program;
  program.consts = {state};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::AsgpDp1d, 0),
                  ins(Opcode::Return)};
  program.asgp_dp1d_segments = {std::move(segment)};
  return program;
}

bool test_dp1_success_boundary_memo_errors_and_fuel() {
  const BytecodeProgram recurrence = dp1_program(
      Value::from_int(4), dp1_segment(false, Value::from_int(0)));
  const ExecResult recurrence_out = run_verified(recurrence);
  if (!check(!recurrence_out.is_error && recurrence_out.value.i == 11,
             "ASGP-DP1D backward recurrence mismatch")) return false;

  const ExecResult boundary = run_verified(dp1_program(
      Value::from_int(-1), dp1_segment(false, Value::from_int(99))));
  if (!check(!boundary.is_error && boundary.value.i == 99,
             "ASGP-DP1D should return its boundary value")) return false;

  const BytecodeProgram memo = dp1_program(
      Value::from_int(20), dp1_segment(true, Value::from_int(0)));
  const ExecResult memo_out = run_verified(memo, 500);
  if (!check(!memo_out.is_error && memo_out.value.i == 10946,
             "ASGP-DP1D should memoize overlapping dependencies")) return false;

  const ExecResult state_type = run_verified(dp1_program(
      Value::from_float(1.5), dp1_segment(false, Value::from_int(0))));
  if (!check(state_type.is_error && state_type.err.code == ErrCode::Type,
             "ASGP-DP1D state must be Int")) return false;

  const ExecResult result_type = run_verified(dp1_program(
      Value::from_int(1),
      dp1_segment(false, Value::from_int(0), false, true)));
  if (!check(result_type.is_error && result_type.err.code == ErrCode::Type,
             "ASGP-DP1D transition type must match dependency type")) return false;

  AsgpDp1dSegment mixed_dependencies = dp1_segment(true, Value::from_int(0));
  mixed_dependencies.solve = phase(
      {Value::from_bool(true)},
      {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)}, 1, {{20, 0}});
  const ExecResult dependency_type = run_verified(
      dp1_program(Value::from_int(1), std::move(mixed_dependencies)));
  if (!check(dependency_type.is_error && dependency_type.err.code == ErrCode::Type,
             "ASGP-DP1D dependencies must have matching types")) return false;

  const ExecResult hidden = run_verified(dp1_program(
      Value::from_int(0),
      dp1_segment(false, Value::from_int(0), true, false)));
  if (!check(hidden.is_error && hidden.err.code == ErrCode::Name,
             "ASGP-DP1D phases must not see ordinary locals")) return false;

  const ExecResult timeout = run_verified(memo, 10);
  return check(timeout.is_error && timeout.err.code == ErrCode::Timeout,
               "ASGP-DP1D recursion should consume fuel");
}

AsgpDp2dSegment dp2_segment(Value boundary, bool hidden_solve = false,
                            bool bool_transition = false) {
  AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 5;
  segment.j_lo = 0;
  segment.j_hi = 5;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = boundary;
  segment.dep_kind = 4;
  segment.solve_i_name = 30;
  segment.solve_j_name = 31;
  segment.transition_i_name = 32;
  segment.transition_j_name = 33;
  segment.transition_dep_names = {34, 35, 36};
  segment.solve = hidden_solve
      ? phase({}, {ins_a(Opcode::Load, 2), ins(Opcode::Return)}, 3,
              {{30, 0}, {31, 1}})
      : phase({Value::from_int(1)},
              {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)}, 2,
              {{30, 0}, {31, 1}});
  if (bool_transition) {
    segment.transition = phase(
        {Value::from_bool(true)},
        {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)}, 5,
        {{32, 0}, {33, 1}, {34, 2}, {35, 3}, {36, 4}});
  } else {
    segment.transition = phase(
        {}, {ins_a(Opcode::Load, 2), ins_a(Opcode::Load, 3), ins(Opcode::Add),
             ins_a(Opcode::Load, 4), ins(Opcode::Add), ins(Opcode::Return)},
        5, {{32, 0}, {33, 1}, {34, 2}, {35, 3}, {36, 4}});
  }
  return segment;
}

BytecodeProgram dp2_program(Value i, Value j, AsgpDp2dSegment segment) {
  BytecodeProgram program;
  program.consts = {i, j};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1),
                  ins_a(Opcode::AsgpDp2d, 0), ins(Opcode::Return)};
  program.asgp_dp2d_segments = {std::move(segment)};
  return program;
}

bool test_dp2_success_boundary_memo_errors_and_fuel() {
  const BytecodeProgram recurrence = dp2_program(
      Value::from_int(2), Value::from_int(2), dp2_segment(Value::from_int(0)));
  const ExecResult recurrence_out = run_verified(recurrence);
  if (!check(!recurrence_out.is_error && recurrence_out.value.i == 13,
             "ASGP-DP2D neighborhood recurrence mismatch")) return false;

  const ExecResult boundary = run_verified(dp2_program(
      Value::from_int(-1), Value::from_int(0), dp2_segment(Value::from_int(99))));
  if (!check(!boundary.is_error && boundary.value.i == 99,
             "ASGP-DP2D should return its boundary value")) return false;

  const BytecodeProgram memo = dp2_program(
      Value::from_int(5), Value::from_int(5), dp2_segment(Value::from_int(0)));
  const ExecResult memo_out = run_verified(memo, 1200);
  if (!check(!memo_out.is_error && memo_out.value.i == 1683,
             "ASGP-DP2D should memoize overlapping dependencies")) return false;

  const ExecResult state_type = run_verified(dp2_program(
      Value::from_float(1.5), Value::from_int(0), dp2_segment(Value::from_int(0))));
  if (!check(state_type.is_error && state_type.err.code == ErrCode::Type,
             "ASGP-DP2D states must be Int")) return false;

  const ExecResult result_type = run_verified(dp2_program(
      Value::from_int(1), Value::from_int(1),
      dp2_segment(Value::from_int(0), false, true)));
  if (!check(result_type.is_error && result_type.err.code == ErrCode::Type,
             "ASGP-DP2D transition type must match dependency type")) return false;

  AsgpDp2dSegment mixed_dependencies = dp2_segment(Value::from_int(0));
  mixed_dependencies.solve = phase(
      {Value::from_bool(true)},
      {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)}, 2,
      {{30, 0}, {31, 1}});
  const ExecResult dependency_type = run_verified(dp2_program(
      Value::from_int(1), Value::from_int(1), std::move(mixed_dependencies)));
  if (!check(dependency_type.is_error && dependency_type.err.code == ErrCode::Type,
             "ASGP-DP2D dependencies must have matching types")) return false;

  const ExecResult hidden = run_verified(dp2_program(
      Value::from_int(0), Value::from_int(0),
      dp2_segment(Value::from_int(0), true, false)));
  if (!check(hidden.is_error && hidden.err.code == ErrCode::Name,
             "ASGP-DP2D phases must not see ordinary locals")) return false;

  const ExecResult timeout = run_verified(memo, 10);
  return check(timeout.is_error && timeout.err.code == ErrCode::Timeout,
               "ASGP-DP2D recursion should consume fuel");
}

}  // namespace

int main() {
  if (!test_dc_success_errors_visibility_and_fuel()) return 1;
  if (!test_dp1_success_boundary_memo_errors_and_fuel()) return 1;
  if (!test_dp2_success_boundary_memo_errors_and_fuel()) return 1;
  std::cout << "g3pvm_test_asgp_semantics: OK\n";
  return 0;
}
