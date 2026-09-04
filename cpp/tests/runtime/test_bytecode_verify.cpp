#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "gagp/core/builtin.hpp"
#include "gagp/core/bytecode_verify.hpp"
#include "gagp/evolution/compiler.hpp"
#include "gagp/evolution/genome_generation.hpp"

namespace {

using namespace gagp;

Instr ins(Opcode op) { return Instr{op, 0, 0, false, false}; }
Instr ins_a(Opcode op, int a) { return Instr{op, a, 0, true, false}; }
Instr ins_ab(Opcode op, int a, int b) { return Instr{op, a, b, true, true}; }

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool expect_code(const BytecodeProgram& program, BytecodeVerifyCode code,
                 const std::string& label,
                 const BytecodeVerifyOptions& options = BytecodeVerifyOptions{}) {
  const BytecodeVerifyResult result = verify_bytecode(program, options);
  return check(!result, label + " should fail") &&
         check(result.diagnostic.code == code,
               label + " expected " + bytecode_verify_code_name(code) + " but got " +
                   bytecode_verify_code_name(result.diagnostic.code) + " at " +
                   result.diagnostic.path + ": " + result.diagnostic.message);
}

BytecodeProgram constant_program() {
  BytecodeProgram program;
  program.consts = {Value::from_int(7)};
  program.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return program;
}

PhaseProgram phase_load(const std::vector<int>& names, int selected) {
  PhaseProgram phase;
  phase.n_locals = static_cast<int>(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    phase.binder_locals[names[i]] = static_cast<int>(i);
  }
  phase.code = {ins_a(Opcode::Load, selected)};
  return phase;
}

BytecodeProgram dc_program() {
  BytecodeProgram program;
  program.consts = {Value::from_int_list_hash_len(1, 2)};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::AsgpDc, 0), ins(Opcode::Return)};
  AsgpDcSegment segment;
  segment.solve_xs_name = 0;
  segment.solve_n_name = 1;
  segment.solve_lo_name = 2;
  segment.divide_n_name = 3;
  segment.combine_left_name = 4;
  segment.combine_right_name = 5;
  segment.solve = phase_load({0, 1, 2}, 1);
  segment.divide = phase_load({3}, 0);
  segment.combine = phase_load({4, 5}, 0);
  program.asgp_dc_segments.push_back(segment);
  return program;
}

BytecodeProgram dp1_program() {
  BytecodeProgram program;
  program.consts = {Value::from_int(3)};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::AsgpDp1d, 0), ins(Opcode::Return)};
  AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 3;
  segment.base_state = 0;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = -1;
  segment.dep_offsets = {1};
  segment.solve_state_name = 0;
  segment.transition_state_name = 1;
  segment.transition_dep_names = {2};
  segment.solve = phase_load({0}, 0);
  segment.transition = phase_load({1, 2}, 1);
  program.asgp_dp1d_segments.push_back(segment);
  return program;
}

BytecodeProgram dp2_program() {
  BytecodeProgram program;
  program.consts = {Value::from_int(2), Value::from_int(3)};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1),
                  ins_a(Opcode::AsgpDp2d, 0), ins(Opcode::Return)};
  AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = 2;
  segment.solve_i_name = 0;
  segment.solve_j_name = 1;
  segment.transition_i_name = 2;
  segment.transition_j_name = 3;
  segment.transition_dep_names = {4};
  segment.solve = phase_load({0, 1}, 0);
  segment.transition = phase_load({2, 3, 4}, 2);
  program.asgp_dp2d_segments.push_back(segment);
  return program;
}

}  // namespace

int main() {
  using namespace gagp;

  BytecodeProgram program = constant_program();
  BytecodeVerifyResult result = verify_bytecode(program);
  if (!check(result.ok && result.verified.has_reachable_return &&
                 result.verified.max_stack_depth == 1,
             "simple constant bytecode verifies")) return 1;

  program.code[0].has_a = false;
  if (!expect_code(program, BytecodeVerifyCode::MissingOperand, "missing operand")) return 1;

  program = constant_program();
  program.code[0].a = 2;
  if (!expect_code(program, BytecodeVerifyCode::InvalidConstantIndex, "constant range")) return 1;

  program = constant_program();
  program.code[0].op = static_cast<Opcode>(255);
  if (!expect_code(program, BytecodeVerifyCode::UnknownOpcode, "unknown opcode")) return 1;

  program = constant_program();
  program.n_locals = 1;
  program.code[0] = ins_a(Opcode::Load, 1);
  if (!expect_code(program, BytecodeVerifyCode::InvalidLocalIndex, "local range")) return 1;

  program = constant_program();
  program.n_locals = -1;
  if (!expect_code(program, BytecodeVerifyCode::InvalidLocalCount, "negative local count")) return 1;

  program = constant_program();
  program.consts[0] = Value::invalid();
  if (!expect_code(program, BytecodeVerifyCode::InvalidConstant, "private constant tag")) return 1;

  program = constant_program();
  program.consts[0] = Value::from_char(0xD800);
  if (!expect_code(program, BytecodeVerifyCode::InvalidConstant, "invalid Char scalar")) return 1;

  program = constant_program();
  program.n_locals = 1;
  program.var2idx["x"] = 1;
  if (!expect_code(program, BytecodeVerifyCode::InvalidVarMapping, "variable mapping range")) return 1;

  program = constant_program();
  program.code = {ins_a(Opcode::Jmp, 3)};
  if (!expect_code(program, BytecodeVerifyCode::InvalidJumpTarget, "jump range")) return 1;

  program = constant_program();
  program.code = {ins(Opcode::Add), ins(Opcode::Return)};
  if (!expect_code(program, BytecodeVerifyCode::StackUnderflow, "stack underflow")) return 1;

  program = constant_program();
  program.code = {ins_ab(Opcode::CallBuiltin, 999, 1), ins(Opcode::Return)};
  if (!expect_code(program, BytecodeVerifyCode::InvalidBuiltinId, "builtin id")) return 1;

  program.code = {ins_ab(Opcode::CallBuiltin, static_cast<int>(BuiltinId::Abs), 2),
                  ins(Opcode::Return)};
  if (!expect_code(program, BytecodeVerifyCode::InvalidBuiltinArity, "builtin arity")) return 1;

  program.consts = {Value::from_bool(true), Value::from_int(1)};
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::JmpIfFalse, 4),
                  ins_a(Opcode::PushConst, 1), ins_a(Opcode::Jmp, 5),
                  ins_a(Opcode::Jmp, 5), ins_a(Opcode::Jmp, 5)};
  if (!expect_code(program, BytecodeVerifyCode::StackJoinMismatch, "join depth")) return 1;

  program.consts.push_back(Value::from_float(1.0));
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::JmpIfFalse, 4),
                  ins_a(Opcode::PushConst, 1), ins_a(Opcode::Jmp, 5),
                  ins_a(Opcode::PushConst, 2), ins_a(Opcode::Jmp, 5)};
  if (!expect_code(program, BytecodeVerifyCode::StackJoinMismatch, "join type")) return 1;

  program = constant_program();
  program.code = {ins_a(Opcode::PushConst, 0)};
  if (!expect_code(program, BytecodeVerifyCode::InvalidFallthrough, "main fallthrough")) return 1;

  program.consts = {Value::from_bool(true)};
  program.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Neg)};
  if (!check(verify_bytecode(program).ok, "guaranteed runtime TypeError remains valid bytecode")) return 1;

  program.consts.clear();
  program.code = {ins_a(Opcode::Jmp, 0)};
  if (!check(verify_bytecode(program).ok, "timeout-only loop remains valid bytecode")) return 1;

  program = dc_program();
  if (!check(verify_bytecode(program).ok, "valid ASGP-DC segments verify")) return 1;
  BytecodeVerifyOptions public_only;
  public_only.allow_private_opcodes = false;
  if (!expect_code(program, BytecodeVerifyCode::InvalidPrivateOpcode,
                   "private opcode profile", public_only)) return 1;

  program = constant_program();
  program.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::AsgpDc, 0), ins(Opcode::Return)};
  if (!expect_code(program, BytecodeVerifyCode::InvalidSegmentIndex, "segment index")) return 1;

  program = dp1_program();
  result = verify_bytecode(program);
  if (!check(result.ok && result.verified.phase_program_count == 2,
             "valid ASGP-DP1D segments verify")) return 1;
  program.asgp_dp1d_segments[0].dep_offsets[0] = 0;
  if (!expect_code(program, BytecodeVerifyCode::InvalidSegmentMetadata,
                   "DP1 dependency metadata")) return 1;

  program = dp1_program();
  program.asgp_dp1d_segments[0].solve.binder_locals.clear();
  if (!expect_code(program, BytecodeVerifyCode::InvalidBinderLocal,
                   "phase binder mapping")) return 1;

  program = dp1_program();
  program.asgp_dp1d_segments[0].solve.code = {ins_a(Opcode::AsgpDp1d, 0)};
  if (!expect_code(program, BytecodeVerifyCode::InvalidPrivateOpcode,
                   "nested ASGP phase call")) return 1;

  program = dp2_program();
  if (!check(verify_bytecode(program).ok, "valid ASGP-DP2D segments verify")) return 1;
  program.asgp_dp2d_segments[0].transition_dep_names.push_back(5);
  if (!expect_code(program, BytecodeVerifyCode::InvalidSegmentMetadata,
                   "DP2 dependency metadata")) return 1;

  program = constant_program();
  BytecodeVerifyOptions limited;
  limited.max_instructions_per_code = 1;
  if (!expect_code(program, BytecodeVerifyCode::ResourceLimit, "instruction limit", limited)) return 1;

  const gagp::evo::Limits limits;
  const std::vector<gagp::evo::InputSpec> inputs{{"x", gagp::evo::RType::Int}};
  for (std::uint64_t seed = 0; seed < 128; ++seed) {
    const gagp::evo::ProgramGenome genome =
        gagp::evo::generate_random_genome(seed, limits, inputs);
    const BytecodeProgram compiled = gagp::evo::compile_for_eval(genome, {"x"});
    result = verify_bytecode(compiled);
    if (!check(result.ok, "compiler output verifies for seed " + std::to_string(seed) +
                              " at " + result.diagnostic.path + ": " +
                              result.diagnostic.message)) return 1;
  }

  return 0;
}
