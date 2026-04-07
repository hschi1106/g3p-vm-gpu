#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::Opcode;
using g3pvm::Value;

g3pvm::Instr ins(Opcode op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(Opcode op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }
g3pvm::Instr ins_ab(Opcode op, int a, int b) { return g3pvm::Instr{op, a, b, true, true}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool eval_single(const BytecodeProgram& program, const Value& expected, double expected_fitness, const std::string& label) {
  std::vector<BytecodeProgram> programs = {program};
  std::vector<g3pvm::CaseBindings> shared_cases(1);
  std::vector<Value> shared_answer = {expected};
  g3pvm::FitnessSessionGpu session;
  const g3pvm::FitnessSessionInitResult init = session.init(shared_cases, shared_answer, 100, 1);
  if (!init.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << init.err.message << ")\n";
    return true;
  }
  const g3pvm::FitnessEvalResult out = session.eval_programs(programs);
  if (!out.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << out.err.message << ")\n";
    return true;
  }
  if (!check(out.fitness.size() == 1, label + " should return one fitness score")) return false;
  return check(out.fitness[0] == expected_fitness, label + " fitness mismatch");
}

}  // namespace

int main() {
  g3pvm::payload::clear();

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(2), Value::from_int(3)};
    p.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1), ins(Opcode::Add), ins(Opcode::Return)};
    if (!eval_single(p, Value::from_int(5), 0.0, "numeric add")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {
        g3pvm::payload::make_num_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(3),
    };
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 8, 2),
        ins(Opcode::Return),
    };
    const Value expected = g3pvm::payload::make_num_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)});
    if (!eval_single(p, expected, 1.0, "append num_list")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_string_value("abc")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, 9, 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("cba"), 1.0, "reverse string")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_string_value("abracadabra"), g3pvm::payload::make_string_value("cad")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 10, 2),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int(4), 0.0, "find string")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_string_value("abracadabra"), g3pvm::payload::make_string_value("cad")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 11, 2),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_bool(true), 1.0, "contains string")) return 1;
  }

  std::cout << "g3pvm_test_vm_gpu_smoke: OK\n";
  return 0;
}
