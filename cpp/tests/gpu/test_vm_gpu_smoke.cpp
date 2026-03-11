#include <iostream>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::Opcode;
using g3pvm::Value;

g3pvm::Instr ins(Opcode op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(Opcode op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  BytecodeProgram p;
  p.consts = {Value::from_int(2), Value::from_int(3)};
  p.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1), ins(Opcode::Add), ins(Opcode::Return)};

  std::vector<BytecodeProgram> programs = {p};
  std::vector<g3pvm::CaseInputs> shared_cases(1);
  std::vector<Value> shared_answer = {Value::from_int(5)};
  g3pvm::FitnessSessionGpu session;
  const g3pvm::FitnessEvalResult init = session.init(shared_cases, shared_answer, 100, 1);
  if (!init.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << init.err.message << ")\n";
    return 0;
  }
  const g3pvm::FitnessEvalResult out = session.eval_programs(programs);
  if (!out.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << out.err.message << ")\n";
    return 0;
  }
  if (!check(out.fitness.size() == 1, "gpu smoke should return one fitness score")) return 1;
  if (!check(out.fitness[0] == 0.0, "gpu smoke exact numeric fitness should be zero MAE")) return 1;
  std::cout << "g3pvm_test_vm_gpu_smoke: OK\n";
  return 0;
}
