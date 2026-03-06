#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/fitness_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::Value;

g3pvm::Instr ins(const std::string& op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(const std::string& op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }

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
  p.code = {ins_a("PUSH_CONST", 0), ins_a("PUSH_CONST", 1), ins("ADD"), ins("RETURN")};

  std::vector<BytecodeProgram> programs = {p};
  std::vector<g3pvm::CaseInputs> shared_cases(1);
  std::vector<Value> shared_answer = {Value::from_int(5)};
  const g3pvm::FitnessEvalResult out =
      g3pvm::eval_fitness_gpu_profiled(programs, shared_cases, shared_answer, 100, 1);
  if (!out.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << out.err.message << ")\n";
    return 0;
  }
  if (!check(out.fitness.size() == 1, "gpu smoke should return one fitness score")) return 1;
  if (!check(out.fitness[0] == 0.0, "gpu smoke exact numeric fitness should be zero MAE")) return 1;
  std::cout << "g3pvm_test_vm_gpu_smoke: OK\n";
  return 0;
}
