#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"
#include "g3pvm/vm_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::InputCase;
using g3pvm::LocalBinding;
using g3pvm::Value;

g3pvm::Instr ins(const std::string& op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(const std::string& op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }

BytecodeProgram make_add_one_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(1)};
  p.code = {ins_a("LOAD", 0), ins_a("PUSH_CONST", 0), ins("ADD"), ins("RETURN")};
  return p;
}

BytecodeProgram make_type_error_program() {
  BytecodeProgram p;
  p.consts = {Value::from_bool(true)};
  p.code = {ins_a("PUSH_CONST", 0), ins("NEG")};
  return p;
}

BytecodeProgram make_timeout_program() {
  BytecodeProgram p;
  p.code = {ins_a("JMP", 0)};
  return p;
}

}  // namespace

int main() {
  std::vector<BytecodeProgram> programs;
  programs.push_back(make_add_one_program());
  programs.push_back(make_type_error_program());
  programs.push_back(make_timeout_program());

  std::vector<InputCase> shared_cases;
  std::vector<Value> shared_answer;
  for (int i = 0; i < 64; ++i) {
    shared_cases.push_back(InputCase{LocalBinding{0, Value::from_int(i)}});
    shared_answer.push_back(Value::from_int(i + 1));
  }

  const std::vector<int> cpu_fit =
      g3pvm::run_bytecode_cpu_multi_fitness_shared_cases(programs, shared_cases, shared_answer, 64);
  const g3pvm::GPUFitnessEvalResult gpu_fit = g3pvm::run_bytecode_gpu_multi_fitness_shared_cases_debug(
      programs, shared_cases, shared_answer, 64, 128);

  if (!gpu_fit.ok) {
    if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
      return 0;
    }
    std::cerr << "FAIL: gpu fitness run failed: " << gpu_fit.err.message << "\n";
    return 1;
  }

  if (cpu_fit != gpu_fit.fitness) {
    std::cerr << "FAIL: cpu/gpu fitness mismatch\n";
    return 1;
  }

  std::cout << "g3pvm_test_fitness_cpu_gpu_parity: OK\n";
  return 0;
}
