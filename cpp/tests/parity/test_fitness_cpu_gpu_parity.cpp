#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/exec_cpu.hpp"
#include "g3pvm/runtime/fitness_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::CaseInputs;
using g3pvm::InputBinding;
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

BytecodeProgram make_return_const_program(int v) {
  BytecodeProgram p;
  p.consts = {Value::from_int(v)};
  p.code = {ins_a("PUSH_CONST", 0), ins("RETURN")};
  return p;
}

BytecodeProgram make_return_string_program() {
  BytecodeProgram p;
  p.consts = {Value::from_string_hash_len(0x1234ULL, 3)};
  p.code = {ins_a("PUSH_CONST", 0), ins("RETURN")};
  return p;
}

}  // namespace

int main() {
  std::vector<BytecodeProgram> programs;
  programs.push_back(make_add_one_program());
  programs.push_back(make_return_const_program(7));
  programs.push_back(make_return_string_program());
  programs.push_back(make_type_error_program());
  programs.push_back(make_timeout_program());

  std::vector<CaseInputs> shared_cases;
  std::vector<Value> shared_answer;
  for (int i = 0; i < 64; ++i) {
    shared_cases.push_back(CaseInputs{InputBinding{0, Value::from_int(i)}});
    shared_answer.push_back(Value::from_int(i + 1));
  }

  const std::vector<double> cpu_fit =
      g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, 64);
  const g3pvm::FitnessEvalResult gpu_fit = g3pvm::eval_fitness_gpu_profiled(
      programs, shared_cases, shared_answer, 64, 128);

  if (!gpu_fit.ok) {
    if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
      return 0;
    }
    std::cerr << "FAIL: gpu fitness run failed: " << gpu_fit.err.message << "\n";
    return 1;
  }

  if (cpu_fit.size() != gpu_fit.fitness.size()) {
    std::cerr << "FAIL: cpu/gpu fitness size mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
    if (std::fabs(cpu_fit[i] - gpu_fit.fitness[i]) > 1e-9) {
      std::cerr << "FAIL: cpu/gpu fitness mismatch at " << i << "\n";
      return 1;
    }
  }
  if (std::fabs(cpu_fit[0] - 0.0) > 1e-9) {
    std::cerr << "FAIL: exact numeric program should score 0 MAE\n";
    return 1;
  }
  if (std::fabs(cpu_fit[1] + 1674.0) > 1e-9) {
    std::cerr << "FAIL: constant numeric program should accumulate negative MAE\n";
    return 1;
  }
  if (std::fabs(cpu_fit[2] + 64.0) > 1e-9) {
    std::cerr << "FAIL: non-numeric actual against numeric expected should accumulate fixed penalty\n";
    return 1;
  }
  if (std::fabs(cpu_fit[3] - 0.0) > 1e-9 || std::fabs(cpu_fit[4] - 0.0) > 1e-9) {
    std::cerr << "FAIL: erroring programs should score 0\n";
    return 1;
  }

  std::cout << "g3pvm_test_fitness_cpu_gpu_parity: OK\n";
  return 0;
}
