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

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

BytecodeProgram make_add_one_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(1)};
  p.code = {
      ins_a("LOAD", 0),
      ins_a("PUSH_CONST", 0),
      ins("ADD"),
      ins("RETURN"),
  };
  return p;
}

BytecodeProgram make_const_program(int v) {
  BytecodeProgram p;
  p.consts = {Value::from_int(v)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins("RETURN"),
  };
  return p;
}

BytecodeProgram make_type_error_program() {
  BytecodeProgram p;
  p.consts = {Value::from_bool(true)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins("NEG"),
  };
  return p;
}

}  // namespace

int main() {
  std::vector<BytecodeProgram> programs = {
      make_add_one_program(),
      make_const_program(7),
      make_type_error_program(),
  };

  std::vector<InputCase> shared_cases = {
      InputCase{LocalBinding{0, Value::from_int(10)}},
      InputCase{LocalBinding{0, Value::from_int(20)}},
      InputCase{LocalBinding{0, Value::from_int(30)}},
  };

  std::vector<Value> shared_answer = {
      Value::from_int(11),  // p0 correct, p1 wrong
      Value::from_int(7),   // p0 wrong, p1 correct
      Value::from_int(31),  // p0 correct, p1 wrong
  };

  const std::vector<int> cpu_fitness =
      g3pvm::run_bytecode_cpu_multi_fitness_shared_cases(programs, shared_cases, shared_answer, 100);
  if (!check(cpu_fitness.size() == 3, "cpu fitness size mismatch")) return 1;
  if (!check(cpu_fitness[0] == 2, "cpu fitness program0 mismatch")) return 1;
  if (!check(cpu_fitness[1] == 1, "cpu fitness program1 mismatch")) return 1;
  if (!check(cpu_fitness[2] == -30, "cpu fitness program2 mismatch")) return 1;

  const std::vector<int> gpu_fitness = g3pvm::run_bytecode_gpu_multi_fitness_shared_cases(
      programs, shared_cases, shared_answer, 100, 128);
  if (gpu_fitness.empty()) {
    std::cout << "g3pvm_test_vm_gpu_fitness: SKIP (cuda device unavailable)\n";
    return 0;
  }

  if (!check(gpu_fitness.size() == cpu_fitness.size(), "gpu fitness size mismatch")) return 1;
  for (std::size_t i = 0; i < cpu_fitness.size(); ++i) {
    if (!check(gpu_fitness[i] == cpu_fitness[i], "cpu/gpu fitness mismatch")) return 1;
  }

  std::cout << "g3pvm_test_vm_gpu_fitness: OK\n";
  return 0;
}
