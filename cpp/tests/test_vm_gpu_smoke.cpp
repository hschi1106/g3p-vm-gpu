#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::InputCase;
using g3pvm::VMResult;
using g3pvm::Value;
using g3pvm::ValueTag;

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
  std::vector<InputCase> shared_cases(1);
  std::vector<std::vector<VMResult>> out =
      g3pvm::run_bytecode_gpu_multi_batch(programs, shared_cases, 100, 1);
  if (out.size() != 1 || out[0].size() != 1) {
    std::cerr << "FAIL: gpu smoke should return one program with one result\n";
    return 1;
  }
  VMResult one = out[0][0];
  if (one.is_error) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << one.err.message << ")\n";
    return 0;
  }
  if (!check(one.value.tag == ValueTag::Int, "gpu smoke result should be int")) return 1;
  if (!check(one.value.i == 5, "gpu smoke result should be 5")) return 1;
  std::cout << "g3pvm_test_vm_gpu_smoke: OK\n";
  return 0;
}
