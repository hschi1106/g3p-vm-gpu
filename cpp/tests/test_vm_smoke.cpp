#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::ErrCode;
using g3pvm::Instr;
using g3pvm::VMResult;
using g3pvm::Value;
using g3pvm::ValueTag;

Instr ins(const std::string& op) { return Instr{op, 0, 0, false, false}; }

Instr ins_a(const std::string& op, int a) { return Instr{op, a, 0, true, false}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool test_basic_arithmetic() {
  BytecodeProgram p;
  p.consts = {Value::from_int(1), Value::from_int(2)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins("ADD"),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 100);
  if (!check(!out.is_error, "basic arithmetic should return")) return false;
  if (!check(out.value.tag == ValueTag::Int, "basic arithmetic result should be int")) return false;
  if (!check(out.value.i == 3, "basic arithmetic result should be 3")) return false;
  return true;
}

bool test_loop_like_control_flow() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(0), Value::from_int(1), Value::from_int(5)};
  p.code = {
      ins_a("PUSH_CONST", 0),   // x = 0
      ins_a("STORE", 0),
      ins_a("LOAD", 0),         // loop_head:
      ins_a("PUSH_CONST", 2),   // x < 5 ?
      ins("LT"),
      ins_a("JMP_IF_FALSE", 12),
      ins_a("LOAD", 0),
      ins_a("PUSH_CONST", 1),
      ins("ADD"),               // x = x + 1
      ins_a("STORE", 0),
      ins_a("JMP", 2),
      ins_a("PUSH_CONST", 0),   // unreachable padding
      ins_a("LOAD", 0),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 1000);
  if (!check(!out.is_error, "control-flow program should return")) return false;
  if (!check(out.value.tag == ValueTag::Int, "control-flow result should be int")) return false;
  if (!check(out.value.i == 5, "control-flow result should be 5")) return false;
  return true;
}

}  // namespace

int main() {
  if (!test_basic_arithmetic()) return 1;
  if (!test_loop_like_control_flow()) return 1;
  std::cout << "g3pvm_test_vm_smoke: OK\n";
  return 0;
}
