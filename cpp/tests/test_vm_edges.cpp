#include <iostream>
#include <string>

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

Instr ins(const std::string& op) { return Instr{op, 0, 0, false, false}; }
Instr ins_a(const std::string& op, int a) { return Instr{op, a, 0, true, false}; }
Instr ins_ab(const std::string& op, int a, int b) { return Instr{op, a, b, true, true}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool expect_err(const VMResult& out, ErrCode code, const std::string& msg) {
  if (!check(out.is_error, msg + " (expected error)")) return false;
  if (!check(out.err.code == code, msg + " (error code mismatch)")) return false;
  return true;
}

bool test_jump_target_out_of_range() {
  BytecodeProgram p;
  p.code = {ins_a("JMP", 99)};
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Value, "jump target out of range");
}

bool test_stack_underflow() {
  BytecodeProgram p;
  p.code = {ins("ADD")};
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Value, "stack underflow on ADD");
}

bool test_builtin_arity_error() {
  BytecodeProgram p;
  p.consts = {Value::from_int(7)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_ab("CALL_BUILTIN", 0, 2),  // abs expects 1 arg
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Value, "builtin arity triggers stack underflow");
}

bool test_builtin_arity_type_error() {
  BytecodeProgram p;
  p.consts = {Value::from_int(7)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_ab("CALL_BUILTIN", 0, 0),  // abs with argc=0
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Type, "builtin arity type error");
}

bool test_timeout() {
  BytecodeProgram p;
  p.consts = {Value::from_int(1)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("JMP", 0),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 1);
  return expect_err(out, ErrCode::Timeout, "timeout");
}

}  // namespace

int main() {
  if (!test_jump_target_out_of_range()) return 1;
  if (!test_stack_underflow()) return 1;
  if (!test_builtin_arity_error()) return 1;
  if (!test_builtin_arity_type_error()) return 1;
  if (!test_timeout()) return 1;
  std::cout << "g3pvm_test_vm_edges: OK\n";
  return 0;
}
