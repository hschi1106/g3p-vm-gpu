#include <iostream>
#include <string>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/payload.hpp"
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

bool test_builtin_len_string_ok() {
  BytecodeProgram p;
  p.consts = {Value::from_string_hash_len(0x1234ULL, 7U)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_ab("CALL_BUILTIN", 4, 1),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "len(string) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Int, "len(string) should return int")) return false;
  return check(out.value.i == 7, "len(string) should return encoded length");
}

bool test_builtin_len_list_ok() {
  BytecodeProgram p;
  p.consts = {Value::from_list_hash_len(0x5678ULL, 3U)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_ab("CALL_BUILTIN", 4, 1),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "len(list) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Int, "len(list) should return int")) return false;
  return check(out.value.i == 3, "len(list) should return encoded length");
}

bool test_builtin_len_type_error() {
  BytecodeProgram p;
  p.consts = {Value::from_int(11)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_ab("CALL_BUILTIN", 4, 1),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Type, "len(int) should error");
}

bool test_compare_string_eq_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 2U),
      Value::from_string_hash_len(0x1111ULL, 2U),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins("EQ"),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "string EQ should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Bool, "string EQ should return bool")) return false;
  return check(out.value.b, "string EQ should be true for same payload");
}

bool test_builtin_concat_string_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 2U),
      Value::from_string_hash_len(0x2222ULL, 3U),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 5, 2),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "concat(string,string) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::String, "concat(string,string) should return string")) return false;
  return check(g3pvm::Value::container_len(out.value) == 5U, "concat(string,string) length mismatch");
}

bool test_builtin_concat_list_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_list_hash_len(0x3333ULL, 1U),
      Value::from_list_hash_len(0x4444ULL, 4U),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 5, 2),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "concat(list,list) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::List, "concat(list,list) should return list")) return false;
  return check(g3pvm::Value::container_len(out.value) == 5U, "concat(list,list) length mismatch");
}

bool test_builtin_concat_type_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 2U),
      Value::from_list_hash_len(0x4444ULL, 4U),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 5, 2),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Type, "concat(string,list) should error");
}

bool test_builtin_slice_string_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_int(2),
      Value::from_int(6),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_a("PUSH_CONST", 2),
      ins_ab("CALL_BUILTIN", 6, 3),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "slice(string,2,6) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::String, "slice(string,2,6) should return string")) return false;
  return check(g3pvm::Value::container_len(out.value) == 4U, "slice(string,2,6) length mismatch");
}

bool test_builtin_slice_list_negative_idx_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_list_hash_len(0x2222ULL, 8U),
      Value::from_int(-5),
      Value::from_int(-1),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_a("PUSH_CONST", 2),
      ins_ab("CALL_BUILTIN", 6, 3),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "slice(list,-5,-1) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::List, "slice(list,-5,-1) should return list")) return false;
  return check(g3pvm::Value::container_len(out.value) == 4U, "slice(list,-5,-1) length mismatch");
}

bool test_builtin_slice_type_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_float(1.5),
      Value::from_int(6),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_a("PUSH_CONST", 2),
      ins_ab("CALL_BUILTIN", 6, 3),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Type, "slice(string,float,int) should error");
}

bool test_builtin_index_string_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_int(3),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 7, 2),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "index(string,3) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::Int, "index(string,3) should return int token");
}

bool test_builtin_index_list_negative_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_list_hash_len(0x2222ULL, 8U),
      Value::from_int(-2),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 7, 2),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "index(list,-2) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::Int, "index(list,-2) should return int token");
}

bool test_builtin_index_oob_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 3U),
      Value::from_int(5),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 7, 2),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Value, "index(string,5) should error");
}

bool test_builtin_index_type_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_float(1.5),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 7, 2),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  return expect_err(out, ErrCode::Type, "index(string,float) should error");
}

bool test_builtin_index_string_payload_exact_ok() {
  g3pvm::payload::clear();
  BytecodeProgram p;
  p.consts = {
      g3pvm::payload::make_string_value("abcdef"),
      Value::from_int(2),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 7, 2),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "index(payload_string,2) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::String, "index(payload_string,2) should return string")) return false;
  return check(g3pvm::Value::container_len(out.value) == 1U, "index(payload_string,2) should return len=1 string");
}

bool test_builtin_index_list_payload_exact_ok() {
  g3pvm::payload::clear();
  BytecodeProgram p;
  std::vector<Value> elems = {Value::from_int(11), Value::from_int(22), Value::from_int(33)};
  p.consts = {
      g3pvm::payload::make_list_value(elems),
      Value::from_int(1),
  };
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins_a("PUSH_CONST", 1),
      ins_ab("CALL_BUILTIN", 7, 2),
      ins("RETURN"),
  };
  VMResult out = g3pvm::run_bytecode(p, {}, 10);
  if (!check(!out.is_error, "index(payload_list,1) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Int, "index(payload_list,1) should return int")) return false;
  return check(out.value.i == 22, "index(payload_list,1) element mismatch");
}

}  // namespace

int main() {
  if (!test_jump_target_out_of_range()) return 1;
  if (!test_stack_underflow()) return 1;
  if (!test_builtin_arity_error()) return 1;
  if (!test_builtin_arity_type_error()) return 1;
  if (!test_timeout()) return 1;
  if (!test_builtin_len_string_ok()) return 1;
  if (!test_builtin_len_list_ok()) return 1;
  if (!test_builtin_len_type_error()) return 1;
  if (!test_compare_string_eq_ok()) return 1;
  if (!test_builtin_concat_string_ok()) return 1;
  if (!test_builtin_concat_list_ok()) return 1;
  if (!test_builtin_concat_type_error()) return 1;
  if (!test_builtin_slice_string_ok()) return 1;
  if (!test_builtin_slice_list_negative_idx_ok()) return 1;
  if (!test_builtin_slice_type_error()) return 1;
  if (!test_builtin_index_string_ok()) return 1;
  if (!test_builtin_index_list_negative_ok()) return 1;
  if (!test_builtin_index_oob_error()) return 1;
  if (!test_builtin_index_type_error()) return 1;
  if (!test_builtin_index_string_payload_exact_ok()) return 1;
  if (!test_builtin_index_list_payload_exact_ok()) return 1;
  std::cout << "g3pvm_test_vm_edges: OK\n";
  return 0;
}
