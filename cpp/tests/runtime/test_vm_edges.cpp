#include <iostream>
#include <stdexcept>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/runtime/payload/payload.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::ErrCode;
using g3pvm::Instr;
using g3pvm::ExecResult;
using g3pvm::Opcode;
using g3pvm::Value;

Instr ins(Opcode op) { return Instr{op, 0, 0, false, false}; }
Instr ins_a(Opcode op, int a) { return Instr{op, a, 0, true, false}; }
Instr ins_ab(Opcode op, int a, int b) { return Instr{op, a, b, true, true}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool expect_err(const ExecResult& out, ErrCode code, const std::string& msg) {
  if (!check(out.is_error, msg + " (expected error)")) return false;
  if (!check(out.err.code == code, msg + " (error code mismatch)")) return false;
  return true;
}

bool test_jump_target_out_of_range() {
  BytecodeProgram p;
  p.code = {ins_a(Opcode::Jmp, 99)};
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Value, "jump target out of range");
}

bool test_stack_underflow() {
  BytecodeProgram p;
  p.code = {ins(Opcode::Add)};
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Value, "stack underflow on ADD");
}

bool test_builtin_arity_error() {
  BytecodeProgram p;
  p.consts = {Value::from_int(7)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, 0, 2),  // abs expects 1 arg
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Value, "builtin arity triggers stack underflow");
}

bool test_builtin_arity_type_error() {
  BytecodeProgram p;
  p.consts = {Value::from_int(7)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, 0, 0),  // abs with argc=0
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Type, "builtin arity type error");
}

bool test_timeout() {
  BytecodeProgram p;
  p.consts = {Value::from_int(1)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::Jmp, 0),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 1);
  return expect_err(out, ErrCode::Timeout, "timeout");
}

bool test_builtin_len_string_ok() {
  BytecodeProgram p;
  p.consts = {Value::from_string_hash_len(0x1234ULL, 7U)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, 4, 1),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "len(string) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Int, "len(string) should return int")) return false;
  return check(out.value.i == 7, "len(string) should return encoded length");
}

bool test_builtin_len_list_ok() {
  BytecodeProgram p;
  p.consts = {Value::from_int_list_hash_len(0x5678ULL, 3U)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, 4, 1),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "len(list) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Int, "len(list) should return int")) return false;
  return check(out.value.i == 3, "len(list) should return encoded length");
}

bool test_builtin_len_type_error() {
  BytecodeProgram p;
  p.consts = {Value::from_int(11)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, 4, 1),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Type, "len(int) should error");
}

bool test_mixed_int_float_numeric_ops_error() {
  {
    BytecodeProgram p;
    p.consts = {Value::from_int(1), Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins(Opcode::Add),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!expect_err(out, ErrCode::Type, "mixed int/float ADD should error")) return false;
  }
  {
    BytecodeProgram p;
    p.consts = {Value::from_int(1), Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins(Opcode::Lt),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!expect_err(out, ErrCode::Type, "mixed int/float LT should error")) return false;
  }
  {
    BytecodeProgram p;
    p.consts = {Value::from_int(1), Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 1, 2),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!expect_err(out, ErrCode::Type, "mixed int/float min should error")) return false;
  }
  {
    BytecodeProgram p;
    p.consts = {Value::from_int(1), Value::from_int(0), Value::from_float(2.0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_a(Opcode::PushConst, 2),
        ins_ab(Opcode::CallBuiltin, 3, 3),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!expect_err(out, ErrCode::Type, "mixed int/float clip should error")) return false;
  }
  return true;
}

bool test_builtin_clip_reversed_bounds_returns_hi() {
  {
    BytecodeProgram p;
    p.consts = {Value::from_int(5), Value::from_int(10), Value::from_int(0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_a(Opcode::PushConst, 2),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Clip), 3),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!check(!out.is_error, "clip(int, reversed bounds) should not error")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int, "clip(int, reversed bounds) should return int")) return false;
    if (!check(out.value.i == 0, "clip(int, reversed bounds) should return hi")) return false;
  }
  {
    BytecodeProgram p;
    p.consts = {Value::from_float(5.0), Value::from_float(10.0), Value::from_float(0.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_a(Opcode::PushConst, 2),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Clip), 3),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!check(!out.is_error, "clip(float, reversed bounds) should not error")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Float, "clip(float, reversed bounds) should return float")) return false;
    if (!check(out.value.f == 0.5, "clip(float, reversed bounds) should return hi")) return false;
  }
  return true;
}

bool test_compare_string_eq_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 2U),
      Value::from_string_hash_len(0x1111ULL, 2U),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins(Opcode::Eq),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
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
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 5, 2),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "concat(string,string) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::FallbackToken,
               "concat(string,string) fallback should return fallback token");
}

bool test_builtin_concat_list_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_int_list_hash_len(0x3333ULL, 1U),
      Value::from_int_list_hash_len(0x4444ULL, 4U),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 5, 2),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "concat(list,list) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::IntList,
               "concat(int_list,int_list) fallback should return IntList");
}

bool test_builtin_concat_type_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 2U),
      Value::from_int_list_hash_len(0x4444ULL, 4U),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 5, 2),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
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
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::PushConst, 2),
      ins_ab(Opcode::CallBuiltin, 6, 3),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "slice(string,2,6) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::FallbackToken,
               "slice(string,2,6) fallback should return fallback token");
}

bool test_builtin_slice_string_payload_empty_ok() {
  g3pvm::payload::clear();
  BytecodeProgram p;
  p.consts = {
      g3pvm::payload::make_string_value("we"),
      Value::from_int(1),
      Value::from_int(-4),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::PushConst, 2),
      ins_ab(Opcode::CallBuiltin, 6, 3),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "slice(payload_string,1,-4) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::String,
             "slice(payload_string,1,-4) should return string")) {
    return false;
  }
  std::string exact;
  if (!check(g3pvm::payload::lookup_string(out.value, &exact),
             "slice(payload_string,1,-4) should keep exact empty payload")) {
    return false;
  }
  if (!check(exact.empty(), "slice(payload_string,1,-4) should produce empty string")) return false;
  return check(g3pvm::Value::container_len(out.value) == 0U,
               "slice(payload_string,1,-4) should return len=0 string");
}

bool test_builtin_slice_list_negative_idx_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_int_list_hash_len(0x2222ULL, 8U),
      Value::from_int(-5),
      Value::from_int(-1),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::PushConst, 2),
      ins_ab(Opcode::CallBuiltin, 6, 3),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "slice(list,-5,-1) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::IntList,
               "slice(int_list,-5,-1) fallback should return IntList");
}

bool test_builtin_slice_type_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_float(1.5),
      Value::from_int(6),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::PushConst, 2),
      ins_ab(Opcode::CallBuiltin, 6, 3),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Type, "slice(string,float,int) should error");
}

bool test_builtin_index_string_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_int(3),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 7, 2),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "index(string,3) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::FallbackToken,
               "index(string,3) should return fallback token");
}

bool test_builtin_index_list_negative_ok() {
  BytecodeProgram p;
  p.consts = {
      Value::from_int_list_hash_len(0x2222ULL, 8U),
      Value::from_int(-2),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 7, 2),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "index(list,-2) should not error")) return false;
  return check(out.value.tag == g3pvm::ValueTag::FallbackToken,
               "index(list,-2) should return fallback token");
}

bool test_builtin_index_oob_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 3U),
      Value::from_int(5),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 7, 2),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  return expect_err(out, ErrCode::Value, "index(string,5) should error");
}

bool test_builtin_index_type_error() {
  BytecodeProgram p;
  p.consts = {
      Value::from_string_hash_len(0x1111ULL, 8U),
      Value::from_float(1.5),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 7, 2),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
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
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 7, 2),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "index(payload_string,2) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Char, "index(payload_string,2) should return char")) return false;
  return check(out.value.i == static_cast<unsigned char>('c'), "index(payload_string,2) char mismatch");
}

bool test_builtin_index_list_payload_exact_ok() {
  g3pvm::payload::clear();
  BytecodeProgram p;
  std::vector<Value> elems = {Value::from_int(11), Value::from_int(22), Value::from_int(33)};
  p.consts = {
      g3pvm::payload::make_int_list_value(elems),
      Value::from_int(1),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, 7, 2),
      ins(Opcode::Return),
  };
  ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
  if (!check(!out.is_error, "index(payload_list,1) should not error")) return false;
  if (!check(out.value.tag == g3pvm::ValueTag::Int, "index(payload_list,1) should return int")) return false;
  return check(out.value.i == 22, "index(payload_list,1) element mismatch");
}

bool test_first_wave_sequence_builtins_payload_exact_ok() {
  g3pvm::payload::clear();
  {
    BytecodeProgram p;
    p.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(3),
    };
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 8, 2),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!check(!out.is_error, "append(int_list,int) should not error")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::IntList, "append should return IntList")) return false;
    std::vector<Value> elems;
    if (!check(g3pvm::payload::lookup_list(out.value, &elems), "append should keep exact payload")) return false;
    if (!check(elems.size() == 3 && elems[2].tag == g3pvm::ValueTag::Int && elems[2].i == 3,
               "append payload element mismatch")) return false;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_string_value("abc")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, 9, 1),
        ins(Opcode::Return),
    };
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!check(!out.is_error, "reverse(string) should not error")) return false;
    std::string exact;
    if (!check(g3pvm::payload::lookup_string(out.value, &exact), "reverse should keep exact string payload")) return false;
    if (!check(exact == "cba", "reverse string payload mismatch")) return false;
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
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!check(!out.is_error, "find(string,string) should not error")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 4, "find result mismatch")) return false;
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
    ExecResult out = g3pvm::execute_bytecode_cpu(p, {}, 10);
    if (!check(!out.is_error, "contains(string,string) should not error")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Bool && out.value.b, "contains result mismatch")) return false;
  }
  return true;
}

bool test_payload_retain_only_keeps_live_closure() {
  g3pvm::payload::clear();
  const Value keep_a = g3pvm::payload::make_string_value("keep-a");
  const Value keep_b = g3pvm::payload::make_string_value("keep-b");
  const Value drop_s = g3pvm::payload::make_string_value("drop-s");
  const Value keep_list = g3pvm::payload::make_string_list_value({keep_a, keep_b});
  const Value drop_list = g3pvm::payload::make_int_list_value({Value::from_int(7), Value::from_int(9)});

  g3pvm::payload::retain_only({keep_list});

  std::string exact;
  if (!check(g3pvm::payload::lookup_string(keep_a, &exact) && exact == "keep-a",
             "retain_only should keep transitive string payloads")) {
    return false;
  }
  if (!check(g3pvm::payload::lookup_string(keep_b, &exact) && exact == "keep-b",
             "retain_only should keep all transitive string payloads")) {
    return false;
  }
  if (!check(!g3pvm::payload::lookup_string(drop_s, &exact),
             "retain_only should drop unreferenced string payloads")) {
    return false;
  }

  std::vector<Value> elems;
  if (!check(g3pvm::payload::lookup_list(keep_list, &elems) && elems.size() == 2,
             "retain_only should keep live list payloads")) {
    return false;
  }
  if (!check(!g3pvm::payload::lookup_list(drop_list, &elems),
             "retain_only should drop unreferenced list payloads")) {
    return false;
  }

  const g3pvm::payload::PayloadStats stats = g3pvm::payload::stats();
  if (!check(stats.string_entries == 2, "retain_only stats string_entries mismatch")) return false;
  if (!check(stats.list_entries == 1, "retain_only stats list_entries mismatch")) return false;
  return true;
}

bool test_payload_typed_list_construction_rejects_wrong_element_tags() {
  g3pvm::payload::clear();
  bool rejected = false;
  try {
    (void)g3pvm::payload::make_int_list_value({Value::from_float(1.0)});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!check(rejected, "make_int_list_value should reject float elements")) return false;

  rejected = false;
  try {
    (void)g3pvm::payload::make_float_list_value({Value::from_int(1)});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!check(rejected, "make_float_list_value should reject int elements")) return false;

  rejected = false;
  try {
    (void)g3pvm::payload::make_string_list_value({Value::from_char('x')});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!check(rejected, "make_string_list_value should reject char elements")) return false;

  const Value s = g3pvm::payload::make_string_value("ok");
  const Value list = g3pvm::payload::make_string_list_value({s});
  std::vector<Value> elems;
  if (!check(g3pvm::payload::lookup_list(list, &elems), "valid string list should be registered")) {
    return false;
  }
  return check(elems.size() == 1 && elems[0].tag == g3pvm::ValueTag::String,
               "valid string list payload should preserve string element");
}

bool test_payload_packed_lookup_uses_exact_typed_list_tag() {
  g3pvm::payload::clear();
  constexpr std::uint64_t h = 0x12345ULL;
  constexpr std::uint32_t len = 2U;
  const Value int_key = Value::from_int_list_hash_len(h, len);
  const Value float_key = Value::from_float_list_hash_len(h, len);
  const Value string_list_key = Value::from_string_list_hash_len(h, len);

  g3pvm::payload::register_list(int_key, {Value::from_int(10), Value::from_int(20)});

  std::vector<Value> elems;
  if (!check(g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::IntList, int_key.i, &elems),
             "lookup_list_packed should find matching IntList token")) {
    return false;
  }
  if (!check(elems.size() == 2 && elems[0].tag == g3pvm::ValueTag::Int && elems[0].i == 10,
             "IntList lookup should return IntList payload")) {
    return false;
  }
  if (!check(!g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::FloatList, int_key.i, &elems),
             "FloatList lookup should not alias an IntList token")) {
    return false;
  }
  if (!check(!g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::StringList, int_key.i, &elems),
             "StringList lookup should not alias an IntList token")) {
    return false;
  }

  const Value s0 = g3pvm::payload::make_string_value("a");
  const Value s1 = g3pvm::payload::make_string_value("b");
  g3pvm::payload::register_list(float_key, {Value::from_float(1.5), Value::from_float(2.5)});
  g3pvm::payload::register_list(string_list_key, {s0, s1});

  if (!check(g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::FloatList, float_key.i, &elems),
             "lookup_list_packed should find matching FloatList token")) {
    return false;
  }
  if (!check(elems.size() == 2 && elems[0].tag == g3pvm::ValueTag::Float && elems[0].f == 1.5,
             "FloatList lookup should return FloatList payload")) {
    return false;
  }
  if (!check(g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::StringList, string_list_key.i,
                                                &elems),
             "lookup_list_packed should find matching StringList token")) {
    return false;
  }
  if (!check(elems.size() == 2 && elems[0].tag == g3pvm::ValueTag::String,
             "StringList lookup should return StringList payload")) {
    return false;
  }

  const g3pvm::payload::PayloadStats stats = g3pvm::payload::stats();
  return check(stats.list_entries == 3, "registry should keep one entry per typed-list tag");
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
  if (!test_mixed_int_float_numeric_ops_error()) return 1;
  if (!test_builtin_clip_reversed_bounds_returns_hi()) return 1;
  if (!test_compare_string_eq_ok()) return 1;
  if (!test_builtin_concat_string_ok()) return 1;
  if (!test_builtin_concat_list_ok()) return 1;
  if (!test_builtin_concat_type_error()) return 1;
  if (!test_builtin_slice_string_ok()) return 1;
  if (!test_builtin_slice_string_payload_empty_ok()) return 1;
  if (!test_builtin_slice_list_negative_idx_ok()) return 1;
  if (!test_builtin_slice_type_error()) return 1;
  if (!test_builtin_index_string_ok()) return 1;
  if (!test_builtin_index_list_negative_ok()) return 1;
  if (!test_builtin_index_oob_error()) return 1;
  if (!test_builtin_index_type_error()) return 1;
  if (!test_builtin_index_string_payload_exact_ok()) return 1;
  if (!test_builtin_index_list_payload_exact_ok()) return 1;
  if (!test_first_wave_sequence_builtins_payload_exact_ok()) return 1;
  if (!test_payload_retain_only_keeps_live_closure()) return 1;
  if (!test_payload_typed_list_construction_rejects_wrong_element_tags()) return 1;
  if (!test_payload_packed_lookup_uses_exact_typed_list_tag()) return 1;
  std::cout << "g3pvm_test_vm_edges: OK\n";
  return 0;
}
