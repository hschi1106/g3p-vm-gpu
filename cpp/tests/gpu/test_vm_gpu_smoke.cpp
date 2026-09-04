#include <iostream>
#include <string>
#include <vector>

#include "gagp/core/builtin.hpp"
#include "gagp/core/bytecode.hpp"
#include "gagp/core/errors.hpp"
#include "gagp/core/value.hpp"
#include "gagp/runtime/gpu/fitness_gpu.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace {

using gagp::BytecodeProgram;
using gagp::Opcode;
using gagp::Value;

gagp::Instr ins(Opcode op) { return gagp::Instr{op, 0, 0, false, false}; }
gagp::Instr ins_a(Opcode op, int a) { return gagp::Instr{op, a, 0, true, false}; }
gagp::Instr ins_ab(Opcode op, int a, int b) { return gagp::Instr{op, a, b, true, true}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool eval_single(const BytecodeProgram& program, const Value& expected, double expected_fitness, const std::string& label) {
  std::vector<BytecodeProgram> programs = {program};
  std::vector<gagp::CaseBindings> shared_cases(1);
  std::vector<Value> shared_answer = {expected};
  gagp::FitnessSessionGpu session;
  const gagp::FitnessSessionInitResult init = session.init(shared_cases, shared_answer, 100, 1);
  if (!init.ok) {
    std::cout << "gagp_test_vm_gpu_smoke: SKIP (" << init.err.message << ")\n";
    return true;
  }
  const gagp::FitnessEvalResult out = session.eval_programs(programs);
  if (!out.ok) {
    std::cout << "gagp_test_vm_gpu_smoke: SKIP (" << out.err.message << ")\n";
    return true;
  }
  if (!check(out.fitness.size() == 1, label + " should return one fitness score")) return false;
  return check(out.fitness[0] == expected_fitness, label + " fitness mismatch");
}

}  // namespace

int main() {
  gagp::payload::clear();

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(2), Value::from_int(3)};
    p.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1), ins(Opcode::Add), ins(Opcode::Return)};
    if (!eval_single(p, Value::from_int(5), 0.0, "numeric add")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(5), Value::from_int(10), Value::from_int(0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_a(Opcode::PushConst, 2),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Clip), 3),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int(0), 0.0, "clip reversed bounds")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {
        gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(3),
    };
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 8, 2),
        ins(Opcode::Return),
    };
    const Value expected = gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)});
    if (!eval_single(p, expected, 1.0, "append int_list")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(7)};
    p.code = {
        ins_a(Opcode::EmptyList, 1),
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Append), 2),
        ins(Opcode::Return),
    };
    const Value expected = gagp::payload::make_int_list_value({Value::from_int(7)});
    if (!eval_single(p, expected, 1.0, "empty list append helper")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {gagp::payload::make_int_list_value({Value::from_int(1)}), Value::from_int(7)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins(Opcode::CheckList),
        ins(Opcode::EmptyListLike),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Append), 2),
        ins(Opcode::Return),
    };
    const Value expected = gagp::payload::make_int_list_value({Value::from_int(7)});
    if (!eval_single(p, expected, 1.0, "empty list like helper")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(7)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins(Opcode::CheckInt),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int(7), 0.0, "check int helper")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {gagp::payload::make_string_value("abc")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, 9, 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("cba"), 1.0, "reverse string")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {gagp::payload::make_string_value("abracadabra"), gagp::payload::make_string_value("cad")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 10, 2),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int(4), 0.0, "find string")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {gagp::payload::make_string_value("abracadabra"), gagp::payload::make_string_value("cad")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, 11, 2),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_bool(true), 1.0, "contains string")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {
        gagp::payload::make_int_list_value({Value::from_int(2), Value::from_int(3)}),
        Value::from_int(1),
    };
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Prepend), 2),
        ins(Opcode::Return),
    };
    const Value expected = gagp::payload::make_int_list_value({
        Value::from_int(1),
        Value::from_int(2),
        Value::from_int(3),
    });
    if (!eval_single(p, expected, 1.0, "prepend int_list")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {gagp::payload::make_string_value("abc"), Value::from_int(1)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), 2),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_char('b'), 1.0, "index string returns char")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(97)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Chr), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToUpper), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::CharToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("A"), 1.0, "char conversion chain")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {gagp::payload::make_string_value("E")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::StringToChar), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToLower), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Ord), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int('e'), 0.0, "string char ord chain")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_char('7')};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::IsDigit), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_bool(true), 1.0, "char digit predicate")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(123)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("123"), 1.0, "to_string int")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("1.5"), 1.0, "to_string float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(2.0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("2"), 1.0, "to_string whole float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(-0.0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("0"), 1.0, "to_string negative zero float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(1.2345678)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("1.234568"), 1.0, "to_string rounded float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Singleton), 1),
        ins(Opcode::Return),
    };
    const Value expected = gagp::payload::make_float_list_value({Value::from_float(1.5)});
    if (!eval_single(p, expected, 1.0, "singleton float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_char('z')};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Singleton), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, gagp::payload::make_string_value("z"), 1.0, "singleton char")) return 1;
  }

  std::cout << "gagp_test_vm_gpu_smoke: OK\n";
  return 0;
}
