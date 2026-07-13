#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::Opcode;
using g3pvm::Value;

g3pvm::Instr ins(Opcode op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(Opcode op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }
g3pvm::Instr ins_ab(Opcode op, int a, int b) { return g3pvm::Instr{op, a, b, true, true}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool eval_single(const BytecodeProgram& program, const Value& expected, double expected_fitness, const std::string& label) {
  std::vector<BytecodeProgram> programs = {program};
  std::vector<g3pvm::CaseBindings> shared_cases(1);
  std::vector<Value> shared_answer = {expected};
  g3pvm::FitnessSessionGpu session;
  const g3pvm::FitnessSessionInitResult init = session.init(shared_cases, shared_answer, 100, 1);
  if (!init.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << init.err.message << ")\n";
    return true;
  }
  const g3pvm::FitnessEvalResult out = session.eval_programs(programs);
  if (!out.ok) {
    std::cout << "g3pvm_test_vm_gpu_smoke: SKIP (" << out.err.message << ")\n";
    return true;
  }
  if (!check(out.fitness.size() == 1, label + " should return one fitness score")) return false;
  return check(out.fitness[0] == expected_fitness, label + " fitness mismatch");
}

}  // namespace

int main() {
  g3pvm::payload::clear();

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
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Clip), 3),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int(0), 0.0, "clip reversed bounds")) return 1;
  }

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
    const Value expected = g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)});
    if (!eval_single(p, expected, 1.0, "append int_list")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(7)};
    p.code = {
        ins_a(Opcode::EmptyList, 1),
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Append), 2),
        ins(Opcode::Return),
    };
    const Value expected = g3pvm::payload::make_int_list_value({Value::from_int(7)});
    if (!eval_single(p, expected, 1.0, "empty list append helper")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_int_list_value({Value::from_int(1)}), Value::from_int(7)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins(Opcode::CheckList),
        ins(Opcode::EmptyListLike),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Append), 2),
        ins(Opcode::Return),
    };
    const Value expected = g3pvm::payload::make_int_list_value({Value::from_int(7)});
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
    p.consts = {g3pvm::payload::make_string_value("abc")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, 9, 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("cba"), 1.0, "reverse string")) return 1;
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
    if (!eval_single(p, Value::from_int(4), 0.0, "find string")) return 1;
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
    if (!eval_single(p, Value::from_bool(true), 1.0, "contains string")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(2), Value::from_int(3)}),
        Value::from_int(1),
    };
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Prepend), 2),
        ins(Opcode::Return),
    };
    const Value expected = g3pvm::payload::make_int_list_value({
        Value::from_int(1),
        Value::from_int(2),
        Value::from_int(3),
    });
    if (!eval_single(p, expected, 1.0, "prepend int_list")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_string_value("abc"), Value::from_int(1)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_a(Opcode::PushConst, 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Index), 2),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_char('b'), 1.0, "index string returns char")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(97)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Chr), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToUpper), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::CharToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("A"), 1.0, "char conversion chain")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {g3pvm::payload::make_string_value("E")};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::StringToChar), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToLower), 1),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Ord), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_int('e'), 0.0, "string char ord chain")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_char('7')};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::IsDigit), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, Value::from_bool(true), 1.0, "char digit predicate")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_int(123)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("123"), 1.0, "to_string int")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("1.5"), 1.0, "to_string float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(2.0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("2"), 1.0, "to_string whole float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(-0.0)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("0"), 1.0, "to_string negative zero float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(1.2345678)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::ToString), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("1.234568"), 1.0, "to_string rounded float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_float(1.5)};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Singleton), 1),
        ins(Opcode::Return),
    };
    const Value expected = g3pvm::payload::make_float_list_value({Value::from_float(1.5)});
    if (!eval_single(p, expected, 1.0, "singleton float")) return 1;
  }

  {
    BytecodeProgram p;
    p.consts = {Value::from_char('z')};
    p.code = {
        ins_a(Opcode::PushConst, 0),
        ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Singleton), 1),
        ins(Opcode::Return),
    };
    if (!eval_single(p, g3pvm::payload::make_string_value("z"), 1.0, "singleton char")) return 1;
  }

  std::cout << "g3pvm_test_vm_gpu_smoke: OK\n";
  return 0;
}
