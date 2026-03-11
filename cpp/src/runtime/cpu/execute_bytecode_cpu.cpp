#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/core/opcode.hpp"
#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/runtime/cpu/builtins_cpu.hpp"

namespace g3pvm {

namespace {

struct LocalSlot {
  bool is_set = false;
  Value value = Value::none();
};

ExecResult fail(ErrCode code, const std::string& message) {
  ExecResult out;
  out.is_error = true;
  out.err = Err{code, message};
  return out;
}

bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                     bool& any_float) {
  return vm_semantics::to_numeric_pair(a, b, a_out, b_out, any_float);
}

bool value_to_bool(const Value& v, bool& out) {
  if (v.tag != ValueTag::Bool) {
    return false;
  }
  out = v.b;
  return true;
}

ExecResult compare_values(const Opcode op, const Value& a, const Value& b) {
  vm_semantics::CmpOp cmp_op = vm_semantics::CmpOp::EQ;
  if (op == Opcode::Lt) cmp_op = vm_semantics::CmpOp::LT;
  else if (op == Opcode::Le) cmp_op = vm_semantics::CmpOp::LE;
  else if (op == Opcode::Gt) cmp_op = vm_semantics::CmpOp::GT;
  else if (op == Opcode::Ge) cmp_op = vm_semantics::CmpOp::GE;
  else if (op == Opcode::Eq) cmp_op = vm_semantics::CmpOp::EQ;
  else if (op == Opcode::Ne) cmp_op = vm_semantics::CmpOp::NE;
  else return fail(ErrCode::Type, "unknown comparison op");

  bool out_bool = false;
  const vm_semantics::CompareStatus status = vm_semantics::compare_values(cmp_op, a, b, out_bool);
  if (status == vm_semantics::CompareStatus::Ok) {
    return ExecResult{false, Value::from_bool(out_bool), Err{ErrCode::Value, ""}};
  }
  if (status == vm_semantics::CompareStatus::BoolOrderingNotSupported) {
    return fail(ErrCode::Type, "ordering comparison on bool not supported");
  }
  if (status == vm_semantics::CompareStatus::NoneOrderingNotSupported) {
    return fail(ErrCode::Type, "ordering comparison on None not supported");
  }
  if (status == vm_semantics::CompareStatus::UnsupportedTypes) {
    return fail(ErrCode::Type, "unsupported comparison operand types");
  }
  return fail(ErrCode::Type, "unknown comparison op");
}

}  // namespace

ExecResult execute_bytecode_cpu(const BytecodeProgram& program,
                                const std::vector<std::pair<int, Value>>& inputs,
                                int fuel) {
  std::vector<Value> stack;
  std::vector<LocalSlot> locals;
  locals.resize(static_cast<std::size_t>(program.n_locals));

  for (const auto& item : inputs) {
    const int idx = item.first;
    if (idx >= 0 && idx < program.n_locals) {
      locals[static_cast<std::size_t>(idx)].is_set = true;
      locals[static_cast<std::size_t>(idx)].value = item.second;
    }
  }

  int ip = 0;
  while (ip < static_cast<int>(program.code.size())) {
    if (fuel <= 0) {
      return fail(ErrCode::Timeout, "out of fuel");
    }
    fuel -= 1;

    const Instr& ins = program.code[static_cast<std::size_t>(ip)];
    ip += 1;
    const Opcode op = ins.op;

    if (op == Opcode::PushConst) {
      if (!ins.has_a || ins.a < 0 || ins.a >= static_cast<int>(program.consts.size())) {
        return fail(ErrCode::Value, "const index out of range");
      }
      stack.push_back(program.consts[static_cast<std::size_t>(ins.a)]);
      continue;
    }

    if (op == Opcode::Load) {
      if (!ins.has_a || ins.a < 0 || ins.a >= static_cast<int>(locals.size())) {
        return fail(ErrCode::Name, "local index out of range");
      }
      const LocalSlot& slot = locals[static_cast<std::size_t>(ins.a)];
      if (!slot.is_set) {
        return fail(ErrCode::Name, "read of uninitialized local");
      }
      stack.push_back(slot.value);
      continue;
    }

    if (op == Opcode::Store) {
      if (!ins.has_a || ins.a < 0 || ins.a >= static_cast<int>(locals.size())) {
        return fail(ErrCode::Name, "local index out of range");
      }
      if (stack.empty()) {
        return fail(ErrCode::Value, "stack underflow");
      }
      locals[static_cast<std::size_t>(ins.a)].is_set = true;
      locals[static_cast<std::size_t>(ins.a)].value = stack.back();
      stack.pop_back();
      continue;
    }

    if (op == Opcode::Neg || op == Opcode::Not) {
      if (stack.empty()) {
        return fail(ErrCode::Value, "stack underflow");
      }
      const Value x = stack.back();
      stack.pop_back();
      if (op == Opcode::Neg) {
        if (!is_numeric(x)) {
          return fail(ErrCode::Type, "NEG expects numeric");
        }
        if (x.tag == ValueTag::Float) {
          stack.push_back(Value::from_float(vm_semantics::canonicalize_vm_float(-x.f)));
        } else {
          stack.push_back(Value::from_int(vm_semantics::wrap_int_neg(x.i)));
        }
      } else {
        if (x.tag != ValueTag::Bool) {
          return fail(ErrCode::Type, "NOT expects bool");
        }
        stack.push_back(Value::from_bool(!x.b));
      }
      continue;
    }

    if (op == Opcode::Add || op == Opcode::Sub || op == Opcode::Mul || op == Opcode::Div ||
        op == Opcode::Mod) {
      if (stack.size() < 2) {
        return fail(ErrCode::Value, "stack underflow");
      }
      const Value b = stack.back();
      stack.pop_back();
      const Value a = stack.back();
      stack.pop_back();

      double a_num = 0.0;
      double b_num = 0.0;
      bool any_float = false;
      if (!to_numeric_pair(a, b, a_num, b_num, any_float)) {
        return fail(ErrCode::Type, std::string(opcode_name(op)) + " expects numeric operands");
      }

      if ((op == Opcode::Div || op == Opcode::Mod) && b_num == 0.0) {
        return fail(ErrCode::ZeroDiv, (op == Opcode::Div) ? "division by zero" : "modulo by zero");
      }

      if (op == Opcode::Add) {
        if (any_float) {
          stack.push_back(Value::from_float(vm_semantics::canonicalize_vm_float(a_num + b_num)));
        } else {
          stack.push_back(Value::from_int(
              vm_semantics::wrap_int_add(static_cast<long long>(a_num), static_cast<long long>(b_num))));
        }
      } else if (op == Opcode::Sub) {
        if (any_float) {
          stack.push_back(Value::from_float(vm_semantics::canonicalize_vm_float(a_num - b_num)));
        } else {
          stack.push_back(Value::from_int(
              vm_semantics::wrap_int_sub(static_cast<long long>(a_num), static_cast<long long>(b_num))));
        }
      } else if (op == Opcode::Mul) {
        if (any_float) {
          stack.push_back(Value::from_float(vm_semantics::canonicalize_vm_float(a_num * b_num)));
        } else {
          stack.push_back(Value::from_int(
              vm_semantics::wrap_int_mul(static_cast<long long>(a_num), static_cast<long long>(b_num))));
        }
      } else if (op == Opcode::Div) {
        stack.push_back(Value::from_float(vm_semantics::canonicalize_vm_float(a_num / b_num)));
      } else if (any_float) {
        stack.push_back(
            Value::from_float(vm_semantics::canonicalize_vm_float(vm_semantics::py_float_mod(a_num, b_num))));
      } else {
        const long long ai = static_cast<long long>(a_num);
        const long long bi = static_cast<long long>(b_num);
        stack.push_back(Value::from_int(vm_semantics::py_int_mod(ai, bi)));
      }
      continue;
    }

    if (op == Opcode::Lt || op == Opcode::Le || op == Opcode::Gt || op == Opcode::Ge || op == Opcode::Eq ||
        op == Opcode::Ne) {
      if (stack.size() < 2) {
        return fail(ErrCode::Value, "stack underflow");
      }
      const Value b = stack.back();
      stack.pop_back();
      const Value a = stack.back();
      stack.pop_back();
      ExecResult cmp = compare_values(op, a, b);
      if (cmp.is_error) {
        return cmp;
      }
      stack.push_back(cmp.value);
      continue;
    }

    if (op == Opcode::Jmp) {
      if (!ins.has_a || ins.a < 0 || ins.a > static_cast<int>(program.code.size())) {
        return fail(ErrCode::Value, "jump target out of range");
      }
      ip = ins.a;
      continue;
    }

    if (op == Opcode::JmpIfFalse || op == Opcode::JmpIfTrue) {
      if (stack.empty()) {
        return fail(ErrCode::Value, "stack underflow");
      }
      if (!ins.has_a || ins.a < 0 || ins.a > static_cast<int>(program.code.size())) {
        return fail(ErrCode::Value, "jump target out of range");
      }
      const Value c = stack.back();
      stack.pop_back();
      bool cond = false;
      if (!value_to_bool(c, cond)) {
        return fail(ErrCode::Type, "jump condition must be bool");
      }
      if (op == Opcode::JmpIfFalse && !cond) ip = ins.a;
      if (op == Opcode::JmpIfTrue && cond) ip = ins.a;
      continue;
    }

    if (op == Opcode::CallBuiltin) {
      const int bid = ins.has_a ? ins.a : -1;
      const int argc = ins.has_b ? ins.b : -1;
      if (argc < 0) {
        return fail(ErrCode::Type, "invalid builtin argc");
      }
      if (static_cast<int>(stack.size()) < argc) {
        return fail(ErrCode::Value, "stack underflow");
      }
      std::vector<Value> args;
      args.reserve(static_cast<std::size_t>(argc));
      const std::size_t start = stack.size() - static_cast<std::size_t>(argc);
      for (std::size_t i = start; i < stack.size(); ++i) {
        args.push_back(stack[i]);
      }
      stack.resize(start);

      BuiltinId builtin_id = BuiltinId::Abs;
      if (!builtin_id_from_int(bid, builtin_id)) {
        return fail(ErrCode::Name, "unknown builtin id");
      }

      BuiltinResult out = builtin_call(builtin_id, args);
      if (out.is_error) {
        return ExecResult{true, Value::none(), out.err};
      }
      stack.push_back(out.value);
      continue;
    }

    if (op == Opcode::Return) {
      if (stack.empty()) {
        return fail(ErrCode::Value, "return requires value on stack");
      }
      ExecResult out;
      out.value = stack.back();
      return out;
    }
  }

  return fail(ErrCode::Value, "program finished without return");
}

}  // namespace g3pvm
