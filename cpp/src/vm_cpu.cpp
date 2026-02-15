#include "g3pvm/vm_cpu.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "g3pvm/builtins.hpp"

namespace g3pvm {

namespace {

struct LocalSlot {
  bool is_set = false;
  Value value = Value::none();
};

VMResult fail(ErrCode code, const std::string& message) {
  VMResult out;
  out.is_error = true;
  out.err = Err{code, message};
  return out;
}

bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                     bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
    return false;
  }
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

double py_float_mod(double a, double b) {
  return a - std::floor(a / b) * b;
}

long long py_int_mod(long long a, long long b) {
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}

bool value_to_bool(const Value& v, bool& out) {
  if (v.tag != ValueTag::Bool) {
    return false;
  }
  out = v.b;
  return true;
}

VMResult compare_values(const std::string& op, const Value& a, const Value& b) {
  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (to_numeric_pair(a, b, a_num, b_num, any_float)) {
    VMResult out;
    if (op == "LT") out.value = Value::from_bool(a_num < b_num);
    else if (op == "LE") out.value = Value::from_bool(a_num <= b_num);
    else if (op == "GT") out.value = Value::from_bool(a_num > b_num);
    else if (op == "GE") out.value = Value::from_bool(a_num >= b_num);
    else if (op == "EQ") out.value = Value::from_bool(a_num == b_num);
    else if (op == "NE") out.value = Value::from_bool(a_num != b_num);
    else return fail(ErrCode::Type, "unknown comparison op");
    return out;
  }

  if (a.tag == ValueTag::Bool && b.tag == ValueTag::Bool) {
    if (op == "EQ") return VMResult{false, Value::from_bool(a.b == b.b), Err{ErrCode::Value, ""}};
    if (op == "NE") return VMResult{false, Value::from_bool(a.b != b.b), Err{ErrCode::Value, ""}};
    return fail(ErrCode::Type, "ordering comparison on bool not supported");
  }

  if (a.tag == ValueTag::None || b.tag == ValueTag::None) {
    if (op == "EQ") {
      return VMResult{false, Value::from_bool(a.tag == ValueTag::None && b.tag == ValueTag::None),
                      Err{ErrCode::Value, ""}};
    }
    if (op == "NE") {
      return VMResult{false, Value::from_bool(!(a.tag == ValueTag::None && b.tag == ValueTag::None)),
                      Err{ErrCode::Value, ""}};
    }
    return fail(ErrCode::Type, "ordering comparison on None not supported");
  }

  return fail(ErrCode::Type, "unsupported comparison operand types");
}

bool values_equal_for_fitness(const Value& a, const Value& b) {
  if (a.tag != b.tag) {
    return false;
  }
  if (a.tag == ValueTag::None) {
    return true;
  }
  if (a.tag == ValueTag::Bool) {
    return a.b == b.b;
  }
  if (a.tag == ValueTag::Int) {
    return a.i == b.i;
  }
  if (a.tag == ValueTag::Float) {
    return std::fabs(a.f - b.f) <= 1e-12;
  }
  return false;
}

}  // namespace

VMResult run_bytecode(const BytecodeProgram& program, const std::vector<std::pair<int, Value>>& inputs,
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

    if (ins.op == "PUSH_CONST") {
      if (!ins.has_a || ins.a < 0 || ins.a >= static_cast<int>(program.consts.size())) {
        return fail(ErrCode::Value, "const index out of range");
      }
      stack.push_back(program.consts[static_cast<std::size_t>(ins.a)]);
      continue;
    }

    if (ins.op == "LOAD") {
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

    if (ins.op == "STORE") {
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

    if (ins.op == "NEG" || ins.op == "NOT") {
      if (stack.empty()) {
        return fail(ErrCode::Value, "stack underflow");
      }
      const Value x = stack.back();
      stack.pop_back();
      if (ins.op == "NEG") {
        if (!is_numeric(x)) {
          return fail(ErrCode::Type, "NEG expects numeric");
        }
        if (x.tag == ValueTag::Float) stack.push_back(Value::from_float(-x.f));
        else stack.push_back(Value::from_int(-x.i));
      } else {
        if (x.tag != ValueTag::Bool) {
          return fail(ErrCode::Type, "NOT expects bool");
        }
        stack.push_back(Value::from_bool(!x.b));
      }
      continue;
    }

    if (ins.op == "ADD" || ins.op == "SUB" || ins.op == "MUL" || ins.op == "DIV" ||
        ins.op == "MOD") {
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
        return fail(ErrCode::Type, ins.op + " expects numeric operands");
      }

      if (ins.op == "DIV" || ins.op == "MOD") {
        if (b_num == 0.0) {
          return fail(ErrCode::ZeroDiv, (ins.op == "DIV") ? "division by zero" : "modulo by zero");
        }
      }

      if (ins.op == "ADD") {
        if (any_float) stack.push_back(Value::from_float(a_num + b_num));
        else stack.push_back(Value::from_int(static_cast<long long>(a_num) + static_cast<long long>(b_num)));
      } else if (ins.op == "SUB") {
        if (any_float) stack.push_back(Value::from_float(a_num - b_num));
        else stack.push_back(Value::from_int(static_cast<long long>(a_num) - static_cast<long long>(b_num)));
      } else if (ins.op == "MUL") {
        if (any_float) stack.push_back(Value::from_float(a_num * b_num));
        else stack.push_back(Value::from_int(static_cast<long long>(a_num) * static_cast<long long>(b_num)));
      } else if (ins.op == "DIV") {
        stack.push_back(Value::from_float(a_num / b_num));
      } else {
        if (any_float) {
          stack.push_back(Value::from_float(py_float_mod(a_num, b_num)));
        } else {
          const long long ai = static_cast<long long>(a_num);
          const long long bi = static_cast<long long>(b_num);
          stack.push_back(Value::from_int(py_int_mod(ai, bi)));
        }
      }
      continue;
    }

    if (ins.op == "LT" || ins.op == "LE" || ins.op == "GT" || ins.op == "GE" || ins.op == "EQ" ||
        ins.op == "NE") {
      if (stack.size() < 2) {
        return fail(ErrCode::Value, "stack underflow");
      }
      const Value b = stack.back();
      stack.pop_back();
      const Value a = stack.back();
      stack.pop_back();
      VMResult cmp = compare_values(ins.op, a, b);
      if (cmp.is_error) {
        return cmp;
      }
      stack.push_back(cmp.value);
      continue;
    }

    if (ins.op == "JMP") {
      if (!ins.has_a || ins.a < 0 || ins.a > static_cast<int>(program.code.size())) {
        return fail(ErrCode::Value, "jump target out of range");
      }
      ip = ins.a;
      continue;
    }

    if (ins.op == "JMP_IF_FALSE" || ins.op == "JMP_IF_TRUE") {
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
      if (ins.op == "JMP_IF_FALSE" && !cond) ip = ins.a;
      if (ins.op == "JMP_IF_TRUE" && cond) ip = ins.a;
      continue;
    }

    if (ins.op == "CALL_BUILTIN") {
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

      std::string name;
      if (bid == 0) name = "abs";
      else if (bid == 1) name = "min";
      else if (bid == 2) name = "max";
      else if (bid == 3) name = "clip";
      else return fail(ErrCode::Name, "unknown builtin id");

      BuiltinResult out = builtin_call(name, args);
      if (out.is_error) {
        return VMResult{true, Value::none(), out.err};
      }
      stack.push_back(out.value);
      continue;
    }

    if (ins.op == "RETURN") {
      if (stack.empty()) {
        return fail(ErrCode::Value, "return requires value on stack");
      }
      VMResult out;
      out.value = stack.back();
      return out;
    }

    return fail(ErrCode::Type, "unknown opcode: " + ins.op);
  }

  return fail(ErrCode::Value, "program finished without return");
}

std::vector<int> run_bytecode_cpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel) {
  if (programs.empty() || shared_cases.empty()) {
    return {};
  }
  if (shared_answer.size() != shared_cases.size()) {
    return {};
  }

  std::vector<int> fitness(programs.size(), 0);
  for (std::size_t p = 0; p < programs.size(); ++p) {
    for (std::size_t c = 0; c < shared_cases.size(); ++c) {
      std::vector<std::pair<int, Value>> inputs;
      inputs.reserve(shared_cases[c].size());
      for (const LocalBinding& binding : shared_cases[c]) {
        inputs.push_back({binding.idx, binding.value});
      }
      const VMResult out = run_bytecode(programs[p], inputs, fuel);
      if (out.is_error) {
        fitness[p] -= 10;
      } else if (values_equal_for_fitness(out.value, shared_answer[c])) {
        fitness[p] += 1;
      }
    }
  }

  return fitness;
}

}  // namespace g3pvm
