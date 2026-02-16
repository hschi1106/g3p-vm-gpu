#include "codec.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace g3pvm::cli_detail {

Value decode_typed_value(const JsonValue& v) {
  const JsonValue& t_node = require_object_field(v, "type");
  const std::string t = require_string(t_node, "type");
  if (t == "none") {
    return Value::none();
  }
  if (t == "bool") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Bool) {
      throw std::runtime_error("bool typed value requires bool payload");
    }
    return Value::from_bool(raw.bool_v);
  }
  if (t == "int") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("int typed value requires numeric payload");
    }
    const long long i = static_cast<long long>(raw.number_v);
    if (static_cast<double>(i) != raw.number_v) {
      throw std::runtime_error("int typed value must be integral");
    }
    return Value::from_int(i);
  }
  if (t == "float") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("float typed value requires numeric payload");
    }
    return Value::from_float(raw.number_v);
  }
  throw std::runtime_error("unknown typed value type");
}

BytecodeProgram decode_program(const JsonValue& bc) {
  BytecodeProgram program;
  program.n_locals = require_int(require_object_field(bc, "n_locals"), "n_locals");

  const JsonValue& consts = require_object_field(bc, "consts");
  if (consts.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("bytecode.consts must be array");
  }
  for (const JsonValue& c : consts.array_v) {
    program.consts.push_back(decode_typed_value(c));
  }

  const JsonValue& code = require_object_field(bc, "code");
  if (code.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("bytecode.code must be array");
  }
  for (const JsonValue& ci : code.array_v) {
    Instr ins;
    ins.op = require_string(require_object_field(ci, "op"), "op");

    auto a_it = ci.object_v.find("a");
    if (a_it != ci.object_v.end() && a_it->second.kind != JsonValue::Kind::Null) {
      ins.a = require_int(a_it->second, "a");
      ins.has_a = true;
    }

    auto b_it = ci.object_v.find("b");
    if (b_it != ci.object_v.end() && b_it->second.kind != JsonValue::Kind::Null) {
      ins.b = require_int(b_it->second, "b");
      ins.has_b = true;
    }
    program.code.push_back(ins);
  }
  return program;
}

InputCase decode_input_case(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input case must be array");
  }
  InputCase one_case;
  one_case.reserve(v.array_v.size());
  for (const JsonValue& item : v.array_v) {
    const int idx = require_int(require_object_field(item, "idx"), "idx");
    const Value value = decode_typed_value(require_object_field(item, "value"));
    one_case.push_back(LocalBinding{idx, value});
  }
  return one_case;
}

std::vector<InputCase> decode_cases(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("shared_cases must be array");
  }
  std::vector<InputCase> out;
  out.reserve(v.array_v.size());
  for (const JsonValue& case_node : v.array_v) {
    out.push_back(decode_input_case(case_node));
  }
  return out;
}

std::vector<Value> decode_shared_answer(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("shared_answer must be array");
  }
  std::vector<Value> out;
  out.reserve(v.array_v.size());
  for (const JsonValue& item : v.array_v) {
    out.push_back(decode_typed_value(item));
  }
  return out;
}

std::vector<BytecodeProgram> decode_programs(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("programs must be array");
  }
  std::vector<BytecodeProgram> out;
  out.reserve(v.array_v.size());
  for (const JsonValue& p : v.array_v) {
    out.push_back(decode_program(p));
  }
  return out;
}

void print_value(const Value& v) {
  if (v.tag == ValueTag::Int) {
    std::cout << "int " << v.i << "\n";
    return;
  }
  if (v.tag == ValueTag::Float) {
    std::cout << "float " << std::setprecision(17) << v.f << "\n";
    return;
  }
  if (v.tag == ValueTag::Bool) {
    std::cout << "bool " << (v.b ? 1 : 0) << "\n";
    return;
  }
  std::cout << "none\n";
}

}  // namespace g3pvm::cli_detail
