#include "g3pvm/cli/codec.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cstdint>

#include "g3pvm/runtime/payload/payload.hpp"

namespace g3pvm::cli_detail {

namespace {

std::uint64_t fnv1a64_bytes(const unsigned char* p, std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= static_cast<std::uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

std::uint64_t fnv1a64_mix(std::uint64_t h, std::uint64_t x) {
  for (int i = 0; i < 8; ++i) {
    const unsigned char b = static_cast<unsigned char>((x >> (i * 8)) & 0xffULL);
    h ^= static_cast<std::uint64_t>(b);
    h *= 1099511628211ULL;
  }
  return h;
}

std::uint64_t stable_typed_hash(const JsonValue& v);

std::uint64_t stable_scalar_hash(const std::string& t, const JsonValue& raw) {
  std::uint64_t h = fnv1a64_bytes(reinterpret_cast<const unsigned char*>(t.data()), t.size());
  if (t == "none") return h;
  if (t == "bool") {
    if (raw.kind != JsonValue::Kind::Bool) {
      throw std::runtime_error("bool typed value requires bool payload");
    }
    return fnv1a64_mix(h, raw.bool_v ? 1ULL : 0ULL);
  }
  if (t == "int") {
    if (raw.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("int typed value requires numeric payload");
    }
    const long long i = static_cast<long long>(raw.number_v);
    if (static_cast<double>(i) != raw.number_v) {
      throw std::runtime_error("int typed value must be integral");
    }
    return fnv1a64_mix(h, static_cast<std::uint64_t>(i));
  }
  if (t == "float") {
    if (raw.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("float typed value requires numeric payload");
    }
    const double d = raw.number_v;
    const auto* p = reinterpret_cast<const unsigned char*>(&d);
    return fnv1a64_bytes(p, sizeof(double));
  }
  if (t == "string") {
    if (raw.kind != JsonValue::Kind::String) {
      throw std::runtime_error("string typed value requires string payload");
    }
    return fnv1a64_bytes(reinterpret_cast<const unsigned char*>(raw.string_v.data()), raw.string_v.size());
  }
  throw std::runtime_error("unknown scalar typed value type");
}

std::uint64_t stable_typed_hash(const JsonValue& v) {
  const JsonValue& t_node = require_object_field(v, "type");
  const std::string t = require_string(t_node, "type");
  if (t == "list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("list typed value requires array payload");
    }
    std::uint64_t h = fnv1a64_bytes(reinterpret_cast<const unsigned char*>("list"), 4);
    h = fnv1a64_mix(h, static_cast<std::uint64_t>(raw.array_v.size()));
    for (const JsonValue& e : raw.array_v) {
      h = fnv1a64_mix(h, stable_typed_hash(e));
    }
    return h;
  }
  auto it = v.object_v.find("value");
  if (t == "none") {
    const JsonValue none_v;
    return stable_scalar_hash(t, none_v);
  }
  if (it == v.object_v.end()) {
    throw std::runtime_error("typed value requires field: value");
  }
  return stable_scalar_hash(t, it->second);
}

}  // namespace

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
  if (t == "string") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::String) {
      throw std::runtime_error("string typed value requires string payload");
    }
    return payload::make_string_value(raw.string_v);
  }
  if (t == "list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("list typed value requires array payload");
    }
    std::vector<Value> elems;
    elems.reserve(raw.array_v.size());
    for (const JsonValue& e : raw.array_v) {
      elems.push_back(decode_typed_value(e));
    }
    return payload::make_list_value(elems);
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
    const std::string op_name = require_string(require_object_field(ci, "op"), "op");
    if (!opcode_from_name(op_name, ins.op)) {
      throw std::runtime_error("unknown opcode: " + op_name);
    }

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

CaseBindings decode_input_case(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input case must be array");
  }
  CaseBindings one_case;
  one_case.reserve(v.array_v.size());
  for (const JsonValue& item : v.array_v) {
    const int idx = require_int(require_object_field(item, "idx"), "idx");
    const Value value = decode_typed_value(require_object_field(item, "value"));
    one_case.push_back(InputBinding{idx, value});
  }
  return one_case;
}

std::vector<CaseBindings> decode_cases(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("shared_cases must be array");
  }
  std::vector<CaseBindings> out;
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
  if (v.tag == ValueTag::String) {
    std::cout << "string_hash48 " << Value::container_hash48(v) << " len " << Value::container_len(v) << "\n";
    return;
  }
  if (v.tag == ValueTag::List) {
    std::cout << "list_hash48 " << Value::container_hash48(v) << " len " << Value::container_len(v) << "\n";
    return;
  }
  std::cout << "none\n";
}

}  // namespace g3pvm::cli_detail
