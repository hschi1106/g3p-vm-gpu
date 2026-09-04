#include "gagp/cli/codec.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <utility>

#include "gagp/core/bytecode_verify.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace gagp::cli_detail {

Value decode_typed_value(const JsonValue& v);

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

Value decode_legacy_typed_list(const JsonValue& raw);

int expected_dp2d_transition_arity(int dep_kind) {
  switch (dep_kind) {
    case 0:
    case 1:
      return 2;
    case 2:
    case 3:
      return 1;
    case 4:
    case 5:
      return 3;
    default:
      throw std::runtime_error("ASGP-DP2D dep_kind must be in [0, 5]");
  }
}

void validate_asgp_dp1d_segment_arity(const AsgpDp1dSegment& segment) {
  if (segment.dep_kind != -1 && segment.dep_kind != 1) {
    throw std::runtime_error("ASGP-DP1D dep_kind must be -1 or 1");
  }
  if (segment.dep_offsets.empty()) {
    throw std::runtime_error("ASGP-DP1D dep_offsets must not be empty");
  }
  if (segment.dep_offsets.size() != segment.transition_dep_names.size()) {
    throw std::runtime_error("ASGP-DP1D dependency arity mismatch");
  }
}

void validate_asgp_dp2d_segment_arity(const AsgpDp2dSegment& segment) {
  const int expected_arity = expected_dp2d_transition_arity(segment.dep_kind);
  if (segment.transition_dep_names.size() != static_cast<std::size_t>(expected_arity)) {
    throw std::runtime_error("ASGP-DP2D dependency arity mismatch");
  }
}

std::uint64_t stable_scalar_hash(const std::string& t, const JsonValue& raw) {
  std::uint64_t h = fnv1a64_bytes(reinterpret_cast<const unsigned char*>(t.data()), t.size());
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
  if (t == "char") {
    if (raw.kind != JsonValue::Kind::String || raw.string_v.size() != 1U) {
      throw std::runtime_error("char typed value requires length-1 string payload");
    }
    return fnv1a64_mix(h, static_cast<unsigned char>(raw.string_v[0]));
  }
  throw std::runtime_error("unknown scalar typed value type");
}

std::uint64_t stable_typed_hash(const JsonValue& v) {
  const JsonValue& t_node = require_object_field(v, "type");
  const std::string t = require_string(t_node, "type");
  if (t == "int_list" || t == "float_list" || t == "num_list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error(t + " typed value requires array payload");
    }
    std::uint64_t h = fnv1a64_bytes(reinterpret_cast<const unsigned char*>(t.data()), t.size());
    h = fnv1a64_mix(h, static_cast<std::uint64_t>(raw.array_v.size()));
    for (const JsonValue& e : raw.array_v) {
      if (e.kind != JsonValue::Kind::Number) {
        throw std::runtime_error(t + " elements must be numeric");
      }
      if (t == "int_list") {
        h = fnv1a64_mix(h, stable_scalar_hash("int", e));
      } else if (t == "float_list") {
        h = fnv1a64_mix(h, stable_scalar_hash("float", e));
      } else if (static_cast<double>(static_cast<long long>(e.number_v)) == e.number_v) {
        h = fnv1a64_mix(h, stable_scalar_hash("int", e));
      } else {
        h = fnv1a64_mix(h, stable_scalar_hash("float", e));
      }
    }
    return h;
  }
  if (t == "string_list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("string_list typed value requires array payload");
    }
    std::uint64_t h = fnv1a64_bytes(reinterpret_cast<const unsigned char*>("string_list"), 11);
    h = fnv1a64_mix(h, static_cast<std::uint64_t>(raw.array_v.size()));
    for (const JsonValue& e : raw.array_v) {
      if (e.kind != JsonValue::Kind::String) {
        throw std::runtime_error("string_list elements must be strings");
      }
      h = fnv1a64_mix(h, stable_scalar_hash("string", e));
    }
    return h;
  }
  if (t == "list") {
    const Value legacy = decode_legacy_typed_list(require_object_field(v, "value"));
    if (legacy.tag == ValueTag::IntList) {
      return fnv1a64_mix(fnv1a64_bytes(reinterpret_cast<const unsigned char*>("legacy_int_list"), 15),
                         static_cast<std::uint64_t>(legacy.i));
    }
    if (legacy.tag == ValueTag::FloatList) {
      return fnv1a64_mix(fnv1a64_bytes(reinterpret_cast<const unsigned char*>("legacy_float_list"), 17),
                         static_cast<std::uint64_t>(legacy.i));
    }
    return fnv1a64_mix(fnv1a64_bytes(reinterpret_cast<const unsigned char*>("legacy_string_list"), 18),
                       static_cast<std::uint64_t>(legacy.i));
  }
  auto it = v.object_v.find("value");
  if (it == v.object_v.end()) {
    throw std::runtime_error("typed value requires field: value");
  }
  return stable_scalar_hash(t, it->second);
}

Value decode_legacy_typed_list(const JsonValue& raw) {
  if (raw.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("legacy list typed value requires array payload");
  }
  std::vector<Value> elems;
  elems.reserve(raw.array_v.size());
  bool all_numeric = true;
  bool all_string = true;
  bool any_float = false;
  for (const JsonValue& e : raw.array_v) {
    const Value v = decode_typed_value(e);
    elems.push_back(v);
    if (!is_numeric(v)) {
      all_numeric = false;
    } else if (v.tag == ValueTag::Float) {
      any_float = true;
    }
    if (v.tag != ValueTag::String) {
      all_string = false;
    }
  }
  if (all_numeric) {
    if (!any_float) {
      return payload::make_int_list_value(elems);
    }
    std::vector<Value> float_elems;
    float_elems.reserve(elems.size());
    for (const Value& elem : elems) {
      float_elems.push_back(elem.tag == ValueTag::Float ? elem : Value::from_float(static_cast<double>(elem.i)));
    }
    return payload::make_float_list_value(float_elems);
  }
  if (all_string) {
    return payload::make_string_list_value(elems);
  }
  throw std::runtime_error("legacy list typed value must infer to int_list, float_list, or string_list");
}

}  // namespace

Value decode_typed_value(const JsonValue& v) {
  const JsonValue& t_node = require_object_field(v, "type");
  const std::string t = require_string(t_node, "type");
  if (t == "none") {
    throw std::runtime_error("none is not a public current typed value");
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
  if (t == "char") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::String || raw.string_v.size() != 1U) {
      throw std::runtime_error("char typed value requires length-1 string payload");
    }
    return Value::from_char(static_cast<unsigned char>(raw.string_v[0]));
  }
  if (t == "int_list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("int_list typed value requires array payload");
    }
    std::vector<Value> elems;
    elems.reserve(raw.array_v.size());
    for (const JsonValue& e : raw.array_v) {
      if (e.kind != JsonValue::Kind::Number) {
        throw std::runtime_error("int_list elements must be numeric");
      }
      const long long i = static_cast<long long>(e.number_v);
      if (static_cast<double>(i) != e.number_v) {
        throw std::runtime_error("int_list elements must be integral");
      }
      elems.push_back(Value::from_int(i));
    }
    return payload::make_int_list_value(elems);
  }
  if (t == "float_list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("float_list typed value requires array payload");
    }
    std::vector<Value> elems;
    elems.reserve(raw.array_v.size());
    for (const JsonValue& e : raw.array_v) {
      if (e.kind != JsonValue::Kind::Number) {
        throw std::runtime_error("float_list elements must be numeric");
      }
      elems.push_back(Value::from_float(e.number_v));
    }
    return payload::make_float_list_value(elems);
  }
  if (t == "num_list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("legacy num_list typed value requires array payload");
    }
    bool any_float = false;
    std::vector<double> numbers;
    numbers.reserve(raw.array_v.size());
    for (const JsonValue& e : raw.array_v) {
      if (e.kind != JsonValue::Kind::Number) {
        throw std::runtime_error("legacy num_list elements must be numeric");
      }
      const long long i = static_cast<long long>(e.number_v);
      if (static_cast<double>(i) != e.number_v) {
        any_float = true;
      }
      numbers.push_back(e.number_v);
    }
    std::vector<Value> elems;
    elems.reserve(numbers.size());
    if (any_float) {
      for (double number : numbers) {
        elems.push_back(Value::from_float(number));
      }
      return payload::make_float_list_value(elems);
    }
    for (double number : numbers) {
      elems.push_back(Value::from_int(static_cast<long long>(number)));
    }
    return payload::make_int_list_value(elems);
  }
  if (t == "string_list") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("string_list typed value requires array payload");
    }
    std::vector<Value> elems;
    elems.reserve(raw.array_v.size());
    for (const JsonValue& e : raw.array_v) {
      if (e.kind != JsonValue::Kind::String) {
        throw std::runtime_error("string_list elements must be strings");
      }
      elems.push_back(payload::make_string_value(e.string_v));
    }
    return payload::make_string_list_value(elems);
  }
  if (t == "list") {
    return decode_legacy_typed_list(require_object_field(v, "value"));
  }
  throw std::runtime_error("unknown typed value type");
}

std::vector<Instr> decode_code(const JsonValue& code) {
  if (code.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("bytecode.code must be array");
  }
  std::vector<Instr> out;
  out.reserve(code.array_v.size());
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
    out.push_back(ins);
  }
  return out;
}

std::vector<Value> decode_const_array(const JsonValue& consts) {
  if (consts.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("bytecode.consts must be array");
  }
  std::vector<Value> out;
  out.reserve(consts.array_v.size());
  for (const JsonValue& c : consts.array_v) {
    out.push_back(decode_typed_value(c));
  }
  return out;
}

std::vector<int> decode_int_array(const JsonValue& raw, const char* field_name) {
  if (raw.kind != JsonValue::Kind::Array) {
    throw std::runtime_error(std::string(field_name) + " must be array");
  }
  std::vector<int> out;
  out.reserve(raw.array_v.size());
  for (const JsonValue& item : raw.array_v) {
    out.push_back(require_int(item, field_name));
  }
  return out;
}

PhaseProgram decode_phase_program(const JsonValue& raw) {
  PhaseProgram phase;
  phase.n_locals = require_int(require_object_field(raw, "n_locals"), "phase.n_locals");
  phase.consts = decode_const_array(require_object_field(raw, "consts"));
  phase.code = decode_code(require_object_field(raw, "code"));

  auto binders_it = raw.object_v.find("binder_locals");
  if (binders_it != raw.object_v.end() && binders_it->second.kind != JsonValue::Kind::Null) {
    if (binders_it->second.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("phase.binder_locals must be array");
    }
    for (const JsonValue& item : binders_it->second.array_v) {
      const int name_id = require_int(require_object_field(item, "name"), "binder_locals.name");
      const int local = require_int(require_object_field(item, "local"), "binder_locals.local");
      phase.binder_locals[name_id] = local;
    }
  }
  return phase;
}

BytecodeProgram decode_program(const JsonValue& bc) {
  BytecodeProgram program;
  program.n_locals = require_int(require_object_field(bc, "n_locals"), "n_locals");

  program.consts = decode_const_array(require_object_field(bc, "consts"));
  program.code = decode_code(require_object_field(bc, "code"));

  auto segments_it = bc.object_v.find("segments");
  if (segments_it != bc.object_v.end() && segments_it->second.kind != JsonValue::Kind::Null) {
    if (segments_it->second.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("bytecode.segments must be object");
    }
    auto dc_it = segments_it->second.object_v.find("asgp_dc");
    if (dc_it != segments_it->second.object_v.end()) {
      if (dc_it->second.kind != JsonValue::Kind::Array) {
        throw std::runtime_error("segments.asgp_dc must be array");
      }
      for (const JsonValue& raw_segment : dc_it->second.array_v) {
        AsgpDcSegment segment;
        segment.solve_xs_name = require_int(require_object_field(raw_segment, "solve_xs_name"), "solve_xs_name");
        segment.solve_n_name = require_int(require_object_field(raw_segment, "solve_n_name"), "solve_n_name");
        segment.solve_lo_name = require_int(require_object_field(raw_segment, "solve_lo_name"), "solve_lo_name");
        segment.divide_n_name = require_int(require_object_field(raw_segment, "divide_n_name"), "divide_n_name");
        segment.combine_left_name =
            require_int(require_object_field(raw_segment, "combine_left_name"), "combine_left_name");
        segment.combine_right_name =
            require_int(require_object_field(raw_segment, "combine_right_name"), "combine_right_name");
        segment.solve = decode_phase_program(require_object_field(raw_segment, "solve"));
        segment.divide = decode_phase_program(require_object_field(raw_segment, "divide"));
        segment.combine = decode_phase_program(require_object_field(raw_segment, "combine"));
        program.asgp_dc_segments.push_back(std::move(segment));
      }
    }

    auto dp1_it = segments_it->second.object_v.find("asgp_dp1d");
    if (dp1_it != segments_it->second.object_v.end()) {
      if (dp1_it->second.kind != JsonValue::Kind::Array) {
        throw std::runtime_error("segments.asgp_dp1d must be array");
      }
      for (const JsonValue& raw_segment : dp1_it->second.array_v) {
        AsgpDp1dSegment segment;
        segment.lo = require_int(require_object_field(raw_segment, "lo"), "lo");
        segment.hi = require_int(require_object_field(raw_segment, "hi"), "hi");
        segment.base_state = require_int(require_object_field(raw_segment, "base_state"), "base_state");
        segment.boundary_value = decode_typed_value(require_object_field(raw_segment, "boundary_value"));
        segment.dep_kind = require_int(require_object_field(raw_segment, "dep_kind"), "dep_kind");
        segment.dep_offsets = decode_int_array(require_object_field(raw_segment, "dep_offsets"), "dep_offsets");
        segment.solve_state_name =
            require_int(require_object_field(raw_segment, "solve_state_name"), "solve_state_name");
        segment.transition_state_name =
            require_int(require_object_field(raw_segment, "transition_state_name"), "transition_state_name");
        segment.transition_dep_names =
            decode_int_array(require_object_field(raw_segment, "transition_dep_names"), "transition_dep_names");
        segment.solve = decode_phase_program(require_object_field(raw_segment, "solve"));
        segment.transition = decode_phase_program(require_object_field(raw_segment, "transition"));
        validate_asgp_dp1d_segment_arity(segment);
        program.asgp_dp1d_segments.push_back(std::move(segment));
      }
    }

    auto dp2_it = segments_it->second.object_v.find("asgp_dp2d");
    if (dp2_it != segments_it->second.object_v.end()) {
      if (dp2_it->second.kind != JsonValue::Kind::Array) {
        throw std::runtime_error("segments.asgp_dp2d must be array");
      }
      for (const JsonValue& raw_segment : dp2_it->second.array_v) {
        AsgpDp2dSegment segment;
        segment.i_lo = require_int(require_object_field(raw_segment, "i_lo"), "i_lo");
        segment.i_hi = require_int(require_object_field(raw_segment, "i_hi"), "i_hi");
        segment.j_lo = require_int(require_object_field(raw_segment, "j_lo"), "j_lo");
        segment.j_hi = require_int(require_object_field(raw_segment, "j_hi"), "j_hi");
        segment.base_i = require_int(require_object_field(raw_segment, "base_i"), "base_i");
        segment.base_j = require_int(require_object_field(raw_segment, "base_j"), "base_j");
        segment.boundary_value = decode_typed_value(require_object_field(raw_segment, "boundary_value"));
        segment.dep_kind = require_int(require_object_field(raw_segment, "dep_kind"), "dep_kind");
        segment.solve_i_name = require_int(require_object_field(raw_segment, "solve_i_name"), "solve_i_name");
        segment.solve_j_name = require_int(require_object_field(raw_segment, "solve_j_name"), "solve_j_name");
        segment.transition_i_name =
            require_int(require_object_field(raw_segment, "transition_i_name"), "transition_i_name");
        segment.transition_j_name =
            require_int(require_object_field(raw_segment, "transition_j_name"), "transition_j_name");
        segment.transition_dep_names =
            decode_int_array(require_object_field(raw_segment, "transition_dep_names"), "transition_dep_names");
        segment.solve = decode_phase_program(require_object_field(raw_segment, "solve"));
        segment.transition = decode_phase_program(require_object_field(raw_segment, "transition"));
        validate_asgp_dp2d_segment_arity(segment);
        program.asgp_dp2d_segments.push_back(std::move(segment));
      }
    }
  }
  const BytecodeVerifyResult verified = verify_bytecode(program);
  if (!verified) {
    throw std::runtime_error(
        std::string("invalid bytecode (") +
        bytecode_verify_code_name(verified.diagnostic.code) + ") at " +
        verified.diagnostic.path + ": " + verified.diagnostic.message);
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
  if (v.tag == ValueTag::Char) {
    std::cout << "char " << v.i << "\n";
    return;
  }
  if (v.tag == ValueTag::String) {
    std::cout << "string_hash48 " << Value::container_hash48(v) << " len " << Value::container_len(v) << "\n";
    return;
  }
  if (v.tag == ValueTag::IntList) {
    std::cout << "int_list_hash48 " << Value::container_hash48(v) << " len " << Value::container_len(v) << "\n";
    return;
  }
  if (v.tag == ValueTag::FloatList) {
    std::cout << "float_list_hash48 " << Value::container_hash48(v) << " len " << Value::container_len(v) << "\n";
    return;
  }
  if (v.tag == ValueTag::StringList) {
    std::cout << "string_list_hash48 " << Value::container_hash48(v) << " len " << Value::container_len(v) << "\n";
    return;
  }
  if (v.tag == ValueTag::FallbackToken) {
    std::cout << "fallback_token " << static_cast<std::uint64_t>(v.i) << "\n";
    return;
  }
  std::cout << "invalid\n";
}

}  // namespace gagp::cli_detail
