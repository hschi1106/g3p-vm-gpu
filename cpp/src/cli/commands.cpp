#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "g3pvm/evolution/repro/pack.hpp"
#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/commands.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/cli/options.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::Value;
using g3pvm::ValueTag;
using g3pvm::cli_detail::CliOptions;
using g3pvm::cli_detail::JsonValue;

std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    if (c == '"') {
      oss << "\\\"";
    } else if (c == '\\') {
      oss << "\\\\";
    } else if (c == '\n') {
      oss << "\\n";
    } else if (c == '\r') {
      oss << "\\r";
    } else if (c == '\t') {
      oss << "\\t";
    } else {
      oss << c;
    }
  }
  return oss.str();
}

void write_value_json(std::ostream& out, const Value& v) {
  if (v.tag == ValueTag::Invalid) {
    out << "null";
    return;
  }
  if (v.tag == ValueTag::Bool) {
    out << (v.b ? "true" : "false");
    return;
  }
  if (v.tag == ValueTag::Int) {
    out << v.i;
    return;
  }
  if (v.tag == ValueTag::FallbackToken) {
    out << "null";
    return;
  }
  if (v.tag == ValueTag::String || v.tag == ValueTag::IntList ||
      v.tag == ValueTag::FloatList || v.tag == ValueTag::StringList) {
    out << "null";
    return;
  }
  if (std::isfinite(v.f)) {
    out << std::setprecision(17) << v.f;
    return;
  }
  out << "null";
}

void write_typed_value_json(std::ostream& out, const Value& v);

void write_typed_list_json(std::ostream& out, const char* type, const std::vector<Value>& elems) {
  out << "{\"type\":\"" << type << "\",\"value\":[";
  for (std::size_t i = 0; i < elems.size(); ++i) {
    if (i > 0) out << ",";
    if (std::string(type) == "int_list") {
      if (elems[i].tag != ValueTag::Int) {
        throw std::runtime_error("IntList AST constant contains non-int element");
      }
      out << elems[i].i;
    } else if (std::string(type) == "float_list") {
      if (elems[i].tag != ValueTag::Float) {
        throw std::runtime_error("FloatList AST constant contains non-float element");
      }
      out << std::setprecision(17) << elems[i].f;
    } else {
      if (elems[i].tag != ValueTag::String) {
        throw std::runtime_error("StringList AST constant contains non-string element");
      }
      std::string s;
      if (!g3pvm::payload::lookup_string(elems[i], &s)) {
        throw std::runtime_error("missing string element payload while writing AST JSON");
      }
      out << "\"" << json_escape(s) << "\"";
    }
  }
  out << "]}";
}

void write_typed_value_json(std::ostream& out, const Value& v) {
  if (v.tag == ValueTag::Bool) {
    out << "{\"type\":\"bool\",\"value\":" << (v.b ? "true" : "false") << "}";
    return;
  }
  if (v.tag == ValueTag::Int) {
    out << "{\"type\":\"int\",\"value\":" << v.i << "}";
    return;
  }
  if (v.tag == ValueTag::Float) {
    out << "{\"type\":\"float\",\"value\":";
    if (std::isfinite(v.f)) {
      out << std::setprecision(17) << v.f;
    } else {
      out << "0";
    }
    out << "}";
    return;
  }
  if (v.tag == ValueTag::Char) {
    std::string s;
    s.push_back(static_cast<char>(v.i & 0xff));
    out << "{\"type\":\"char\",\"value\":\"" << json_escape(s) << "\"}";
    return;
  }
  if (v.tag == ValueTag::String) {
    std::string s;
    if (!g3pvm::payload::lookup_string(v, &s)) {
      throw std::runtime_error("missing string payload while writing AST JSON");
    }
    out << "{\"type\":\"string\",\"value\":\"" << json_escape(s) << "\"}";
    return;
  }
  if (v.tag == ValueTag::IntList || v.tag == ValueTag::FloatList || v.tag == ValueTag::StringList) {
    std::vector<Value> elems;
    if (!g3pvm::payload::lookup_list(v, &elems)) {
      throw std::runtime_error("missing list payload while writing AST JSON");
    }
    if (v.tag == ValueTag::IntList) {
      write_typed_list_json(out, "int_list", elems);
    } else if (v.tag == ValueTag::FloatList) {
      write_typed_list_json(out, "float_list", elems);
    } else {
      write_typed_list_json(out, "string_list", elems);
    }
    return;
  }
  throw std::runtime_error("unsupported AST constant value tag");
}

void write_ast_json(std::ostream& out, const g3pvm::evo::AstProgram& ast) {
  out << "{";
  out << "\"version\":\"" << json_escape(ast.version) << "\",";
  out << "\"nodes\":[";
  for (std::size_t i = 0; i < ast.nodes.size(); ++i) {
    if (i > 0) out << ",";
    const g3pvm::evo::AstNode& node = ast.nodes[i];
    out << "{\"kind\":" << static_cast<int>(node.kind)
        << ",\"i0\":" << node.i0
        << ",\"i1\":" << node.i1 << "}";
  }
  out << "],\"names\":[";
  for (std::size_t i = 0; i < ast.names.size(); ++i) {
    if (i > 0) out << ",";
    out << "\"" << json_escape(ast.names[i]) << "\"";
  }
  out << "],\"consts\":[";
  for (std::size_t i = 0; i < ast.consts.size(); ++i) {
    if (i > 0) out << ",";
    write_typed_value_json(out, ast.consts[i]);
  }
  out << "],\"linear_rec_binders\":[";
  for (std::size_t i = 0; i < ast.linear_rec_binders.size(); ++i) {
    if (i > 0) out << ",";
    const g3pvm::evo::LinearRecBinders& binders = ast.linear_rec_binders[i];
    out << "{\"node_index\":" << binders.node_index
        << ",\"elem_name\":" << binders.elem_name
        << ",\"accum_name\":" << binders.accum_name
        << ",\"index_name\":" << binders.index_name << "}";
  }
  out << "],\"asgp_dc_binders\":[";
  for (std::size_t i = 0; i < ast.asgp_dc_binders.size(); ++i) {
    if (i > 0) out << ",";
    const g3pvm::evo::AsgpDcBinders& binders = ast.asgp_dc_binders[i];
    out << "{\"node_index\":" << binders.node_index
        << ",\"solve_xs_name\":" << binders.solve_xs_name
        << ",\"solve_n_name\":" << binders.solve_n_name
        << ",\"solve_lo_name\":" << binders.solve_lo_name
        << ",\"divide_n_name\":" << binders.divide_n_name
        << ",\"combine_left_name\":" << binders.combine_left_name
        << ",\"combine_right_name\":" << binders.combine_right_name << "}";
  }
  out << "],\"asgp_dp1d_specs\":[";
  for (std::size_t i = 0; i < ast.asgp_dp1d_specs.size(); ++i) {
    if (i > 0) out << ",";
    const g3pvm::evo::AsgpDp1dSpec& spec = ast.asgp_dp1d_specs[i];
    out << "{\"node_index\":" << spec.node_index
        << ",\"lo\":" << spec.lo
        << ",\"hi\":" << spec.hi
        << ",\"base_state\":" << spec.base_state
        << ",\"boundary_const\":" << spec.boundary_const
        << ",\"dep_kind\":" << static_cast<int>(spec.dep_kind)
        << ",\"dep_offsets\":[";
    for (std::size_t j = 0; j < spec.dep_offsets.size(); ++j) {
      if (j > 0) out << ",";
      out << spec.dep_offsets[j];
    }
    out << "],\"solve_state_name\":" << spec.solve_state_name
        << ",\"transition_state_name\":" << spec.transition_state_name
        << ",\"transition_dep_names\":[";
    for (std::size_t j = 0; j < spec.transition_dep_names.size(); ++j) {
      if (j > 0) out << ",";
      out << spec.transition_dep_names[j];
    }
    out << "]}";
  }
  out << "],\"asgp_dp2d_specs\":[";
  for (std::size_t i = 0; i < ast.asgp_dp2d_specs.size(); ++i) {
    if (i > 0) out << ",";
    const g3pvm::evo::AsgpDp2dSpec& spec = ast.asgp_dp2d_specs[i];
    out << "{\"node_index\":" << spec.node_index
        << ",\"i_lo\":" << spec.i_lo
        << ",\"i_hi\":" << spec.i_hi
        << ",\"j_lo\":" << spec.j_lo
        << ",\"j_hi\":" << spec.j_hi
        << ",\"base_i\":" << spec.base_i
        << ",\"base_j\":" << spec.base_j
        << ",\"boundary_const\":" << spec.boundary_const
        << ",\"dep_kind\":" << static_cast<int>(spec.dep_kind)
        << ",\"solve_i_name\":" << spec.solve_i_name
        << ",\"solve_j_name\":" << spec.solve_j_name
        << ",\"transition_i_name\":" << spec.transition_i_name
        << ",\"transition_j_name\":" << spec.transition_j_name
        << ",\"transition_dep_names\":[";
    for (std::size_t j = 0; j < spec.transition_dep_names.size(); ++j) {
      if (j > 0) out << ",";
      out << spec.transition_dep_names[j];
    }
    out << "]}";
  }
  out << "]}";
}

bool is_integer_number(double x) {
  const long long i = static_cast<long long>(x);
  return static_cast<double>(i) == x;
}

double canonicalize_metric(double x) {
  if (!std::isfinite(x) || x == 0.0) {
    return x;
  }
  int exp = 0;
  const double frac = std::frexp(x, &exp);
  constexpr int keep_mantissa_bits = 48;
  const double scaled = std::ldexp(frac, keep_mantissa_bits);
  return std::ldexp(std::nearbyint(scaled), exp - keep_mantissa_bits);
}

Value decode_typed_or_raw_value(const JsonValue& v, bool strict_format = false) {
  if (v.kind == JsonValue::Kind::Object) {
    auto it = v.object_v.find("type");
    if (it != v.object_v.end()) {
      if (it->second.kind != JsonValue::Kind::String) {
        throw std::runtime_error("typed value field type must be string");
      }
      if (strict_format &&
          (it->second.string_v == "none" || it->second.string_v == "num_list" || it->second.string_v == "list")) {
        throw std::runtime_error("fitness-cases rejects legacy typed value type: " + it->second.string_v);
      }
      return g3pvm::cli_detail::decode_typed_value(v);
    }
  }

  if (strict_format) {
    throw std::runtime_error("fitness-cases values must be explicitly typed");
  }

  if (v.kind == JsonValue::Kind::Null) {
    throw std::runtime_error("null is not a public current value");
  }
  if (v.kind == JsonValue::Kind::Bool) {
    return Value::from_bool(v.bool_v);
  }
  if (v.kind == JsonValue::Kind::Number) {
    if (is_integer_number(v.number_v)) {
      return Value::from_int(static_cast<long long>(v.number_v));
    }
    return Value::from_float(v.number_v);
  }
  throw std::runtime_error("unsupported raw value type");
}

g3pvm::evo::NamedInputs decode_inputs(const JsonValue& raw, bool strict_format = false) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("case.inputs must be an object");
  }
  g3pvm::evo::NamedInputs out;
  for (const auto& kv : raw.object_v) {
    out[kv.first] = decode_typed_or_raw_value(kv.second, strict_format);
  }
  return out;
}

int require_int_field_local(const JsonValue& raw, const char* key, const char* section) {
  auto it = raw.object_v.find(key);
  if (it == raw.object_v.end() || it->second.kind != JsonValue::Kind::Number ||
      !is_integer_number(it->second.number_v)) {
    throw std::runtime_error(std::string("expected integer field: ") + section + "." + key);
  }
  return static_cast<int>(it->second.number_v);
}

std::size_t require_node_index_field_local(const JsonValue& raw, const char* section) {
  const int index = require_int_field_local(raw, "node_index", section);
  if (index < 0) {
    throw std::runtime_error(std::string(section) + ".node_index must be non-negative");
  }
  return static_cast<std::size_t>(index);
}

std::vector<int> require_int_array_field_local(const JsonValue& raw,
                                               const char* key,
                                               const char* section) {
  auto it = raw.object_v.find(key);
  if (it == raw.object_v.end() || it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error(std::string("expected integer array field: ") + section + "." + key);
  }
  std::vector<int> out;
  out.reserve(it->second.array_v.size());
  for (const JsonValue& item : it->second.array_v) {
    if (item.kind != JsonValue::Kind::Number || !is_integer_number(item.number_v)) {
      throw std::runtime_error(std::string("expected integer elements: ") + section + "." + key);
    }
    out.push_back(static_cast<int>(item.number_v));
  }
  return out;
}

g3pvm::evo::AstProgram decode_ast_json_impl(const JsonValue& raw) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("AST JSON must be an object");
  }
  g3pvm::evo::AstProgram ast;
  auto version_it = raw.object_v.find("version");
  if (version_it == raw.object_v.end() || version_it->second.kind != JsonValue::Kind::String) {
    throw std::runtime_error("AST JSON missing string field: version");
  }
  ast.version = version_it->second.string_v;

  auto nodes_it = raw.object_v.find("nodes");
  if (nodes_it == raw.object_v.end() || nodes_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("AST JSON missing array field: nodes");
  }
  ast.nodes.reserve(nodes_it->second.array_v.size());
  for (const JsonValue& row : nodes_it->second.array_v) {
    if (row.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("AST node must be an object");
    }
    const int kind = require_int_field_local(row, "kind", "node");
    ast.nodes.push_back(g3pvm::evo::AstNode{
        static_cast<g3pvm::evo::NodeKind>(kind),
        require_int_field_local(row, "i0", "node"),
        require_int_field_local(row, "i1", "node"),
    });
  }

  auto names_it = raw.object_v.find("names");
  if (names_it == raw.object_v.end() || names_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("AST JSON missing array field: names");
  }
  ast.names.reserve(names_it->second.array_v.size());
  for (const JsonValue& item : names_it->second.array_v) {
    if (item.kind != JsonValue::Kind::String) {
      throw std::runtime_error("AST names must be strings");
    }
    ast.names.push_back(item.string_v);
  }

  auto consts_it = raw.object_v.find("consts");
  if (consts_it == raw.object_v.end() || consts_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("AST JSON missing array field: consts");
  }
  ast.consts.reserve(consts_it->second.array_v.size());
  for (const JsonValue& item : consts_it->second.array_v) {
    ast.consts.push_back(g3pvm::cli_detail::decode_typed_value(item));
  }

  auto binders_it = raw.object_v.find("linear_rec_binders");
  if (binders_it != raw.object_v.end()) {
    if (binders_it->second.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("AST linear_rec_binders must be an array");
    }
    ast.linear_rec_binders.reserve(binders_it->second.array_v.size());
    for (const JsonValue& row : binders_it->second.array_v) {
      if (row.kind != JsonValue::Kind::Object) {
        throw std::runtime_error("AST linear_rec_binders item must be an object");
      }
      ast.linear_rec_binders.push_back(g3pvm::evo::LinearRecBinders{
          require_node_index_field_local(row, "linear_rec_binders"),
          require_int_field_local(row, "elem_name", "linear_rec_binders"),
          require_int_field_local(row, "accum_name", "linear_rec_binders"),
          require_int_field_local(row, "index_name", "linear_rec_binders"),
      });
    }
  }

  auto dc_it = raw.object_v.find("asgp_dc_binders");
  if (dc_it != raw.object_v.end()) {
    if (dc_it->second.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("AST asgp_dc_binders must be an array");
    }
    for (const JsonValue& row : dc_it->second.array_v) {
      if (row.kind != JsonValue::Kind::Object) {
        throw std::runtime_error("AST asgp_dc_binders item must be an object");
      }
      ast.asgp_dc_binders.push_back(g3pvm::evo::AsgpDcBinders{
          require_node_index_field_local(row, "asgp_dc_binders"),
          require_int_field_local(row, "solve_xs_name", "asgp_dc_binders"),
          require_int_field_local(row, "solve_n_name", "asgp_dc_binders"),
          require_int_field_local(row, "solve_lo_name", "asgp_dc_binders"),
          require_int_field_local(row, "divide_n_name", "asgp_dc_binders"),
          require_int_field_local(row, "combine_left_name", "asgp_dc_binders"),
          require_int_field_local(row, "combine_right_name", "asgp_dc_binders"),
      });
    }
  }

  auto dp1_it = raw.object_v.find("asgp_dp1d_specs");
  if (dp1_it != raw.object_v.end()) {
    if (dp1_it->second.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("AST asgp_dp1d_specs must be an array");
    }
    for (const JsonValue& row : dp1_it->second.array_v) {
      if (row.kind != JsonValue::Kind::Object) {
        throw std::runtime_error("AST asgp_dp1d_specs item must be an object");
      }
      ast.asgp_dp1d_specs.push_back(g3pvm::evo::AsgpDp1dSpec{
          require_node_index_field_local(row, "asgp_dp1d_specs"),
          require_int_field_local(row, "lo", "asgp_dp1d_specs"),
          require_int_field_local(row, "hi", "asgp_dp1d_specs"),
          require_int_field_local(row, "base_state", "asgp_dp1d_specs"),
          require_int_field_local(row, "boundary_const", "asgp_dp1d_specs"),
          static_cast<g3pvm::evo::NodeKind>(
              require_int_field_local(row, "dep_kind", "asgp_dp1d_specs")),
          require_int_array_field_local(row, "dep_offsets", "asgp_dp1d_specs"),
          require_int_field_local(row, "solve_state_name", "asgp_dp1d_specs"),
          require_int_field_local(row, "transition_state_name", "asgp_dp1d_specs"),
          require_int_array_field_local(row, "transition_dep_names", "asgp_dp1d_specs"),
      });
    }
  }

  auto dp2_it = raw.object_v.find("asgp_dp2d_specs");
  if (dp2_it != raw.object_v.end()) {
    if (dp2_it->second.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("AST asgp_dp2d_specs must be an array");
    }
    for (const JsonValue& row : dp2_it->second.array_v) {
      if (row.kind != JsonValue::Kind::Object) {
        throw std::runtime_error("AST asgp_dp2d_specs item must be an object");
      }
      ast.asgp_dp2d_specs.push_back(g3pvm::evo::AsgpDp2dSpec{
          require_node_index_field_local(row, "asgp_dp2d_specs"),
          require_int_field_local(row, "i_lo", "asgp_dp2d_specs"),
          require_int_field_local(row, "i_hi", "asgp_dp2d_specs"),
          require_int_field_local(row, "j_lo", "asgp_dp2d_specs"),
          require_int_field_local(row, "j_hi", "asgp_dp2d_specs"),
          require_int_field_local(row, "base_i", "asgp_dp2d_specs"),
          require_int_field_local(row, "base_j", "asgp_dp2d_specs"),
          require_int_field_local(row, "boundary_const", "asgp_dp2d_specs"),
          static_cast<g3pvm::evo::NodeKind>(
              require_int_field_local(row, "dep_kind", "asgp_dp2d_specs")),
          require_int_field_local(row, "solve_i_name", "asgp_dp2d_specs"),
          require_int_field_local(row, "solve_j_name", "asgp_dp2d_specs"),
          require_int_field_local(row, "transition_i_name", "asgp_dp2d_specs"),
          require_int_field_local(row, "transition_j_name", "asgp_dp2d_specs"),
          require_int_array_field_local(row, "transition_dep_names", "asgp_dp2d_specs"),
      });
    }
  }
  return ast;
}

bool paths_match(const std::string& lhs, const std::string& rhs) {
  if (lhs == rhs) {
    return true;
  }
  try {
    const auto lhs_path = std::filesystem::absolute(std::filesystem::path(lhs)).lexically_normal();
    const auto rhs_path = std::filesystem::absolute(std::filesystem::path(rhs)).lexically_normal();
    return lhs_path == rhs_path;
  } catch (const std::exception&) {
    return false;
  }
}

std::string fnv1a64_hex(const std::string& text) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : text) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  std::ostringstream oss;
  oss << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

const JsonValue& require_object_section(const JsonValue& raw, const char* key) {
  auto it = raw.object_v.find(key);
  if (it == raw.object_v.end() || it->second.kind != JsonValue::Kind::Object) {
    throw std::runtime_error(std::string("grammar config missing object section: ") + key);
  }
  return it->second;
}

void reject_unknown_fields(const JsonValue& raw, const std::vector<std::string>& allowed, const char* section) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error(std::string("grammar config section is not an object: ") + section);
  }
  for (const auto& kv : raw.object_v) {
    if (std::find(allowed.begin(), allowed.end(), kv.first) == allowed.end()) {
      throw std::runtime_error(std::string("grammar config unknown field: ") + section + "." + kv.first);
    }
  }
}

bool require_bool_field(const JsonValue& raw, const char* key, const char* section) {
  auto it = raw.object_v.find(key);
  if (it == raw.object_v.end() || it->second.kind != JsonValue::Kind::Bool) {
    throw std::runtime_error(std::string("grammar config expected boolean field: ") + section + "." + key);
  }
  return it->second.bool_v;
}

void require_bool_fields(const JsonValue& raw, const std::vector<std::string>& keys, const char* section) {
  for (const std::string& key : keys) {
    (void)require_bool_field(raw, key.c_str(), section);
  }
}

const JsonValue* optional_object_section(const JsonValue& raw, const char* key, const char* owner) {
  auto it = raw.object_v.find(key);
  if (it == raw.object_v.end()) {
    return nullptr;
  }
  if (it->second.kind != JsonValue::Kind::Object) {
    throw std::runtime_error(std::string("grammar config expected object field: ") + owner + "." + key);
  }
  return &it->second;
}

void validate_optional_metadata(const JsonValue& payload) {
  if (const JsonValue* structured = optional_object_section(payload, "structured", "root")) {
    reject_unknown_fields(*structured,
                          {"max_nested_binders", "max_map_body_depth", "max_filter_pred_depth",
                           "max_linear_rec_body_depth"},
                          "structured");
  }
  if (const JsonValue* limits = optional_object_section(payload, "limits", "root")) {
    reject_unknown_fields(*limits,
                          {"max_expr_depth", "max_stmts_per_block", "max_total_nodes", "max_for_k",
                           "max_call_args"},
                          "limits");
  }
  if (const JsonValue* asgp = optional_object_section(payload, "asgp", "root")) {
    reject_unknown_fields(*asgp, {"max_scheme_nesting", "dc", "dp1d", "dp2d"}, "asgp");
    if (const JsonValue* dc = optional_object_section(*asgp, "dc", "asgp")) {
      reject_unknown_fields(*dc, {"enabled_source_elems", "max_depth"}, "asgp.dc");
    }
    if (const JsonValue* dp1d = optional_object_section(*asgp, "dp1d", "asgp")) {
      reject_unknown_fields(*dp1d, {"max_states", "max_step", "dependency_patterns"}, "asgp.dp1d");
    }
    if (const JsonValue* dp2d = optional_object_section(*asgp, "dp2d", "asgp")) {
      reject_unknown_fields(*dp2d, {"max_cells", "dependency_patterns"}, "asgp.dp2d");
    }
  }
  auto compat_it = payload.object_v.find("compat");
  if (compat_it != payload.object_v.end() &&
      compat_it->second.kind != JsonValue::Kind::Null &&
      compat_it->second.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("grammar config expected root.compat to be null or object");
  }
}

bool config_requests_legacy_num_list_input_compat(const JsonValue& payload) {
  auto compat_it = payload.object_v.find("compat");
  if (compat_it == payload.object_v.end() || compat_it->second.kind == JsonValue::Kind::Null) {
    return false;
  }
  const JsonValue& compat = compat_it->second;
  if (compat.kind != JsonValue::Kind::Object) {
    return false;
  }
  auto mode_it = compat.object_v.find("mode");
  auto num_list_mode_it = compat.object_v.find("num_list_mode");
  if (mode_it == compat.object_v.end() || num_list_mode_it == compat.object_v.end() ||
      mode_it->second.kind != JsonValue::Kind::String ||
      num_list_mode_it->second.kind != JsonValue::Kind::String) {
    return false;
  }
  return mode_it->second.string_v == "compact" &&
         num_list_mode_it->second.string_v == "both";
}

g3pvm::evo::GrammarConfig parse_grammar_config_current_payload(const JsonValue& payload) {
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("grammar config must be a JSON object");
  }
  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "grammar-config") {
    throw std::runtime_error("grammar config must include format_version=grammar-config");
  }

  const JsonValue& statements = require_object_section(payload, "statements");
  const JsonValue& expressions = require_object_section(payload, "expressions");
  const JsonValue& builtins = require_object_section(payload, "builtins");
  const JsonValue& values = require_object_section(payload, "values");

  reject_unknown_fields(payload,
                        {"format_version", "profile", "statements", "expressions", "builtins", "values",
                         "structured", "asgp", "limits", "compat"},
                        "root");
  reject_unknown_fields(statements, {"assign", "if_stmt", "for_range", "return"}, "statements");
  reject_unknown_fields(expressions,
                        {"const", "var", "bound_var", "unary", "binary", "if_expr", "call",
                         "map_list", "filter_list", "linear_rec", "asgp_dc", "asgp_dp1d", "asgp_dp2d"},
                        "expressions");
  reject_unknown_fields(builtins,
                        {"abs", "min", "max", "clip", "idiv0", "imod0", "len", "concat", "slice",
                         "index", "append", "prepend", "reverse", "find", "contains", "singleton",
                         "char_to_string", "string_to_char", "ord", "chr", "is_letter", "is_digit",
                         "is_space", "is_vowel", "to_lower", "to_upper", "to_string"},
                        "builtins");
  reject_unknown_fields(values,
                        {"int", "float", "bool", "char", "string", "int_list", "float_list",
                         "string_list"},
                        "values");
  validate_optional_metadata(payload);

  g3pvm::evo::GrammarConfig cfg;
  cfg.statement_assign = require_bool_field(statements, "assign", "statements");
  cfg.statement_if_stmt = require_bool_field(statements, "if_stmt", "statements");
  cfg.statement_for_range = require_bool_field(statements, "for_range", "statements");
  cfg.statement_return = require_bool_field(statements, "return", "statements");

  cfg.expression_const = require_bool_field(expressions, "const", "expressions");
  cfg.expression_var = require_bool_field(expressions, "var", "expressions");
  (void)require_bool_field(expressions, "bound_var", "expressions");
  const bool unary_enabled = require_bool_field(expressions, "unary", "expressions");
  const bool binary_enabled = require_bool_field(expressions, "binary", "expressions");
  cfg.expression_if_expr = require_bool_field(expressions, "if_expr", "expressions");
  const bool call_enabled = require_bool_field(expressions, "call", "expressions");
  cfg.expression_map_list = require_bool_field(expressions, "map_list", "expressions");
  cfg.expression_filter_list = require_bool_field(expressions, "filter_list", "expressions");
  cfg.expression_linear_rec = require_bool_field(expressions, "linear_rec", "expressions");
  cfg.expression_asgp_dc = require_bool_field(expressions, "asgp_dc", "expressions");
  cfg.expression_asgp_dp1d = require_bool_field(expressions, "asgp_dp1d", "expressions");
  cfg.expression_asgp_dp2d = require_bool_field(expressions, "asgp_dp2d", "expressions");

  cfg.unary_neg = unary_enabled;
  cfg.unary_not = unary_enabled;
  cfg.binary_add = binary_enabled;
  cfg.binary_sub = binary_enabled;
  cfg.binary_mul = binary_enabled;
  cfg.binary_div = binary_enabled;
  cfg.binary_mod = binary_enabled;
  cfg.binary_lt = binary_enabled;
  cfg.binary_le = binary_enabled;
  cfg.binary_gt = binary_enabled;
  cfg.binary_ge = binary_enabled;
  cfg.binary_eq = binary_enabled;
  cfg.binary_ne = binary_enabled;
  cfg.binary_and = binary_enabled;
  cfg.binary_or = binary_enabled;

  const bool builtin_abs = require_bool_field(builtins, "abs", "builtins");
  const bool builtin_min = require_bool_field(builtins, "min", "builtins");
  const bool builtin_max = require_bool_field(builtins, "max", "builtins");
  const bool builtin_clip = require_bool_field(builtins, "clip", "builtins");
  const bool builtin_idiv0 = require_bool_field(builtins, "idiv0", "builtins");
  const bool builtin_imod0 = require_bool_field(builtins, "imod0", "builtins");
  const bool builtin_len = require_bool_field(builtins, "len", "builtins");
  const bool builtin_concat = require_bool_field(builtins, "concat", "builtins");
  const bool builtin_slice = require_bool_field(builtins, "slice", "builtins");
  const bool builtin_index = require_bool_field(builtins, "index", "builtins");
  const bool builtin_append = require_bool_field(builtins, "append", "builtins");
  const bool builtin_prepend = require_bool_field(builtins, "prepend", "builtins");
  const bool builtin_reverse = require_bool_field(builtins, "reverse", "builtins");
  const bool builtin_find = require_bool_field(builtins, "find", "builtins");
  const bool builtin_contains = require_bool_field(builtins, "contains", "builtins");
  const bool builtin_singleton = require_bool_field(builtins, "singleton", "builtins");
  const bool builtin_char_to_string = require_bool_field(builtins, "char_to_string", "builtins");
  const bool builtin_string_to_char = require_bool_field(builtins, "string_to_char", "builtins");
  const bool builtin_ord = require_bool_field(builtins, "ord", "builtins");
  const bool builtin_chr = require_bool_field(builtins, "chr", "builtins");
  const bool builtin_is_letter = require_bool_field(builtins, "is_letter", "builtins");
  const bool builtin_is_digit = require_bool_field(builtins, "is_digit", "builtins");
  const bool builtin_is_space = require_bool_field(builtins, "is_space", "builtins");
  const bool builtin_is_vowel = require_bool_field(builtins, "is_vowel", "builtins");
  const bool builtin_to_lower = require_bool_field(builtins, "to_lower", "builtins");
  const bool builtin_to_upper = require_bool_field(builtins, "to_upper", "builtins");
  const bool builtin_to_string = require_bool_field(builtins, "to_string", "builtins");
  cfg.builtin_abs = call_enabled && builtin_abs;
  cfg.builtin_min = call_enabled && builtin_min;
  cfg.builtin_max = call_enabled && builtin_max;
  cfg.builtin_clip = call_enabled && builtin_clip;
  cfg.builtin_idiv0 = call_enabled && builtin_idiv0;
  cfg.builtin_imod0 = call_enabled && builtin_imod0;
  cfg.builtin_len = call_enabled && builtin_len;
  cfg.builtin_concat = call_enabled && builtin_concat;
  cfg.builtin_slice = call_enabled && builtin_slice;
  cfg.builtin_index = call_enabled && builtin_index;
  cfg.builtin_append = call_enabled && builtin_append;
  cfg.builtin_prepend = call_enabled && builtin_prepend;
  cfg.builtin_reverse = call_enabled && builtin_reverse;
  cfg.builtin_find = call_enabled && builtin_find;
  cfg.builtin_contains = call_enabled && builtin_contains;
  cfg.builtin_singleton = call_enabled && builtin_singleton;
  cfg.builtin_char_to_string = call_enabled && builtin_char_to_string;
  cfg.builtin_string_to_char = call_enabled && builtin_string_to_char;
  cfg.builtin_ord = call_enabled && builtin_ord;
  cfg.builtin_chr = call_enabled && builtin_chr;
  cfg.builtin_is_letter = call_enabled && builtin_is_letter;
  cfg.builtin_is_digit = call_enabled && builtin_is_digit;
  cfg.builtin_is_space = call_enabled && builtin_is_space;
  cfg.builtin_is_vowel = call_enabled && builtin_is_vowel;
  cfg.builtin_to_lower = call_enabled && builtin_to_lower;
  cfg.builtin_to_upper = call_enabled && builtin_to_upper;
  cfg.builtin_to_string = call_enabled && builtin_to_string;

  cfg.value_int = require_bool_field(values, "int", "values");
  cfg.value_float = require_bool_field(values, "float", "values");
  cfg.value_bool = require_bool_field(values, "bool", "values");
  cfg.value_char = require_bool_field(values, "char", "values");
  cfg.value_string = require_bool_field(values, "string", "values");
  cfg.value_int_list = require_bool_field(values, "int_list", "values");
  cfg.value_float_list = require_bool_field(values, "float_list", "values");
  cfg.value_string_list = require_bool_field(values, "string_list", "values");
  cfg.compat_legacy_num_list_inputs_as_any =
      config_requests_legacy_num_list_input_compat(payload);
  cfg.validate();
  return cfg;
}

g3pvm::evo::GrammarConfig parse_grammar_config_payload(const JsonValue& payload) {
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("grammar config must be a JSON object");
  }
  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String) {
    throw std::runtime_error("grammar config must include format_version=grammar-config");
  }
  return parse_grammar_config_current_payload(payload);
}

std::vector<g3pvm::evo::EvalCase> parse_cases(const JsonValue& payload) {
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("input JSON must be object");
  }

  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String) {
    throw std::runtime_error("input JSON must include format_version=fitness-cases");
  }
  if (fv_it->second.string_v != "fitness-cases") {
    throw std::runtime_error("input JSON must include format_version=fitness-cases");
  }

  auto cases_it = payload.object_v.find("cases");
  if (cases_it == payload.object_v.end() || cases_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input JSON must include list field: cases");
  }

  std::vector<g3pvm::evo::EvalCase> out;
  out.reserve(cases_it->second.array_v.size());
  for (const JsonValue& row : cases_it->second.array_v) {
    if (row.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("cases[i] must be object");
    }
    auto inputs_it = row.object_v.find("inputs");
    auto expected_it = row.object_v.find("expected");
    if (inputs_it == row.object_v.end() || expected_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] must include inputs/expected");
    }
    out.push_back(g3pvm::evo::EvalCase{decode_inputs(inputs_it->second, true),
                                          decode_typed_or_raw_value(expected_it->second, true)});
  }
  if (out.empty()) {
    throw std::runtime_error("cases must not be empty");
  }
  return out;
}

std::string read_text_file(const std::string& path);

g3pvm::evo::Limits parse_limits_object(const JsonValue& raw) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("population seed set limits must be an object");
  }
  auto read_int = [&](const char* key) -> int {
    auto it = raw.object_v.find(key);
    if (it == raw.object_v.end() || it->second.kind != JsonValue::Kind::Number) {
      throw std::runtime_error(std::string("population seed set missing numeric limits.") + key);
    }
    return static_cast<int>(it->second.number_v);
  };
  return g3pvm::evo::Limits{
      read_int("max_expr_depth"),
      read_int("max_stmts_per_block"),
      read_int("max_total_nodes"),
      read_int("max_for_k"),
      read_int("max_call_args"),
  };
}

struct LoadedPopulation {
  g3pvm::evo::Limits limits;
  std::vector<std::uint64_t> seeds;
  std::vector<g3pvm::evo::ProgramGenome> genomes;
};

LoadedPopulation load_population_from_seed_set(const std::string& population_json,
                                               const std::string& cases_path,
                                               const g3pvm::evo::GrammarConfig& grammar,
                                               const std::string& grammar_config_path,
                                               const std::string& grammar_config_hash) {
  const JsonValue payload = g3pvm::cli_detail::JsonParser(read_text_file(population_json)).parse();
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("population seed set must be a JSON object");
  }

  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "population-seeds") {
    throw std::runtime_error("population seed set must include format_version=population-seeds");
  }

  auto cases_it = payload.object_v.find("cases_path");
  if (cases_it != payload.object_v.end() && cases_it->second.kind == JsonValue::Kind::String &&
      !cases_it->second.string_v.empty() && !paths_match(cases_it->second.string_v, cases_path)) {
    throw std::runtime_error("population seed set cases_path does not match --cases");
  }

  auto grammar_it = payload.object_v.find("grammar_config");
  if (grammar_it != payload.object_v.end() && grammar_it->second.kind == JsonValue::Kind::Object) {
    const JsonValue& seed_grammar = grammar_it->second;
    std::string seed_hash;
    std::string seed_path;
    auto hash_it = seed_grammar.object_v.find("hash");
    if (hash_it != seed_grammar.object_v.end() && hash_it->second.kind == JsonValue::Kind::String) {
      seed_hash = hash_it->second.string_v;
    }
    auto path_it = seed_grammar.object_v.find("path");
    if (path_it != seed_grammar.object_v.end() && path_it->second.kind == JsonValue::Kind::String) {
      seed_path = path_it->second.string_v;
    }
    if (!seed_hash.empty()) {
      if (grammar_config_hash.empty()) {
        throw std::runtime_error("population seed set requires matching --grammar-config hash");
      }
      if (seed_hash != grammar_config_hash) {
        throw std::runtime_error("population seed set grammar_config.hash does not match --grammar-config");
      }
    } else if (!seed_path.empty()) {
      if (grammar_config_path.empty() || !paths_match(seed_path, grammar_config_path)) {
        throw std::runtime_error("population seed set grammar_config.path does not match --grammar-config");
      }
    }
  }

  auto limits_it = payload.object_v.find("limits");
  if (limits_it == payload.object_v.end()) {
    throw std::runtime_error("population seed set missing limits");
  }
  auto seeds_it = payload.object_v.find("seeds");
  if (seeds_it == payload.object_v.end() || seeds_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("population seed set missing seeds array");
  }

  LoadedPopulation out;
  out.limits = parse_limits_object(limits_it->second);
  out.seeds.reserve(seeds_it->second.array_v.size());
  out.genomes.reserve(seeds_it->second.array_v.size());
  for (const JsonValue& row : seeds_it->second.array_v) {
    if (row.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("population seed set seeds[i] must be object");
    }
    auto seed_it = row.object_v.find("seed");
    if (seed_it == row.object_v.end() || seed_it->second.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("population seed set seeds[i] missing numeric seed");
    }
    const std::uint64_t seed = static_cast<std::uint64_t>(seed_it->second.number_v);
    out.seeds.push_back(seed);
    out.genomes.push_back(g3pvm::evo::generate_random_genome(seed, out.limits, grammar));
  }
  if (out.genomes.empty()) {
    throw std::runtime_error("population seed set must contain at least one seed");
  }
  return out;
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("missing input file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

struct LoadedCommandInputs {
  g3pvm::evo::GrammarConfig grammar;
  std::string grammar_hash;
  std::vector<g3pvm::evo::EvalCase> cases;
};

LoadedCommandInputs load_command_inputs(const CliOptions& args) {
  LoadedCommandInputs inputs;
  if (!args.grammar_config_path.empty()) {
    const std::string grammar_text = read_text_file(args.grammar_config_path);
    inputs.grammar =
        parse_grammar_config_payload(g3pvm::cli_detail::JsonParser(grammar_text).parse());
    inputs.grammar_hash = fnv1a64_hex(grammar_text);
  }

  const std::string cases_text = read_text_file(args.cases_path);
  inputs.cases =
      parse_cases(g3pvm::cli_detail::JsonParser(cases_text).parse());
  return inputs;
}

g3pvm::evo::EvolutionConfig make_evolution_config(
    const CliOptions& args, const g3pvm::evo::GrammarConfig& grammar) {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = args.population_size;
  cfg.generations = args.generations;
  cfg.mutation_rate = args.mutation_rate;
  cfg.mutation_subtree_prob = args.mutation_subtree_prob;
  cfg.penalty = args.penalty;
  cfg.eval_engine =
      (args.engine == "gpu") ? g3pvm::evo::EvalEngine::GPU : g3pvm::evo::EvalEngine::CPU;
  cfg.reproduction_backend =
      g3pvm::evo::repro::parse_reproduction_backend_name(args.repro_backend);
  cfg.cpu_repro_ablation =
      g3pvm::evo::repro::parse_cpu_repro_ablation_name(args.cpu_repro_ablation);
  cfg.repro_overlap = args.repro_overlap;
  cfg.gpu_blocksize = args.blocksize;
  cfg.selection_pressure = args.selection_pressure;
  cfg.seed = args.seed;
  cfg.fuel = args.fuel;
  cfg.skip_final_eval = args.skip_final_eval;
  cfg.retain_final_population = args.retain_final_population;
  cfg.grammar = grammar;
  return cfg;
}

}  // namespace

g3pvm::evo::AstProgram g3pvm::cli_detail::decode_ast_json(const JsonValue& raw) {
  return decode_ast_json_impl(raw);
}

int g3pvm::cli_detail::run_eval_ast_command(const CliOptions& args) {
      const LoadedCommandInputs inputs = load_command_inputs(args);
      const g3pvm::evo::EvolutionConfig cfg = make_evolution_config(args, inputs.grammar);
      if (cfg.eval_engine != g3pvm::evo::EvalEngine::CPU) {
        throw std::runtime_error("--eval-ast-json currently supports --engine cpu only");
      }
      const JsonValue ast_payload = g3pvm::cli_detail::JsonParser(read_text_file(args.eval_ast_json)).parse();
      g3pvm::evo::ProgramGenome genome;
      genome.ast = decode_ast_json(ast_payload);
      const g3pvm::evo::AstVerifyResult verified = g3pvm::evo::verify_ast(
          genome.ast, g3pvm::evo::canonical_input_specs(inputs.cases, cfg.grammar));
      if (!verified) {
        throw std::runtime_error(
            std::string("invalid AST (") +
            g3pvm::evo::verify_code_name(verified.diagnostic.code) + ") at " +
            verified.diagnostic.path + ": " + verified.diagnostic.message);
      }
      genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
      const std::vector<g3pvm::evo::ScoredGenome> scored =
          g3pvm::evo::evaluate_population({genome}, inputs.cases, cfg);
      if (scored.empty()) {
        throw std::runtime_error("AST evaluation produced no score");
      }
      const double fitness = canonicalize_metric(scored[0].fitness);
      std::cout << "AST_EVAL fitness=" << std::fixed << std::setprecision(6)
                << fitness << " program_key=" << scored[0].genome.meta.program_key << "\n";
      if (!args.out_json.empty()) {
        std::ofstream out(args.out_json);
        if (!out) {
          throw std::runtime_error("failed to open out-json path");
        }
        out << "{\n";
        out << "  \"format_version\": \"ast-eval-result\",\n";
        out << "  \"meta\": {\n";
        out << "    \"cases_path\": \"" << json_escape(args.cases_path) << "\",\n";
        out << "    \"ast_json\": \"" << json_escape(args.eval_ast_json) << "\",\n";
        out << "    \"eval_engine\": \"" << g3pvm::evo::eval_engine_name(cfg.eval_engine) << "\",\n";
        out << "    \"fuel\": " << cfg.fuel << ",\n";
        out << "    \"penalty\": " << std::setprecision(17) << cfg.penalty << "\n";
        out << "  },\n";
        out << "  \"result\": {\n";
        out << "    \"fitness\": " << std::setprecision(17) << scored[0].fitness << ",\n";
        out << "    \"program_key\": \"" << json_escape(scored[0].genome.meta.program_key) << "\"\n";
        out << "  }\n";
        out << "}\n";
      }
      return 0;
}

int g3pvm::cli_detail::run_evolve_command(const CliOptions& args) {
    const LoadedCommandInputs inputs = load_command_inputs(args);
    g3pvm::evo::EvolutionConfig cfg = make_evolution_config(args, inputs.grammar);
    const std::vector<g3pvm::evo::EvalCase>& cases = inputs.cases;
    const std::string& grammar_config_hash = inputs.grammar_hash;

    std::vector<g3pvm::evo::ProgramGenome> initial_population;
    const std::vector<g3pvm::evo::ProgramGenome>* initial_population_ptr = nullptr;
    std::string population_source = "generated";
    if (!args.population_json.empty()) {
      LoadedPopulation loaded = load_population_from_seed_set(args.population_json,
                                                              args.cases_path,
                                                              cfg.grammar,
                                                              args.grammar_config_path,
                                                              grammar_config_hash);
      cfg.population_size = static_cast<int>(loaded.genomes.size());
      cfg.limits = loaded.limits;
      initial_population = std::move(loaded.genomes);
      initial_population_ptr = &initial_population;
      population_source = "population_json";
    } else {
      cfg.limits = g3pvm::evo::Limits{args.max_expr_depth,
                                      args.max_stmts_per_block,
                                      args.max_total_nodes,
                                      args.max_for_k,
                                      args.max_call_args};
    }

    const g3pvm::evo::EvolutionResult result =
        g3pvm::evo::evolve_population(cases, cfg, initial_population_ptr);
    const char* selection_label = "round_based_tournament";
    const char* crossover_label = "typed_subtree";

    struct HistoryRow {
      int generation = 0;
      double best_fitness = 0.0;
      double mean_fitness = 0.0;
      std::string program_key;
    };
    std::vector<HistoryRow> history_rows;
    history_rows.reserve(result.history_best.size());

    for (int i = 0; i < static_cast<int>(result.history_best.size()); ++i) {
      const auto& best = result.history_best[static_cast<std::size_t>(i)];
      const double best_fit = canonicalize_metric(result.history_best_fitness[static_cast<std::size_t>(i)]);
      const double mean_fit = canonicalize_metric(result.history_mean_fitness[static_cast<std::size_t>(i)]);
      const std::string& program_key = best.genome.meta.program_key;
      history_rows.push_back(HistoryRow{i, best_fit, mean_fit, program_key});

      std::cout << "GEN " << std::setfill('0') << std::setw(3) << i << std::setfill(' ') << " best="
                << std::fixed << std::setprecision(6) << best_fit << " mean=" << std::fixed
                << std::setprecision(6) << mean_fit << " program_key=" << program_key << "\n";

      if (args.show_program == "ast" || args.show_program == "both") {
        std::cout << "AST " << std::setfill('0') << std::setw(3) << i << std::setfill(' ') << ": "
                  << g3pvm::evo::ast_to_string(best.genome.ast) << "\n";
      }
      if (args.show_program == "bytecode" || args.show_program == "both") {
        const g3pvm::BytecodeProgram bc = g3pvm::evo::compile_for_eval(best.genome);
        std::cout << "BYTECODE " << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                  << ": n_locals=" << bc.n_locals << " consts=" << bc.consts.size() << " code="
                  << bc.code.size() << "\n";

        std::cout << "BYTECODE_HEAD";
        const std::size_t cap = std::min<std::size_t>(12, bc.code.size());
        for (std::size_t j = 0; j < cap; ++j) {
          std::cout << " " << j << ":" << g3pvm::opcode_name(bc.code[j].op);
        }
        std::cout << "\n";
      }
    }

    if (result.final_eval_skipped) {
      const HistoryRow& last = history_rows.back();
      std::cout << "FINAL skipped=true last_history_best=" << std::fixed << std::setprecision(6)
                << canonicalize_metric(last.best_fitness)
                << " program_key=" << last.program_key
                << " repro_backend=" << g3pvm::evo::repro::reproduction_backend_name(cfg.reproduction_backend)
                << " cpu_repro_ablation=" << g3pvm::evo::repro::cpu_repro_ablation_name(cfg.cpu_repro_ablation)
                << " repro_overlap=" << (cfg.repro_overlap ? "on" : "off")
                << " selection=" << selection_label
                << " crossover=" << crossover_label << "\n";
    } else {
      std::cout << "FINAL best=" << std::fixed << std::setprecision(6) << canonicalize_metric(result.best.fitness)
                << " program_key=" << result.best.genome.meta.program_key
                << " repro_backend=" << g3pvm::evo::repro::reproduction_backend_name(cfg.reproduction_backend)
                << " cpu_repro_ablation=" << g3pvm::evo::repro::cpu_repro_ablation_name(cfg.cpu_repro_ablation)
                << " repro_overlap=" << (cfg.repro_overlap ? "on" : "off")
                << " selection=" << selection_label
                << " crossover=" << crossover_label << "\n";
    }

    if (args.timing == "summary" || args.timing == "all") {
      double gen_eval_sum = 0.0;
      double gen_repro_sum = 0.0;
      for (const auto& generation : result.timing.generations) {
        gen_eval_sum += generation.eval_ms;
        gen_repro_sum += generation.repro_ms;
      }
      const auto& eval = result.timing.evaluation_totals;
      const auto& repro = result.timing.reproduction_totals;
      std::cout << "TIMING phase=init_population ms=" << std::fixed << std::setprecision(3)
                << result.timing.init_population_ms << "\n";
      std::cout << "TIMING phase=generations_eval_total ms=" << std::fixed << std::setprecision(3)
                << gen_eval_sum << "\n";
      std::cout << "TIMING phase=generations_repro_total ms=" << std::fixed << std::setprecision(3)
                << gen_repro_sum << "\n";
      std::cout << "TIMING phase=generations_selection_total ms=" << std::fixed << std::setprecision(3)
                << repro.selection_ms << "\n";
      std::cout << "TIMING phase=generations_crossover_total ms=" << std::fixed << std::setprecision(3)
                << repro.crossover_ms << "\n";
      std::cout << "TIMING phase=generations_mutation_total ms=" << std::fixed << std::setprecision(3)
                << repro.mutation_ms << "\n";
      std::cout << "TIMING phase=generations_repro_prepare_inputs_total ms=" << std::fixed
                << std::setprecision(3) << repro.prepare_inputs_ms << "\n";
      std::cout << "TIMING phase=generations_repro_setup_total ms=" << std::fixed << std::setprecision(3)
                << repro.setup_ms << "\n";
      std::cout << "TIMING phase=generations_repro_preprocess_total ms=" << std::fixed
                << std::setprecision(3) << repro.preprocess_ms << "\n";
      std::cout << "TIMING phase=generations_repro_pack_total ms=" << std::fixed << std::setprecision(3)
                << repro.pack_ms << "\n";
      std::cout << "TIMING phase=generations_repro_upload_total ms=" << std::fixed << std::setprecision(3)
                << repro.upload_ms << "\n";
      std::cout << "TIMING phase=generations_repro_kernel_total ms=" << std::fixed << std::setprecision(3)
                << repro.kernel_ms << "\n";
      std::cout << "TIMING phase=generations_repro_copyback_total ms=" << std::fixed
                << std::setprecision(3) << repro.copyback_ms << "\n";
      std::cout << "TIMING phase=generations_repro_decode_total ms=" << std::fixed
                << std::setprecision(3) << repro.decode_ms << "\n";
      std::cout << "TIMING phase=generations_repro_teardown_total ms=" << std::fixed
                << std::setprecision(3) << repro.teardown_ms << "\n";
      std::cout << "TIMING phase=generations_repro_selection_kernel_total ms=" << std::fixed
                << std::setprecision(3) << repro.selection_kernel_ms << "\n";
      std::cout << "TIMING phase=generations_repro_variation_kernel_total ms=" << std::fixed
                << std::setprecision(3) << repro.variation_kernel_ms << "\n";
      std::cout << "TIMING phase=cpu_compile_total ms=" << std::fixed << std::setprecision(3)
                << eval.cpu_compile_ms << "\n";
      std::cout << "TIMING phase=final_eval ms=" << std::fixed << std::setprecision(3)
                << result.timing.final_eval_ms << "\n";
      if (cfg.eval_engine == g3pvm::evo::EvalEngine::GPU) {
        std::cout << "TIMING phase=gpu_eval_init ms=" << std::fixed << std::setprecision(3)
                  << result.timing.gpu_eval_init_ms << "\n";
        std::cout << "TIMING phase=gpu_compile_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_compile_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_call_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_call_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_pack_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_pack_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_launch_prep_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_launch_prep_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_upload_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_upload_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_pack_upload_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_pack_upload_ms() << "\n";
        std::cout << "TIMING phase=gpu_eval_kernel_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_kernel_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_copyback_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_copyback_ms << "\n";
        std::cout << "TIMING phase=gpu_eval_teardown_total ms=" << std::fixed << std::setprecision(3)
                  << eval.gpu_eval_teardown_ms << "\n";
      }
      std::cout << "TIMING phase=total ms=" << std::fixed << std::setprecision(3)
                << result.timing.total_ms << "\n";
    }

    if (args.timing == "per_gen" || args.timing == "all") {
      for (std::size_t i = 0; i < result.timing.generations.size(); ++i) {
        const auto& generation = result.timing.generations[i];
        const auto& eval = generation.evaluation;
        const auto& repro = generation.reproduction;
        std::cout << "TIMING gen=" << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                  << " eval_ms=" << std::fixed << std::setprecision(3) << generation.eval_ms
                  << " repro_ms=" << generation.repro_ms
                  << " total_ms=" << generation.total_ms
                  << " selection_ms=" << repro.selection_ms
                  << " crossover_ms=" << repro.crossover_ms
                  << " mutation_ms=" << repro.mutation_ms
                  << " repro_prepare_inputs_ms=" << repro.prepare_inputs_ms
                  << " repro_setup_ms=" << repro.setup_ms
                  << " repro_preprocess_ms=" << repro.preprocess_ms
                  << " repro_pack_ms=" << repro.pack_ms
                  << " repro_upload_ms=" << repro.upload_ms
                  << " repro_kernel_ms=" << repro.kernel_ms
                  << " repro_copyback_ms=" << repro.copyback_ms
                  << " repro_decode_ms=" << repro.decode_ms
                  << " repro_teardown_ms=" << repro.teardown_ms
                  << " repro_selection_kernel_ms=" << repro.selection_kernel_ms
                  << " repro_variation_kernel_ms=" << repro.variation_kernel_ms
                  << " cpu_compile_ms=" << eval.cpu_compile_ms << "\n";
        if (cfg.eval_engine == g3pvm::evo::EvalEngine::GPU) {
          std::cout << "TIMING gpu_gen=" << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                    << " gpu_compile_ms=" << std::fixed << std::setprecision(3)
                    << eval.gpu_compile_ms
                    << " gpu_eval_call_ms=" << eval.gpu_eval_call_ms
                    << " gpu_eval_pack_ms=" << eval.gpu_eval_pack_ms
                    << " gpu_eval_launch_prep_ms=" << eval.gpu_eval_launch_prep_ms
                    << " gpu_eval_upload_ms=" << eval.gpu_eval_upload_ms
                    << " gpu_eval_pack_upload_ms=" << eval.gpu_eval_pack_upload_ms()
                    << " gpu_eval_kernel_ms=" << eval.gpu_eval_kernel_ms
                    << " gpu_eval_copyback_ms=" << eval.gpu_eval_copyback_ms
                    << " gpu_eval_teardown_ms=" << eval.gpu_eval_teardown_ms << "\n";
        }
      }
    }

    if (!args.out_json.empty()) {
      std::ofstream out(args.out_json);
      if (!out) {
        throw std::runtime_error("failed to open out-json path");
      }
      const auto& timing = result.timing;
      const auto& eval = timing.evaluation_totals;
      const auto& repro = timing.reproduction_totals;

      out << "{\n";
      out << "  \"meta\": {\n";
      out << "    \"cases_path\": \"" << json_escape(args.cases_path) << "\",\n";
      out << "    \"population_size\": " << cfg.population_size << ",\n";
      out << "    \"generations\": " << cfg.generations << ",\n";
      out << "    \"population_source\": \"" << population_source << "\",\n";
      if (!args.population_json.empty()) {
        out << "    \"population_json\": \"" << json_escape(args.population_json) << "\",\n";
      } else {
        out << "    \"population_json\": null,\n";
      }
      out << "    \"grammar_config\": {\n";
      if (!args.grammar_config_path.empty()) {
        out << "      \"path\": \"" << json_escape(args.grammar_config_path) << "\",\n";
        out << "      \"hash\": \"" << json_escape(grammar_config_hash) << "\"\n";
      } else {
        out << "      \"path\": null,\n";
        out << "      \"hash\": null\n";
      }
      out << "    },\n";
      out << "    \"selection\": \"" << selection_label << "\",\n";
      out << "    \"crossover_method\": \"" << crossover_label << "\",\n";
      out << "    \"eval_engine\": \"" << g3pvm::evo::eval_engine_name(cfg.eval_engine) << "\",\n";
      out << "    \"reproduction_backend\": \""
          << g3pvm::evo::repro::reproduction_backend_name(cfg.reproduction_backend) << "\",\n";
      out << "    \"cpu_repro_ablation\": \""
          << g3pvm::evo::repro::cpu_repro_ablation_name(cfg.cpu_repro_ablation) << "\",\n";
      out << "    \"repro_overlap\": " << (cfg.repro_overlap ? "true" : "false") << ",\n";
      out << "    \"skip_final_eval\": " << (cfg.skip_final_eval ? "true" : "false") << ",\n";
      out << "    \"retain_final_population\": " << (cfg.retain_final_population ? "true" : "false") << ",\n";
      out << "    \"gpu_blocksize\": " << cfg.gpu_blocksize << ",\n";
      out << "    \"seed\": " << cfg.seed << ",\n";
      out << "    \"timing\": {\n";
      out << "      \"init_population_ms\": " << std::setprecision(17) << timing.init_population_ms << ",\n";
      out << "      \"gpu_eval_init_ms\": " << timing.gpu_eval_init_ms << ",\n";
      out << "      \"final_eval_ms\": " << timing.final_eval_ms << ",\n";
      out << "      \"cpu_compile_ms_total\": " << eval.cpu_compile_ms << ",\n";
      out << "      \"gpu_compile_ms_total\": " << eval.gpu_compile_ms << ",\n";
      out << "      \"gpu_eval_call_ms_total\": " << eval.gpu_eval_call_ms << ",\n";
      out << "      \"gpu_eval_pack_ms_total\": " << eval.gpu_eval_pack_ms << ",\n";
      out << "      \"gpu_eval_launch_prep_ms_total\": " << eval.gpu_eval_launch_prep_ms << ",\n";
      out << "      \"gpu_eval_upload_ms_total\": " << eval.gpu_eval_upload_ms << ",\n";
      out << "      \"gpu_eval_pack_upload_ms_total\": " << eval.gpu_eval_pack_upload_ms() << ",\n";
      out << "      \"gpu_eval_kernel_ms_total\": " << eval.gpu_eval_kernel_ms << ",\n";
      out << "      \"gpu_eval_copyback_ms_total\": " << eval.gpu_eval_copyback_ms << ",\n";
      out << "      \"gpu_eval_teardown_ms_total\": " << eval.gpu_eval_teardown_ms << ",\n";
      out << "      \"generations_selection_ms_total\": " << repro.selection_ms << ",\n";
      out << "      \"generations_crossover_ms_total\": " << repro.crossover_ms << ",\n";
      out << "      \"generations_mutation_ms_total\": " << repro.mutation_ms << ",\n";
      out << "      \"generations_repro_prepare_inputs_ms_total\": "
          << repro.prepare_inputs_ms << ",\n";
      out << "      \"generations_repro_setup_ms_total\": " << repro.setup_ms << ",\n";
      out << "      \"generations_repro_preprocess_ms_total\": " << repro.preprocess_ms
          << ",\n";
      out << "      \"generations_repro_pack_ms_total\": " << repro.pack_ms << ",\n";
      out << "      \"generations_repro_upload_ms_total\": " << repro.upload_ms << ",\n";
      out << "      \"generations_repro_kernel_ms_total\": " << repro.kernel_ms << ",\n";
      out << "      \"generations_repro_copyback_ms_total\": " << repro.copyback_ms
          << ",\n";
      out << "      \"generations_repro_decode_ms_total\": " << repro.decode_ms
          << ",\n";
      out << "      \"generations_repro_teardown_ms_total\": " << repro.teardown_ms
          << ",\n";
      out << "      \"generations_repro_selection_kernel_ms_total\": "
          << repro.selection_kernel_ms << ",\n";
      out << "      \"generations_repro_variation_kernel_ms_total\": "
          << repro.variation_kernel_ms << ",\n";
      out << "      \"total_ms\": " << timing.total_ms << "\n";
      out << "    }\n";
      out << "  },\n";

      out << "  \"history\": [\n";
      for (std::size_t i = 0; i < history_rows.size(); ++i) {
        const auto& row = history_rows[i];
        out << "    {\"generation\": " << row.generation << ", \"best_fitness\": "
            << std::setprecision(17) << row.best_fitness << ", \"mean_fitness\": " << row.mean_fitness
            << ", \"program_key\": \"" << json_escape(row.program_key) << "\"}";
        if (i + 1 < history_rows.size()) {
          out << ",";
        }
        out << "\n";
      }
      out << "  ],\n";

      auto dump_series = [&](const char* name, auto value_at, bool last) {
        out << "    \"" << name << "\": [";
        for (std::size_t i = 0; i < timing.generations.size(); ++i) {
          if (i > 0) out << ", ";
          out << std::setprecision(17) << value_at(timing.generations[i]);
        }
        out << "]";
        if (!last) out << ",";
        out << "\n";
      };

      out << "  \"timing\": {\n";
      dump_series("generation_eval_ms", [](const auto& g) { return g.eval_ms; }, false);
      dump_series("generation_repro_ms", [](const auto& g) { return g.repro_ms; }, false);
      dump_series("generation_cpu_compile_ms", [](const auto& g) { return g.evaluation.cpu_compile_ms; }, false);
      dump_series("generation_gpu_compile_ms", [](const auto& g) { return g.evaluation.gpu_compile_ms; }, false);
      dump_series("generation_gpu_eval_call_ms", [](const auto& g) { return g.evaluation.gpu_eval_call_ms; }, false);
      dump_series("generation_gpu_eval_pack_ms", [](const auto& g) { return g.evaluation.gpu_eval_pack_ms; }, false);
      dump_series("generation_gpu_eval_launch_prep_ms", [](const auto& g) { return g.evaluation.gpu_eval_launch_prep_ms; }, false);
      dump_series("generation_gpu_eval_upload_ms", [](const auto& g) { return g.evaluation.gpu_eval_upload_ms; }, false);
      dump_series("generation_gpu_eval_pack_upload_ms", [](const auto& g) { return g.evaluation.gpu_eval_pack_upload_ms(); }, false);
      dump_series("generation_gpu_eval_kernel_ms", [](const auto& g) { return g.evaluation.gpu_eval_kernel_ms; }, false);
      dump_series("generation_gpu_eval_copyback_ms", [](const auto& g) { return g.evaluation.gpu_eval_copyback_ms; }, false);
      dump_series("generation_gpu_eval_teardown_ms", [](const auto& g) { return g.evaluation.gpu_eval_teardown_ms; }, false);
      dump_series("generation_selection_ms", [](const auto& g) { return g.reproduction.selection_ms; }, false);
      dump_series("generation_crossover_ms", [](const auto& g) { return g.reproduction.crossover_ms; }, false);
      dump_series("generation_mutation_ms", [](const auto& g) { return g.reproduction.mutation_ms; }, false);
      dump_series("generation_repro_prepare_inputs_ms", [](const auto& g) { return g.reproduction.prepare_inputs_ms; }, false);
      dump_series("generation_repro_setup_ms", [](const auto& g) { return g.reproduction.setup_ms; }, false);
      dump_series("generation_repro_preprocess_ms", [](const auto& g) { return g.reproduction.preprocess_ms; }, false);
      dump_series("generation_repro_pack_ms", [](const auto& g) { return g.reproduction.pack_ms; }, false);
      dump_series("generation_repro_upload_ms", [](const auto& g) { return g.reproduction.upload_ms; }, false);
      dump_series("generation_repro_kernel_ms", [](const auto& g) { return g.reproduction.kernel_ms; }, false);
      dump_series("generation_repro_copyback_ms", [](const auto& g) { return g.reproduction.copyback_ms; }, false);
      dump_series("generation_repro_decode_ms", [](const auto& g) { return g.reproduction.decode_ms; }, false);
      dump_series("generation_repro_teardown_ms", [](const auto& g) { return g.reproduction.teardown_ms; }, false);
      dump_series("generation_repro_selection_kernel_ms", [](const auto& g) { return g.reproduction.selection_kernel_ms; }, false);
      dump_series("generation_repro_variation_kernel_ms", [](const auto& g) { return g.reproduction.variation_kernel_ms; }, false);
      dump_series("generation_total_ms", [](const auto& g) { return g.total_ms; }, true);
      out << "  },\n";

      out << "  \"final\": {\n";
      out << "    \"skipped\": " << (result.final_eval_skipped ? "true" : "false");
      if (!result.final_eval_skipped) {
        const g3pvm::evo::ProgramGenome best_output =
            g3pvm::evo::repro::compact_genome_tables(result.best.genome);
        out << ",\n";
        out << "    \"best_fitness\": " << std::setprecision(17) << result.best.fitness << ",\n";
        out << "    \"program_key\": \"" << json_escape(best_output.meta.program_key) << "\",\n";
        out << "    \"ast_repr\": \"" << json_escape(g3pvm::evo::ast_to_string(best_output.ast)) << "\",\n";
        out << "    \"ast_names\": [";
        for (std::size_t i = 0; i < best_output.ast.names.size(); ++i) {
          if (i > 0) out << ", ";
          out << "\"" << json_escape(best_output.ast.names[i]) << "\"";
        }
        out << "],\n";
        out << "    \"ast_consts\": [";
        for (std::size_t i = 0; i < best_output.ast.consts.size(); ++i) {
          if (i > 0) out << ", ";
          write_value_json(out, best_output.ast.consts[i]);
        }
        out << "],\n";
        out << "    \"ast\": ";
        write_ast_json(out, best_output.ast);
        out << "\n";
      } else {
        out << "\n";
      }
      out << "  }\n";
      out << "}\n";
    }

    return 0;
}
