#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/runtime/payload/payload.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/gpu/payload_flavor_gpu.hpp"

// Keep directly buildable.
#include "json.cpp"
#include "codec.cpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::CaseBindings;
using g3pvm::ExecResult;
using g3pvm::InputBinding;
using g3pvm::Value;
using g3pvm::cli_detail::JsonValue;
using g3pvm::evo::EvalCase;
using g3pvm::evo::NamedInputs;
using g3pvm::evo::ProgramGenome;
using g3pvm::gpu_detail::DPayloadFlavor;

struct CliOptions {
  std::string cases_path;
  std::string out_population_json;
  std::string out_metadata_json;
  std::string target_payload_flavor;
  std::string generator_root_type = "any";
  std::string generator_mode = "synthetic";
  int population_size = 1024;
  std::uint64_t seed_start = 0;
  int probe_cases = 32;
  double min_success_rate = 0.5;
  int max_attempts = 500000;
  int fuel = 20000;
  int max_expr_depth = 5;
  int max_stmts_per_block = 6;
  int max_total_nodes = 80;
  int max_for_k = 16;
  int max_call_args = 3;
  int target_depth = 5;
  int target_node_count = 0;
};

struct AcceptedProgram {
  std::uint64_t seed = 0;
  int probe_successes = 0;
  int actual_depth = 0;
  int node_count = 0;
  int code_len = 0;
  std::string payload_flavor;
  std::string program_key;
};

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("missing file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool is_integer_number(double x) {
  const long long i = static_cast<long long>(x);
  return static_cast<double>(i) == x;
}

Value decode_typed_or_raw_value(const JsonValue& v) {
  if (v.kind == JsonValue::Kind::Object) {
    auto it = v.object_v.find("type");
    if (it != v.object_v.end()) {
      return g3pvm::cli_detail::decode_typed_value(v);
    }
  }
  if (v.kind == JsonValue::Kind::Null) return Value::none();
  if (v.kind == JsonValue::Kind::Bool) return Value::from_bool(v.bool_v);
  if (v.kind == JsonValue::Kind::Number) {
    if (is_integer_number(v.number_v)) return Value::from_int(static_cast<long long>(v.number_v));
    return Value::from_float(v.number_v);
  }
  throw std::runtime_error("unsupported raw value type");
}

NamedInputs decode_inputs(const JsonValue& raw) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("case.inputs must be an object");
  }
  NamedInputs out;
  for (const auto& kv : raw.object_v) {
    out[kv.first] = decode_typed_or_raw_value(kv.second);
  }
  return out;
}

std::vector<EvalCase> parse_cases_v1(const JsonValue& payload) {
  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "fitness-cases-v1") {
    throw std::runtime_error("input JSON must include format_version=fitness-cases-v1");
  }
  auto cases_it = payload.object_v.find("cases");
  if (cases_it == payload.object_v.end() || cases_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input JSON must include list field: cases");
  }
  std::vector<EvalCase> out;
  out.reserve(cases_it->second.array_v.size());
  for (const JsonValue& row : cases_it->second.array_v) {
    const auto inputs_it = row.object_v.find("inputs");
    const auto expected_it = row.object_v.find("expected");
    if (inputs_it == row.object_v.end() || expected_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] must include inputs/expected");
    }
    out.push_back(EvalCase{decode_inputs(inputs_it->second), decode_typed_or_raw_value(expected_it->second)});
  }
  if (out.empty()) {
    throw std::runtime_error("cases must not be empty");
  }
  return out;
}

std::vector<std::string> build_canonical_input_names(const std::vector<EvalCase>& cases) {
  std::set<std::string> names;
  for (const EvalCase& one_case : cases) {
    for (const auto& kv : one_case.inputs) {
      names.insert(kv.first);
    }
  }
  return std::vector<std::string>(names.begin(), names.end());
}

std::vector<CaseBindings> build_shared_case_bindings(const std::vector<EvalCase>& cases,
                                                     const std::vector<std::string>& input_names) {
  std::vector<CaseBindings> out;
  out.reserve(cases.size());
  for (const EvalCase& one_case : cases) {
    CaseBindings bindings;
    bindings.reserve(input_names.size());
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      auto it = one_case.inputs.find(input_names[i]);
      if (it != one_case.inputs.end()) {
        bindings.push_back(InputBinding{static_cast<int>(i), it->second});
      }
    }
    out.push_back(std::move(bindings));
  }
  return out;
}

std::vector<std::pair<int, Value>> case_inputs_to_locals(const EvalCase& one_case,
                                                         const std::vector<std::string>& input_names) {
  std::vector<std::pair<int, Value>> out;
  out.reserve(input_names.size());
  for (std::size_t i = 0; i < input_names.size(); ++i) {
    auto it = one_case.inputs.find(input_names[i]);
    if (it != one_case.inputs.end()) {
      out.push_back({static_cast<int>(i), it->second});
    }
  }
  return out;
}

std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    if (c == '"') oss << "\\\"";
    else if (c == '\\') oss << "\\\\";
    else if (c == '\n') oss << "\\n";
    else if (c == '\r') oss << "\\r";
    else if (c == '\t') oss << "\\t";
    else oss << c;
  }
  return oss.str();
}

unsigned value_payload_mask(const Value& v) {
  if (v.tag == g3pvm::ValueTag::String) return 1U << 0;
  if (v.tag == g3pvm::ValueTag::List) return 1U << 1;
  return 0U;
}

unsigned shared_cases_payload_mask(const std::vector<CaseBindings>& shared_cases) {
  unsigned mask = 0U;
  for (const CaseBindings& case_bindings : shared_cases) {
    for (const InputBinding& binding : case_bindings) {
      mask |= value_payload_mask(binding.value);
    }
  }
  return mask;
}

std::string payload_flavor_name(DPayloadFlavor flavor) {
  switch (flavor) {
    case DPayloadFlavor::None: return "none";
    case DPayloadFlavor::StringOnly: return "string";
    case DPayloadFlavor::ListOnly: return "list";
    case DPayloadFlavor::Mixed: return "mixed";
  }
  return "unknown";
}

int payload_flavor_index(DPayloadFlavor flavor) {
  switch (flavor) {
    case DPayloadFlavor::None: return 0;
    case DPayloadFlavor::StringOnly: return 1;
    case DPayloadFlavor::ListOnly: return 2;
    case DPayloadFlavor::Mixed: return 3;
  }
  return 0;
}

DPayloadFlavor parse_payload_flavor(const std::string& raw) {
  if (raw == "any") return DPayloadFlavor::None;
  if (raw == "none") return DPayloadFlavor::None;
  if (raw == "string") return DPayloadFlavor::StringOnly;
  if (raw == "list") return DPayloadFlavor::ListOnly;
  if (raw == "mixed") return DPayloadFlavor::Mixed;
  throw std::runtime_error("--target-payload-flavor must be one of: any,none,string,list,mixed");
}

g3pvm::evo::RType parse_root_type(const std::string& raw) {
  if (raw == "any") return g3pvm::evo::RType::Any;
  if (raw == "num") return g3pvm::evo::RType::Num;
  if (raw == "bool") return g3pvm::evo::RType::Bool;
  if (raw == "none") return g3pvm::evo::RType::NoneType;
  if (raw == "string") return g3pvm::evo::RType::String;
  if (raw == "list") return g3pvm::evo::RType::List;
  throw std::runtime_error("--generator-root-type must be one of: any,num,bool,none,string,list");
}

std::string random_string_literal(std::mt19937_64& rng) {
  static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz";
  const int len = std::uniform_int_distribution<int>(0, 8)(rng);
  std::string out;
  out.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    out.push_back(kAlphabet[std::uniform_int_distribution<int>(0, 25)(rng)]);
  }
  return out;
}

int append_const_id(g3pvm::evo::AstProgram& program, const Value& value) {
  program.consts.push_back(value);
  return static_cast<int>(program.consts.size() - 1);
}

void emit_int_leaf(std::mt19937_64& rng, g3pvm::evo::AstProgram& program) {
  program.nodes.push_back(
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::CONST,
                          append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng))),
                          0});
}

void emit_num_leaf(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, bool allow_x_var) {
  if (allow_x_var && std::bernoulli_distribution(0.35)(rng)) {
    program.nodes.push_back(g3pvm::evo::AstNode{g3pvm::evo::NodeKind::VAR, 0, 0});
    return;
  }
  if (std::bernoulli_distribution(0.5)(rng)) {
    program.nodes.push_back(
        g3pvm::evo::AstNode{g3pvm::evo::NodeKind::CONST,
                            append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng))),
                            0});
  } else {
    const double value =
        std::round(std::uniform_real_distribution<double>(-8.0, 8.0)(rng) * 1000.0) / 1000.0;
    program.nodes.push_back(
        g3pvm::evo::AstNode{g3pvm::evo::NodeKind::CONST, append_const_id(program, Value::from_float(value)), 0});
  }
}

void emit_string_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth);
void emit_list_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth);
void emit_num_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes);
void emit_string_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes);
void emit_list_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes);

void emit_num_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth) {
  using namespace g3pvm::evo;
  if (depth <= 1) {
    emit_num_leaf(rng, program, true);
    return;
  }
  const int choice = std::uniform_int_distribution<int>(0, 2)(rng);
  if (choice == 0) {
    program.nodes.push_back(AstNode{NodeKind::NEG, 0, 0});
    emit_num_expr(rng, program, depth - 1);
    return;
  }
  program.nodes.push_back(AstNode{std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? NodeKind::ADD : NodeKind::MUL, 0, 0});
  emit_num_expr(rng, program, depth - 1);
  emit_num_leaf(rng, program, true);
}

void emit_string_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth) {
  using namespace g3pvm::evo;
  if (depth <= 1) {
    program.nodes.push_back(
        AstNode{NodeKind::CONST,
                append_const_id(program, g3pvm::payload::make_string_value(random_string_literal(rng))),
                0});
    return;
  }
  const int choice = std::uniform_int_distribution<int>(0, 1)(rng);
  if (choice == 0) {
    program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
    emit_string_expr(rng, program, depth - 1);
    program.nodes.push_back(
        AstNode{NodeKind::CONST,
                append_const_id(program, g3pvm::payload::make_string_value(random_string_literal(rng))),
                0});
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
  emit_string_expr(rng, program, depth - 1);
  emit_int_leaf(rng, program);
  emit_int_leaf(rng, program);
}

void emit_list_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth) {
  using namespace g3pvm::evo;
  if (depth <= 1) {
    const int len = std::uniform_int_distribution<int>(0, 4)(rng);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      const int kind = std::uniform_int_distribution<int>(0, 2)(rng);
      if (kind == 0) elems.push_back(Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng)));
      else if (kind == 1) elems.push_back(Value::from_bool(std::bernoulli_distribution(0.5)(rng)));
      else elems.push_back(Value::none());
    }
    program.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(program, g3pvm::payload::make_list_value(elems)), 0});
    return;
  }
  const int choice = std::uniform_int_distribution<int>(0, 1)(rng);
  if (choice == 0) {
    program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
    emit_list_expr(rng, program, depth - 1);
    emit_list_expr(rng, program, 1);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
  emit_list_expr(rng, program, depth - 1);
  emit_int_leaf(rng, program);
  emit_int_leaf(rng, program);
}

void emit_string_num_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth) {
  using namespace g3pvm::evo;
  if (depth <= 1) {
    emit_num_leaf(rng, program, true);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
  emit_string_expr(rng, program, depth - 1);
}

void emit_list_num_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth) {
  using namespace g3pvm::evo;
  if (depth <= 1) {
    emit_num_leaf(rng, program, true);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
  emit_list_expr(rng, program, depth - 1);
}

void emit_mixed_num_expr(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int depth) {
  using namespace g3pvm::evo;
  if (depth <= 1) {
    emit_num_leaf(rng, program, true);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::ADD, 0, 0});
  emit_string_num_expr(rng, program, depth - 1);
  emit_list_num_expr(rng, program, std::max(1, depth - 2));
}

int random_split_two(std::mt19937_64& rng, int total) {
  return std::uniform_int_distribution<int>(1, total - 1)(rng);
}

bool string_like_expr_size_possible(int nodes) {
  return nodes == 1 || nodes >= 3;
}

std::vector<std::pair<int, int>> string_like_concat_splits(int remaining) {
  std::vector<std::pair<int, int>> out;
  for (int left = 1; left < remaining; ++left) {
    const int right = remaining - left;
    if (string_like_expr_size_possible(left) && string_like_expr_size_possible(right)) {
      out.push_back({left, right});
    }
  }
  return out;
}

void emit_string_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes == 1) {
    program.nodes.push_back(
        AstNode{NodeKind::CONST,
                append_const_id(program, g3pvm::payload::make_string_value(random_string_literal(rng))),
                0});
    return;
  }
  if (!string_like_expr_size_possible(nodes)) {
    throw std::runtime_error("invalid exact string node count");
  }

  const auto concat_splits = string_like_concat_splits(nodes - 1);
  const bool can_concat = !concat_splits.empty();
  const bool can_slice = nodes >= 4 && string_like_expr_size_possible(nodes - 3);

  if (!can_slice || (can_concat && std::bernoulli_distribution(0.6)(rng))) {
    program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
    const auto& split = concat_splits[static_cast<std::size_t>(
        std::uniform_int_distribution<int>(0, static_cast<int>(concat_splits.size()) - 1)(rng))];
    emit_string_expr_exact_nodes(rng, program, split.first);
    emit_string_expr_exact_nodes(rng, program, split.second);
    return;
  }

  program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
  emit_string_expr_exact_nodes(rng, program, nodes - 3);
  emit_int_leaf(rng, program);
  emit_int_leaf(rng, program);
}

void emit_list_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes == 1) {
    const int len = std::uniform_int_distribution<int>(0, 4)(rng);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      const int kind = std::uniform_int_distribution<int>(0, 2)(rng);
      if (kind == 0) elems.push_back(Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng)));
      else if (kind == 1) elems.push_back(Value::from_bool(std::bernoulli_distribution(0.5)(rng)));
      else elems.push_back(Value::none());
    }
    program.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(program, g3pvm::payload::make_list_value(elems)), 0});
    return;
  }
  if (!string_like_expr_size_possible(nodes)) {
    throw std::runtime_error("invalid exact list node count");
  }

  const auto concat_splits = string_like_concat_splits(nodes - 1);
  const bool can_concat = !concat_splits.empty();
  const bool can_slice = nodes >= 4 && string_like_expr_size_possible(nodes - 3);

  if (!can_slice || (can_concat && std::bernoulli_distribution(0.6)(rng))) {
    program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
    const auto& split = concat_splits[static_cast<std::size_t>(
        std::uniform_int_distribution<int>(0, static_cast<int>(concat_splits.size()) - 1)(rng))];
    emit_list_expr_exact_nodes(rng, program, split.first);
    emit_list_expr_exact_nodes(rng, program, split.second);
    return;
  }

  program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
  emit_list_expr_exact_nodes(rng, program, nodes - 3);
  emit_int_leaf(rng, program);
  emit_int_leaf(rng, program);
}

void emit_num_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes <= 1) {
    emit_num_leaf(rng, program, true);
    return;
  }

  const bool payload_len_size_possible = string_like_expr_size_possible(nodes - 1);
  if (payload_len_size_possible && std::bernoulli_distribution(0.28)(rng)) {
    program.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
    if (std::bernoulli_distribution(0.5)(rng)) {
      emit_string_expr_exact_nodes(rng, program, nodes - 1);
    } else {
      emit_list_expr_exact_nodes(rng, program, nodes - 1);
    }
    return;
  }

  if (nodes == 2 || std::bernoulli_distribution(0.25)(rng)) {
    program.nodes.push_back(AstNode{std::bernoulli_distribution(0.5)(rng) ? NodeKind::NEG : NodeKind::CALL_ABS, 0, 0});
    emit_num_expr_exact_nodes(rng, program, nodes - 1);
    return;
  }

  if (nodes >= 4 && std::bernoulli_distribution(0.15)(rng)) {
    program.nodes.push_back(AstNode{NodeKind::CALL_CLIP, 0, 0});
    const int remaining = nodes - 1;
    const int first = std::uniform_int_distribution<int>(1, remaining - 2)(rng);
    const int remaining_after_first = remaining - first;
    const int second = std::uniform_int_distribution<int>(1, remaining_after_first - 1)(rng);
    const int third = remaining_after_first - second;
    emit_num_expr_exact_nodes(rng, program, first);
    emit_num_expr_exact_nodes(rng, program, second);
    emit_num_expr_exact_nodes(rng, program, third);
    return;
  }

  program.nodes.push_back(
      AstNode{std::bernoulli_distribution(0.5)(rng) ? NodeKind::ADD : NodeKind::MUL, 0, 0});
  const int remaining = nodes - 1;
  const int left_nodes = random_split_two(rng, remaining);
  const int right_nodes = remaining - left_nodes;
  emit_num_expr_exact_nodes(rng, program, left_nodes);
  emit_num_expr_exact_nodes(rng, program, right_nodes);
}

void emit_num_expr_exact_nodes_no_payload(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes <= 1) {
    emit_num_leaf(rng, program, true);
    return;
  }
  if (nodes == 2 || std::bernoulli_distribution(0.25)(rng)) {
    program.nodes.push_back(
        AstNode{std::bernoulli_distribution(0.5)(rng) ? NodeKind::NEG : NodeKind::CALL_ABS, 0, 0});
    emit_num_expr_exact_nodes_no_payload(rng, program, nodes - 1);
    return;
  }
  if (nodes >= 4 && std::bernoulli_distribution(0.15)(rng)) {
    program.nodes.push_back(AstNode{NodeKind::CALL_CLIP, 0, 0});
    const int remaining = nodes - 1;
    const int first = std::uniform_int_distribution<int>(1, remaining - 2)(rng);
    const int remaining_after_first = remaining - first;
    const int second = std::uniform_int_distribution<int>(1, remaining_after_first - 1)(rng);
    const int third = remaining_after_first - second;
    emit_num_expr_exact_nodes_no_payload(rng, program, first);
    emit_num_expr_exact_nodes_no_payload(rng, program, second);
    emit_num_expr_exact_nodes_no_payload(rng, program, third);
    return;
  }
  program.nodes.push_back(
      AstNode{std::bernoulli_distribution(0.5)(rng) ? NodeKind::ADD : NodeKind::MUL, 0, 0});
  const int remaining = nodes - 1;
  const int left_nodes = random_split_two(rng, remaining);
  const int right_nodes = remaining - left_nodes;
  emit_num_expr_exact_nodes_no_payload(rng, program, left_nodes);
  emit_num_expr_exact_nodes_no_payload(rng, program, right_nodes);
}

void emit_string_num_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes < 2) throw std::runtime_error("invalid exact string-num node count");
  program.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
  emit_string_expr_exact_nodes(rng, program, nodes - 1);
}

void emit_list_num_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes < 2) throw std::runtime_error("invalid exact list-num node count");
  program.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
  emit_list_expr_exact_nodes(rng, program, nodes - 1);
}

bool payload_num_expr_size_possible(int nodes) {
  return nodes == 2 || nodes >= 4;
}

void emit_mixed_num_expr_exact_nodes(std::mt19937_64& rng, g3pvm::evo::AstProgram& program, int nodes) {
  using namespace g3pvm::evo;
  if (nodes < 5) throw std::runtime_error("invalid exact mixed-num node count");
  if ((nodes % 2) == 0) {
    program.nodes.push_back(
        AstNode{std::bernoulli_distribution(0.5)(rng) ? NodeKind::NEG : NodeKind::CALL_ABS, 0, 0});
    emit_mixed_num_expr_exact_nodes(rng, program, nodes - 1);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::ADD, 0, 0});
  const int remaining = nodes - 1;
  std::vector<std::pair<int, int>> splits;
  for (int left = 2; left <= remaining - 2; ++left) {
    const int right = nodes - 1 - left;
    if (payload_num_expr_size_possible(left) && payload_num_expr_size_possible(right)) {
      splits.push_back({left, right});
    }
  }
  const auto& split =
      splits[static_cast<std::size_t>(std::uniform_int_distribution<int>(0, static_cast<int>(splits.size()) - 1)(rng))];
  emit_string_num_expr_exact_nodes(rng, program, split.first);
  emit_list_num_expr_exact_nodes(rng, program, split.second);
}

std::vector<DPayloadFlavor> feasible_exact_node_flavors(int target_node_count) {
  const int expr_nodes = target_node_count - 4;
  std::vector<DPayloadFlavor> out;
  if (expr_nodes >= 1) out.push_back(DPayloadFlavor::None);
  if (expr_nodes >= 2) out.push_back(DPayloadFlavor::StringOnly);
  if (expr_nodes >= 2) out.push_back(DPayloadFlavor::ListOnly);
  if (expr_nodes >= 7) out.push_back(DPayloadFlavor::Mixed);
  return out;
}

DPayloadFlavor choose_balanced_any_flavor(std::uint64_t seed,
                                          int target_node_count,
                                          const std::array<int, 4>& accepted_counts) {
  const auto feasible = feasible_exact_node_flavors(target_node_count);
  int best = std::numeric_limits<int>::max();
  std::vector<DPayloadFlavor> choices;
  for (DPayloadFlavor flavor : feasible) {
    const int count = accepted_counts[static_cast<std::size_t>(payload_flavor_index(flavor))];
    if (count < best) {
      best = count;
      choices.clear();
      choices.push_back(flavor);
    } else if (count == best) {
      choices.push_back(flavor);
    }
  }
  return choices[static_cast<std::size_t>(seed % choices.size())];
}

ProgramGenome build_synthetic_bucket_genome(std::uint64_t seed,
                                            DPayloadFlavor flavor,
                                            int target_depth,
                                            int target_node_count,
                                            const g3pvm::evo::Limits& limits) {
  using namespace g3pvm::evo;
  std::mt19937_64 rng(seed);
  AstProgram program;
  program.version = "ast-prefix-v1";
  program.names.push_back("x");
  program.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
  program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
  program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  if (target_node_count > 0) {
    const int expr_nodes = std::max(1, target_node_count - 4);
    switch (flavor) {
      case DPayloadFlavor::None:
        emit_num_expr_exact_nodes_no_payload(rng, program, expr_nodes);
        break;
      case DPayloadFlavor::StringOnly:
        emit_string_num_expr_exact_nodes(rng, program, expr_nodes);
        break;
      case DPayloadFlavor::ListOnly:
        emit_list_num_expr_exact_nodes(rng, program, expr_nodes);
        break;
      case DPayloadFlavor::Mixed:
        emit_mixed_num_expr_exact_nodes(rng, program, expr_nodes);
        break;
    }
  } else {
    switch (flavor) {
      case DPayloadFlavor::None:
        emit_num_expr(rng, program, target_depth);
        break;
      case DPayloadFlavor::StringOnly:
        emit_string_expr(rng, program, target_depth);
        break;
      case DPayloadFlavor::ListOnly:
        emit_list_expr(rng, program, target_depth);
        break;
      case DPayloadFlavor::Mixed:
        emit_mixed_num_expr(rng, program, target_depth);
        break;
    }
  }
  program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});

  ProgramGenome genome;
  genome.ast = std::move(program);
  genome.meta = build_genome_meta(genome.ast);
  if (genome.meta.node_count > limits.max_total_nodes || genome.meta.max_depth > limits.max_expr_depth) {
    return generate_random_genome(seed, limits);
  }
  return genome;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* key) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + key);
      return argv[++i];
    };
    if (arg == "--cases") opts.cases_path = need_value("--cases");
    else if (arg == "--out-population-json") opts.out_population_json = need_value("--out-population-json");
    else if (arg == "--out-metadata-json") opts.out_metadata_json = need_value("--out-metadata-json");
    else if (arg == "--target-payload-flavor") opts.target_payload_flavor = need_value("--target-payload-flavor");
    else if (arg == "--generator-root-type") opts.generator_root_type = need_value("--generator-root-type");
    else if (arg == "--generator-mode") opts.generator_mode = need_value("--generator-mode");
    else if (arg == "--target-depth") opts.target_depth = std::stoi(need_value("--target-depth"));
    else if (arg == "--target-node-count") opts.target_node_count = std::stoi(need_value("--target-node-count"));
    else if (arg == "--population-size") opts.population_size = std::stoi(need_value("--population-size"));
    else if (arg == "--seed-start") opts.seed_start = static_cast<std::uint64_t>(std::stoull(need_value("--seed-start")));
    else if (arg == "--probe-cases") opts.probe_cases = std::stoi(need_value("--probe-cases"));
    else if (arg == "--min-success-rate") opts.min_success_rate = std::stod(need_value("--min-success-rate"));
    else if (arg == "--max-attempts") opts.max_attempts = std::stoi(need_value("--max-attempts"));
    else if (arg == "--fuel") opts.fuel = std::stoi(need_value("--fuel"));
    else if (arg == "--max-expr-depth") opts.max_expr_depth = std::stoi(need_value("--max-expr-depth"));
    else if (arg == "--max-stmts-per-block") opts.max_stmts_per_block = std::stoi(need_value("--max-stmts-per-block"));
    else if (arg == "--max-total-nodes") opts.max_total_nodes = std::stoi(need_value("--max-total-nodes"));
    else if (arg == "--max-for-k") opts.max_for_k = std::stoi(need_value("--max-for-k"));
    else if (arg == "--max-call-args") opts.max_call_args = std::stoi(need_value("--max-call-args"));
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (opts.cases_path.empty()) throw std::runtime_error("--cases is required");
  if (opts.out_population_json.empty()) throw std::runtime_error("--out-population-json is required");
  if (opts.out_metadata_json.empty()) throw std::runtime_error("--out-metadata-json is required");
  if (opts.target_payload_flavor.empty()) throw std::runtime_error("--target-payload-flavor is required");
  if (opts.generator_mode != "native" && opts.generator_mode != "synthetic") {
    throw std::runtime_error("--generator-mode must be native or synthetic");
  }
  if (opts.population_size <= 0) throw std::runtime_error("--population-size must be > 0");
  if (opts.target_node_count < 0) throw std::runtime_error("--target-node-count must be >= 0");
  if (opts.target_depth < 0) throw std::runtime_error("--target-depth must be >= 0");
  if (opts.probe_cases <= 0) throw std::runtime_error("--probe-cases must be > 0");
  if (opts.min_success_rate < 0.0 || opts.min_success_rate > 1.0) {
    throw std::runtime_error("--min-success-rate must be in [0, 1]");
  }
  if (opts.max_attempts <= 0) throw std::runtime_error("--max-attempts must be > 0");
  if (opts.target_node_count == 0 && opts.target_depth > 0 && opts.max_expr_depth < opts.target_depth) {
    throw std::runtime_error("--max-expr-depth must be >= --target-depth");
  }
  return opts;
}

void write_population_seed_set(const std::string& path,
                               const std::string& cases_path,
                               const g3pvm::evo::Limits& limits,
                               const std::vector<AcceptedProgram>& accepted,
                               int probe_case_count,
                               double min_success_rate,
                               int fuel,
                               int attempts,
                               int target_depth,
                               int target_node_count,
                               const std::string& target_payload_flavor) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open out-population-json path");
  }
  out << "{\n";
  out << "  \"format_version\": \"population-seeds-v1\",\n";
  out << "  \"cases_path\": \"" << json_escape(cases_path) << "\",\n";
  out << "  \"population_size\": " << accepted.size() << ",\n";
  out << "  \"probe_cases\": " << probe_case_count << ",\n";
  out << "  \"min_success_rate\": " << std::setprecision(17) << min_success_rate << ",\n";
  out << "  \"fuel\": " << fuel << ",\n";
  out << "  \"attempts\": " << attempts << ",\n";
  out << "  \"target_depth\": " << target_depth << ",\n";
  out << "  \"target_node_count\": " << target_node_count << ",\n";
  out << "  \"target_payload_flavor\": \"" << target_payload_flavor << "\",\n";
  out << "  \"limits\": {\n";
  out << "    \"max_expr_depth\": " << limits.max_expr_depth << ",\n";
  out << "    \"max_stmts_per_block\": " << limits.max_stmts_per_block << ",\n";
  out << "    \"max_total_nodes\": " << limits.max_total_nodes << ",\n";
  out << "    \"max_for_k\": " << limits.max_for_k << ",\n";
  out << "    \"max_call_args\": " << limits.max_call_args << "\n";
  out << "  },\n";
  out << "  \"seeds\": [\n";
  for (std::size_t i = 0; i < accepted.size(); ++i) {
    const AcceptedProgram& row = accepted[i];
    out << "    {\"seed\": " << row.seed
        << ", \"probe_successes\": " << row.probe_successes
        << ", \"actual_depth\": " << row.actual_depth
        << ", \"node_count\": " << row.node_count
        << ", \"code_len\": " << row.code_len
        << ", \"payload_flavor\": \"" << row.payload_flavor << "\""
        << ", \"program_key\": \"" << json_escape(row.program_key) << "\"}";
    if (i + 1 < accepted.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

void write_metadata_json(const std::string& path,
                         const std::string& cases_path,
                         const g3pvm::evo::Limits& limits,
                         const std::vector<AcceptedProgram>& accepted,
                         int probe_case_count,
                         double min_success_rate,
                         int fuel,
                         int attempts,
                         int target_depth,
                         int target_node_count,
                         const std::string& target_payload_flavor) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open out-metadata-json path");
  }
  out << "{\n";
  out << "  \"format_version\": \"population-bucket-metadata-v1\",\n";
  out << "  \"cases_path\": \"" << json_escape(cases_path) << "\",\n";
  out << "  \"population_size\": " << accepted.size() << ",\n";
  out << "  \"probe_cases\": " << probe_case_count << ",\n";
  out << "  \"min_success_rate\": " << std::setprecision(17) << min_success_rate << ",\n";
  out << "  \"fuel\": " << fuel << ",\n";
  out << "  \"attempts\": " << attempts << ",\n";
  out << "  \"target_depth\": " << target_depth << ",\n";
  out << "  \"target_node_count\": " << target_node_count << ",\n";
  out << "  \"target_payload_flavor\": \"" << target_payload_flavor << "\",\n";
  out << "  \"limits\": {\n";
  out << "    \"max_expr_depth\": " << limits.max_expr_depth << ",\n";
  out << "    \"max_stmts_per_block\": " << limits.max_stmts_per_block << ",\n";
  out << "    \"max_total_nodes\": " << limits.max_total_nodes << ",\n";
  out << "    \"max_for_k\": " << limits.max_for_k << ",\n";
  out << "    \"max_call_args\": " << limits.max_call_args << "\n";
  out << "  },\n";
  out << "  \"programs\": [\n";
  for (std::size_t i = 0; i < accepted.size(); ++i) {
    const AcceptedProgram& row = accepted[i];
    out << "    {\"seed\": " << row.seed
        << ", \"probe_successes\": " << row.probe_successes
        << ", \"actual_depth\": " << row.actual_depth
        << ", \"node_count\": " << row.node_count
        << ", \"code_len\": " << row.code_len
        << ", \"payload_flavor\": \"" << row.payload_flavor << "\""
        << ", \"program_key\": \"" << json_escape(row.program_key) << "\"}";
    if (i + 1 < accepted.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions args = parse_cli(argc, argv);
    const bool match_any_payload_flavor = args.target_payload_flavor == "any";
    const DPayloadFlavor target_flavor = parse_payload_flavor(args.target_payload_flavor);
    const g3pvm::evo::RType generator_root_type = parse_root_type(args.generator_root_type);
    const JsonValue cases_payload = g3pvm::cli_detail::JsonParser(read_text_file(args.cases_path)).parse();
    const std::vector<EvalCase> cases = parse_cases_v1(cases_payload);
    const std::vector<std::string> input_names = build_canonical_input_names(cases);
    const std::vector<CaseBindings> shared_case_bindings = build_shared_case_bindings(cases, input_names);
    const unsigned shared_input_payload_mask = shared_cases_payload_mask(shared_case_bindings);

    const g3pvm::evo::Limits limits{
        args.max_expr_depth,
        args.max_stmts_per_block,
        args.max_total_nodes,
        args.max_for_k,
        args.max_call_args,
    };

    const int probe_case_count = std::min<int>(args.probe_cases, static_cast<int>(cases.size()));
    const int min_successes =
        std::max(1, static_cast<int>(std::ceil(args.min_success_rate * static_cast<double>(probe_case_count))));

    std::vector<AcceptedProgram> accepted;
    accepted.reserve(static_cast<std::size_t>(args.population_size));
    std::uint64_t candidate_seed = args.seed_start;
    int attempts = 0;
    std::array<int, 4> accepted_flavor_counts{0, 0, 0, 0};

    while (static_cast<int>(accepted.size()) < args.population_size && attempts < args.max_attempts) {
      ++attempts;
      const DPayloadFlavor generation_flavor =
          (match_any_payload_flavor && args.generator_mode == "synthetic" && args.target_node_count > 0)
              ? choose_balanced_any_flavor(candidate_seed, args.target_node_count, accepted_flavor_counts)
              : target_flavor;
      const ProgramGenome genome =
          args.generator_mode == "synthetic"
              ? build_synthetic_bucket_genome(
                    candidate_seed, generation_flavor, args.target_depth, args.target_node_count, limits)
              : (generator_root_type == g3pvm::evo::RType::Any
                     ? g3pvm::evo::generate_random_genome(candidate_seed, limits)
                     : g3pvm::evo::generate_random_genome_for_return_type(candidate_seed, generator_root_type, limits));
      if (args.target_depth > 0 && genome.meta.max_depth != args.target_depth) {
        ++candidate_seed;
        continue;
      }
      if (args.target_node_count > 0 && genome.meta.node_count != args.target_node_count) {
        ++candidate_seed;
        continue;
      }

      const BytecodeProgram program = g3pvm::evo::compile_for_eval(genome, input_names);
      const DPayloadFlavor flavor =
          g3pvm::gpu_detail::classify_payload_flavor_for_program(program, shared_input_payload_mask);
      if (!match_any_payload_flavor && flavor != target_flavor) {
        ++candidate_seed;
        continue;
      }

      int successes = 0;
      for (int i = 0; i < probe_case_count; ++i) {
        const ExecResult out = g3pvm::execute_bytecode_cpu(
            program, case_inputs_to_locals(cases[static_cast<std::size_t>(i)], input_names), args.fuel);
        if (!out.is_error) {
          ++successes;
        }
      }
      if (successes >= min_successes) {
        accepted_flavor_counts[static_cast<std::size_t>(payload_flavor_index(flavor))] += 1;
        accepted.push_back(AcceptedProgram{
            candidate_seed,
            successes,
            genome.meta.max_depth,
            genome.meta.node_count,
            static_cast<int>(program.code.size()),
            payload_flavor_name(flavor),
            genome.meta.program_key,
        });
      }
      ++candidate_seed;
    }

    if (static_cast<int>(accepted.size()) != args.population_size) {
      throw std::runtime_error("failed to collect enough accepted genomes for requested depth/payload bucket");
    }

    write_population_seed_set(args.out_population_json,
                              args.cases_path,
                              limits,
                              accepted,
                              probe_case_count,
                              args.min_success_rate,
                              args.fuel,
                              attempts,
                              args.target_depth,
                              args.target_node_count,
                              args.target_payload_flavor);
    write_metadata_json(args.out_metadata_json,
                        args.cases_path,
                        limits,
                        accepted,
                        probe_case_count,
                        args.min_success_rate,
                        args.fuel,
                        attempts,
                        args.target_depth,
                        args.target_node_count,
                        args.target_payload_flavor);

    std::cout << "BUCKET"
              << " cases=" << args.cases_path
              << " population_size=" << accepted.size()
              << " target_depth=" << args.target_depth
              << " target_node_count=" << args.target_node_count
              << " target_payload_flavor=" << args.target_payload_flavor
              << " generator_mode=" << args.generator_mode
              << " generator_root_type=" << args.generator_root_type
              << " attempts=" << attempts
              << " out_population_json=" << args.out_population_json
              << " out_metadata_json=" << args.out_metadata_json
              << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
