#include "subtree_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gagp/evolution/node_descriptor.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace gagp::evo::subtree {

namespace {

int rand_int(std::mt19937_64& rng, int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(rng);
}

double rand_real(std::mt19937_64& rng, double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(rng);
}

bool rand_prob(std::mt19937_64& rng, double p) {
  std::bernoulli_distribution dist(std::max(0.0, std::min(1.0, p)));
  return dist(rng);
}

bool value_equal(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::Invalid) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int || a.tag == ValueTag::Char || a.tag == ValueTag::FallbackToken) return a.i == b.i;
  if (a.tag == ValueTag::Float) return a.f == b.f;
  if (a.tag == ValueTag::String || a.tag == ValueTag::IntList || a.tag == ValueTag::FloatList ||
      a.tag == ValueTag::StringList) return a.i == b.i;
  return false;
}

std::size_t fill_subtree_end_at(const AstProgram& program, std::size_t idx, std::vector<std::size_t>& out) {
  if (idx >= program.nodes.size()) throw std::runtime_error("prefix traversal out of range");
  std::size_t cur = idx + 1;
  for (int i = 0; i < node_arity(program.nodes[idx].kind); ++i) {
    cur = fill_subtree_end_at(program, cur, out);
  }
  out[idx] = cur;
  return cur;
}

int find_or_add_name(AstProgram& target, const std::string& name, std::unordered_map<std::string, int>& name2idx) {
  auto it = name2idx.find(name);
  if (it != name2idx.end()) return it->second;
  const int idx = static_cast<int>(target.names.size());
  target.names.push_back(name);
  name2idx[name] = idx;
  return idx;
}

int find_or_add_const(AstProgram& target, const Value& value) {
  for (std::size_t i = 0; i < target.consts.size(); ++i) {
    if (value_equal(target.consts[i], value)) return static_cast<int>(i);
  }
  const int idx = static_cast<int>(target.consts.size());
  target.consts.push_back(value);
  return idx;
}

RType infer_name_type_from_string(const std::string& name) {
  if (name == "strings" || name == "words" || name == "string_list" ||
      (name.size() >= 8 && name.compare(name.size() - 8, 8, "_strings") == 0) ||
      (name.size() >= 6 && name.compare(name.size() - 6, 6, "_words") == 0)) {
    return RType::StringList;
  }
  if (name == "fs" || name == "floats" || name == "float_list" ||
      (name.size() >= 6 && name.compare(name.size() - 6, 6, "_float") == 0) ||
      (name.size() >= 7 && name.compare(name.size() - 7, 7, "_floats") == 0)) {
    return RType::FloatList;
  }
  if (name == "xs" || name == "items" || name == "list" ||
      (name.size() >= 5 && name.compare(name.size() - 5, 5, "_list") == 0)) {
    return RType::IntList;
  }
  if (name == "s" || name == "str" || name == "text" ||
      (name.size() >= 7 && name.compare(name.size() - 7, 7, "_string") == 0)) {
    return RType::String;
  }
  if (name == "f" || name == "flt" || name == "float" ||
      (name.size() >= 2 && name.compare(name.size() - 2, 2, "_f") == 0)) {
    return RType::Float;
  }
  return RType::Int;
}

std::vector<int> collect_name_ids_for_type(const AstProgram& target, RType type) {
  std::vector<int> out;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    const RType inferred = infer_name_type_from_string(target.names[i]);
    if (type == RType::Any || inferred == type) {
      out.push_back(static_cast<int>(i));
    }
  }
  return out;
}

int choose_name_id(std::mt19937_64& rng, const std::vector<int>& ids) {
  if (ids.empty()) {
    return -1;
  }
  return ids[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(ids.size()) - 1))];
}

std::vector<AstNode> map_subtree_nodes_into(AstProgram& target,
                                            const AstProgram& donor,
                                            std::size_t start,
                                            std::size_t stop) {
  std::unordered_map<std::string, int> name2idx;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    name2idx[target.names[i]] = static_cast<int>(i);
  }
  std::unordered_map<int, int> name_map;
  std::unordered_map<int, int> const_map;
  std::vector<AstNode> out;
  out.reserve(stop - start);
  for (std::size_t i = start; i < stop; ++i) {
    AstNode node = donor.nodes[i];
    if (node.kind == NodeKind::CONST) {
      auto it = const_map.find(node.i0);
      if (it == const_map.end()) {
        const int mapped = find_or_add_const(target, donor.consts.at(static_cast<std::size_t>(node.i0)));
        const_map[node.i0] = mapped;
        node.i0 = mapped;
      } else {
        node.i0 = it->second;
      }
    } else if (node.kind == NodeKind::VAR || node.kind == NodeKind::BOUND_VAR ||
               node.kind == NodeKind::ASSIGN || node.kind == NodeKind::FOR_RANGE ||
               node.kind == NodeKind::MAP_LIST || node.kind == NodeKind::FILTER_LIST) {
      auto it = name_map.find(node.i0);
      if (it == name_map.end()) {
        const int mapped = find_or_add_name(target, donor.names.at(static_cast<std::size_t>(node.i0)), name2idx);
        name_map[node.i0] = mapped;
        node.i0 = mapped;
      } else {
        node.i0 = it->second;
      }
    }
    out.push_back(node);
  }
  return out;
}

int remap_name_from_donor(AstProgram& target, const AstProgram& donor, int donor_name_id) {
  std::unordered_map<std::string, int> name2idx;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    name2idx[target.names[i]] = static_cast<int>(i);
  }
  return find_or_add_name(target, donor.names.at(static_cast<std::size_t>(donor_name_id)), name2idx);
}

int remap_const_from_donor(AstProgram& target, const AstProgram& donor, int donor_const_id) {
  return find_or_add_const(target, donor.consts.at(static_cast<std::size_t>(donor_const_id)));
}

std::vector<int> remap_names_from_donor(AstProgram& target,
                                        const AstProgram& donor,
                                        const std::vector<int>& donor_name_ids) {
  std::vector<int> out;
  out.reserve(donor_name_ids.size());
  for (int donor_name_id : donor_name_ids) {
    out.push_back(remap_name_from_donor(target, donor, donor_name_id));
  }
  return out;
}

bool can_emit_asgp_dc_for_type(RType type, const GrammarConfig& grammar) {
  if (!grammar.expression_asgp_dc || !grammar.builtin_index) {
    return false;
  }
  return (grammar.binary_add &&
          ((type == RType::Int && grammar.value_int && grammar.value_int_list) ||
           (type == RType::Float && grammar.value_float && grammar.value_float_list))) ||
         (grammar.builtin_concat && grammar.builtin_singleton &&
          type == RType::String && grammar.value_string && grammar.value_char);
}

bool can_emit_asgp_dp1d_for_type(RType type, const GrammarConfig& grammar) {
  if (!grammar.expression_asgp_dp1d || !grammar.value_int) {
    return false;
  }
  return (grammar.binary_add &&
          ((type == RType::Int && grammar.value_int) || (type == RType::Float && grammar.value_float))) ||
         (type == RType::String && grammar.value_string && grammar.builtin_concat);
}

bool can_emit_asgp_dp2d_for_type(RType type, const GrammarConfig& grammar) {
  if (!grammar.expression_asgp_dp2d || !grammar.value_int) {
    return false;
  }
  return (grammar.binary_add &&
          ((type == RType::Int && grammar.value_int) || (type == RType::Float && grammar.value_float))) ||
         (type == RType::String && grammar.value_string && grammar.builtin_concat);
}

Value asgp_dc_source_literal(RType type) {
  if (type == RType::Float) {
    return gagp::payload::make_float_list_value({
        Value::from_float(1.0),
        Value::from_float(2.0),
        Value::from_float(3.0),
    });
  }
  if (type == RType::String) {
    return gagp::payload::make_string_value("abc");
  }
  return gagp::payload::make_int_list_value({
      Value::from_int(1),
      Value::from_int(2),
      Value::from_int(3),
  });
}

int find_or_add_name_in_program(AstProgram& target, const std::string& name) {
  std::unordered_map<std::string, int> name2idx;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    name2idx[target.names[i]] = static_cast<int>(i);
  }
  return find_or_add_name(target, name, name2idx);
}

std::vector<AstNode> make_asgp_dc_nodes_for_type(AstProgram& target, RType type, int source_name_id = -1) {
  const std::size_t suffix = target.names.size();
  const int solve_xs_name = find_or_add_name_in_program(target, "__asgp_xs" + std::to_string(suffix));
  const int solve_n_name = find_or_add_name_in_program(target, "__asgp_n" + std::to_string(suffix));
  const int solve_lo_name = find_or_add_name_in_program(target, "__asgp_lo" + std::to_string(suffix));
  const int divide_n_name = find_or_add_name_in_program(target, "__asgp_dn" + std::to_string(suffix));
  const int combine_left_name = find_or_add_name_in_program(target, "__asgp_l" + std::to_string(suffix));
  const int combine_right_name = find_or_add_name_in_program(target, "__asgp_r" + std::to_string(suffix));
  const int index_const = find_or_add_const(target, Value::from_int(0));
  const int split_const = find_or_add_const(target, Value::from_int(1));

  target.asgp_dc_binders.push_back(AsgpDcBinders{
      0,
      solve_xs_name,
      solve_n_name,
      solve_lo_name,
      divide_n_name,
      combine_left_name,
      combine_right_name,
  });
  std::vector<AstNode> out = {
      AstNode{NodeKind::ASGP_DC, 0, 0},
  };
  if (source_name_id >= 0 && static_cast<std::size_t>(source_name_id) < target.names.size()) {
    out.push_back(AstNode{NodeKind::VAR, source_name_id, 0});
  } else {
    const int source_const = find_or_add_const(target, asgp_dc_source_literal(type));
    out.push_back(AstNode{NodeKind::CONST, source_const, 0});
  }
  out.insert(out.end(), {
      AstNode{type == RType::String ? NodeKind::CALL_SINGLETON : NodeKind::CALL_INDEX, 0, 0},
  });
  if (type == RType::String) {
    out.push_back(AstNode{NodeKind::CALL_INDEX, 0, 0});
  }
  out.insert(out.end(), {
      AstNode{NodeKind::BOUND_VAR, solve_xs_name, 0},
      AstNode{NodeKind::CONST, index_const, 0},
      AstNode{NodeKind::CONST, split_const, 0},
      AstNode{type == RType::String ? NodeKind::CALL_CONCAT : NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, combine_left_name, 0},
      AstNode{NodeKind::BOUND_VAR, combine_right_name, 0},
  });
  return out;
}

Value scalar_one_literal(RType type) {
  return type == RType::Float ? Value::from_float(1.0) : Value::from_int(1);
}

Value scalar_zero_literal(RType type) {
  return type == RType::Float ? Value::from_float(0.0) : Value::from_int(0);
}

Value dp_one_literal(RType type) {
  return type == RType::String ? gagp::payload::make_string_value("a") : scalar_one_literal(type);
}

Value dp_zero_literal(RType type) {
  return type == RType::String ? gagp::payload::make_string_value("") : scalar_zero_literal(type);
}

std::vector<AstNode> make_asgp_dp1d_nodes_for_type(AstProgram& target, RType type) {
  const std::size_t suffix = target.names.size();
  const int solve_state_name = find_or_add_name_in_program(target, "__asgp_s" + std::to_string(suffix));
  const int transition_state_name = find_or_add_name_in_program(target, "__asgp_ts" + std::to_string(suffix));
  const int dep_name = find_or_add_name_in_program(target, "__asgp_d" + std::to_string(suffix));
  const int state_const = find_or_add_const(target, Value::from_int(4));
  const int solve_const = find_or_add_const(target, dp_one_literal(type));
  const int one_const = find_or_add_const(target, dp_one_literal(type));
  const int boundary_const = find_or_add_const(target, dp_zero_literal(type));

  target.asgp_dp1d_specs.push_back(AsgpDp1dSpec{
      0,
      0,
      5,
      0,
      boundary_const,
      NodeKind::DP1_BACKWARD1,
      {1},
      solve_state_name,
      transition_state_name,
      {dep_name},
  });
  return {
      AstNode{NodeKind::ASGP_DP1D, 0, 0},
      AstNode{NodeKind::CONST, state_const, 0},
      AstNode{NodeKind::CONST, solve_const, 0},
      AstNode{type == RType::String ? NodeKind::CALL_CONCAT : NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, dep_name, 0},
      AstNode{NodeKind::CONST, one_const, 0},
  };
}

std::vector<AstNode> make_asgp_dp2d_nodes_for_type(AstProgram& target, RType type) {
  const std::size_t suffix = target.names.size();
  const int solve_i_name = find_or_add_name_in_program(target, "__asgp_i" + std::to_string(suffix));
  const int solve_j_name = find_or_add_name_in_program(target, "__asgp_j" + std::to_string(suffix));
  const int transition_i_name = find_or_add_name_in_program(target, "__asgp_ti" + std::to_string(suffix));
  const int transition_j_name = find_or_add_name_in_program(target, "__asgp_tj" + std::to_string(suffix));
  const int dep_a_name = find_or_add_name_in_program(target, "__asgp_da" + std::to_string(suffix));
  const int dep_b_name = find_or_add_name_in_program(target, "__asgp_db" + std::to_string(suffix));
  const int state_i_const = find_or_add_const(target, Value::from_int(2));
  const int state_j_const = find_or_add_const(target, Value::from_int(2));
  const int solve_const = find_or_add_const(target, dp_one_literal(type));
  const int boundary_const = find_or_add_const(target, dp_zero_literal(type));

  target.asgp_dp2d_specs.push_back(AsgpDp2dSpec{
      0,
      0,
      3,
      0,
      3,
      0,
      0,
      boundary_const,
      NodeKind::DP2_CROSS_BACKWARD,
      solve_i_name,
      solve_j_name,
      transition_i_name,
      transition_j_name,
      {dep_a_name, dep_b_name},
  });
  return {
      AstNode{NodeKind::ASGP_DP2D, 0, 0},
      AstNode{NodeKind::CONST, state_i_const, 0},
      AstNode{NodeKind::CONST, state_j_const, 0},
      AstNode{NodeKind::CONST, solve_const, 0},
      AstNode{type == RType::String ? NodeKind::CALL_CONCAT : NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, dep_a_name, 0},
      AstNode{NodeKind::BOUND_VAR, dep_b_name, 0},
  };
}

}  // namespace

int node_arity(NodeKind kind) {
  return node_descriptor(kind).prefix_arity;
}

std::vector<std::size_t> build_subtree_end(const AstProgram& program) {
  std::vector<std::size_t> out(program.nodes.size(), 0);
  if (!program.nodes.empty()) {
    const std::size_t end = fill_subtree_end_at(program, 0, out);
    if (end != program.nodes.size()) throw std::runtime_error("prefix trailing tokens");
  }
  return out;
}

std::vector<AstNode> make_random_expr_nodes_for_type(std::mt19937_64& rng,
                                                     AstProgram& target,
                                                     RType type,
                                                     int depth,
                                                     const GrammarConfig& grammar,
                                                     bool allow_asgp) {
  (void)depth;
  grammar.validate();
  if (!grammar.allows_type(type)) {
    type = RType::Int;
  }
  AstProgram donor;
  const std::vector<int> int_name_ids =
      (grammar.expression_var && grammar.allows_type(RType::Int)) ? collect_name_ids_for_type(target, RType::Int)
                                                                  : std::vector<int>{};
  const std::vector<int> float_name_ids =
      (grammar.expression_var && grammar.allows_type(RType::Float)) ? collect_name_ids_for_type(target, RType::Float)
                                                                    : std::vector<int>{};
  const std::vector<int> string_name_ids =
      (grammar.expression_var && grammar.value_string) ? collect_name_ids_for_type(target, RType::String)
                                                       : std::vector<int>{};
  const std::vector<int> int_list_name_ids =
      (grammar.expression_var && grammar.value_int_list) ? collect_name_ids_for_type(target, RType::IntList)
                                                         : std::vector<int>{};
  const std::vector<int> float_list_name_ids =
      (grammar.expression_var && grammar.value_float_list) ? collect_name_ids_for_type(target, RType::FloatList)
                                                           : std::vector<int>{};
  const std::vector<int> string_list_name_ids =
      (grammar.expression_var && grammar.value_string_list) ? collect_name_ids_for_type(target, RType::StringList)
                                                            : std::vector<int>{};
  const std::vector<int> container_name_ids = [&]() {
    std::vector<int> out = string_name_ids;
    out.insert(out.end(), int_list_name_ids.begin(), int_list_name_ids.end());
    out.insert(out.end(), float_list_name_ids.begin(), float_list_name_ids.end());
    out.insert(out.end(), string_list_name_ids.begin(), string_list_name_ids.end());
    return out;
  }();
  const std::vector<int> numeric_name_ids = [&]() {
    std::vector<int> out = int_name_ids;
    out.insert(out.end(), float_name_ids.begin(), float_name_ids.end());
    return out;
  }();
  const int int_name_id = choose_name_id(rng, int_name_ids);
  const int float_name_id = choose_name_id(rng, float_name_ids);
  const int numeric_name_id = choose_name_id(rng, numeric_name_ids);
  const int container_name_id = choose_name_id(rng, container_name_ids);

  if (allow_asgp && (type == RType::Int || type == RType::Float || type == RType::String) &&
      depth > 1 && rand_prob(rng, 0.18)) {
    std::vector<int> asgp_forms;
    if (can_emit_asgp_dc_for_type(type, grammar)) asgp_forms.push_back(0);
    if (can_emit_asgp_dp1d_for_type(type, grammar)) asgp_forms.push_back(1);
    if (can_emit_asgp_dp2d_for_type(type, grammar)) asgp_forms.push_back(2);
    if (!asgp_forms.empty()) {
      const int form = asgp_forms[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(asgp_forms.size()) - 1))];
      if (form == 0) {
        const std::vector<int>& source_names = type == RType::Float ? float_list_name_ids :
                                               (type == RType::String ? string_name_ids
                                                                      : int_list_name_ids);
        const int source_name_id = rand_prob(rng, 0.65) ? choose_name_id(rng, source_names) : -1;
        return make_asgp_dc_nodes_for_type(target, type, source_name_id);
      }
      if (form == 1) return make_asgp_dp1d_nodes_for_type(target, type);
      return make_asgp_dp2d_nodes_for_type(target, type);
    }
  }

  if ((type == RType::IntList || type == RType::FloatList || type == RType::StringList) && depth > 0 &&
      grammar.expression_var && (grammar.expression_map_list || grammar.expression_filter_list) &&
      rand_prob(rng, 0.35)) {
    const bool use_filter = grammar.expression_filter_list && grammar.value_bool &&
                            (!grammar.expression_map_list || rand_prob(rng, 0.5));
    donor.names = {"__map_u"};
    if (type == RType::IntList) {
      donor.consts.push_back(gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}));
    } else if (type == RType::FloatList) {
      donor.consts.push_back(
          gagp::payload::make_float_list_value({Value::from_float(1.0), Value::from_float(2.0)}));
    } else {
      donor.consts.push_back(gagp::payload::make_string_list_value({
          gagp::payload::make_string_value("a"),
          gagp::payload::make_string_value("b"),
      }));
    }
    if (use_filter) {
      donor.consts.push_back(Value::from_bool(true));
      donor.nodes.push_back(AstNode{NodeKind::FILTER_LIST, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::CONST, 1, 0});
    } else {
      const int out_tag = type == RType::IntList ? static_cast<int>(ListTypeTag::Int) :
                          (type == RType::FloatList ? static_cast<int>(ListTypeTag::Float)
                                                     : static_cast<int>(ListTypeTag::String));
      donor.nodes.push_back(AstNode{NodeKind::MAP_LIST, 0, out_tag});
      donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::BOUND_VAR, 0, 0});
    }
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }

  if (type == RType::Bool && grammar.binary_lt && numeric_name_id >= 0 && rand_prob(rng, 0.6)) {
    const RType numeric_type = infer_name_type_from_string(target.names[static_cast<std::size_t>(numeric_name_id)]);
    donor.names.push_back(target.names[static_cast<std::size_t>(numeric_name_id)]);
    donor.nodes.push_back(AstNode{NodeKind::LT, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    donor.consts.push_back(numeric_type == RType::Float
                               ? Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0)
                               : Value::from_int(rand_int(rng, -8, 8)));
    donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if (type == RType::Bool && grammar.binary_lt && grammar.builtin_len && container_name_id >= 0 && rand_prob(rng, 0.4)) {
    donor.names.push_back(target.names[static_cast<std::size_t>(container_name_id)]);
    donor.nodes.push_back(AstNode{NodeKind::LT, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    donor.consts.push_back(Value::from_int(rand_int(rng, -8, 8)));
    donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  const int same_type_name_id = (type == RType::Float) ? float_name_id : int_name_id;
  if ((type == RType::Int || type == RType::Float) && grammar.expression_var && same_type_name_id >= 0 &&
      rand_prob(rng, 0.45)) {
    donor.names.push_back(target.names[static_cast<std::size_t>(same_type_name_id)]);
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if ((type == RType::Int || type == RType::Float) && container_name_id >= 0 &&
      (grammar.builtin_len || (grammar.builtin_index && (grammar.value_int_list || grammar.value_float_list))) &&
      rand_prob(rng, 0.55)) {
    const RType container_type = infer_name_type_from_string(target.names[static_cast<std::size_t>(container_name_id)]);
    donor.names.push_back(target.names[static_cast<std::size_t>(container_name_id)]);
    const bool indexable_numeric_list =
        container_type == RType::IntList || container_type == RType::FloatList;
    if (type == RType::Int && grammar.builtin_len &&
        (!indexable_numeric_list || !grammar.builtin_index || rand_prob(rng, 0.5))) {
      donor.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    } else if (grammar.builtin_index &&
               ((type == RType::Int && container_type == RType::IntList) ||
                (type == RType::Float && container_type == RType::FloatList))) {
      donor.nodes.push_back(AstNode{NodeKind::CALL_INDEX, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
      donor.consts.push_back(Value::from_int(rand_int(rng, -6, 6)));
      donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    } else {
      donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
      donor.consts.push_back(grammar.value_int ? Value::from_int(0) : Value::from_float(0.0));
    }
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  Value value = grammar.value_int ? Value::from_int(0) : Value::from_float(0.0);
  if (type == RType::Int) {
    value = Value::from_int(rand_int(rng, -8, 8));
  } else if (type == RType::Bool) value = Value::from_bool(rand_prob(rng, 0.5));
  else if (type == RType::Float) {
    value = Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0);
  } else if (type == RType::Char) {
    static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz";
    value = Value::from_char(kAlphabet[rand_int(rng, 0, 25)]);
  }
  else if (type == RType::String) {
    static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz";
    const int len = rand_int(rng, 0, 8);
    std::string s;
    s.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      s.push_back(kAlphabet[rand_int(rng, 0, 25)]);
    }
    value = gagp::payload::make_string_value(s);
  } else if (type == RType::IntList) {
    const int len = rand_int(rng, 0, 4);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      elems.push_back(Value::from_int(rand_int(rng, -8, 8)));
    }
    value = gagp::payload::make_int_list_value(elems);
  } else if (type == RType::FloatList) {
    const int len = rand_int(rng, 0, 4);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      elems.push_back(Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0));
    }
    value = gagp::payload::make_float_list_value(elems);
  } else if (type == RType::StringList) {
    static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz";
    const int len = rand_int(rng, 0, 4);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      const int s_len = rand_int(rng, 0, 5);
      std::string s;
      s.reserve(static_cast<std::size_t>(s_len));
      for (int j = 0; j < s_len; ++j) s.push_back(kAlphabet[rand_int(rng, 0, 25)]);
      elems.push_back(gagp::payload::make_string_value(s));
    }
    value = gagp::payload::make_string_list_value(elems);
  } else if (grammar.value_int && (!grammar.value_float || rand_prob(rng, 0.5))) {
    value = Value::from_int(rand_int(rng, -8, 8));
  } else {
    value = Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0);
  }
  donor.consts.push_back(value);
  donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
  return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
}

AstProgram replace_subtree(const AstProgram& base,
                           std::size_t target_start,
                           std::size_t target_stop,
                           const AstProgram& donor,
                           std::size_t donor_start,
                           std::size_t donor_stop) {
  AstProgram out;
  out.version = k_ast_prefix_version_current;
  out.names = base.names;
  out.consts = base.consts;
  std::vector<AstNode> donor_nodes = map_subtree_nodes_into(out, donor, donor_start, donor_stop);
  const std::size_t removed = target_stop - target_start;
  const std::size_t inserted = donor_nodes.size();
  for (const LinearRecBinders& binders : base.linear_rec_binders) {
    if (binders.node_index >= target_start && binders.node_index < target_stop) {
      continue;
    }
    LinearRecBinders shifted = binders;
    if (shifted.node_index >= target_stop) {
      shifted.node_index = shifted.node_index - removed + inserted;
    }
    out.linear_rec_binders.push_back(shifted);
  }
  for (const LinearRecBinders& binders : donor.linear_rec_binders) {
    if (binders.node_index < donor_start || binders.node_index >= donor_stop) {
      continue;
    }
    out.linear_rec_binders.push_back(LinearRecBinders{
        target_start + (binders.node_index - donor_start),
        remap_name_from_donor(out, donor, binders.elem_name),
        remap_name_from_donor(out, donor, binders.accum_name),
        remap_name_from_donor(out, donor, binders.index_name),
    });
  }
  for (const AsgpDcBinders& binders : base.asgp_dc_binders) {
    if (binders.node_index >= target_start && binders.node_index < target_stop) {
      continue;
    }
    AsgpDcBinders shifted = binders;
    if (shifted.node_index >= target_stop) {
      shifted.node_index = shifted.node_index - removed + inserted;
    }
    out.asgp_dc_binders.push_back(shifted);
  }
  for (const AsgpDcBinders& binders : donor.asgp_dc_binders) {
    if (binders.node_index < donor_start || binders.node_index >= donor_stop) {
      continue;
    }
    out.asgp_dc_binders.push_back(AsgpDcBinders{
        target_start + (binders.node_index - donor_start),
        remap_name_from_donor(out, donor, binders.solve_xs_name),
        remap_name_from_donor(out, donor, binders.solve_n_name),
        remap_name_from_donor(out, donor, binders.solve_lo_name),
        remap_name_from_donor(out, donor, binders.divide_n_name),
        remap_name_from_donor(out, donor, binders.combine_left_name),
        remap_name_from_donor(out, donor, binders.combine_right_name),
    });
  }
  for (const AsgpDp1dSpec& spec : base.asgp_dp1d_specs) {
    if (spec.node_index >= target_start && spec.node_index < target_stop) {
      continue;
    }
    AsgpDp1dSpec shifted = spec;
    if (shifted.node_index >= target_stop) {
      shifted.node_index = shifted.node_index - removed + inserted;
    }
    out.asgp_dp1d_specs.push_back(shifted);
  }
  for (const AsgpDp1dSpec& spec : donor.asgp_dp1d_specs) {
    if (spec.node_index < donor_start || spec.node_index >= donor_stop) {
      continue;
    }
    out.asgp_dp1d_specs.push_back(AsgpDp1dSpec{
        target_start + (spec.node_index - donor_start),
        spec.lo,
        spec.hi,
        spec.base_state,
        remap_const_from_donor(out, donor, spec.boundary_const),
        spec.dep_kind,
        spec.dep_offsets,
        remap_name_from_donor(out, donor, spec.solve_state_name),
        remap_name_from_donor(out, donor, spec.transition_state_name),
        remap_names_from_donor(out, donor, spec.transition_dep_names),
    });
  }
  for (const AsgpDp2dSpec& spec : base.asgp_dp2d_specs) {
    if (spec.node_index >= target_start && spec.node_index < target_stop) {
      continue;
    }
    AsgpDp2dSpec shifted = spec;
    if (shifted.node_index >= target_stop) {
      shifted.node_index = shifted.node_index - removed + inserted;
    }
    out.asgp_dp2d_specs.push_back(shifted);
  }
  for (const AsgpDp2dSpec& spec : donor.asgp_dp2d_specs) {
    if (spec.node_index < donor_start || spec.node_index >= donor_stop) {
      continue;
    }
    out.asgp_dp2d_specs.push_back(AsgpDp2dSpec{
        target_start + (spec.node_index - donor_start),
        spec.i_lo,
        spec.i_hi,
        spec.j_lo,
        spec.j_hi,
        spec.base_i,
        spec.base_j,
        remap_const_from_donor(out, donor, spec.boundary_const),
        spec.dep_kind,
        remap_name_from_donor(out, donor, spec.solve_i_name),
        remap_name_from_donor(out, donor, spec.solve_j_name),
        remap_name_from_donor(out, donor, spec.transition_i_name),
        remap_name_from_donor(out, donor, spec.transition_j_name),
        remap_names_from_donor(out, donor, spec.transition_dep_names),
    });
  }
  out.nodes.reserve(base.nodes.size() - (target_stop - target_start) + donor_nodes.size());
  out.nodes.insert(out.nodes.end(), base.nodes.begin(), base.nodes.begin() + static_cast<std::ptrdiff_t>(target_start));
  out.nodes.insert(out.nodes.end(), donor_nodes.begin(), donor_nodes.end());
  out.nodes.insert(out.nodes.end(), base.nodes.begin() + static_cast<std::ptrdiff_t>(target_stop), base.nodes.end());
  return out;
}

}  // namespace gagp::evo::subtree
