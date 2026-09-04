#include "gagp/evolution/genome_generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gagp/evolution/ast_verify.hpp"
#include "gagp/runtime/payload/payload.hpp"
#include "subtree_utils.hpp"

namespace gagp::evo {

namespace {

struct PrefixGenCtx {
  std::set<int> int_names;
  std::set<int> float_names;
  std::set<int> bool_names;
  std::set<int> char_names;
  std::set<int> string_names;
  std::set<int> int_list_names;
  std::set<int> float_list_names;
  std::set<int> string_list_names;
  std::set<int> any_names;
  int tmp_idx = 0;
};

template <typename T>
const T& choose_one(std::mt19937_64& rng, const std::vector<T>& values) {
  if (values.empty()) {
    throw std::runtime_error("choose_one on empty vector");
  }
  std::uniform_int_distribution<int> dist(0, static_cast<int>(values.size()) - 1);
  const int idx = dist(rng);
  return values[static_cast<std::size_t>(idx)];
}

std::vector<RType> filter_types(const GrammarConfig& grammar, const std::vector<RType>& values) {
  std::vector<RType> out;
  out.reserve(values.size());
  for (RType type : values) {
    if (grammar.allows_type(type)) {
      out.push_back(type);
    }
  }
  return out;
}

std::vector<NodeKind> filter_node_kinds(const GrammarConfig& grammar, const std::vector<NodeKind>& values) {
  std::vector<NodeKind> out;
  out.reserve(values.size());
  for (NodeKind kind : values) {
    if (grammar.allows_node_kind(kind)) {
      out.push_back(kind);
    }
  }
  return out;
}

RType choose_type(std::mt19937_64& rng, const GrammarConfig& grammar, const std::vector<RType>& values) {
  const std::vector<RType> enabled = filter_types(grammar, values);
  if (!enabled.empty()) {
    return choose_one(rng, enabled);
  }
  return grammar.value_int ? RType::Int : RType::Float;
}

RType choose_any_type(std::mt19937_64& rng, const GrammarConfig& grammar) {
  return choose_type(rng, grammar, {RType::Int, RType::Float, RType::Bool, RType::Char, RType::String, RType::IntList, RType::FloatList, RType::StringList});
}

RType coerce_type(std::mt19937_64& rng, const GrammarConfig& grammar, RType type) {
  if (type == RType::Any) {
    return choose_any_type(rng, grammar);
  }
  if (grammar.allows_type(type)) {
    return type;
  }
  return choose_any_type(rng, grammar);
}

int ensure_name(AstProgram& program, const std::string& name) {
  for (std::size_t i = 0; i < program.names.size(); ++i) {
    if (program.names[i] == name) {
      return static_cast<int>(i);
    }
  }
  program.names.push_back(name);
  return static_cast<int>(program.names.size() - 1);
}

int append_const_id(AstProgram& program, const Value& value) {
  program.consts.push_back(value);
  return static_cast<int>(program.consts.size() - 1);
}

ProgramGenome as_genome_prefix(const AstProgram& ast) {
  ProgramGenome genome;
  genome.ast = ast;
  genome.meta = build_genome_meta(genome.ast);
  return genome;
}

int choose_name_for_type(std::mt19937_64& rng, const PrefixGenCtx& ctx, RType type) {
  std::vector<int> names;
  if (type == RType::Int) names.assign(ctx.int_names.begin(), ctx.int_names.end());
  else if (type == RType::Float) names.assign(ctx.float_names.begin(), ctx.float_names.end());
  else if (type == RType::Bool) names.assign(ctx.bool_names.begin(), ctx.bool_names.end());
  else if (type == RType::Char) names.assign(ctx.char_names.begin(), ctx.char_names.end());
  else if (type == RType::String) names.assign(ctx.string_names.begin(), ctx.string_names.end());
  else if (type == RType::IntList) names.assign(ctx.int_list_names.begin(), ctx.int_list_names.end());
  else if (type == RType::FloatList) names.assign(ctx.float_list_names.begin(), ctx.float_list_names.end());
  else if (type == RType::StringList) names.assign(ctx.string_list_names.begin(), ctx.string_list_names.end());
  else if (type == RType::Any) {
    names.assign(ctx.int_names.begin(), ctx.int_names.end());
    names.insert(names.end(), ctx.float_names.begin(), ctx.float_names.end());
    names.insert(names.end(), ctx.bool_names.begin(), ctx.bool_names.end());
    names.insert(names.end(), ctx.char_names.begin(), ctx.char_names.end());
    names.insert(names.end(), ctx.string_names.begin(), ctx.string_names.end());
    names.insert(names.end(), ctx.int_list_names.begin(), ctx.int_list_names.end());
    names.insert(names.end(), ctx.float_list_names.begin(), ctx.float_list_names.end());
    names.insert(names.end(), ctx.string_list_names.begin(), ctx.string_list_names.end());
    names.insert(names.end(), ctx.any_names.begin(), ctx.any_names.end());
  }
  if (type != RType::Any) {
    names.insert(names.end(), ctx.any_names.begin(), ctx.any_names.end());
  }
  if (names.empty()) {
    return -1;
  }
  return choose_one(rng, names);
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

Value random_int_list_literal(std::mt19937_64& rng) {
  const int len = std::uniform_int_distribution<int>(0, 5)(rng);
  std::vector<Value> elems;
  elems.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    elems.push_back(Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng)));
  }
  return gagp::payload::make_int_list_value(elems);
}

Value random_float_list_literal(std::mt19937_64& rng) {
  const int len = std::uniform_int_distribution<int>(0, 5)(rng);
  std::vector<Value> elems;
  elems.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    elems.push_back(Value::from_float(
        std::round(std::uniform_real_distribution<double>(-8.0, 8.0)(rng) * 1000.0) / 1000.0));
  }
  return gagp::payload::make_float_list_value(elems);
}

Value random_string_list_literal(std::mt19937_64& rng) {
  const int len = std::uniform_int_distribution<int>(0, 5)(rng);
  std::vector<Value> elems;
  elems.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    elems.push_back(gagp::payload::make_string_value(random_string_literal(rng)));
  }
  return gagp::payload::make_string_list_value(elems);
}

bool can_emit_asgp_dc_for_type(RType target, const GrammarConfig& grammar) {
  if (!grammar.expression_asgp_dc || !grammar.builtin_index) {
    return false;
  }
  return (grammar.binary_add &&
          ((target == RType::Int && grammar.value_int && grammar.value_int_list) ||
           (target == RType::Float && grammar.value_float && grammar.value_float_list))) ||
         (grammar.builtin_concat && grammar.builtin_singleton &&
          target == RType::String && grammar.value_string && grammar.value_char);
}

bool can_emit_asgp_dp1d_for_type(RType target, const GrammarConfig& grammar) {
  if (!grammar.expression_asgp_dp1d || !grammar.value_int) {
    return false;
  }
  return (grammar.binary_add &&
          ((target == RType::Int && grammar.value_int) || (target == RType::Float && grammar.value_float))) ||
         (target == RType::String && grammar.value_string && grammar.builtin_concat);
}

bool can_emit_asgp_dp2d_for_type(RType target, const GrammarConfig& grammar) {
  if (!grammar.expression_asgp_dp2d || !grammar.value_int) {
    return false;
  }
  return (grammar.binary_add &&
          ((target == RType::Int && grammar.value_int) || (target == RType::Float && grammar.value_float))) ||
         (target == RType::String && grammar.value_string && grammar.builtin_concat);
}

Value asgp_dc_source_literal(RType target) {
  if (target == RType::Float) {
    return gagp::payload::make_float_list_value({
        Value::from_float(1.0),
        Value::from_float(2.0),
        Value::from_float(3.0),
    });
  }
  if (target == RType::String) {
    return gagp::payload::make_string_value("abc");
  }
  return gagp::payload::make_int_list_value({
      Value::from_int(1),
      Value::from_int(2),
      Value::from_int(3),
  });
}

Value scalar_one_literal(RType target) {
  return target == RType::Float ? Value::from_float(1.0) : Value::from_int(1);
}

Value scalar_zero_literal(RType target) {
  return target == RType::Float ? Value::from_float(0.0) : Value::from_int(0);
}

Value dp_one_literal(RType target) {
  return target == RType::String ? gagp::payload::make_string_value("a") : scalar_one_literal(target);
}

Value dp_zero_literal(RType target) {
  return target == RType::String ? gagp::payload::make_string_value("") : scalar_zero_literal(target);
}

void assign_name_type(PrefixGenCtx& ctx, int name_id, RType type) {
  ctx.int_names.erase(name_id);
  ctx.float_names.erase(name_id);
  ctx.bool_names.erase(name_id);
  ctx.char_names.erase(name_id);
  ctx.string_names.erase(name_id);
  ctx.int_list_names.erase(name_id);
  ctx.float_list_names.erase(name_id);
  ctx.string_list_names.erase(name_id);
  ctx.any_names.erase(name_id);
  if (type == RType::Int) ctx.int_names.insert(name_id);
  if (type == RType::Float) ctx.float_names.insert(name_id);
  if (type == RType::Bool) ctx.bool_names.insert(name_id);
  if (type == RType::Char) ctx.char_names.insert(name_id);
  if (type == RType::String) ctx.string_names.insert(name_id);
  if (type == RType::IntList) ctx.int_list_names.insert(name_id);
  if (type == RType::FloatList) ctx.float_list_names.insert(name_id);
  if (type == RType::StringList) ctx.string_list_names.insert(name_id);
  if (type == RType::Any) ctx.any_names.insert(name_id);
}

void seed_input_names(AstProgram& program,
                      PrefixGenCtx& ctx,
                      const std::vector<InputSpec>& input_specs,
                      const GrammarConfig& grammar) {
  if (input_specs.empty()) {
    if (grammar.allows_type(RType::Int) && grammar.expression_var) {
      ctx.int_names.insert(ensure_name(program, "x"));
    } else if (grammar.allows_type(RType::Float) && grammar.expression_var) {
      ctx.float_names.insert(ensure_name(program, "x"));
    }
    return;
  }
  for (const InputSpec& spec : input_specs) {
    if (spec.name.empty()) {
      continue;
    }
    const int name_id = ensure_name(program, spec.name);
    const RType type = spec.type == RType::Invalid ? RType::Any : spec.type;
    if (grammar.allows_type(type)) {
      assign_name_type(ctx, name_id, type);
    }
  }
}

void emit_random_leaf(std::mt19937_64& rng,
                      AstProgram& program,
                      PrefixGenCtx& ctx,
                      RType target,
                      const GrammarConfig& grammar) {
  target = coerce_type(rng, grammar, target);
  const int var_id = choose_name_for_type(rng, ctx, target);
  if (grammar.expression_var && var_id >= 0 && std::bernoulli_distribution(0.45)(rng)) {
    program.nodes.push_back(AstNode{NodeKind::VAR, var_id, 0});
    return;
  }
  if (target == RType::Int) {
    program.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng))), 0});
    return;
  }
  if (target == RType::Float) {
    program.nodes.push_back(AstNode{NodeKind::CONST,
                                    append_const_id(program, Value::from_float(
                                                                 std::round(std::uniform_real_distribution<double>(-8.0, 8.0)(rng) * 1000.0) / 1000.0)),
                                    0});
    return;
  }
  if (target == RType::Bool) {
    program.nodes.push_back(AstNode{
        NodeKind::CONST, append_const_id(program, Value::from_bool(std::bernoulli_distribution(0.5)(rng))), 0});
    return;
  }
  if (target == RType::Char) {
    static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789 ";
    const int idx = std::uniform_int_distribution<int>(0, static_cast<int>(sizeof(kAlphabet) - 2))(rng);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_char(kAlphabet[idx])), 0});
    return;
  }
  if (target == RType::String) {
    const Value value = gagp::payload::make_string_value(random_string_literal(rng));
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }
  if (target == RType::IntList) {
    const Value value = random_int_list_literal(rng);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }
  if (target == RType::FloatList) {
    const Value value = random_float_list_literal(rng);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }
  if (target == RType::StringList) {
    const Value value = random_string_list_literal(rng);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }

  emit_random_leaf(rng, program, ctx, choose_any_type(rng, grammar), grammar);
}

void emit_random_expr(std::mt19937_64& rng,
                      AstProgram& program,
                      PrefixGenCtx& ctx,
                      int depth,
                      RType target,
                      const GrammarConfig& grammar,
                      bool allow_asgp = false) {
  target = coerce_type(rng, grammar, target);
  if (depth <= 0) {
    emit_random_leaf(rng, program, ctx, target, grammar);
    return;
  }

  if (target == RType::Int || target == RType::Float) {
    const RType numeric_target = target;
    std::vector<int> choices{0};
    std::vector<NodeKind> numeric_op_candidates{NodeKind::ADD, NodeKind::SUB, NodeKind::MUL, NodeKind::MOD};
    if (numeric_target == RType::Float) {
      numeric_op_candidates.push_back(NodeKind::DIV);
    }
    const std::vector<NodeKind> num_ops = filter_node_kinds(grammar, numeric_op_candidates);
    const std::vector<RType> len_arg_types =
        filter_types(grammar, {RType::String, RType::IntList, RType::FloatList, RType::StringList});
    std::vector<NodeKind> builtins = filter_node_kinds(
        grammar, {NodeKind::CALL_ABS, NodeKind::CALL_MIN, NodeKind::CALL_MAX, NodeKind::CALL_CLIP});
    if (numeric_target == RType::Int && grammar.builtin_len && !len_arg_types.empty()) builtins.push_back(NodeKind::CALL_LEN);
    if (grammar.builtin_index &&
        ((numeric_target == RType::Int && grammar.value_int_list) ||
         (numeric_target == RType::Float && grammar.value_float_list))) {
      builtins.push_back(NodeKind::CALL_INDEX);
    }
    if (numeric_target == RType::Int && grammar.builtin_find && grammar.value_string) builtins.push_back(NodeKind::CALL_FIND);
    if (grammar.unary_neg) choices.push_back(1);
    if (!num_ops.empty()) choices.push_back(2);
    if (!builtins.empty()) choices.push_back(3);
    if (grammar.expression_if_expr && grammar.value_bool) choices.push_back(4);
    if (grammar.expression_linear_rec && grammar.expression_var && grammar.binary_add &&
        ((numeric_target == RType::Int && grammar.value_int_list) ||
         (numeric_target == RType::Float && grammar.value_float_list))) {
      choices.push_back(5);
    }
    std::vector<int> asgp_forms;
    if (allow_asgp && depth > 1) {
      if (can_emit_asgp_dc_for_type(numeric_target, grammar)) asgp_forms.push_back(6);
      if (can_emit_asgp_dp1d_for_type(numeric_target, grammar)) asgp_forms.push_back(7);
      if (can_emit_asgp_dp2d_for_type(numeric_target, grammar)) asgp_forms.push_back(8);
    }
    if (!asgp_forms.empty() && std::bernoulli_distribution(0.04)(rng)) {
      choices.push_back(choose_one(rng, asgp_forms));
    }

    const int c = choose_one(rng, choices);
    if (c == 0) return emit_random_leaf(rng, program, ctx, numeric_target, grammar);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::NEG, 0, 0});
      return emit_random_expr(rng, program, ctx, depth - 1, numeric_target, grammar);
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{choose_one(rng, num_ops), 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, numeric_target, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, numeric_target, grammar);
      return;
    }
    if (c == 3) {
      const NodeKind builtin = choose_one(rng, builtins);
      program.nodes.push_back(AstNode{builtin, 0, 0});
      if (builtin == NodeKind::CALL_LEN) {
        emit_random_expr(rng, program, ctx, depth - 1, choose_one(rng, len_arg_types), grammar);
      } else if (builtin == NodeKind::CALL_INDEX) {
        emit_random_expr(rng, program, ctx, depth - 1,
                         numeric_target == RType::Int ? RType::IntList : RType::FloatList,
                         grammar);
        program.nodes.push_back(AstNode{
            NodeKind::CONST,
            append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))),
            0});
      } else if (builtin == NodeKind::CALL_FIND) {
        emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
        emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      } else {
        for (int i = 0; i < subtree::node_arity(builtin); ++i) {
          emit_random_expr(rng, program, ctx, depth - 1, numeric_target, grammar);
        }
      }
      return;
    }
    if (c == 4) {
      program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, numeric_target, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, numeric_target, grammar);
      return;
    }
    if (c == 6) {
      const int solve_xs_name = ensure_name(program, "__asgp_xs" + std::to_string(ctx.tmp_idx++));
      const int solve_n_name = ensure_name(program, "__asgp_n" + std::to_string(ctx.tmp_idx++));
      const int solve_lo_name = ensure_name(program, "__asgp_lo" + std::to_string(ctx.tmp_idx++));
      const int divide_n_name = ensure_name(program, "__asgp_dn" + std::to_string(ctx.tmp_idx++));
      const int combine_left_name = ensure_name(program, "__asgp_l" + std::to_string(ctx.tmp_idx++));
      const int combine_right_name = ensure_name(program, "__asgp_r" + std::to_string(ctx.tmp_idx++));
      const std::size_t node_index = program.nodes.size();
      program.nodes.push_back(AstNode{NodeKind::ASGP_DC, 0, 0});
      program.asgp_dc_binders.push_back(AsgpDcBinders{
          node_index,
          solve_xs_name,
          solve_n_name,
          solve_lo_name,
          divide_n_name,
          combine_left_name,
          combine_right_name,
      });
      std::vector<int> source_names;
      if (numeric_target == RType::Float) {
        source_names.assign(ctx.float_list_names.begin(), ctx.float_list_names.end());
      } else {
        source_names.assign(ctx.int_list_names.begin(), ctx.int_list_names.end());
      }
      if (!source_names.empty() && std::bernoulli_distribution(0.65)(rng)) {
        program.nodes.push_back(AstNode{NodeKind::VAR, choose_one(rng, source_names), 0});
      } else {
        program.nodes.push_back(AstNode{
            NodeKind::CONST,
            append_const_id(program, asgp_dc_source_literal(numeric_target)),
            0});
      }
      program.nodes.push_back(AstNode{NodeKind::CALL_INDEX, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, solve_xs_name, 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(0)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(1)), 0});
      program.nodes.push_back(AstNode{NodeKind::ADD, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, combine_left_name, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, combine_right_name, 0});
      return;
    }
    if (c == 7) {
      const int solve_state_name = ensure_name(program, "__asgp_s" + std::to_string(ctx.tmp_idx++));
      const int transition_state_name = ensure_name(program, "__asgp_ts" + std::to_string(ctx.tmp_idx++));
      const int dep_name = ensure_name(program, "__asgp_d" + std::to_string(ctx.tmp_idx++));
      const std::size_t node_index = program.nodes.size();
      const int boundary_const = append_const_id(program, dp_zero_literal(numeric_target));
      program.nodes.push_back(AstNode{NodeKind::ASGP_DP1D, 0, 0});
      program.asgp_dp1d_specs.push_back(AsgpDp1dSpec{
          node_index,
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
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(4)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, dp_one_literal(numeric_target)), 0});
      program.nodes.push_back(AstNode{NodeKind::ADD, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, dep_name, 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, dp_one_literal(numeric_target)), 0});
      return;
    }
    if (c == 8) {
      const int solve_i_name = ensure_name(program, "__asgp_i" + std::to_string(ctx.tmp_idx++));
      const int solve_j_name = ensure_name(program, "__asgp_j" + std::to_string(ctx.tmp_idx++));
      const int transition_i_name = ensure_name(program, "__asgp_ti" + std::to_string(ctx.tmp_idx++));
      const int transition_j_name = ensure_name(program, "__asgp_tj" + std::to_string(ctx.tmp_idx++));
      const int dep_a_name = ensure_name(program, "__asgp_da" + std::to_string(ctx.tmp_idx++));
      const int dep_b_name = ensure_name(program, "__asgp_db" + std::to_string(ctx.tmp_idx++));
      const std::size_t node_index = program.nodes.size();
      const int boundary_const = append_const_id(program, dp_zero_literal(numeric_target));
      program.nodes.push_back(AstNode{NodeKind::ASGP_DP2D, 0, 0});
      program.asgp_dp2d_specs.push_back(AsgpDp2dSpec{
          node_index,
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
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(2)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(2)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, dp_one_literal(numeric_target)), 0});
      program.nodes.push_back(AstNode{NodeKind::ADD, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, dep_a_name, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, dep_b_name, 0});
      return;
    }
    const int elem_name = ensure_name(program, "__lr_u" + std::to_string(ctx.tmp_idx++));
    const int accum_name = ensure_name(program, "__lr_v" + std::to_string(ctx.tmp_idx++));
    const int index_name = ensure_name(program, "__lr_i" + std::to_string(ctx.tmp_idx++));
    const std::size_t node_index = program.nodes.size();
    program.nodes.push_back(AstNode{NodeKind::LINEAR_REC, 0, 0});
    program.linear_rec_binders.push_back(LinearRecBinders{node_index, elem_name, accum_name, index_name});
    emit_random_expr(rng, program, ctx, depth - 1,
                     numeric_target == RType::Int ? RType::IntList : RType::FloatList,
                     grammar);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(0)), 0});
    program.nodes.push_back(AstNode{
        NodeKind::CONST,
        append_const_id(program, numeric_target == RType::Int ? Value::from_int(0) : Value::from_float(0.0)),
        0});
    program.nodes.push_back(AstNode{NodeKind::ADD, 0, 0});
    program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, elem_name, 0});
    program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, accum_name, 0});
    program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, elem_name, 0});
    return;
  }

  if (target == RType::Bool) {
    std::vector<int> choices{0};
    const std::vector<NodeKind> cmp_ops = filter_node_kinds(
        grammar, {NodeKind::LT, NodeKind::LE, NodeKind::GT, NodeKind::GE, NodeKind::EQ, NodeKind::NE});
    const std::vector<NodeKind> bool_ops = filter_node_kinds(grammar, {NodeKind::AND, NodeKind::OR});
    if (grammar.unary_not) choices.push_back(1);
    if (!cmp_ops.empty() && (grammar.allows_type(RType::Int) || grammar.allows_type(RType::Float))) choices.push_back(2);
    if (!bool_ops.empty()) choices.push_back(3);
    if (grammar.builtin_contains && grammar.value_string) choices.push_back(4);
    if (grammar.expression_if_expr) choices.push_back(5);

    const int c = choose_one(rng, choices);
    if (c == 0) return emit_random_leaf(rng, program, ctx, RType::Bool, grammar);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::NOT, 0, 0});
      return emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
    }
    if (c == 2) {
      const RType numeric_type = choose_type(rng, grammar, {RType::Int, RType::Float});
      program.nodes.push_back(AstNode{choose_one(rng, cmp_ops), 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, numeric_type, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, numeric_type, grammar);
      return;
    }
    if (c == 3) {
      program.nodes.push_back(AstNode{choose_one(rng, bool_ops), 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
      return;
    }
    if (c == 4) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONTAINS, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
    return;
  }

  if (target == RType::String) {
    std::vector<int> choices{0};
    if (grammar.builtin_concat) choices.push_back(1);
    if (grammar.builtin_slice) choices.push_back(2);
    if (grammar.builtin_reverse) choices.push_back(3);
    if (grammar.expression_if_expr && grammar.value_bool) choices.push_back(4);
    std::vector<int> asgp_forms;
    if (allow_asgp && depth > 1) {
      if (can_emit_asgp_dc_for_type(RType::String, grammar)) asgp_forms.push_back(5);
      if (can_emit_asgp_dp1d_for_type(RType::String, grammar)) asgp_forms.push_back(6);
      if (can_emit_asgp_dp2d_for_type(RType::String, grammar)) asgp_forms.push_back(7);
    }
    if (!asgp_forms.empty() && std::bernoulli_distribution(0.04)(rng)) {
      choices.push_back(choose_one(rng, asgp_forms));
    }
    const int c = choose_one(rng, choices);
    if (c == 0) return emit_random_leaf(rng, program, ctx, RType::String, grammar);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      return;
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      program.nodes.push_back(AstNode{
          NodeKind::CONST,
          append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))),
          0});
      program.nodes.push_back(AstNode{
          NodeKind::CONST,
          append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))),
          0});
      return;
    }
    if (c == 3) {
      program.nodes.push_back(AstNode{NodeKind::CALL_REVERSE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
      return;
    }
    if (c == 5) {
      const int solve_xs_name = ensure_name(program, "__asgp_xs" + std::to_string(ctx.tmp_idx++));
      const int solve_n_name = ensure_name(program, "__asgp_n" + std::to_string(ctx.tmp_idx++));
      const int solve_lo_name = ensure_name(program, "__asgp_lo" + std::to_string(ctx.tmp_idx++));
      const int divide_n_name = ensure_name(program, "__asgp_dn" + std::to_string(ctx.tmp_idx++));
      const int combine_left_name = ensure_name(program, "__asgp_l" + std::to_string(ctx.tmp_idx++));
      const int combine_right_name = ensure_name(program, "__asgp_r" + std::to_string(ctx.tmp_idx++));
      const std::size_t node_index = program.nodes.size();
      program.nodes.push_back(AstNode{NodeKind::ASGP_DC, 0, 0});
      program.asgp_dc_binders.push_back(AsgpDcBinders{
          node_index,
          solve_xs_name,
          solve_n_name,
          solve_lo_name,
          divide_n_name,
          combine_left_name,
          combine_right_name,
      });
      std::vector<int> source_names(ctx.string_names.begin(), ctx.string_names.end());
      if (!source_names.empty() && std::bernoulli_distribution(0.65)(rng)) {
        program.nodes.push_back(AstNode{NodeKind::VAR, choose_one(rng, source_names), 0});
      } else {
        program.nodes.push_back(AstNode{
            NodeKind::CONST,
            append_const_id(program, asgp_dc_source_literal(RType::String)),
            0});
      }
      program.nodes.push_back(AstNode{NodeKind::CALL_SINGLETON, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::CALL_INDEX, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, solve_xs_name, 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(0)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(1)), 0});
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, combine_left_name, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, combine_right_name, 0});
      return;
    }
    if (c == 6) {
      const int solve_state_name = ensure_name(program, "__asgp_s" + std::to_string(ctx.tmp_idx++));
      const int transition_state_name = ensure_name(program, "__asgp_ts" + std::to_string(ctx.tmp_idx++));
      const int dep_name = ensure_name(program, "__asgp_d" + std::to_string(ctx.tmp_idx++));
      const std::size_t node_index = program.nodes.size();
      const int boundary_const = append_const_id(program, dp_zero_literal(RType::String));
      program.nodes.push_back(AstNode{NodeKind::ASGP_DP1D, 0, 0});
      program.asgp_dp1d_specs.push_back(AsgpDp1dSpec{
          node_index,
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
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(4)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, dp_one_literal(RType::String)), 0});
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, dep_name, 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, dp_one_literal(RType::String)), 0});
      return;
    }
    if (c == 7) {
      const int solve_i_name = ensure_name(program, "__asgp_i" + std::to_string(ctx.tmp_idx++));
      const int solve_j_name = ensure_name(program, "__asgp_j" + std::to_string(ctx.tmp_idx++));
      const int transition_i_name = ensure_name(program, "__asgp_ti" + std::to_string(ctx.tmp_idx++));
      const int transition_j_name = ensure_name(program, "__asgp_tj" + std::to_string(ctx.tmp_idx++));
      const int dep_a_name = ensure_name(program, "__asgp_da" + std::to_string(ctx.tmp_idx++));
      const int dep_b_name = ensure_name(program, "__asgp_db" + std::to_string(ctx.tmp_idx++));
      const std::size_t node_index = program.nodes.size();
      const int boundary_const = append_const_id(program, dp_zero_literal(RType::String));
      program.nodes.push_back(AstNode{NodeKind::ASGP_DP2D, 0, 0});
      program.asgp_dp2d_specs.push_back(AsgpDp2dSpec{
          node_index,
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
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(2)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(2)), 0});
      program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, dp_one_literal(RType::String)), 0});
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, dep_a_name, 0});
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, dep_b_name, 0});
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
    emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
    emit_random_expr(rng, program, ctx, depth - 1, RType::String, grammar);
    return;
  }

  if (target == RType::IntList || target == RType::FloatList || target == RType::StringList) {
    std::vector<int> choices{0};
    if (grammar.builtin_concat) choices.push_back(1);
    if (grammar.builtin_slice) choices.push_back(2);
    if (grammar.builtin_append &&
        grammar.allows_type(target == RType::StringList ? RType::String :
                            (target == RType::FloatList ? RType::Float : RType::Int))) {
      choices.push_back(3);
    }
    if (grammar.builtin_reverse) choices.push_back(4);
    if (grammar.expression_if_expr && grammar.value_bool) choices.push_back(5);
    if (grammar.expression_map_list && grammar.expression_var) choices.push_back(6);
    if (grammar.expression_filter_list && grammar.expression_var && grammar.value_bool) choices.push_back(7);
    const int c = choose_one(rng, choices);
    if (c == 0) return emit_random_leaf(rng, program, ctx, target, grammar);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      return;
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      program.nodes.push_back(AstNode{
          NodeKind::CONST,
          append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))),
          0});
      program.nodes.push_back(AstNode{
          NodeKind::CONST,
          append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))),
          0});
      return;
    }
    if (c == 3) {
      program.nodes.push_back(AstNode{NodeKind::CALL_APPEND, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      emit_random_expr(rng,
                       program,
                       ctx,
                       depth - 1,
                       target == RType::StringList ? RType::String :
                       (target == RType::FloatList ? RType::Float : RType::Int),
                       grammar);
      return;
    }
    if (c == 4) {
      program.nodes.push_back(AstNode{NodeKind::CALL_REVERSE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      return;
    }
    if (c == 5) {
      program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      return;
    }
    const int binder_name = ensure_name(program, "__map_u" + std::to_string(ctx.tmp_idx++));
    if (c == 6) {
      const int out_tag = target == RType::IntList ? static_cast<int>(ListTypeTag::Int) :
                          (target == RType::FloatList ? static_cast<int>(ListTypeTag::Float)
                                                       : static_cast<int>(ListTypeTag::String));
      program.nodes.push_back(AstNode{NodeKind::MAP_LIST, binder_name, out_tag});
      emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
      program.nodes.push_back(AstNode{NodeKind::BOUND_VAR, binder_name, 0});
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::FILTER_LIST, binder_name, 0});
    emit_random_expr(rng, program, ctx, depth - 1, target, grammar);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::from_bool(true)), 0});
    return;
  }

  emit_random_leaf(rng, program, ctx, RType::Int, grammar);
}

int choose_or_new_name(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx) {
  std::vector<int> all;
  for (std::size_t i = 0; i < program.names.size(); ++i) {
    all.push_back(static_cast<int>(i));
  }
  if (!all.empty() && !std::bernoulli_distribution(0.4)(rng)) {
    return choose_one(rng, all);
  }
  return ensure_name(program, "t" + std::to_string(ctx.tmp_idx++));
}

bool can_emit_statement(const GrammarConfig& grammar, bool allow_return_stmt, int depth) {
  return grammar.statement_assign ||
         (depth > 0 && grammar.statement_if_stmt && grammar.value_bool) ||
         (depth > 0 && grammar.statement_for_range && grammar.value_int) ||
         (allow_return_stmt && grammar.statement_return);
}

void emit_random_block(std::mt19937_64& rng,
                       AstProgram& program,
                       PrefixGenCtx& ctx,
                       int depth,
                       const Limits& limits,
                       bool force_return,
                       const GrammarConfig& grammar,
                       RType* program_return_type,
                       int max_stmts = -1,
                       bool allow_return_stmt = true);

void emit_random_stmt(std::mt19937_64& rng,
                      AstProgram& program,
                      PrefixGenCtx& ctx,
                      int depth,
                      const Limits& limits,
                      bool allow_return_stmt,
                      const GrammarConfig& grammar,
                      RType* program_return_type) {
  const std::vector<RType> allowed_types =
      filter_types(grammar, {RType::Int, RType::Float, RType::Bool, RType::Char, RType::String, RType::IntList, RType::FloatList, RType::StringList});
  if (depth <= 0) {
    if (grammar.statement_assign &&
        (!allow_return_stmt || !grammar.statement_return || std::bernoulli_distribution(0.75)(rng))) {
      const int name_id = choose_or_new_name(rng, program, ctx);
      const RType type = choose_one(rng, allowed_types);
      program.nodes.push_back(AstNode{NodeKind::ASSIGN, name_id, 0});
      emit_random_expr(rng, program, ctx, 0, type, grammar, true);
      assign_name_type(ctx, name_id, type);
      return;
    }
    if (allow_return_stmt && grammar.statement_return) {
      program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
      if (*program_return_type == RType::Invalid) {
        *program_return_type = choose_any_type(rng, grammar);
      } else {
        *program_return_type = coerce_type(rng, grammar, *program_return_type);
      }
      emit_random_expr(rng, program, ctx, 0, *program_return_type, grammar, true);
    }
    return;
  }

  std::vector<int> choices;
  if (grammar.statement_assign) choices.push_back(0);
  if (grammar.statement_if_stmt && grammar.value_bool) choices.push_back(1);
  if (grammar.statement_for_range && grammar.value_int) choices.push_back(2);
  if (allow_return_stmt && grammar.statement_return) choices.push_back(3);
  if (choices.empty()) return;
  const int choice = choose_one(rng, choices);
  if (choice == 0) {
    const int name_id = choose_or_new_name(rng, program, ctx);
    const RType type = choose_one(rng, allowed_types);
    program.nodes.push_back(AstNode{NodeKind::ASSIGN, name_id, 0});
    emit_random_expr(rng, program, ctx, depth - 1, type, grammar, true);
    assign_name_type(ctx, name_id, type);
    return;
  }
  if (choice == 1) {
    program.nodes.push_back(AstNode{NodeKind::IF_STMT, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool, grammar);
    PrefixGenCtx then_ctx = ctx;
    PrefixGenCtx else_ctx = ctx;
    emit_random_block(rng,
                      program,
                      then_ctx,
                      depth - 1,
                      limits,
                      false,
                      grammar,
                      program_return_type,
                      std::max(1, std::min(2, limits.max_stmts_per_block)),
                      allow_return_stmt);
    emit_random_block(rng,
                      program,
                      else_ctx,
                      depth - 1,
                      limits,
                      false,
                      grammar,
                      program_return_type,
                      std::max(1, std::min(2, limits.max_stmts_per_block)),
                      allow_return_stmt);
    return;
  }
  if (choice == 2) {
    const int name_id = ensure_name(
        program, choose_one(rng, std::vector<std::string>{"i", "j", "k"}) + std::to_string(std::uniform_int_distribution<int>(0, 9)(rng)));
    program.nodes.push_back(AstNode{NodeKind::FOR_RANGE, name_id, 0});
    program.nodes.push_back(AstNode{
        NodeKind::CONST,
        append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(0, std::max(0, limits.max_for_k))(rng))),
        0,
    });
    PrefixGenCtx body_ctx = ctx;
    body_ctx.int_names.insert(name_id);
    emit_random_block(rng,
                      program,
                      body_ctx,
                      depth - 1,
                      limits,
                      false,
                      grammar,
                      program_return_type,
                      std::max(1, std::min(2, limits.max_stmts_per_block)),
                      allow_return_stmt);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  if (*program_return_type == RType::Invalid) {
    *program_return_type = choose_one(rng, allowed_types);
  } else {
    *program_return_type = coerce_type(rng, grammar, *program_return_type);
  }
  emit_random_expr(rng,
                   program,
                   ctx,
                   depth - 1,
                   *program_return_type,
                   grammar,
                   true);
}

void emit_random_block(std::mt19937_64& rng,
                       AstProgram& program,
                       PrefixGenCtx& ctx,
                       int depth,
                       const Limits& limits,
                       bool force_return,
                       const GrammarConfig& grammar,
                       RType* program_return_type,
                       int max_stmts,
                       bool allow_return_stmt) {
  const int max_n = (max_stmts < 0) ? limits.max_stmts_per_block : max_stmts;
  const int n = std::uniform_int_distribution<int>(1, std::max(1, max_n))(rng);
  bool has_return = false;
  for (int i = 0; i < n; ++i) {
    if (!can_emit_statement(grammar, allow_return_stmt, depth)) {
      break;
    }
    program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    const std::size_t before = program.nodes.size();
    emit_random_stmt(rng, program, ctx, depth, limits, allow_return_stmt, grammar,
                     program_return_type);
    if (program.nodes[before].kind == NodeKind::RETURN) {
      has_return = true;
      break;
    }
  }
  if (force_return && !has_return) {
    program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    const std::vector<RType> forced_choices{
        RType::Int, RType::Float, RType::Bool, RType::Char, RType::String,
        RType::IntList, RType::FloatList, RType::StringList,
    };
    if (*program_return_type == RType::Invalid) {
      *program_return_type = choose_type(rng, grammar, forced_choices);
    } else {
      *program_return_type = coerce_type(rng, grammar, *program_return_type);
    }
    emit_random_expr(rng,
                     program,
                     ctx,
                     std::max(0, depth - 1),
                     *program_return_type,
                     grammar,
                     true);
  }
  program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
}

}  // namespace

ProgramGenome generate_random_genome(std::uint64_t seed, const Limits& limits) {
  return generate_random_genome(seed, limits, {}, GrammarConfig{});
}

ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const GrammarConfig& grammar) {
  return generate_random_genome(seed, limits, {}, grammar);
}

ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const std::vector<InputSpec>& input_specs) {
  return generate_random_genome(seed, limits, input_specs, GrammarConfig{});
}

ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const std::vector<InputSpec>& input_specs,
                                     const GrammarConfig& grammar) {
  grammar.validate();
  std::mt19937_64 rng(seed);
  for (int i = 0; i < 128; ++i) {
    AstProgram program;
    program.version = k_ast_prefix_version_current;
    program.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
    PrefixGenCtx ctx;
    seed_input_names(program, ctx, input_specs, grammar);
    RType program_return_type = RType::Invalid;
    emit_random_block(rng, program, ctx, limits.max_expr_depth, limits, true, grammar,
                      &program_return_type);
    ProgramGenome genome = as_genome_prefix(program);
    if (genome.meta.node_count <= limits.max_total_nodes &&
        verify_ast(genome.ast, input_specs)) {
      return genome;
    }
  }

  AstProgram fallback;
  fallback.version = k_ast_prefix_version_current;
  fallback.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  fallback.nodes.push_back(AstNode{
      NodeKind::CONST,
      append_const_id(fallback, grammar.value_int ? Value::from_int(0) : Value::from_float(0.0)),
      0});
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return as_genome_prefix(fallback);
}

ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed, RType return_type, const Limits& limits) {
  return generate_random_genome_for_return_type(seed, return_type, limits, {}, GrammarConfig{});
}

ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const GrammarConfig& grammar) {
  return generate_random_genome_for_return_type(seed, return_type, limits, {}, grammar);
}

ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const std::vector<InputSpec>& input_specs) {
  return generate_random_genome_for_return_type(seed, return_type, limits, input_specs, GrammarConfig{});
}

ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const std::vector<InputSpec>& input_specs,
                                                     const GrammarConfig& grammar) {
  grammar.validate();
  std::mt19937_64 rng(seed);
  const int typed_depth = std::max(0, std::min(limits.max_expr_depth, 4));
  const int typed_max_stmts = std::max(1, std::min(limits.max_stmts_per_block, 3));
  for (int i = 0; i < 16; ++i) {
    AstProgram program;
    program.version = k_ast_prefix_version_current;
    program.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
    PrefixGenCtx ctx;
    seed_input_names(program, ctx, input_specs, grammar);
    RType program_return_type = return_type;
    emit_random_block(rng,
                      program,
                      ctx,
                      typed_depth,
                      limits,
                      true,
                      grammar,
                      &program_return_type,
                      typed_max_stmts,
                      false);
    ProgramGenome genome = as_genome_prefix(program);
    if (genome.meta.node_count <= limits.max_total_nodes &&
        verify_ast(genome.ast, input_specs)) {
      return genome;
    }
  }

  AstProgram fallback;
  fallback.version = k_ast_prefix_version_current;
  fallback.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  const RType fallback_type = coerce_type(rng, grammar, return_type);
  if (fallback_type == RType::Bool) {
    fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::from_bool(false)), 0});
  } else if (fallback_type == RType::Float) {
    fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::from_float(0.0)), 0});
  } else if (fallback_type == RType::String) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, gagp::payload::make_string_value("")), 0});
  } else if (fallback_type == RType::Char) {
    fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::from_char('a')), 0});
  } else if (fallback_type == RType::IntList) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, gagp::payload::make_int_list_value({})), 0});
  } else if (fallback_type == RType::FloatList) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, gagp::payload::make_float_list_value({})), 0});
  } else if (fallback_type == RType::StringList) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, gagp::payload::make_string_list_value({})), 0});
  } else {
    fallback.nodes.push_back(AstNode{
        NodeKind::CONST,
        append_const_id(fallback, grammar.value_int ? Value::from_int(0) : Value::from_float(0.0)),
        0});
  }
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return as_genome_prefix(fallback);
}

}  // namespace gagp::evo
