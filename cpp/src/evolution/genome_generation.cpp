#include "g3pvm/evolution/genome_generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/runtime/payload/payload.hpp"
#include "subtree_utils.hpp"

namespace g3pvm::evo {

namespace {

struct PrefixGenCtx {
  std::set<int> num_names;
  std::set<int> bool_names;
  std::set<int> none_names;
  std::set<int> string_names;
  std::set<int> num_list_names;
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
  if (type == RType::Num) names.assign(ctx.num_names.begin(), ctx.num_names.end());
  else if (type == RType::Bool) names.assign(ctx.bool_names.begin(), ctx.bool_names.end());
  else if (type == RType::NoneType) names.assign(ctx.none_names.begin(), ctx.none_names.end());
  else if (type == RType::String) names.assign(ctx.string_names.begin(), ctx.string_names.end());
  else if (type == RType::NumList) names.assign(ctx.num_list_names.begin(), ctx.num_list_names.end());
  else if (type == RType::StringList) names.assign(ctx.string_list_names.begin(), ctx.string_list_names.end());
  else if (type == RType::Any) {
    names.assign(ctx.num_names.begin(), ctx.num_names.end());
    names.insert(names.end(), ctx.bool_names.begin(), ctx.bool_names.end());
    names.insert(names.end(), ctx.none_names.begin(), ctx.none_names.end());
    names.insert(names.end(), ctx.string_names.begin(), ctx.string_names.end());
    names.insert(names.end(), ctx.num_list_names.begin(), ctx.num_list_names.end());
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

Value random_num_list_literal(std::mt19937_64& rng) {
  const int len = std::uniform_int_distribution<int>(0, 5)(rng);
  std::vector<Value> elems;
  elems.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    if (std::bernoulli_distribution(0.65)(rng)) {
      elems.push_back(Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng)));
    } else {
      elems.push_back(Value::from_float(
          std::round(std::uniform_real_distribution<double>(-8.0, 8.0)(rng) * 1000.0) / 1000.0));
    }
  }
  return g3pvm::payload::make_num_list_value(elems);
}

Value random_string_list_literal(std::mt19937_64& rng) {
  const int len = std::uniform_int_distribution<int>(0, 5)(rng);
  std::vector<Value> elems;
  elems.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    elems.push_back(g3pvm::payload::make_string_value(random_string_literal(rng)));
  }
  return g3pvm::payload::make_string_list_value(elems);
}

void assign_name_type(PrefixGenCtx& ctx, int name_id, RType type) {
  ctx.num_names.erase(name_id);
  ctx.bool_names.erase(name_id);
  ctx.none_names.erase(name_id);
  ctx.string_names.erase(name_id);
  ctx.num_list_names.erase(name_id);
  ctx.string_list_names.erase(name_id);
  ctx.any_names.erase(name_id);
  if (type == RType::Num) ctx.num_names.insert(name_id);
  if (type == RType::Bool) ctx.bool_names.insert(name_id);
  if (type == RType::NoneType) ctx.none_names.insert(name_id);
  if (type == RType::String) ctx.string_names.insert(name_id);
  if (type == RType::NumList) ctx.num_list_names.insert(name_id);
  if (type == RType::StringList) ctx.string_list_names.insert(name_id);
  if (type == RType::Any) ctx.any_names.insert(name_id);
}

void seed_input_names(AstProgram& program, PrefixGenCtx& ctx, const std::vector<InputSpec>& input_specs) {
  if (input_specs.empty()) {
    ctx.num_names.insert(ensure_name(program, "x"));
    return;
  }
  for (const InputSpec& spec : input_specs) {
    if (spec.name.empty()) {
      continue;
    }
    const int name_id = ensure_name(program, spec.name);
    assign_name_type(ctx, name_id, spec.type == RType::Invalid ? RType::Any : spec.type);
  }
}

void emit_random_leaf(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx, RType target) {
  const int var_id = choose_name_for_type(rng, ctx, target);
  if (var_id >= 0 && std::bernoulli_distribution(0.45)(rng)) {
    program.nodes.push_back(AstNode{NodeKind::VAR, var_id, 0});
    return;
  }
  if (target == RType::Num) {
    if (std::bernoulli_distribution(0.5)(rng)) {
      program.nodes.push_back(
          AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng))), 0});
    } else {
      program.nodes.push_back(AstNode{NodeKind::CONST,
                                      append_const_id(program, Value::from_float(
                                                                   std::round(std::uniform_real_distribution<double>(-8.0, 8.0)(rng) * 1000.0) / 1000.0)),
                                      0});
    }
    return;
  }
  if (target == RType::Bool) {
    program.nodes.push_back(AstNode{
        NodeKind::CONST, append_const_id(program, Value::from_bool(std::bernoulli_distribution(0.5)(rng))), 0});
    return;
  }
  if (target == RType::NoneType) {
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::none()), 0});
    return;
  }
  if (target == RType::String) {
    const Value value = g3pvm::payload::make_string_value(random_string_literal(rng));
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }
  if (target == RType::NumList) {
    const Value value = random_num_list_literal(rng);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }
  if (target == RType::StringList) {
    const Value value = random_string_list_literal(rng);
    program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, value), 0});
    return;
  }

  const int mode = std::uniform_int_distribution<int>(0, 2)(rng);
  if (mode == 0) program.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(program, Value::none()), 0});
  else if (mode == 1) {
    program.nodes.push_back(AstNode{
        NodeKind::CONST, append_const_id(program, Value::from_bool(std::bernoulli_distribution(0.5)(rng))), 0});
  } else {
    program.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-8, 8)(rng))), 0});
  }
}

void emit_random_expr(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx, int depth, RType target) {
  if (target == RType::Any) {
    target = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType, RType::String, RType::NumList, RType::StringList});
  }
  if (depth <= 0) {
    emit_random_leaf(rng, program, ctx, target);
    return;
  }
  if (target == RType::Num) {
    const int c = std::uniform_int_distribution<int>(0, 5)(rng);
    if (c == 0) return emit_random_leaf(rng, program, ctx, RType::Num);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::NEG, 0, 0});
      return emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{
          choose_one(rng, std::vector<NodeKind>{NodeKind::ADD, NodeKind::SUB, NodeKind::MUL, NodeKind::DIV, NodeKind::MOD}),
          0,
          0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
      emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
      return;
    }
    if (c == 3) {
      const NodeKind builtin = choose_one(
          rng, std::vector<NodeKind>{NodeKind::CALL_ABS,
                                     NodeKind::CALL_MIN,
                                     NodeKind::CALL_MAX,
                                     NodeKind::CALL_CLIP,
                                     NodeKind::CALL_LEN,
                                     NodeKind::CALL_INDEX,
                                     NodeKind::CALL_FIND});
      program.nodes.push_back(AstNode{builtin, 0, 0});
      if (builtin == NodeKind::CALL_LEN) {
        emit_random_expr(rng,
                         program,
                         ctx,
                         depth - 1,
                         choose_one(rng, std::vector<RType>{RType::String, RType::NumList, RType::StringList}));
      } else if (builtin == NodeKind::CALL_INDEX) {
        emit_random_expr(rng,
                         program,
                         ctx,
                         depth - 1,
                         choose_one(rng, std::vector<RType>{RType::NumList}));
        program.nodes.push_back(AstNode{
            NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      } else if (builtin == NodeKind::CALL_FIND) {
        emit_random_expr(rng, program, ctx, depth - 1, RType::String);
        emit_random_expr(rng, program, ctx, depth - 1, RType::String);
      } else {
        for (int i = 0; i < subtree::node_arity(builtin); ++i) {
          emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
        }
      }
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
    return;
  }
  if (target == RType::Bool) {
    const int c = std::uniform_int_distribution<int>(0, 5)(rng);
    if (c == 0) return emit_random_leaf(rng, program, ctx, RType::Bool);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::NOT, 0, 0});
      return emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{
          choose_one(rng, std::vector<NodeKind>{NodeKind::LT, NodeKind::LE, NodeKind::GT, NodeKind::GE, NodeKind::EQ, NodeKind::NE}),
          0,
          0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
      emit_random_expr(rng, program, ctx, depth - 1, RType::Num);
      return;
    }
    if (c == 3) {
      program.nodes.push_back(
          AstNode{choose_one(rng, std::vector<NodeKind>{NodeKind::AND, NodeKind::OR}), 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
      emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
      return;
    }
    if (c == 4) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONTAINS, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String);
      emit_random_expr(rng, program, ctx, depth - 1, RType::String);
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    return;
  }
  if (target == RType::String) {
    const int c = std::uniform_int_distribution<int>(0, 3)(rng);
    if (c == 0) return emit_random_leaf(rng, program, ctx, RType::String);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String);
      emit_random_expr(rng, program, ctx, depth - 1, RType::String);
      return;
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::String);
      program.nodes.push_back(AstNode{
          NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      program.nodes.push_back(AstNode{
          NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::CALL_REVERSE, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::String);
    return;
  }
  if (target == RType::NumList || target == RType::StringList) {
    const int c = std::uniform_int_distribution<int>(0, 5)(rng);
    if (c == 0) return emit_random_leaf(rng, program, ctx, target);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target);
      emit_random_expr(rng, program, ctx, depth - 1, target);
      return;
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target);
      program.nodes.push_back(AstNode{
          NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      program.nodes.push_back(AstNode{
          NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      return;
    }
    if (c == 3) {
      program.nodes.push_back(AstNode{NodeKind::CALL_APPEND, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target);
      emit_random_expr(rng, program, ctx, depth - 1, target == RType::NumList ? RType::Num : RType::String);
      return;
    }
    if (c == 4) {
      program.nodes.push_back(AstNode{NodeKind::CALL_REVERSE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, target);
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, target);
    emit_random_expr(rng, program, ctx, depth - 1, target);
    return;
  }
  emit_random_leaf(rng, program, ctx, RType::NoneType);
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

void emit_random_block(std::mt19937_64& rng,
                       AstProgram& program,
                       PrefixGenCtx& ctx,
                       int depth,
                       const Limits& limits,
                       bool force_return,
                       int max_stmts = -1,
                       bool allow_return_stmt = true,
                       RType forced_return_type = RType::Invalid);

void emit_random_stmt(std::mt19937_64& rng,
                      AstProgram& program,
                      PrefixGenCtx& ctx,
                      int depth,
                      const Limits& limits,
                      bool allow_return_stmt) {
  if (depth <= 0) {
    if (!allow_return_stmt || std::bernoulli_distribution(0.75)(rng)) {
      const int name_id = choose_or_new_name(rng, program, ctx);
      const RType type = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType, RType::String, RType::NumList, RType::StringList});
      program.nodes.push_back(AstNode{NodeKind::ASSIGN, name_id, 0});
      emit_random_expr(rng, program, ctx, 0, type);
      assign_name_type(ctx, name_id, type);
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    emit_random_expr(rng, program, ctx, 0, RType::Any);
    return;
  }

  const int max_choice = allow_return_stmt ? 3 : 2;
  const int choice = std::uniform_int_distribution<int>(0, max_choice)(rng);
  if (choice == 0) {
    const int name_id = choose_or_new_name(rng, program, ctx);
    const RType type = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType, RType::String, RType::NumList, RType::StringList});
    program.nodes.push_back(AstNode{NodeKind::ASSIGN, name_id, 0});
    emit_random_expr(rng, program, ctx, depth - 1, type);
    assign_name_type(ctx, name_id, type);
    return;
  }
  if (choice == 1) {
    program.nodes.push_back(AstNode{NodeKind::IF_STMT, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    PrefixGenCtx then_ctx = ctx;
    PrefixGenCtx else_ctx = ctx;
    emit_random_block(rng,
                      program,
                      then_ctx,
                      depth - 1,
                      limits,
                      false,
                      std::max(1, std::min(2, limits.max_stmts_per_block)),
                      allow_return_stmt);
    emit_random_block(rng,
                      program,
                      else_ctx,
                      depth - 1,
                      limits,
                      false,
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
    body_ctx.num_names.insert(name_id);
    emit_random_block(rng,
                      program,
                      body_ctx,
                      depth - 1,
                      limits,
                      false,
                      std::max(1, std::min(2, limits.max_stmts_per_block)),
                      allow_return_stmt);
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  emit_random_expr(rng,
                   program,
                   ctx,
                   depth - 1,
                   choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType, RType::String, RType::NumList, RType::StringList}));
}

void emit_random_block(std::mt19937_64& rng,
                       AstProgram& program,
                       PrefixGenCtx& ctx,
                       int depth,
                       const Limits& limits,
                       bool force_return,
                       int max_stmts,
                       bool allow_return_stmt,
                       RType forced_return_type) {
  const int max_n = (max_stmts < 0) ? limits.max_stmts_per_block : max_stmts;
  const int n = std::uniform_int_distribution<int>(1, std::max(1, max_n))(rng);
  bool has_return = false;
  for (int i = 0; i < n; ++i) {
    program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    const std::size_t before = program.nodes.size();
    emit_random_stmt(rng, program, ctx, depth, limits, allow_return_stmt);
    if (program.nodes[before].kind == NodeKind::RETURN) {
      has_return = true;
      break;
    }
  }
  if (force_return && !has_return) {
    program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    const std::vector<RType> forced_choices{RType::Num, RType::Bool, RType::String, RType::NumList, RType::StringList};
    const RType return_type = forced_return_type != RType::Invalid ? forced_return_type
                                                                   : choose_one(rng, forced_choices);
    emit_random_expr(rng,
                     program,
                     ctx,
                     std::max(0, depth - 1),
                     return_type);
  }
  program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
}

}  // namespace

ProgramGenome generate_random_genome(std::uint64_t seed, const Limits& limits) {
  return generate_random_genome(seed, limits, {});
}

ProgramGenome generate_random_genome(std::uint64_t seed,
                                     const Limits& limits,
                                     const std::vector<InputSpec>& input_specs) {
  std::mt19937_64 rng(seed);
  for (int i = 0; i < 128; ++i) {
    AstProgram program;
    program.version = "ast-prefix-v1";
    program.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
    PrefixGenCtx ctx;
    seed_input_names(program, ctx, input_specs);
    emit_random_block(rng, program, ctx, limits.max_expr_depth, limits, true);
    ProgramGenome genome = as_genome_prefix(program);
    if (genome.meta.node_count <= limits.max_total_nodes) {
      return genome;
    }
  }

  AstProgram fallback;
  fallback.version = "ast-prefix-v1";
  fallback.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::from_int(0)), 0});
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return as_genome_prefix(fallback);
}

ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed, RType return_type, const Limits& limits) {
  return generate_random_genome_for_return_type(seed, return_type, limits, {});
}

ProgramGenome generate_random_genome_for_return_type(std::uint64_t seed,
                                                     RType return_type,
                                                     const Limits& limits,
                                                     const std::vector<InputSpec>& input_specs) {
  std::mt19937_64 rng(seed);
  for (int i = 0; i < 128; ++i) {
    AstProgram program;
    program.version = "ast-prefix-v1";
    program.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
    PrefixGenCtx ctx;
    seed_input_names(program, ctx, input_specs);
    emit_random_block(rng,
                      program,
                      ctx,
                      limits.max_expr_depth,
                      limits,
                      true,
                      -1,
                      false,
                      return_type);
    ProgramGenome genome = as_genome_prefix(program);
    if (genome.meta.node_count <= limits.max_total_nodes && genome.meta.max_depth <= limits.max_expr_depth) {
      return genome;
    }
  }

  AstProgram fallback;
  fallback.version = "ast-prefix-v1";
  fallback.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
  fallback.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  if (return_type == RType::Bool) {
    fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::from_bool(false)), 0});
  } else if (return_type == RType::NoneType) {
    fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::none()), 0});
  } else if (return_type == RType::String) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, g3pvm::payload::make_string_value("")), 0});
  } else if (return_type == RType::NumList) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, g3pvm::payload::make_num_list_value({})), 0});
  } else if (return_type == RType::StringList) {
    fallback.nodes.push_back(
        AstNode{NodeKind::CONST, append_const_id(fallback, g3pvm::payload::make_string_list_value({})), 0});
  } else {
    fallback.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(fallback, Value::from_int(0)), 0});
  }
  fallback.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return as_genome_prefix(fallback);
}

}  // namespace g3pvm::evo
