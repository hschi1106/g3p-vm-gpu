#include "g3pvm/evolution/genome.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "genome_meta.hpp"
#include "subtree_utils.hpp"

namespace g3pvm::evo {

namespace {

struct PrefixGenCtx {
  std::set<int> num_names;
  std::set<int> bool_names;
  std::set<int> none_names;
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
  genome.meta = genome_meta::build_meta_fast(genome.ast);
  return genome;
}

int choose_name_for_type(std::mt19937_64& rng, const PrefixGenCtx& ctx, RType type) {
  std::vector<int> names;
  if (type == RType::Num) names.assign(ctx.num_names.begin(), ctx.num_names.end());
  else if (type == RType::Bool) names.assign(ctx.bool_names.begin(), ctx.bool_names.end());
  else if (type == RType::NoneType) names.assign(ctx.none_names.begin(), ctx.none_names.end());
  else if (type == RType::Any) {
    names.assign(ctx.num_names.begin(), ctx.num_names.end());
    names.insert(names.end(), ctx.bool_names.begin(), ctx.bool_names.end());
    names.insert(names.end(), ctx.none_names.begin(), ctx.none_names.end());
  }
  if (names.empty()) {
    return -1;
  }
  return choose_one(rng, names);
}

void emit_random_expr(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx, int depth, RType target);

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
  if (target == RType::Container) {
    const std::uint32_t n = static_cast<std::uint32_t>(std::uniform_int_distribution<int>(0, 8)(rng));
    const std::uint64_t h = static_cast<std::uint64_t>(rng());
    const Value value = std::bernoulli_distribution(0.5)(rng) ? Value::from_string_hash_len(h, n)
                                                              : Value::from_list_hash_len(h, n);
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
    target = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType, RType::Container});
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
                                     NodeKind::CALL_INDEX});
      program.nodes.push_back(AstNode{builtin, 0, 0});
      if (builtin == NodeKind::CALL_LEN) {
        emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
      } else if (builtin == NodeKind::CALL_INDEX) {
        emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
        program.nodes.push_back(AstNode{
            NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
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
    const int c = std::uniform_int_distribution<int>(0, 4)(rng);
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
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    return;
  }
  if (target == RType::Container) {
    const int c = std::uniform_int_distribution<int>(0, 3)(rng);
    if (c == 0) return emit_random_leaf(rng, program, ctx, RType::Container);
    if (c == 1) {
      program.nodes.push_back(AstNode{NodeKind::CALL_CONCAT, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
      emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
      return;
    }
    if (c == 2) {
      program.nodes.push_back(AstNode{NodeKind::CALL_SLICE, 0, 0});
      emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
      program.nodes.push_back(AstNode{
          NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      program.nodes.push_back(AstNode{
          NodeKind::CONST, append_const_id(program, Value::from_int(std::uniform_int_distribution<int>(-6, 6)(rng))), 0});
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
    emit_random_expr(rng, program, ctx, depth - 1, RType::Container);
    return;
  }
  emit_random_leaf(rng, program, ctx, RType::NoneType);
}

int choose_or_new_name(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx) {
  const std::vector<std::string> base = {"x", "y", "z", "w", "u", "v"};
  std::vector<int> all;
  for (std::size_t i = 0; i < program.names.size(); ++i) {
    all.push_back(static_cast<int>(i));
  }
  if (!all.empty() && !std::bernoulli_distribution(0.4)(rng)) {
    return choose_one(rng, all);
  }
  for (const std::string& name : base) {
    bool found = false;
    for (const std::string& existing : program.names) {
      if (existing == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      return ensure_name(program, name);
    }
  }
  return ensure_name(program, "t" + std::to_string(ctx.tmp_idx++));
}

void emit_random_stmt(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx, int depth, const Limits& limits);

void emit_random_block(std::mt19937_64& rng,
                       AstProgram& program,
                       PrefixGenCtx& ctx,
                       int depth,
                       const Limits& limits,
                       bool force_return,
                       int max_stmts = -1) {
  const int max_n = (max_stmts < 0) ? limits.max_stmts_per_block : max_stmts;
  const int n = std::uniform_int_distribution<int>(1, std::max(1, max_n))(rng);
  bool has_return = false;
  for (int i = 0; i < n; ++i) {
    program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    const std::size_t before = program.nodes.size();
    emit_random_stmt(rng, program, ctx, depth, limits);
    if (program.nodes[before].kind == NodeKind::RETURN) {
      has_return = true;
      break;
    }
  }
  if (force_return && !has_return) {
    program.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    emit_random_expr(rng,
                     program,
                     ctx,
                     std::max(0, depth - 1),
                     choose_one(rng, std::vector<RType>{RType::Num, RType::Bool}));
  }
  program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
}

void emit_random_stmt(std::mt19937_64& rng, AstProgram& program, PrefixGenCtx& ctx, int depth, const Limits& limits) {
  if (depth <= 0) {
    if (std::bernoulli_distribution(0.75)(rng)) {
      const int name_id = choose_or_new_name(rng, program, ctx);
      const RType type = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType});
      program.nodes.push_back(AstNode{NodeKind::ASSIGN, name_id, 0});
      emit_random_expr(rng, program, ctx, 0, type);
      ctx.num_names.erase(name_id);
      ctx.bool_names.erase(name_id);
      ctx.none_names.erase(name_id);
      if (type == RType::Num) ctx.num_names.insert(name_id);
      if (type == RType::Bool) ctx.bool_names.insert(name_id);
      if (type == RType::NoneType) ctx.none_names.insert(name_id);
      return;
    }
    program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    emit_random_expr(rng, program, ctx, 0, RType::Any);
    return;
  }

  const int choice = std::uniform_int_distribution<int>(0, 3)(rng);
  if (choice == 0) {
    const int name_id = choose_or_new_name(rng, program, ctx);
    const RType type = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType});
    program.nodes.push_back(AstNode{NodeKind::ASSIGN, name_id, 0});
    emit_random_expr(rng, program, ctx, depth - 1, type);
    ctx.num_names.erase(name_id);
    ctx.bool_names.erase(name_id);
    ctx.none_names.erase(name_id);
    if (type == RType::Num) ctx.num_names.insert(name_id);
    if (type == RType::Bool) ctx.bool_names.insert(name_id);
    if (type == RType::NoneType) ctx.none_names.insert(name_id);
    return;
  }
  if (choice == 1) {
    program.nodes.push_back(AstNode{NodeKind::IF_STMT, 0, 0});
    emit_random_expr(rng, program, ctx, depth - 1, RType::Bool);
    PrefixGenCtx then_ctx = ctx;
    PrefixGenCtx else_ctx = ctx;
    emit_random_block(rng, program, then_ctx, depth - 1, limits, false, std::max(1, std::min(2, limits.max_stmts_per_block)));
    emit_random_block(rng, program, else_ctx, depth - 1, limits, false, std::max(1, std::min(2, limits.max_stmts_per_block)));
    return;
  }
  if (choice == 2) {
    const int name_id = ensure_name(
        program, choose_one(rng, std::vector<std::string>{"i", "j", "k"}) + std::to_string(std::uniform_int_distribution<int>(0, 9)(rng)));
    program.nodes.push_back(
        AstNode{NodeKind::FOR_RANGE, name_id, std::uniform_int_distribution<int>(0, std::max(0, limits.max_for_k))(rng)});
    PrefixGenCtx body_ctx = ctx;
    body_ctx.num_names.insert(name_id);
    emit_random_block(rng, program, body_ctx, depth - 1, limits, false, std::max(1, std::min(2, limits.max_stmts_per_block)));
    return;
  }
  program.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  emit_random_expr(rng,
                   program,
                   ctx,
                   depth - 1,
                   choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType}));
}

}  // namespace

ProgramGenome make_random_genome(std::uint64_t seed, const Limits& limits) {
  std::mt19937_64 rng(seed);
  for (int i = 0; i < 128; ++i) {
    AstProgram program;
    program.version = "ast-prefix-v1";
    program.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
    PrefixGenCtx ctx;
    ctx.num_names.insert(ensure_name(program, "x"));
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

}  // namespace g3pvm::evo
