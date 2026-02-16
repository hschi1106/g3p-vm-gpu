#include "g3pvm/evo_ast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace g3pvm::evo {

namespace {

struct GenContext {
  std::map<RType, std::set<std::string>> vars_by_type;
  std::set<std::string> all_vars;
  int tmp_idx = 0;

  static GenContext empty() {
    GenContext ctx;
    ctx.vars_by_type[RType::Num] = {};
    ctx.vars_by_type[RType::Bool] = {};
    ctx.vars_by_type[RType::NoneType] = {};
    return ctx;
  }

  GenContext clone() const { return *this; }

  void add_var(const std::string& name, RType type) {
    all_vars.insert(name);
    vars_by_type[RType::Num].erase(name);
    vars_by_type[RType::Bool].erase(name);
    vars_by_type[RType::NoneType].erase(name);
    if (type == RType::Num || type == RType::Bool || type == RType::NoneType) {
      vars_by_type[type].insert(name);
    }
  }

  std::vector<std::string> vars_for(RType type) const {
    if (type == RType::Any) {
      return std::vector<std::string>(all_vars.begin(), all_vars.end());
    }
    auto it = vars_by_type.find(type);
    if (it == vars_by_type.end()) {
      return {};
    }
    return std::vector<std::string>(it->second.begin(), it->second.end());
  }

  std::string next_tmp_name() {
    const std::string out = "t" + std::to_string(tmp_idx);
    tmp_idx += 1;
    return out;
  }
};

constexpr const char* kBaseNames[] = {"x", "y", "z", "w", "u", "v"};

int builtin_arity(Builtin b) {
  if (b == Builtin::ABS) return 1;
  if (b == Builtin::MIN) return 2;
  if (b == Builtin::MAX) return 2;
  return 3;
}

const char* builtin_name(Builtin b) {
  if (b == Builtin::ABS) return "abs";
  if (b == Builtin::MIN) return "min";
  if (b == Builtin::MAX) return "max";
  return "clip";
}

const char* bop_name(BOp op) {
  switch (op) {
    case BOp::ADD:
      return "ADD";
    case BOp::SUB:
      return "SUB";
    case BOp::MUL:
      return "MUL";
    case BOp::DIV:
      return "DIV";
    case BOp::MOD:
      return "MOD";
    case BOp::LT:
      return "LT";
    case BOp::LE:
      return "LE";
    case BOp::GT:
      return "GT";
    case BOp::GE:
      return "GE";
    case BOp::EQ:
      return "EQ";
    case BOp::NE:
      return "NE";
    case BOp::AND:
      return "AND";
    case BOp::OR:
      return "OR";
  }
  return "ADD";
}

const char* uop_name(UOp op) {
  return op == UOp::NEG ? "NEG" : "NOT";
}

bool is_numeric_type(RType t) { return t == RType::Num; }

bool has_return_stmt(const Block& b) {
  for (const StmtPtr& st : b.stmts) {
    if (st && st->kind == Stmt::Kind::Return) {
      return true;
    }
  }
  return false;
}

template <typename T>
const T* as(const ExprPtr& e) {
  return dynamic_cast<const T*>(e.get());
}

template <typename T>
const T* as(const StmtPtr& s) {
  return dynamic_cast<const T*>(s.get());
}

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

template <typename T>
const T& choose_one(std::mt19937_64& rng, const std::vector<T>& v) {
  if (v.empty()) {
    throw std::runtime_error("choose_one on empty vector");
  }
  return v[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(v.size()) - 1))];
}

ExprPtr make_const_num(std::mt19937_64& rng) {
  if (rand_prob(rng, 0.5)) {
    return std::make_shared<ConstExpr>(Value::from_int(rand_int(rng, -8, 8)));
  }
  const double f = std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0;
  return std::make_shared<ConstExpr>(Value::from_float(f));
}

ExprPtr fallback_const_for_type(RType t) {
  if (t == RType::Bool) {
    return std::make_shared<ConstExpr>(Value::from_bool(false));
  }
  if (t == RType::NoneType) {
    return std::make_shared<ConstExpr>(Value::none());
  }
  return std::make_shared<ConstExpr>(Value::from_int(0));
}

ExprPtr gen_expr(std::mt19937_64& rng, GenContext& ctx, int depth, RType target);
Block gen_block(std::mt19937_64& rng,
                GenContext& ctx,
                int depth,
                const Limits& limits,
                bool force_return,
                int max_stmts = -1);

std::string choose_var_name(std::mt19937_64& rng, GenContext& ctx) {
  if (ctx.all_vars.empty()) {
    for (const char* name : kBaseNames) {
      if (ctx.all_vars.count(name) == 0) {
        return name;
      }
    }
    return ctx.next_tmp_name();
  }
  if (rand_prob(rng, 0.4)) {
    for (const char* name : kBaseNames) {
      if (ctx.all_vars.count(name) == 0) {
        return name;
      }
    }
    return ctx.next_tmp_name();
  }
  std::vector<std::string> all(ctx.all_vars.begin(), ctx.all_vars.end());
  return choose_one(rng, all);
}

ExprPtr rand_leaf_expr(std::mt19937_64& rng, GenContext& ctx, RType target) {
  if (target == RType::Num) {
    std::vector<std::string> vars = ctx.vars_for(RType::Num);
    if (!vars.empty() && rand_prob(rng, 0.45)) {
      return std::make_shared<VarExpr>(choose_one(rng, vars));
    }
    return make_const_num(rng);
  }
  if (target == RType::Bool) {
    std::vector<std::string> vars = ctx.vars_for(RType::Bool);
    if (!vars.empty() && rand_prob(rng, 0.45)) {
      return std::make_shared<VarExpr>(choose_one(rng, vars));
    }
    return std::make_shared<ConstExpr>(Value::from_bool(rand_prob(rng, 0.5)));
  }
  if (target == RType::NoneType) {
    std::vector<std::string> vars = ctx.vars_for(RType::NoneType);
    if (!vars.empty() && rand_prob(rng, 0.45)) {
      return std::make_shared<VarExpr>(choose_one(rng, vars));
    }
    return std::make_shared<ConstExpr>(Value::none());
  }

  std::vector<ExprPtr> choices = {
      std::make_shared<ConstExpr>(Value::none()),
      std::make_shared<ConstExpr>(Value::from_bool(rand_prob(rng, 0.5))),
      make_const_num(rng),
  };
  for (RType t : {RType::Num, RType::Bool, RType::NoneType}) {
    std::vector<std::string> vars = ctx.vars_for(t);
    if (!vars.empty()) {
      choices.push_back(std::make_shared<VarExpr>(choose_one(rng, vars)));
    }
  }
  return choose_one(rng, choices);
}

ExprPtr gen_bool_compare(std::mt19937_64& rng, GenContext& ctx, int depth) {
  const int mode = rand_int(rng, 0, 2);
  if (mode == 0) {
    std::vector<BOp> ops = {BOp::LT, BOp::LE, BOp::GT, BOp::GE, BOp::EQ, BOp::NE};
    return std::make_shared<BinaryExpr>(choose_one(rng, ops),
                                        gen_expr(rng, ctx, depth - 1, RType::Num),
                                        gen_expr(rng, ctx, depth - 1, RType::Num));
  }
  if (mode == 1) {
    std::vector<BOp> ops = {BOp::EQ, BOp::NE};
    return std::make_shared<BinaryExpr>(choose_one(rng, ops),
                                        gen_expr(rng, ctx, depth - 1, RType::Bool),
                                        gen_expr(rng, ctx, depth - 1, RType::Bool));
  }
  std::vector<BOp> ops = {BOp::EQ, BOp::NE};
  return std::make_shared<BinaryExpr>(choose_one(rng, ops),
                                      gen_expr(rng, ctx, depth - 1, RType::NoneType),
                                      gen_expr(rng, ctx, depth - 1, RType::NoneType));
}

ExprPtr gen_expr(std::mt19937_64& rng, GenContext& ctx, int depth, RType target) {
  if (target == RType::Any) {
    std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
    target = choose_one(rng, ts);
  }

  if (depth <= 0) {
    return rand_leaf_expr(rng, ctx, target);
  }

  if (target == RType::Num) {
    const int choice = rand_int(rng, 0, 5);
    if (choice == 0) {
      return rand_leaf_expr(rng, ctx, RType::Num);
    }
    if (choice == 1) {
      return std::make_shared<UnaryExpr>(UOp::NEG, gen_expr(rng, ctx, depth - 1, RType::Num));
    }
    if (choice == 2) {
      std::vector<BOp> ops = {BOp::ADD, BOp::SUB, BOp::MUL, BOp::DIV, BOp::MOD};
      return std::make_shared<BinaryExpr>(choose_one(rng, ops),
                                          gen_expr(rng, ctx, depth - 1, RType::Num),
                                          gen_expr(rng, ctx, depth - 1, RType::Num));
    }
    if (choice == 3) {
      std::vector<Builtin> fs = {Builtin::ABS, Builtin::MIN, Builtin::MAX, Builtin::CLIP};
      const Builtin f = choose_one(rng, fs);
      if (f == Builtin::ABS) {
        return std::make_shared<CallExpr>(f, std::vector<ExprPtr>{gen_expr(rng, ctx, depth - 1, RType::Num)});
      }
      if (f == Builtin::MIN || f == Builtin::MAX) {
        return std::make_shared<CallExpr>(
            f,
            std::vector<ExprPtr>{gen_expr(rng, ctx, depth - 1, RType::Num),
                                 gen_expr(rng, ctx, depth - 1, RType::Num)});
      }
      return std::make_shared<CallExpr>(
          f,
          std::vector<ExprPtr>{gen_expr(rng, ctx, depth - 1, RType::Num),
                               gen_expr(rng, ctx, depth - 1, RType::Num),
                               gen_expr(rng, ctx, depth - 1, RType::Num)});
    }
    return std::make_shared<IfExprNode>(gen_expr(rng, ctx, depth - 1, RType::Bool),
                                        gen_expr(rng, ctx, depth - 1, RType::Num),
                                        gen_expr(rng, ctx, depth - 1, RType::Num));
  }

  if (target == RType::Bool) {
    const int choice = rand_int(rng, 0, 4);
    if (choice == 0) {
      return rand_leaf_expr(rng, ctx, RType::Bool);
    }
    if (choice == 1) {
      return std::make_shared<UnaryExpr>(UOp::NOT, gen_expr(rng, ctx, depth - 1, RType::Bool));
    }
    if (choice == 2) {
      return gen_bool_compare(rng, ctx, depth);
    }
    if (choice == 3) {
      std::vector<BOp> ops = {BOp::AND, BOp::OR};
      return std::make_shared<BinaryExpr>(choose_one(rng, ops),
                                          gen_expr(rng, ctx, depth - 1, RType::Bool),
                                          gen_expr(rng, ctx, depth - 1, RType::Bool));
    }
    return std::make_shared<IfExprNode>(gen_expr(rng, ctx, depth - 1, RType::Bool),
                                        gen_expr(rng, ctx, depth - 1, RType::Bool),
                                        gen_expr(rng, ctx, depth - 1, RType::Bool));
  }

  return rand_leaf_expr(rng, ctx, RType::NoneType);
}

StmtPtr gen_assign_stmt(std::mt19937_64& rng, GenContext& ctx, int depth) {
  const std::string name = choose_var_name(rng, ctx);
  std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
  const RType t = choose_one(rng, ts);
  ExprPtr e = gen_expr(rng, ctx, std::max(0, depth), t);
  ctx.add_var(name, t);
  return std::make_shared<AssignStmt>(name, e);
}

StmtPtr gen_stmt(std::mt19937_64& rng, GenContext& ctx, int depth, const Limits& limits) {
  if (depth <= 0) {
    if (rand_prob(rng, 0.75)) {
      return gen_assign_stmt(rng, ctx, depth);
    }
    return std::make_shared<ReturnStmt>(gen_expr(rng, ctx, 0, RType::Any));
  }

  const int choice = rand_int(rng, 0, 3);
  if (choice == 0) {
    return gen_assign_stmt(rng, ctx, depth - 1);
  }
  if (choice == 1) {
    GenContext then_ctx = ctx.clone();
    GenContext else_ctx = ctx.clone();
    return std::make_shared<IfStmtNode>(
        gen_expr(rng, ctx, depth - 1, RType::Bool),
        gen_block(rng,
                  then_ctx,
                  depth - 1,
                  limits,
                  false,
                  std::max(1, std::min(2, limits.max_stmts_per_block))),
        gen_block(rng,
                  else_ctx,
                  depth - 1,
                  limits,
                  false,
                  std::max(1, std::min(2, limits.max_stmts_per_block))));
  }
  if (choice == 2) {
    std::vector<std::string> loop_names = {"i", "j", "k"};
    std::string loop_var = choose_one(rng, loop_names);
    if (ctx.all_vars.count(loop_var) != 0U) {
      loop_var += std::to_string(rand_int(rng, 0, 9));
    }
    GenContext body_ctx = ctx.clone();
    body_ctx.add_var(loop_var, RType::Num);
    return std::make_shared<ForRangeStmt>(
        loop_var,
        rand_int(rng, 0, std::max(0, limits.max_for_k)),
        gen_block(rng,
                  body_ctx,
                  depth - 1,
                  limits,
                  false,
                  std::max(1, std::min(2, limits.max_stmts_per_block))));
  }

  std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
  return std::make_shared<ReturnStmt>(gen_expr(rng, ctx, depth - 1, choose_one(rng, ts)));
}

Block gen_block(std::mt19937_64& rng,
                GenContext& ctx,
                int depth,
                const Limits& limits,
                bool force_return,
                int max_stmts) {
  const int max_n = (max_stmts < 0) ? limits.max_stmts_per_block : max_stmts;
  const int n = rand_int(rng, 1, std::max(1, max_n));
  Block out;
  for (int i = 0; i < n; ++i) {
    StmtPtr st = gen_stmt(rng, ctx, depth, limits);
    out.stmts.push_back(st);
    if (st && st->kind == Stmt::Kind::Return) {
      break;
    }
  }
  if (force_return && !has_return_stmt(out)) {
    std::vector<RType> ts = {RType::Num, RType::Bool};
    out.stmts.push_back(std::make_shared<ReturnStmt>(
        gen_expr(rng, ctx, std::max(0, depth - 1), choose_one(rng, ts))));
  }
  if (static_cast<int>(out.stmts.size()) > max_n) {
    out.stmts.resize(static_cast<std::size_t>(max_n));
  }
  return out;
}

RType infer_expr_type(const ExprPtr& e, const std::unordered_map<std::string, RType>& env) {
  if (!e) {
    return RType::Invalid;
  }

  if (const auto* c = as<ConstExpr>(e)) {
    if (c->value.tag == ValueTag::None) return RType::NoneType;
    if (c->value.tag == ValueTag::Bool) return RType::Bool;
    if (c->value.tag == ValueTag::Int || c->value.tag == ValueTag::Float) return RType::Num;
    return RType::Invalid;
  }

  if (const auto* v = as<VarExpr>(e)) {
    auto it = env.find(v->name);
    return it == env.end() ? RType::Any : it->second;
  }

  if (const auto* u = as<UnaryExpr>(e)) {
    const RType t = infer_expr_type(u->e, env);
    if (u->op == UOp::NEG) {
      return t == RType::Num ? RType::Num : RType::Invalid;
    }
    if (u->op == UOp::NOT) {
      return t == RType::Bool ? RType::Bool : RType::Invalid;
    }
    return RType::Invalid;
  }

  if (const auto* b = as<BinaryExpr>(e)) {
    const RType ta = infer_expr_type(b->a, env);
    const RType tb = infer_expr_type(b->b, env);
    if (b->op == BOp::ADD || b->op == BOp::SUB || b->op == BOp::MUL || b->op == BOp::DIV ||
        b->op == BOp::MOD) {
      return (ta == RType::Num && tb == RType::Num) ? RType::Num : RType::Invalid;
    }
    if (b->op == BOp::AND || b->op == BOp::OR) {
      return (ta == RType::Bool && tb == RType::Bool) ? RType::Bool : RType::Invalid;
    }
    if (b->op == BOp::LT || b->op == BOp::LE || b->op == BOp::GT || b->op == BOp::GE) {
      return (ta == RType::Num && tb == RType::Num) ? RType::Bool : RType::Invalid;
    }
    if (b->op == BOp::EQ || b->op == BOp::NE) {
      if (ta == tb && (ta == RType::Num || ta == RType::Bool || ta == RType::NoneType)) {
        return RType::Bool;
      }
      return RType::Invalid;
    }
    return RType::Invalid;
  }

  if (const auto* ife = as<IfExprNode>(e)) {
    const RType tc = infer_expr_type(ife->cond, env);
    const RType tt = infer_expr_type(ife->then_e, env);
    const RType tf = infer_expr_type(ife->else_e, env);
    if (tc != RType::Bool) {
      return RType::Invalid;
    }
    if (tt == tf && (tt == RType::Num || tt == RType::Bool || tt == RType::NoneType)) {
      return tt;
    }
    return RType::Invalid;
  }

  if (const auto* call = as<CallExpr>(e)) {
    if (static_cast<int>(call->args.size()) != builtin_arity(call->builtin)) {
      return RType::Invalid;
    }
    for (const ExprPtr& arg : call->args) {
      if (infer_expr_type(arg, env) != RType::Num) {
        return RType::Invalid;
      }
    }
    return RType::Num;
  }

  return RType::Invalid;
}

RType validate_expr(const ExprPtr& e,
                    const std::unordered_map<std::string, RType>& env,
                    std::vector<std::string>& errors) {
  const RType t = infer_expr_type(e, env);
  if (t == RType::Invalid) {
    errors.push_back("invalid expression typing");
  }
  return t;
}

int count_expr_nodes(const ExprPtr& e);
int count_stmt_nodes(const StmtPtr& st);
int count_block_nodes(const Block& b);
int expr_depth(const ExprPtr& e);
int stmt_expr_depth(const StmtPtr& st);
int max_block_expr_depth(const Block& b);

void validate_block(const Block& b,
                    const Limits& limits,
                    const std::unordered_map<std::string, RType>& env,
                    std::vector<std::string>& errors,
                    bool top_level = false) {
  if (static_cast<int>(b.stmts.size()) > limits.max_stmts_per_block) {
    errors.push_back("block has too many statements");
  }
  if (top_level && !has_return_stmt(b)) {
    errors.push_back("top-level block must contain at least one Return");
  }

  std::unordered_map<std::string, RType> local = env;
  for (const StmtPtr& st : b.stmts) {
    if (const auto* assign = as<AssignStmt>(st)) {
      const RType t = validate_expr(assign->e, local, errors);
      if (t == RType::Num || t == RType::Bool || t == RType::NoneType) {
        local[assign->name] = t;
      }
    } else if (const auto* ret = as<ReturnStmt>(st)) {
      validate_expr(ret->e, local, errors);
    } else if (const auto* ifs = as<IfStmtNode>(st)) {
      const RType tc = validate_expr(ifs->cond, local, errors);
      if (tc != RType::Bool) {
        errors.push_back("IfStmt condition must be Bool");
      }
      validate_block(ifs->then_block, limits, local, errors, false);
      validate_block(ifs->else_block, limits, local, errors, false);
    } else if (const auto* fr = as<ForRangeStmt>(st)) {
      if (fr->k < 0) {
        errors.push_back("ForRange.k must be non-negative int");
      }
      if (fr->k > limits.max_for_k) {
        errors.push_back("ForRange.k exceeds max_for_k");
      }
      auto body_env = local;
      body_env[fr->var] = RType::Num;
      validate_block(fr->body, limits, body_env, errors, false);
    } else {
      errors.push_back("unknown statement type");
    }
  }
}

int count_expr_nodes(const ExprPtr& e) {
  if (!e) return 1;
  if (e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Var) {
    return 1;
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    return 1 + count_expr_nodes(u->e);
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    return 1 + count_expr_nodes(b->a) + count_expr_nodes(b->b);
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    return 1 + count_expr_nodes(ife->cond) + count_expr_nodes(ife->then_e) + count_expr_nodes(ife->else_e);
  }
  if (const auto* call = as<CallExpr>(e)) {
    int n = 1;
    for (const ExprPtr& arg : call->args) {
      n += count_expr_nodes(arg);
    }
    return n;
  }
  return 1;
}

int count_stmt_nodes(const StmtPtr& st) {
  if (!st) return 1;
  if (const auto* assign = as<AssignStmt>(st)) {
    return 1 + count_expr_nodes(assign->e);
  }
  if (const auto* ret = as<ReturnStmt>(st)) {
    return 1 + count_expr_nodes(ret->e);
  }
  if (const auto* ifs = as<IfStmtNode>(st)) {
    return 1 + count_expr_nodes(ifs->cond) + count_block_nodes(ifs->then_block) + count_block_nodes(ifs->else_block);
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    return 1 + count_block_nodes(fr->body);
  }
  return 1;
}

int count_block_nodes(const Block& b) {
  int n = 1;
  for (const StmtPtr& st : b.stmts) {
    n += count_stmt_nodes(st);
  }
  return n;
}

int expr_depth(const ExprPtr& e) {
  if (!e) return 1;
  if (e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Var) {
    return 1;
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    return 1 + expr_depth(u->e);
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    return 1 + std::max(expr_depth(b->a), expr_depth(b->b));
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    return 1 + std::max({expr_depth(ife->cond), expr_depth(ife->then_e), expr_depth(ife->else_e)});
  }
  if (const auto* call = as<CallExpr>(e)) {
    int d = 1;
    for (const ExprPtr& arg : call->args) {
      d = std::max(d, 1 + expr_depth(arg));
    }
    return d;
  }
  return 1;
}

int stmt_expr_depth(const StmtPtr& st) {
  if (const auto* assign = as<AssignStmt>(st)) {
    return expr_depth(assign->e);
  }
  if (const auto* ret = as<ReturnStmt>(st)) {
    return expr_depth(ret->e);
  }
  if (const auto* ifs = as<IfStmtNode>(st)) {
    return std::max({expr_depth(ifs->cond), max_block_expr_depth(ifs->then_block), max_block_expr_depth(ifs->else_block)});
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    return max_block_expr_depth(fr->body);
  }
  return 0;
}

int max_block_expr_depth(const Block& b) {
  if (b.stmts.empty()) {
    return 0;
  }
  int out = 0;
  for (const StmtPtr& st : b.stmts) {
    out = std::max(out, stmt_expr_depth(st));
  }
  return out;
}

bool block_uses_builtins(const Block& b) {
  std::function<bool(const ExprPtr&)> has_builtin_expr = [&](const ExprPtr& e) {
    if (!e) return false;
    if (e->kind == Expr::Kind::Call) {
      return true;
    }
    if (const auto* u = as<UnaryExpr>(e)) {
      return has_builtin_expr(u->e);
    }
    if (const auto* be = as<BinaryExpr>(e)) {
      return has_builtin_expr(be->a) || has_builtin_expr(be->b);
    }
    if (const auto* ife = as<IfExprNode>(e)) {
      return has_builtin_expr(ife->cond) || has_builtin_expr(ife->then_e) || has_builtin_expr(ife->else_e);
    }
    return false;
  };

  for (const StmtPtr& st : b.stmts) {
    if (const auto* assign = as<AssignStmt>(st)) {
      if (has_builtin_expr(assign->e)) return true;
    } else if (const auto* ret = as<ReturnStmt>(st)) {
      if (has_builtin_expr(ret->e)) return true;
    } else if (const auto* ifs = as<IfStmtNode>(st)) {
      if (has_builtin_expr(ifs->cond) || block_uses_builtins(ifs->then_block) || block_uses_builtins(ifs->else_block)) {
        return true;
      }
    } else if (const auto* fr = as<ForRangeStmt>(st)) {
      if (block_uses_builtins(fr->body)) return true;
    }
  }
  return false;
}

ExprPtr shrink_expr_depth(const ExprPtr& e, int depth, RType expected) {
  if (depth <= 1) {
    return fallback_const_for_type(expected == RType::Any ? RType::Num : expected);
  }
  if (!e || e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Var) {
    return e;
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    const RType target = u->op == UOp::NEG ? RType::Num : RType::Bool;
    return std::make_shared<UnaryExpr>(u->op, shrink_expr_depth(u->e, depth - 1, target));
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    if (b->op == BOp::ADD || b->op == BOp::SUB || b->op == BOp::MUL || b->op == BOp::DIV || b->op == BOp::MOD) {
      return std::make_shared<BinaryExpr>(b->op,
                                          shrink_expr_depth(b->a, depth - 1, RType::Num),
                                          shrink_expr_depth(b->b, depth - 1, RType::Num));
    }
    if (b->op == BOp::AND || b->op == BOp::OR) {
      return std::make_shared<BinaryExpr>(b->op,
                                          shrink_expr_depth(b->a, depth - 1, RType::Bool),
                                          shrink_expr_depth(b->b, depth - 1, RType::Bool));
    }
    return std::make_shared<BinaryExpr>(b->op,
                                        shrink_expr_depth(b->a, depth - 1, RType::Num),
                                        shrink_expr_depth(b->b, depth - 1, RType::Num));
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    const RType t = expected == RType::Any ? RType::Num : expected;
    return std::make_shared<IfExprNode>(shrink_expr_depth(ife->cond, depth - 1, RType::Bool),
                                        shrink_expr_depth(ife->then_e, depth - 1, t),
                                        shrink_expr_depth(ife->else_e, depth - 1, t));
  }
  if (const auto* call = as<CallExpr>(e)) {
    std::vector<ExprPtr> args;
    args.reserve(call->args.size());
    for (const ExprPtr& arg : call->args) {
      args.push_back(shrink_expr_depth(arg, depth - 1, RType::Num));
    }
    return std::make_shared<CallExpr>(call->builtin, args);
  }
  return fallback_const_for_type(expected);
}

StmtPtr shrink_stmt_expr_depth(const StmtPtr& st, int depth);
Block shrink_block_expr_depth(const Block& b, int depth);

StmtPtr shrink_stmt_expr_depth(const StmtPtr& st, int depth) {
  if (const auto* assign = as<AssignStmt>(st)) {
    return std::make_shared<AssignStmt>(assign->name, shrink_expr_depth(assign->e, depth, RType::Any));
  }
  if (const auto* ret = as<ReturnStmt>(st)) {
    return std::make_shared<ReturnStmt>(shrink_expr_depth(ret->e, depth, RType::Any));
  }
  if (const auto* ifs = as<IfStmtNode>(st)) {
    return std::make_shared<IfStmtNode>(shrink_expr_depth(ifs->cond, depth, RType::Bool),
                                        shrink_block_expr_depth(ifs->then_block, depth),
                                        shrink_block_expr_depth(ifs->else_block, depth));
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    return std::make_shared<ForRangeStmt>(fr->var, fr->k, shrink_block_expr_depth(fr->body, depth));
  }
  return st;
}

Block shrink_block_expr_depth(const Block& b, int depth) {
  Block out;
  out.stmts.reserve(b.stmts.size());
  for (const StmtPtr& st : b.stmts) {
    out.stmts.push_back(shrink_stmt_expr_depth(st, depth));
  }
  return out;
}

Block clamp_block_stmt_count(const Block& b, int max_stmts) {
  Block out;
  if (static_cast<int>(b.stmts.size()) <= max_stmts) {
    out.stmts.reserve(b.stmts.size());
    for (const StmtPtr& st : b.stmts) {
      if (const auto* ifs = as<IfStmtNode>(st)) {
        out.stmts.push_back(std::make_shared<IfStmtNode>(ifs->cond,
                                                         clamp_block_stmt_count(ifs->then_block, max_stmts),
                                                         clamp_block_stmt_count(ifs->else_block, max_stmts)));
      } else if (const auto* fr = as<ForRangeStmt>(st)) {
        out.stmts.push_back(
            std::make_shared<ForRangeStmt>(fr->var, fr->k, clamp_block_stmt_count(fr->body, max_stmts)));
      } else {
        out.stmts.push_back(st);
      }
    }
    return out;
  }
  out.stmts.assign(b.stmts.begin(), b.stmts.begin() + max_stmts);
  return out;
}

Block clamp_for_k(const Block& b, int max_for_k) {
  Block out;
  out.stmts.reserve(b.stmts.size());
  for (const StmtPtr& st : b.stmts) {
    if (const auto* ifs = as<IfStmtNode>(st)) {
      out.stmts.push_back(std::make_shared<IfStmtNode>(ifs->cond,
                                                       clamp_for_k(ifs->then_block, max_for_k),
                                                       clamp_for_k(ifs->else_block, max_for_k)));
    } else if (const auto* fr = as<ForRangeStmt>(st)) {
      int k = std::max(0, std::min(max_for_k, fr->k));
      out.stmts.push_back(std::make_shared<ForRangeStmt>(fr->var, k, clamp_for_k(fr->body, max_for_k)));
    } else {
      out.stmts.push_back(st);
    }
  }
  return out;
}

std::unordered_map<std::string, RType> infer_env_from_program(const Block& b) {
  std::unordered_map<std::string, RType> env;
  for (const StmtPtr& st : b.stmts) {
    if (const auto* assign = as<AssignStmt>(st)) {
      const RType t = infer_expr_type(assign->e, env);
      if (t == RType::Num || t == RType::Bool || t == RType::NoneType) {
        env[assign->name] = t;
      }
    } else if (const auto* fr = as<ForRangeStmt>(st)) {
      env[fr->var] = RType::Num;
    }
  }
  return env;
}

GenContext infer_context_from_program(const Block& b) {
  GenContext ctx = GenContext::empty();

  std::function<void(const Block&, std::unordered_map<std::string, RType>)> walk_block;
  walk_block = [&](const Block& block, std::unordered_map<std::string, RType> local) {
    for (const StmtPtr& st : block.stmts) {
      if (const auto* assign = as<AssignStmt>(st)) {
        const RType t = infer_expr_type(assign->e, local);
        if (t == RType::Num || t == RType::Bool || t == RType::NoneType) {
          local[assign->name] = t;
          ctx.add_var(assign->name, t);
        }
      } else if (const auto* ifs = as<IfStmtNode>(st)) {
        walk_block(ifs->then_block, local);
        walk_block(ifs->else_block, local);
      } else if (const auto* fr = as<ForRangeStmt>(st)) {
        local[fr->var] = RType::Num;
        ctx.add_var(fr->var, RType::Num);
        walk_block(fr->body, local);
      }
    }
  };

  walk_block(b, {});
  return ctx;
}

Block repair_limits(const Block& ast, std::mt19937_64& rng, const Limits& limits, const GenContext* ctx_in = nullptr) {
  GenContext ctx = ctx_in == nullptr ? GenContext::empty() : *ctx_in;
  Block fixed = shrink_block_expr_depth(ast, limits.max_expr_depth);
  fixed = clamp_block_stmt_count(fixed, limits.max_stmts_per_block);
  fixed = clamp_for_k(fixed, limits.max_for_k);
  if (!has_return_stmt(fixed)) {
    fixed.stmts.push_back(
        std::make_shared<ReturnStmt>(gen_expr(rng, ctx, std::max(0, limits.max_expr_depth - 1), RType::Num)));
  }
  if (count_block_nodes(fixed) > limits.max_total_nodes) {
    return Block{{std::make_shared<ReturnStmt>(std::make_shared<ConstExpr>(Value::from_int(0)))}};
  }
  return fixed;
}

ExprPtr mutate_expr(const ExprPtr& e, std::mt19937_64& rng, GenContext& ctx, int depth, RType target) {
  if (depth <= 0 || rand_prob(rng, 0.3)) {
    return gen_expr(rng, ctx, std::max(0, depth), target);
  }

  if (as<ConstExpr>(e) != nullptr) {
    return gen_expr(rng, ctx, std::max(0, depth - 1), target);
  }
  if (as<VarExpr>(e) != nullptr) {
    std::vector<std::string> vars = ctx.vars_for(target);
    if (!vars.empty() && rand_prob(rng, 0.6)) {
      return std::make_shared<VarExpr>(choose_one(rng, vars));
    }
    return gen_expr(rng, ctx, std::max(0, depth - 1), target);
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    const RType inner_target = u->op == UOp::NEG ? target : RType::Bool;
    return std::make_shared<UnaryExpr>(u->op, mutate_expr(u->e, rng, ctx, depth - 1, inner_target));
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    if (b->op == BOp::ADD || b->op == BOp::SUB || b->op == BOp::MUL || b->op == BOp::DIV || b->op == BOp::MOD) {
      BOp op = b->op;
      if (rand_prob(rng, 0.3)) {
        std::vector<BOp> ops = {BOp::ADD, BOp::SUB, BOp::MUL, BOp::DIV, BOp::MOD};
        op = choose_one(rng, ops);
      }
      if (rand_prob(rng, 0.5)) {
        return std::make_shared<BinaryExpr>(op, mutate_expr(b->a, rng, ctx, depth - 1, RType::Num), b->b);
      }
      return std::make_shared<BinaryExpr>(op, b->a, mutate_expr(b->b, rng, ctx, depth - 1, RType::Num));
    }
    if (b->op == BOp::AND || b->op == BOp::OR) {
      BOp op = b->op;
      if (rand_prob(rng, 0.3)) {
        std::vector<BOp> ops = {BOp::AND, BOp::OR};
        op = choose_one(rng, ops);
      }
      if (rand_prob(rng, 0.5)) {
        return std::make_shared<BinaryExpr>(op, mutate_expr(b->a, rng, ctx, depth - 1, RType::Bool), b->b);
      }
      return std::make_shared<BinaryExpr>(op, b->a, mutate_expr(b->b, rng, ctx, depth - 1, RType::Bool));
    }
    if (b->op == BOp::LT || b->op == BOp::LE || b->op == BOp::GT || b->op == BOp::GE || b->op == BOp::EQ ||
        b->op == BOp::NE) {
      return gen_expr(rng, ctx, depth - 1, RType::Bool);
    }
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    RType t = infer_expr_type(ife->then_e, {});
    if (t == RType::Invalid) {
      t = target;
    }
    const int mode = rand_int(rng, 0, 2);
    if (mode == 0) {
      return std::make_shared<IfExprNode>(mutate_expr(ife->cond, rng, ctx, depth - 1, RType::Bool), ife->then_e,
                                          ife->else_e);
    }
    if (mode == 1) {
      return std::make_shared<IfExprNode>(ife->cond, mutate_expr(ife->then_e, rng, ctx, depth - 1, t), ife->else_e);
    }
    return std::make_shared<IfExprNode>(ife->cond, ife->then_e, mutate_expr(ife->else_e, rng, ctx, depth - 1, t));
  }
  if (const auto* call = as<CallExpr>(e)) {
    if (call->args.empty()) {
      return gen_expr(rng, ctx, depth - 1, RType::Num);
    }
    const int idx = rand_int(rng, 0, static_cast<int>(call->args.size()) - 1);
    std::vector<ExprPtr> args = call->args;
    args[static_cast<std::size_t>(idx)] = mutate_expr(args[static_cast<std::size_t>(idx)], rng, ctx, depth - 1, RType::Num);
    return std::make_shared<CallExpr>(call->builtin, args);
  }

  return gen_expr(rng, ctx, depth - 1, target);
}

StmtPtr mutate_stmt(const StmtPtr& st, std::mt19937_64& rng, GenContext& ctx, const Limits& limits) {
  if (const auto* assign = as<AssignStmt>(st)) {
    if (rand_prob(rng, 0.3)) {
      const std::string name = choose_var_name(rng, ctx);
      RType t = infer_expr_type(assign->e, infer_env_from_program(Block{{st}}));
      if (t != RType::Num && t != RType::Bool && t != RType::NoneType) {
        std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
        t = choose_one(rng, ts);
      }
      ExprPtr e_new = gen_expr(rng, ctx, limits.max_expr_depth - 1, t);
      ctx.add_var(name, t);
      return std::make_shared<AssignStmt>(name, e_new);
    }
    RType t = infer_expr_type(assign->e, infer_env_from_program(Block{{st}}));
    if (t != RType::Num && t != RType::Bool && t != RType::NoneType) {
      std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
      t = choose_one(rng, ts);
    }
    ExprPtr new_e = mutate_expr(assign->e, rng, ctx, limits.max_expr_depth - 1, t);
    ctx.add_var(assign->name, t);
    return std::make_shared<AssignStmt>(assign->name, new_e);
  }

  if (const auto* ret = as<ReturnStmt>(st)) {
    RType t = infer_expr_type(ret->e, infer_env_from_program(Block{{st}}));
    if (t != RType::Num && t != RType::Bool && t != RType::NoneType) {
      std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
      t = choose_one(rng, ts);
    }
    return std::make_shared<ReturnStmt>(mutate_expr(ret->e, rng, ctx, limits.max_expr_depth - 1, t));
  }

  if (const auto* ifs = as<IfStmtNode>(st)) {
    const int mode = rand_int(rng, 0, 3);
    if (mode == 0) {
      return std::make_shared<IfStmtNode>(mutate_expr(ifs->cond, rng, ctx, limits.max_expr_depth - 1, RType::Bool),
                                          ifs->then_block,
                                          ifs->else_block);
    }
    if (mode == 1) {
      return std::make_shared<IfStmtNode>(ifs->cond, ifs->else_block, ifs->then_block);
    }
    if (mode == 2) {
      GenContext then_ctx = ctx.clone();
      return std::make_shared<IfStmtNode>(
          ifs->cond,
          gen_block(rng,
                    then_ctx,
                    std::max(1, limits.max_expr_depth - 2),
                    limits,
                    false,
                    std::max(1, std::min(2, limits.max_stmts_per_block))),
          ifs->else_block);
    }
    GenContext else_ctx = ctx.clone();
    return std::make_shared<IfStmtNode>(
        ifs->cond,
        ifs->then_block,
        gen_block(rng,
                  else_ctx,
                  std::max(1, limits.max_expr_depth - 2),
                  limits,
                  false,
                  std::max(1, std::min(2, limits.max_stmts_per_block))));
  }

  if (const auto* fr = as<ForRangeStmt>(st)) {
    const int mode = rand_int(rng, 0, 2);
    if (mode == 0) {
      const int delta = choose_one(rng, std::vector<int>{-2, -1, 1, 2});
      const int new_k = std::max(0, std::min(limits.max_for_k, fr->k + delta));
      return std::make_shared<ForRangeStmt>(fr->var, new_k, fr->body);
    }
    if (mode == 1) {
      GenContext body_ctx = ctx.clone();
      body_ctx.add_var(fr->var, RType::Num);
      return std::make_shared<ForRangeStmt>(
          fr->var,
          fr->k,
          gen_block(rng,
                    body_ctx,
                    std::max(1, limits.max_expr_depth - 2),
                    limits,
                    false,
                    std::max(1, std::min(2, limits.max_stmts_per_block))));
    }
    GenContext body_ctx = ctx.clone();
    body_ctx.add_var(fr->var, RType::Num);
    return std::make_shared<ForRangeStmt>(fr->var, rand_int(rng, 0, limits.max_for_k), fr->body);
  }

  return st;
}

int stmt_tree_size(const StmtPtr& st);
int block_stmt_tree_size(const Block& b);
std::pair<StmtPtr, int> mutate_stmt_in_stmt(const StmtPtr& st,
                                            int target,
                                            std::mt19937_64& rng,
                                            GenContext& ctx,
                                            const Limits& limits);
std::pair<Block, int> mutate_block_at_index(const Block& b,
                                            int target,
                                            std::mt19937_64& rng,
                                            GenContext& ctx,
                                            const Limits& limits);

int stmt_tree_size(const StmtPtr& st) {
  if (const auto* ifs = as<IfStmtNode>(st)) {
    return 1 + block_stmt_tree_size(ifs->then_block) + block_stmt_tree_size(ifs->else_block);
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    return 1 + block_stmt_tree_size(fr->body);
  }
  return 1;
}

int block_stmt_tree_size(const Block& b) {
  int out = 0;
  for (const StmtPtr& st : b.stmts) {
    out += stmt_tree_size(st);
  }
  return out;
}

std::pair<StmtPtr, int> mutate_stmt_in_stmt(const StmtPtr& st,
                                            int target,
                                            std::mt19937_64& rng,
                                            GenContext& ctx,
                                            const Limits& limits) {
  if (target == 0) {
    return {mutate_stmt(st, rng, ctx, limits), -1};
  }

  int idx = target - 1;
  if (const auto* ifs = as<IfStmtNode>(st)) {
    const int then_size = block_stmt_tree_size(ifs->then_block);
    if (idx < then_size) {
      GenContext then_ctx = ctx.clone();
      auto [new_then, rem] = mutate_block_at_index(ifs->then_block, idx, rng, then_ctx, limits);
      return {std::make_shared<IfStmtNode>(ifs->cond, new_then, ifs->else_block), rem};
    }
    idx -= then_size;
    const int else_size = block_stmt_tree_size(ifs->else_block);
    if (idx < else_size) {
      GenContext else_ctx = ctx.clone();
      auto [new_else, rem] = mutate_block_at_index(ifs->else_block, idx, rng, else_ctx, limits);
      return {std::make_shared<IfStmtNode>(ifs->cond, ifs->then_block, new_else), rem};
    }
  } else if (const auto* fr = as<ForRangeStmt>(st)) {
    const int body_size = block_stmt_tree_size(fr->body);
    if (idx < body_size) {
      GenContext body_ctx = ctx.clone();
      body_ctx.add_var(fr->var, RType::Num);
      auto [new_body, rem] = mutate_block_at_index(fr->body, idx, rng, body_ctx, limits);
      return {std::make_shared<ForRangeStmt>(fr->var, fr->k, new_body), rem};
    }
  }
  return {st, target};
}

std::pair<Block, int> mutate_block_at_index(const Block& b,
                                            int target,
                                            std::mt19937_64& rng,
                                            GenContext& ctx,
                                            const Limits& limits) {
  Block out;
  out.stmts.reserve(b.stmts.size());
  int cur = target;
  for (const StmtPtr& st : b.stmts) {
    if (cur < 0) {
      out.stmts.push_back(st);
      continue;
    }
    const int size = stmt_tree_size(st);
    if (cur >= size) {
      out.stmts.push_back(st);
      cur -= size;
      continue;
    }
    auto [new_st, rem] = mutate_stmt_in_stmt(st, cur, rng, ctx, limits);
    out.stmts.push_back(new_st);
    cur = rem;
  }
  return {out, cur};
}

Block crossover_top_level_splice_block(const Block& a, const Block& b, std::mt19937_64& rng, const Limits& limits) {
  const int cut_a = a.stmts.empty() ? 0 : rand_int(rng, 0, static_cast<int>(a.stmts.size()));
  const int cut_b = b.stmts.empty() ? 0 : rand_int(rng, 0, static_cast<int>(b.stmts.size()));

  Block child;
  child.stmts.insert(child.stmts.end(), a.stmts.begin(), a.stmts.begin() + cut_a);
  child.stmts.insert(child.stmts.end(), b.stmts.begin() + cut_b, b.stmts.end());
  if (child.stmts.empty()) {
    child.stmts.push_back(std::make_shared<ReturnStmt>(std::make_shared<ConstExpr>(Value::from_int(0))));
  }
  if (static_cast<int>(child.stmts.size()) > limits.max_stmts_per_block) {
    child.stmts.resize(static_cast<std::size_t>(limits.max_stmts_per_block));
  }
  return child;
}

using TypedExpr = std::pair<ExprPtr, RType>;

void collect_typed_expr_nodes_walk_expr(const ExprPtr& e,
                                        const std::unordered_map<std::string, RType>& env,
                                        std::vector<TypedExpr>& out) {
  const RType t = infer_expr_type(e, env);
  if (t == RType::Num || t == RType::Bool || t == RType::NoneType) {
    out.push_back({e, t});
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    collect_typed_expr_nodes_walk_expr(u->e, env, out);
    return;
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    collect_typed_expr_nodes_walk_expr(b->a, env, out);
    collect_typed_expr_nodes_walk_expr(b->b, env, out);
    return;
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    collect_typed_expr_nodes_walk_expr(ife->cond, env, out);
    collect_typed_expr_nodes_walk_expr(ife->then_e, env, out);
    collect_typed_expr_nodes_walk_expr(ife->else_e, env, out);
    return;
  }
  if (const auto* call = as<CallExpr>(e)) {
    for (const ExprPtr& arg : call->args) {
      collect_typed_expr_nodes_walk_expr(arg, env, out);
    }
  }
}

void collect_typed_expr_nodes_walk_block(const Block& block,
                                         const std::unordered_map<std::string, RType>& env,
                                         std::vector<TypedExpr>& out) {
  std::unordered_map<std::string, RType> cur = env;
  for (const StmtPtr& st : block.stmts) {
    if (const auto* assign = as<AssignStmt>(st)) {
      collect_typed_expr_nodes_walk_expr(assign->e, cur, out);
      const RType t = infer_expr_type(assign->e, cur);
      if (t == RType::Num || t == RType::Bool || t == RType::NoneType) {
        cur[assign->name] = t;
      }
    } else if (const auto* ret = as<ReturnStmt>(st)) {
      collect_typed_expr_nodes_walk_expr(ret->e, cur, out);
    } else if (const auto* ifs = as<IfStmtNode>(st)) {
      collect_typed_expr_nodes_walk_expr(ifs->cond, cur, out);
      collect_typed_expr_nodes_walk_block(ifs->then_block, cur, out);
      collect_typed_expr_nodes_walk_block(ifs->else_block, cur, out);
    } else if (const auto* fr = as<ForRangeStmt>(st)) {
      auto body_env = cur;
      body_env[fr->var] = RType::Num;
      collect_typed_expr_nodes_walk_block(fr->body, body_env, out);
    }
  }
}

std::vector<TypedExpr> collect_typed_expr_nodes(const Block& b) {
  std::vector<TypedExpr> out;
  collect_typed_expr_nodes_walk_block(b, {}, out);
  return out;
}

void collect_stmt_nodes_walk_block(const Block& block, std::vector<StmtPtr>& out) {
  for (const StmtPtr& st : block.stmts) {
    out.push_back(st);
    if (const auto* ifs = as<IfStmtNode>(st)) {
      collect_stmt_nodes_walk_block(ifs->then_block, out);
      collect_stmt_nodes_walk_block(ifs->else_block, out);
    } else if (const auto* fr = as<ForRangeStmt>(st)) {
      collect_stmt_nodes_walk_block(fr->body, out);
    }
  }
}

std::vector<StmtPtr> collect_stmt_nodes(const Block& b) {
  std::vector<StmtPtr> out;
  collect_stmt_nodes_walk_block(b, out);
  return out;
}

ExprPtr replace_expr_in_expr(const ExprPtr& e, const ExprPtr& target, const ExprPtr& repl) {
  if (!e) return e;
  if (e.get() == target.get()) {
    return repl;
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    return std::make_shared<UnaryExpr>(u->op, replace_expr_in_expr(u->e, target, repl));
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    return std::make_shared<BinaryExpr>(b->op,
                                        replace_expr_in_expr(b->a, target, repl),
                                        replace_expr_in_expr(b->b, target, repl));
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    return std::make_shared<IfExprNode>(replace_expr_in_expr(ife->cond, target, repl),
                                        replace_expr_in_expr(ife->then_e, target, repl),
                                        replace_expr_in_expr(ife->else_e, target, repl));
  }
  if (const auto* call = as<CallExpr>(e)) {
    std::vector<ExprPtr> args;
    args.reserve(call->args.size());
    for (const ExprPtr& arg : call->args) {
      args.push_back(replace_expr_in_expr(arg, target, repl));
    }
    return std::make_shared<CallExpr>(call->builtin, args);
  }
  return e;
}

StmtPtr replace_expr_in_stmt(const StmtPtr& st, const ExprPtr& target, const ExprPtr& repl);
Block replace_expr_in_block(const Block& b, const ExprPtr& target, const ExprPtr& repl);

StmtPtr replace_expr_in_stmt(const StmtPtr& st, const ExprPtr& target, const ExprPtr& repl) {
  if (const auto* assign = as<AssignStmt>(st)) {
    return std::make_shared<AssignStmt>(assign->name, replace_expr_in_expr(assign->e, target, repl));
  }
  if (const auto* ret = as<ReturnStmt>(st)) {
    return std::make_shared<ReturnStmt>(replace_expr_in_expr(ret->e, target, repl));
  }
  if (const auto* ifs = as<IfStmtNode>(st)) {
    return std::make_shared<IfStmtNode>(replace_expr_in_expr(ifs->cond, target, repl),
                                        replace_expr_in_block(ifs->then_block, target, repl),
                                        replace_expr_in_block(ifs->else_block, target, repl));
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    return std::make_shared<ForRangeStmt>(fr->var, fr->k, replace_expr_in_block(fr->body, target, repl));
  }
  return st;
}

Block replace_expr_in_block(const Block& b, const ExprPtr& target, const ExprPtr& repl) {
  Block out;
  out.stmts.reserve(b.stmts.size());
  for (const StmtPtr& st : b.stmts) {
    out.stmts.push_back(replace_expr_in_stmt(st, target, repl));
  }
  return out;
}

StmtPtr replace_stmt_in_stmt(const StmtPtr& st, const StmtPtr& target, const StmtPtr& repl);
Block replace_stmt_in_block(const Block& b, const StmtPtr& target, const StmtPtr& repl);

StmtPtr replace_stmt_in_stmt(const StmtPtr& st, const StmtPtr& target, const StmtPtr& repl) {
  if (st.get() == target.get()) {
    return repl;
  }
  if (const auto* ifs = as<IfStmtNode>(st)) {
    return std::make_shared<IfStmtNode>(ifs->cond,
                                        replace_stmt_in_block(ifs->then_block, target, repl),
                                        replace_stmt_in_block(ifs->else_block, target, repl));
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    return std::make_shared<ForRangeStmt>(fr->var, fr->k, replace_stmt_in_block(fr->body, target, repl));
  }
  return st;
}

Block replace_stmt_in_block(const Block& b, const StmtPtr& target, const StmtPtr& repl) {
  Block out;
  out.stmts.reserve(b.stmts.size());
  for (const StmtPtr& st : b.stmts) {
    out.stmts.push_back(replace_stmt_in_stmt(st, target, repl));
  }
  return out;
}

void append_value_repr(std::ostringstream& oss, const Value& v) {
  if (v.tag == ValueTag::None) {
    oss << "None";
    return;
  }
  if (v.tag == ValueTag::Bool) {
    oss << (v.b ? "True" : "False");
    return;
  }
  if (v.tag == ValueTag::Int) {
    oss << v.i;
    return;
  }
  std::ostringstream f;
  f << std::setprecision(17) << v.f;
  oss << f.str();
}

void append_expr_repr(std::ostringstream& oss, const ExprPtr& e);
void append_stmt_repr(std::ostringstream& oss, const StmtPtr& st);
void append_block_repr(std::ostringstream& oss, const Block& b);

void append_expr_repr(std::ostringstream& oss, const ExprPtr& e) {
  if (const auto* c = as<ConstExpr>(e)) {
    oss << "Const(";
    append_value_repr(oss, c->value);
    oss << ")";
    return;
  }
  if (const auto* v = as<VarExpr>(e)) {
    oss << "Var('" << v->name << "')";
    return;
  }
  if (const auto* u = as<UnaryExpr>(e)) {
    oss << "Unary(" << uop_name(u->op) << ",";
    append_expr_repr(oss, u->e);
    oss << ")";
    return;
  }
  if (const auto* b = as<BinaryExpr>(e)) {
    oss << "Binary(" << bop_name(b->op) << ",";
    append_expr_repr(oss, b->a);
    oss << ",";
    append_expr_repr(oss, b->b);
    oss << ")";
    return;
  }
  if (const auto* ife = as<IfExprNode>(e)) {
    oss << "IfExpr(";
    append_expr_repr(oss, ife->cond);
    oss << ",";
    append_expr_repr(oss, ife->then_e);
    oss << ",";
    append_expr_repr(oss, ife->else_e);
    oss << ")";
    return;
  }
  if (const auto* call = as<CallExpr>(e)) {
    oss << "Call('" << builtin_name(call->builtin) << "',[";
    for (std::size_t i = 0; i < call->args.size(); ++i) {
      if (i > 0) oss << ",";
      append_expr_repr(oss, call->args[i]);
    }
    oss << "])";
    return;
  }
  oss << "Expr(?)";
}

void append_stmt_repr(std::ostringstream& oss, const StmtPtr& st) {
  if (const auto* assign = as<AssignStmt>(st)) {
    oss << "Assign('" << assign->name << "',";
    append_expr_repr(oss, assign->e);
    oss << ")";
    return;
  }
  if (const auto* ret = as<ReturnStmt>(st)) {
    oss << "Return(";
    append_expr_repr(oss, ret->e);
    oss << ")";
    return;
  }
  if (const auto* ifs = as<IfStmtNode>(st)) {
    oss << "IfStmt(";
    append_expr_repr(oss, ifs->cond);
    oss << ",";
    append_block_repr(oss, ifs->then_block);
    oss << ",";
    append_block_repr(oss, ifs->else_block);
    oss << ")";
    return;
  }
  if (const auto* fr = as<ForRangeStmt>(st)) {
    oss << "ForRange('" << fr->var << "'," << fr->k << ",";
    append_block_repr(oss, fr->body);
    oss << ")";
    return;
  }
  oss << "Stmt(?)";
}

void append_block_repr(std::ostringstream& oss, const Block& b) {
  oss << "Block([";
  for (std::size_t i = 0; i < b.stmts.size(); ++i) {
    if (i > 0) oss << ",";
    append_stmt_repr(oss, b.stmts[i]);
  }
  oss << "])";
}

std::string canonical_serialize(const Block& b) {
  std::ostringstream oss;
  append_block_repr(oss, b);
  return oss.str();
}

std::string stable_hash16(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

GenomeMeta build_meta(const Block& ast) {
  const std::string canon = canonical_serialize(ast);
  GenomeMeta meta;
  meta.node_count = count_block_nodes(ast);
  meta.max_depth = max_block_expr_depth(ast);
  meta.uses_builtins = block_uses_builtins(ast);
  meta.hash_key = stable_hash16(canon);
  return meta;
}

ProgramGenome as_genome(const Block& ast) {
  ProgramGenome g;
  g.ast = ast;
  g.meta = build_meta(ast);
  return g;
}

class Compiler {
 public:
  explicit Compiler(const std::vector<std::string>* preset_locals = nullptr) {
    if (preset_locals != nullptr) {
      for (const std::string& name : *preset_locals) {
        local(name);
      }
    }
  }

  void compile_block(const Block& b) {
    for (const StmtPtr& st : b.stmts) {
      compile_stmt(st);
    }
  }

  BytecodeProgram finalize() {
    BytecodeProgram out;
    out.consts = consts_;
    out.code = code_;
    out.n_locals = static_cast<int>(var2idx_.size());
    out.var2idx = var2idx_;
    return out;
  }

 private:
  int add_const(const Value& v) {
    consts_.push_back(v);
    return static_cast<int>(consts_.size()) - 1;
  }

  int local(const std::string& name) {
    auto it = var2idx_.find(name);
    if (it != var2idx_.end()) {
      return it->second;
    }
    const int idx = static_cast<int>(var2idx_.size());
    var2idx_[name] = idx;
    return idx;
  }

  std::string new_label(const std::string& prefix) {
    return prefix + "_" + std::to_string(label_counter_++);
  }

  std::string new_temp() { return std::string("\x00for_i_") + std::to_string(tmp_counter_++); }

  void emit(const std::string& op, int a = 0, bool has_a = false, int b = 0, bool has_b = false) {
    code_.push_back(Instr{op, a, b, has_a, has_b});
  }

  void emit_jump(const std::string& op, const std::string& label) {
    emit(op, 0, true, 0, false);
    unresolved_.push_back({static_cast<int>(code_.size()) - 1, label});
  }

  void mark_label(const std::string& name) { labels_[name] = static_cast<int>(code_.size()); }

  void patch_jumps() {
    for (const auto& j : unresolved_) {
      auto it = labels_.find(j.label);
      if (it == labels_.end()) {
        throw std::runtime_error("undefined label");
      }
      code_[static_cast<std::size_t>(j.index)].a = it->second;
      code_[static_cast<std::size_t>(j.index)].has_a = true;
    }
  }

  void compile_expr(const ExprPtr& e) {
    if (const auto* c = as<ConstExpr>(e)) {
      emit("PUSH_CONST", add_const(c->value), true);
      return;
    }
    if (const auto* v = as<VarExpr>(e)) {
      emit("LOAD", local(v->name), true);
      return;
    }
    if (const auto* u = as<UnaryExpr>(e)) {
      compile_expr(u->e);
      emit(u->op == UOp::NEG ? "NEG" : "NOT");
      return;
    }
    if (const auto* b = as<BinaryExpr>(e)) {
      if (b->op == BOp::AND) {
        const std::string false_l = new_label("and_false");
        const std::string end_l = new_label("and_end");
        compile_expr(b->a);
        emit_jump("JMP_IF_FALSE", false_l);
        compile_expr(b->b);
        emit("NOT");
        emit("NOT");
        emit_jump("JMP", end_l);
        mark_label(false_l);
        emit("PUSH_CONST", add_const(Value::from_bool(false)), true);
        mark_label(end_l);
        return;
      }
      if (b->op == BOp::OR) {
        const std::string true_l = new_label("or_true");
        const std::string end_l = new_label("or_end");
        compile_expr(b->a);
        emit_jump("JMP_IF_TRUE", true_l);
        compile_expr(b->b);
        emit("NOT");
        emit("NOT");
        emit_jump("JMP", end_l);
        mark_label(true_l);
        emit("PUSH_CONST", add_const(Value::from_bool(true)), true);
        mark_label(end_l);
        return;
      }
      compile_expr(b->a);
      compile_expr(b->b);
      emit(bop_name(b->op));
      return;
    }
    if (const auto* ife = as<IfExprNode>(e)) {
      const std::string else_l = new_label("ifexpr_else");
      const std::string end_l = new_label("ifexpr_end");
      compile_expr(ife->cond);
      emit_jump("JMP_IF_FALSE", else_l);
      compile_expr(ife->then_e);
      emit_jump("JMP", end_l);
      mark_label(else_l);
      compile_expr(ife->else_e);
      mark_label(end_l);
      return;
    }
    if (const auto* call = as<CallExpr>(e)) {
      for (const ExprPtr& arg : call->args) {
        compile_expr(arg);
      }
      int bid = 0;
      if (call->builtin == Builtin::ABS) bid = 0;
      else if (call->builtin == Builtin::MIN) bid = 1;
      else if (call->builtin == Builtin::MAX) bid = 2;
      else bid = 3;
      emit("CALL_BUILTIN", bid, true, static_cast<int>(call->args.size()), true);
      return;
    }

    throw std::runtime_error("unknown Expr node");
  }

  void compile_stmt(const StmtPtr& st) {
    if (const auto* assign = as<AssignStmt>(st)) {
      compile_expr(assign->e);
      emit("STORE", local(assign->name), true);
      return;
    }
    if (const auto* ret = as<ReturnStmt>(st)) {
      compile_expr(ret->e);
      emit("RETURN");
      return;
    }
    if (const auto* ifs = as<IfStmtNode>(st)) {
      const std::string else_l = new_label("if_else");
      const std::string end_l = new_label("if_end");
      compile_expr(ifs->cond);
      emit_jump("JMP_IF_FALSE", else_l);
      compile_block(ifs->then_block);
      emit_jump("JMP", end_l);
      mark_label(else_l);
      compile_block(ifs->else_block);
      mark_label(end_l);
      return;
    }
    if (const auto* fr = as<ForRangeStmt>(st)) {
      if (fr->k < 0) {
        emit("PUSH_CONST", add_const(Value::from_bool(true)), true);
        emit("NEG");
        return;
      }

      const int idx_k = add_const(Value::from_int(fr->k));
      const int idx_0 = add_const(Value::from_int(0));
      const int idx_1 = add_const(Value::from_int(1));
      const int counter_i = local(new_temp());
      const int user_i = local(fr->var);

      const std::string loop_l = new_label("for_loop");
      const std::string end_l = new_label("for_end");

      emit("PUSH_CONST", idx_0, true);
      emit("STORE", counter_i, true);

      mark_label(loop_l);
      emit("LOAD", counter_i, true);
      emit("PUSH_CONST", idx_k, true);
      emit("LT");
      emit_jump("JMP_IF_FALSE", end_l);

      emit("LOAD", counter_i, true);
      emit("STORE", user_i, true);

      compile_block(fr->body);

      emit("LOAD", counter_i, true);
      emit("PUSH_CONST", idx_1, true);
      emit("ADD");
      emit("STORE", counter_i, true);
      emit_jump("JMP", loop_l);
      mark_label(end_l);
      return;
    }

    throw std::runtime_error("unknown Stmt node");
  }

  struct UnresolvedJump {
    int index = 0;
    std::string label;
  };

  std::vector<Value> consts_;
  std::vector<Instr> code_;
  std::vector<UnresolvedJump> unresolved_;
  std::unordered_map<std::string, int> labels_;
  std::unordered_map<std::string, int> var2idx_;
  int label_counter_ = 0;
  int tmp_counter_ = 0;

 public:
  BytecodeProgram build(const Block& b) {
    compile_block(b);
    patch_jumps();
    return finalize();
  }
};

}  // namespace

ProgramGenome make_random_genome(std::uint64_t seed, const Limits& limits) {
  std::mt19937_64 rng(seed);
  for (int i = 0; i < 128; ++i) {
    GenContext ctx = GenContext::empty();
    Block ast = gen_block(rng, ctx, limits.max_expr_depth, limits, true);
    ast = repair_limits(ast, rng, limits, &ctx);
    ProgramGenome g = as_genome(ast);
    if (validate_genome(g, limits).is_valid) {
      return g;
    }
  }
  return as_genome(Block{{std::make_shared<ReturnStmt>(std::make_shared<ConstExpr>(Value::from_int(0)))}});
}

ProgramGenome mutate(const ProgramGenome& genome, std::uint64_t seed, const Limits& limits) {
  std::mt19937_64 rng(seed);
  const int total = block_stmt_tree_size(genome.ast);
  if (total <= 0) {
    return make_random_genome(seed, limits);
  }

  GenContext ctx = infer_context_from_program(genome.ast);
  const int target = rand_int(rng, 0, total - 1);
  auto [mutated, _] = mutate_block_at_index(genome.ast, target, rng, ctx, limits);
  (void)_;

  const GenContext repaired_ctx = infer_context_from_program(mutated);
  const Block repaired = repair_limits(mutated, rng, limits, &repaired_ctx);
  ProgramGenome out = as_genome(repaired);
  if (validate_genome(out, limits).is_valid) {
    return out;
  }
  return make_random_genome(seed + 1, limits);
}

ProgramGenome crossover_top_level(const ProgramGenome& parent_a,
                                  const ProgramGenome& parent_b,
                                  std::uint64_t seed,
                                  const Limits& limits) {
  std::mt19937_64 rng(seed);
  const Block child = crossover_top_level_splice_block(parent_a.ast, parent_b.ast, rng, limits);
  const GenContext ctx = infer_context_from_program(child);
  const Block repaired = repair_limits(child, rng, limits, &ctx);
  ProgramGenome out = as_genome(repaired);
  if (validate_genome(out, limits).is_valid) {
    return out;
  }
  return make_random_genome(seed + 7, limits);
}

ProgramGenome crossover_typed_subtree(const ProgramGenome& parent_a,
                                      const ProgramGenome& parent_b,
                                      std::uint64_t seed,
                                      const Limits& limits) {
  std::mt19937_64 rng(seed);
  const std::vector<TypedExpr> exprs_a = collect_typed_expr_nodes(parent_a.ast);
  const std::vector<TypedExpr> exprs_b = collect_typed_expr_nodes(parent_b.ast);

  bool child_ready = false;
  Block child;

  std::set<RType> types_a;
  std::set<RType> types_b;
  for (const auto& pair : exprs_a) types_a.insert(pair.second);
  for (const auto& pair : exprs_b) types_b.insert(pair.second);

  std::vector<RType> common_types;
  for (RType t : types_a) {
    if (types_b.count(t) != 0U) {
      common_types.push_back(t);
    }
  }

  if (!common_types.empty()) {
    const RType chosen_type = choose_one(rng, common_types);
    std::vector<ExprPtr> pool_a;
    std::vector<ExprPtr> pool_b;
    for (const auto& pair : exprs_a) {
      if (pair.second == chosen_type) pool_a.push_back(pair.first);
    }
    for (const auto& pair : exprs_b) {
      if (pair.second == chosen_type) pool_b.push_back(pair.first);
    }
    if (!pool_a.empty() && !pool_b.empty()) {
      const ExprPtr target = choose_one(rng, pool_a);
      const ExprPtr donor = choose_one(rng, pool_b);
      child = replace_expr_in_block(parent_a.ast, target, donor);
      child_ready = true;
    }
  }

  if (!child_ready) {
    const std::vector<StmtPtr> stmts_a = collect_stmt_nodes(parent_a.ast);
    const std::vector<StmtPtr> stmts_b = collect_stmt_nodes(parent_b.ast);

    std::set<Stmt::Kind> kinds_a;
    std::set<Stmt::Kind> kinds_b;
    for (const StmtPtr& st : stmts_a) kinds_a.insert(st->kind);
    for (const StmtPtr& st : stmts_b) kinds_b.insert(st->kind);

    std::vector<Stmt::Kind> common_kinds;
    for (Stmt::Kind k : kinds_a) {
      if (kinds_b.count(k) != 0U) {
        common_kinds.push_back(k);
      }
    }

    if (!common_kinds.empty()) {
      const Stmt::Kind chosen_kind = choose_one(rng, common_kinds);
      std::vector<StmtPtr> pool_a;
      std::vector<StmtPtr> pool_b;
      for (const StmtPtr& st : stmts_a) {
        if (st->kind == chosen_kind) pool_a.push_back(st);
      }
      for (const StmtPtr& st : stmts_b) {
        if (st->kind == chosen_kind) pool_b.push_back(st);
      }
      if (!pool_a.empty() && !pool_b.empty()) {
        const StmtPtr target = choose_one(rng, pool_a);
        const StmtPtr donor = choose_one(rng, pool_b);
        child = replace_stmt_in_block(parent_a.ast, target, donor);
        child_ready = true;
      }
    }
  }

  if (!child_ready) {
    child = crossover_top_level_splice_block(parent_a.ast, parent_b.ast, rng, limits);
  }

  const GenContext ctx = infer_context_from_program(child);
  const Block repaired = repair_limits(child, rng, limits, &ctx);
  ProgramGenome out = as_genome(repaired);
  if (validate_genome(out, limits).is_valid) {
    return out;
  }
  return make_random_genome(seed + 17, limits);
}

ProgramGenome crossover(const ProgramGenome& parent_a,
                        const ProgramGenome& parent_b,
                        std::uint64_t seed,
                        CrossoverMethod method,
                        const Limits& limits) {
  if (method == CrossoverMethod::TopLevelSplice) {
    return crossover_top_level(parent_a, parent_b, seed, limits);
  }
  if (method == CrossoverMethod::TypedSubtree) {
    return crossover_typed_subtree(parent_a, parent_b, seed, limits);
  }
  if (method == CrossoverMethod::Hybrid) {
    std::mt19937_64 rng(seed);
    if (rand_prob(rng, 0.7)) {
      return crossover_typed_subtree(parent_a, parent_b, seed, limits);
    }
    return crossover_top_level(parent_a, parent_b, seed, limits);
  }
  throw std::invalid_argument("unknown crossover method");
}

ValidationResult validate_genome(const ProgramGenome& genome, const Limits& limits) {
  ValidationResult out;
  if (count_block_nodes(genome.ast) > limits.max_total_nodes) {
    out.errors.push_back("node count exceeds max_total_nodes");
  }
  if (max_block_expr_depth(genome.ast) > limits.max_expr_depth) {
    out.errors.push_back("expression depth exceeds max_expr_depth");
  }
  validate_block(genome.ast, limits, {}, out.errors, true);
  out.is_valid = out.errors.empty();
  return out;
}

BytecodeProgram compile_for_eval(const ProgramGenome& genome) {
  Compiler c;
  return c.build(genome.ast);
}

BytecodeProgram compile_for_eval_with_preset_locals(const ProgramGenome& genome,
                                                    const std::vector<std::string>& preset_locals) {
  Compiler c(&preset_locals);
  return c.build(genome.ast);
}

std::string ast_to_string(const Block& block) { return canonical_serialize(block); }

}  // namespace g3pvm::evo
