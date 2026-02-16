#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/bytecode.hpp"

namespace g3pvm::evo {

enum class UOp {
  NEG,
  NOT,
};

enum class BOp {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  LT,
  LE,
  GT,
  GE,
  EQ,
  NE,
  AND,
  OR,
};

enum class Builtin {
  ABS,
  MIN,
  MAX,
  CLIP,
};

enum class RType {
  Num,
  Bool,
  NoneType,
  Any,
  Invalid,
};

struct Expr;
struct Stmt;

using ExprPtr = std::shared_ptr<Expr>;
using StmtPtr = std::shared_ptr<Stmt>;

struct Block {
  std::vector<StmtPtr> stmts;
};

struct Expr {
  enum class Kind {
    Const,
    Var,
    Unary,
    Binary,
    IfExpr,
    Call,
  };

  explicit Expr(Kind kind) : kind(kind) {}
  virtual ~Expr() = default;

  Kind kind;
};

struct Stmt {
  enum class Kind {
    Assign,
    IfStmt,
    ForRange,
    Return,
  };

  explicit Stmt(Kind kind) : kind(kind) {}
  virtual ~Stmt() = default;

  Kind kind;
};

struct ConstExpr final : Expr {
  explicit ConstExpr(Value value) : Expr(Kind::Const), value(value) {}
  Value value;
};

struct VarExpr final : Expr {
  explicit VarExpr(std::string name) : Expr(Kind::Var), name(std::move(name)) {}
  std::string name;
};

struct UnaryExpr final : Expr {
  UnaryExpr(UOp op, ExprPtr e) : Expr(Kind::Unary), op(op), e(std::move(e)) {}
  UOp op;
  ExprPtr e;
};

struct BinaryExpr final : Expr {
  BinaryExpr(BOp op, ExprPtr a, ExprPtr b) : Expr(Kind::Binary), op(op), a(std::move(a)), b(std::move(b)) {}
  BOp op;
  ExprPtr a;
  ExprPtr b;
};

struct IfExprNode final : Expr {
  IfExprNode(ExprPtr cond, ExprPtr then_e, ExprPtr else_e)
      : Expr(Kind::IfExpr), cond(std::move(cond)), then_e(std::move(then_e)), else_e(std::move(else_e)) {}
  ExprPtr cond;
  ExprPtr then_e;
  ExprPtr else_e;
};

struct CallExpr final : Expr {
  CallExpr(Builtin builtin, std::vector<ExprPtr> args)
      : Expr(Kind::Call), builtin(builtin), args(std::move(args)) {}
  Builtin builtin;
  std::vector<ExprPtr> args;
};

struct AssignStmt final : Stmt {
  AssignStmt(std::string name, ExprPtr e) : Stmt(Kind::Assign), name(std::move(name)), e(std::move(e)) {}
  std::string name;
  ExprPtr e;
};

struct IfStmtNode final : Stmt {
  IfStmtNode(ExprPtr cond, Block then_block, Block else_block)
      : Stmt(Kind::IfStmt), cond(std::move(cond)), then_block(std::move(then_block)), else_block(std::move(else_block)) {}
  ExprPtr cond;
  Block then_block;
  Block else_block;
};

struct ForRangeStmt final : Stmt {
  ForRangeStmt(std::string var, int k, Block body)
      : Stmt(Kind::ForRange), var(std::move(var)), k(k), body(std::move(body)) {}
  std::string var;
  int k;
  Block body;
};

struct ReturnStmt final : Stmt {
  explicit ReturnStmt(ExprPtr e) : Stmt(Kind::Return), e(std::move(e)) {}
  ExprPtr e;
};

struct Limits {
  int max_expr_depth = 5;
  int max_stmts_per_block = 6;
  int max_total_nodes = 80;
  int max_for_k = 16;
  int max_call_args = 3;
};

struct GenomeMeta {
  int node_count = 0;
  int max_depth = 0;
  bool uses_builtins = false;
  std::string hash_key;
};

struct ProgramGenome {
  Block ast;
  GenomeMeta meta;
};

struct ValidationResult {
  bool is_valid = false;
  std::vector<std::string> errors;
};

enum class CrossoverMethod {
  TopLevelSplice,
  TypedSubtree,
  Hybrid,
};

ProgramGenome make_random_genome(std::uint64_t seed, const Limits& limits = Limits{});
ProgramGenome mutate(const ProgramGenome& genome, std::uint64_t seed, const Limits& limits = Limits{});
ProgramGenome crossover_top_level(const ProgramGenome& parent_a,
                                  const ProgramGenome& parent_b,
                                  std::uint64_t seed,
                                  const Limits& limits = Limits{});
ProgramGenome crossover_typed_subtree(const ProgramGenome& parent_a,
                                      const ProgramGenome& parent_b,
                                      std::uint64_t seed,
                                      const Limits& limits = Limits{});
ProgramGenome crossover(const ProgramGenome& parent_a,
                        const ProgramGenome& parent_b,
                        std::uint64_t seed,
                        CrossoverMethod method = CrossoverMethod::TopLevelSplice,
                        const Limits& limits = Limits{});
ValidationResult validate_genome(const ProgramGenome& genome, const Limits& limits = Limits{});
BytecodeProgram compile_for_eval(const ProgramGenome& genome);

std::string ast_to_string(const Block& block);

}  // namespace g3pvm::evo
