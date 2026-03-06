#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/core/bytecode.hpp"

namespace g3pvm::evo {

enum class RType {
  Num,
  Bool,
  NoneType,
  Container,
  Any,
  Invalid,
};

enum class NodeKind {
  PROGRAM,
  BLOCK_NIL,
  BLOCK_CONS,
  ASSIGN,
  IF_STMT,
  FOR_RANGE,
  RETURN,
  CONST,
  VAR,
  NEG,
  NOT,
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
  IF_EXPR,
  CALL_ABS,
  CALL_MIN,
  CALL_MAX,
  CALL_CLIP,
  CALL_LEN,
  CALL_CONCAT,
  CALL_SLICE,
  CALL_INDEX,
};

struct AstNode {
  NodeKind kind = NodeKind::PROGRAM;
  int i0 = 0;
  int i1 = 0;
};

struct AstProgram {
  std::vector<AstNode> nodes;
  std::vector<std::string> names;
  std::vector<Value> consts;
  std::string version = "ast-prefix-v1";
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
  AstProgram ast;
  GenomeMeta meta;
};

ProgramGenome make_random_genome(std::uint64_t seed, const Limits& limits = Limits{});
BytecodeProgram compile_for_eval(const ProgramGenome& genome);
BytecodeProgram compile_for_eval_with_preset_locals(const ProgramGenome& genome,
                                                    const std::vector<std::string>& preset_locals);

std::string ast_to_string(const AstProgram& program);

}  // namespace g3pvm::evo
