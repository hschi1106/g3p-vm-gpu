#include "gagp/evolution/genome.hpp"

#include <algorithm>

#include "gagp/evolution/node_descriptor.hpp"
#include "subtree_utils.hpp"

namespace gagp::evo {

namespace {

struct DepthResult {
  std::size_t next = 0;
  int max_expr_depth = 0;
};

bool is_binary_expr(NodeKind kind) {
  return kind == NodeKind::ADD || kind == NodeKind::SUB || kind == NodeKind::MUL || kind == NodeKind::DIV ||
         kind == NodeKind::MOD || kind == NodeKind::LT || kind == NodeKind::LE || kind == NodeKind::GT ||
         kind == NodeKind::GE || kind == NodeKind::EQ || kind == NodeKind::NE || kind == NodeKind::AND ||
         kind == NodeKind::OR || kind == NodeKind::CALL_MIN || kind == NodeKind::CALL_MAX ||
         kind == NodeKind::CALL_IDIV0 || kind == NodeKind::CALL_IMOD0 ||
         kind == NodeKind::CALL_CONCAT || kind == NodeKind::CALL_INDEX || kind == NodeKind::CALL_APPEND ||
         kind == NodeKind::CALL_PREPEND || kind == NodeKind::CALL_FIND || kind == NodeKind::CALL_CONTAINS;
}

bool is_expr_kind(NodeKind kind) {
  return kind == NodeKind::CONST || kind == NodeKind::VAR || kind == NodeKind::NEG || kind == NodeKind::NOT ||
         kind == NodeKind::IF_EXPR || is_binary_expr(kind) || kind == NodeKind::CALL_ABS ||
         kind == NodeKind::CALL_CLIP || kind == NodeKind::CALL_LEN || kind == NodeKind::CALL_SLICE ||
         kind == NodeKind::CALL_REVERSE || kind == NodeKind::CALL_CHAR_TO_STRING ||
         kind == NodeKind::CALL_STRING_TO_CHAR || kind == NodeKind::CALL_ORD || kind == NodeKind::CALL_CHR ||
         kind == NodeKind::CALL_IS_LETTER || kind == NodeKind::CALL_IS_DIGIT || kind == NodeKind::CALL_IS_SPACE ||
         kind == NodeKind::CALL_IS_VOWEL || kind == NodeKind::CALL_TO_LOWER || kind == NodeKind::CALL_TO_UPPER ||
         kind == NodeKind::CALL_TO_STRING || kind == NodeKind::CALL_SINGLETON ||
         kind == NodeKind::BOUND_VAR || kind == NodeKind::MAP_LIST ||
         kind == NodeKind::FILTER_LIST || kind == NodeKind::LINEAR_REC || kind == NodeKind::ASGP_DC ||
         kind == NodeKind::ASGP_DP1D || kind == NodeKind::ASGP_DP2D;
}

DepthResult compute_expr_depth_prefix(const AstProgram& program, std::size_t idx);
DepthResult compute_block_depth_prefix(const AstProgram& program, std::size_t idx);

DepthResult compute_stmt_depth_prefix(const AstProgram& program, std::size_t idx) {
  if (idx >= program.nodes.size()) return {idx, 0};
  const AstNode& node = program.nodes[idx];
  if (node.kind == NodeKind::ASSIGN || node.kind == NodeKind::RETURN) {
    const DepthResult expr = compute_expr_depth_prefix(program, idx + 1);
    return {expr.next, expr.max_expr_depth};
  }
  if (node.kind == NodeKind::IF_STMT) {
    const DepthResult cond = compute_expr_depth_prefix(program, idx + 1);
    const DepthResult then_block = compute_block_depth_prefix(program, cond.next);
    const DepthResult else_block = compute_block_depth_prefix(program, then_block.next);
    return {else_block.next, std::max(cond.max_expr_depth,
                                      std::max(then_block.max_expr_depth, else_block.max_expr_depth))};
  }
  if (node.kind == NodeKind::FOR_RANGE) {
    const DepthResult bound = compute_expr_depth_prefix(program, idx + 1);
    const DepthResult body = compute_block_depth_prefix(program, bound.next);
    return {body.next, std::max(bound.max_expr_depth, body.max_expr_depth)};
  }
  return {idx + 1, 0};
}

DepthResult compute_block_depth_prefix(const AstProgram& program, std::size_t idx) {
  if (idx >= program.nodes.size()) return {idx, 0};
  const AstNode& node = program.nodes[idx];
  if (node.kind == NodeKind::BLOCK_NIL) {
    return {idx + 1, 0};
  }
  if (node.kind != NodeKind::BLOCK_CONS) {
    return {idx, 0};
  }
  const DepthResult stmt = compute_stmt_depth_prefix(program, idx + 1);
  const DepthResult rest = compute_block_depth_prefix(program, stmt.next);
  return {rest.next, std::max(stmt.max_expr_depth, rest.max_expr_depth)};
}

DepthResult compute_expr_depth_prefix(const AstProgram& program, std::size_t idx) {
  if (idx >= program.nodes.size()) return {idx, 0};
  const AstNode& node = program.nodes[idx];
  if (is_expr_kind(node.kind)) {
    const int arity = subtree::node_arity(node.kind);
    std::size_t cur = idx + 1;
    int child_max = 0;
    for (int i = 0; i < arity; ++i) {
      const DepthResult child = compute_expr_depth_prefix(program, cur);
      cur = child.next;
      child_max = std::max(child_max, child.max_expr_depth);
    }
    return {cur, 1 + child_max};
  }
  return {idx + 1, 0};
}

int compute_max_expr_depth(const AstProgram& program) {
  if (program.nodes.empty()) {
    return 0;
  }
  if (program.nodes.front().kind != NodeKind::PROGRAM) {
    return 0;
  }
  const DepthResult body = compute_block_depth_prefix(program, 1);
  return body.max_expr_depth;
}

}  // namespace

GenomeMeta build_genome_meta(const AstProgram& ast) {
  GenomeMeta meta;
  meta.node_count = static_cast<int>(ast.nodes.size());
  meta.max_depth = compute_max_expr_depth(ast);
  meta.uses_builtins = false;
  for (const AstNode& node : ast.nodes) {
    if (node_descriptor(node.kind).is_builtin()) {
      meta.uses_builtins = true;
      break;
    }
  }
  meta.program_key = ast_cache_key(ast);
  return meta;
}

}  // namespace gagp::evo
