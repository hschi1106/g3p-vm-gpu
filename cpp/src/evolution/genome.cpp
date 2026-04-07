#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

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
         kind == NodeKind::CALL_CONCAT || kind == NodeKind::CALL_INDEX || kind == NodeKind::CALL_APPEND ||
         kind == NodeKind::CALL_FIND || kind == NodeKind::CALL_CONTAINS;
}

bool is_expr_kind(NodeKind kind) {
  return kind == NodeKind::CONST || kind == NodeKind::VAR || kind == NodeKind::NEG || kind == NodeKind::NOT ||
         kind == NodeKind::IF_EXPR || is_binary_expr(kind) || kind == NodeKind::CALL_ABS ||
         kind == NodeKind::CALL_CLIP || kind == NodeKind::CALL_LEN || kind == NodeKind::CALL_SLICE ||
         kind == NodeKind::CALL_REVERSE;
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
  if (node.kind == NodeKind::CONST || node.kind == NodeKind::VAR) {
    return {idx + 1, 1};
  }
  if (node.kind == NodeKind::NEG || node.kind == NodeKind::NOT || node.kind == NodeKind::CALL_ABS ||
      node.kind == NodeKind::CALL_LEN || node.kind == NodeKind::CALL_REVERSE) {
    const DepthResult child = compute_expr_depth_prefix(program, idx + 1);
    return {child.next, 1 + child.max_expr_depth};
  }
  if (is_binary_expr(node.kind)) {
    const DepthResult lhs = compute_expr_depth_prefix(program, idx + 1);
    const DepthResult rhs = compute_expr_depth_prefix(program, lhs.next);
    return {rhs.next, 1 + std::max(lhs.max_expr_depth, rhs.max_expr_depth)};
  }
  if (node.kind == NodeKind::IF_EXPR || node.kind == NodeKind::CALL_CLIP || node.kind == NodeKind::CALL_SLICE) {
    const DepthResult first = compute_expr_depth_prefix(program, idx + 1);
    const DepthResult second = compute_expr_depth_prefix(program, first.next);
    const DepthResult third = compute_expr_depth_prefix(program, second.next);
    return {third.next, 1 + std::max(first.max_expr_depth,
                                     std::max(second.max_expr_depth, third.max_expr_depth))};
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
    if (node.kind == NodeKind::CALL_ABS || node.kind == NodeKind::CALL_MIN || node.kind == NodeKind::CALL_MAX ||
        node.kind == NodeKind::CALL_CLIP || node.kind == NodeKind::CALL_LEN || node.kind == NodeKind::CALL_CONCAT ||
        node.kind == NodeKind::CALL_SLICE || node.kind == NodeKind::CALL_INDEX || node.kind == NodeKind::CALL_APPEND ||
        node.kind == NodeKind::CALL_REVERSE || node.kind == NodeKind::CALL_FIND || node.kind == NodeKind::CALL_CONTAINS) {
      meta.uses_builtins = true;
      break;
    }
  }
  meta.program_key = ast_cache_key(ast);
  return meta;
}

}  // namespace g3pvm::evo
