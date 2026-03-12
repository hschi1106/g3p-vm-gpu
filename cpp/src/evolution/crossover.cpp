#include "g3pvm/evolution/crossover.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "genome_meta.hpp"
#include "subtree_utils.hpp"
#include "typed_expr_analysis.hpp"

namespace g3pvm::evo {

namespace {

template <typename T>
const T& choose_one(std::mt19937_64& rng, const std::vector<T>& values) {
  const std::size_t last = values.size() - 1;
  const std::size_t idx = std::uniform_int_distribution<std::size_t>(0, last)(rng);
  return values[idx];
}

ProgramGenome build_valid_child(const AstProgram& candidate,
                                const ProgramGenome& fallback_parent,
                                const Limits& limits) {
  if (candidate.nodes.empty()) {
    return fallback_parent;
  }

  ProgramGenome out;
  out.ast = candidate;
  out.meta = genome_meta::build_meta_fast(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) {
    return fallback_parent;
  }
  for (const AstNode& node : out.ast.nodes) {
    if (node.kind == NodeKind::FOR_RANGE && (node.i1 < 0 || node.i1 > limits.max_for_k)) {
      return fallback_parent;
    }
  }
  return out;
}

}  // namespace

std::pair<ProgramGenome, ProgramGenome> crossover(const ProgramGenome& parent_a,
                                                  const ProgramGenome& parent_b,
                                                  std::uint64_t seed,
                                                  const Limits& limits) {
  std::mt19937_64 rng(seed);
  const std::vector<std::size_t> end_a = subtree::build_subtree_end(parent_a.ast);
  const std::vector<std::size_t> end_b = subtree::build_subtree_end(parent_b.ast);

  const std::vector<typed_expr::TypedExprRoot> expr_a =
      typed_expr::collect_typed_expr_roots(parent_a.ast, end_a);
  const std::vector<typed_expr::TypedExprRoot> expr_b =
      typed_expr::collect_typed_expr_roots(parent_b.ast, end_b);

  if (expr_a.empty() || expr_b.empty()) {
    return {parent_a, parent_b};
  }

  std::vector<RType> common_types;
  common_types.reserve(expr_a.size());
  for (const typed_expr::TypedExprRoot& root_a : expr_a) {
    const bool in_b = std::any_of(
        expr_b.begin(), expr_b.end(), [&](const typed_expr::TypedExprRoot& root_b) {
          return root_b.type == root_a.type;
        });
    const bool already_seen = std::find(common_types.begin(), common_types.end(), root_a.type) != common_types.end();
    if (in_b && !already_seen) {
      common_types.push_back(root_a.type);
    }
  }

  if (common_types.empty()) {
    return {parent_a, parent_b};
  }

  const RType chosen_type = choose_one(rng, common_types);
  std::vector<typed_expr::TypedExprRoot> roots_a;
  std::vector<typed_expr::TypedExprRoot> roots_b;
  roots_a.reserve(expr_a.size());
  roots_b.reserve(expr_b.size());
  for (const typed_expr::TypedExprRoot& root : expr_a) {
    if (root.type == chosen_type) {
      roots_a.push_back(root);
    }
  }
  for (const typed_expr::TypedExprRoot& root : expr_b) {
    if (root.type == chosen_type) {
      roots_b.push_back(root);
    }
  }

  if (roots_a.empty() || roots_b.empty()) {
    return {parent_a, parent_b};
  }

  const typed_expr::TypedExprRoot& target_a = choose_one(rng, roots_a);
  const typed_expr::TypedExprRoot& target_b = choose_one(rng, roots_b);

  const AstProgram child_a_ast = subtree::replace_subtree(
      parent_a.ast, target_a.start, target_a.stop, parent_b.ast, target_b.start, target_b.stop);
  const AstProgram child_b_ast = subtree::replace_subtree(
      parent_b.ast, target_b.start, target_b.stop, parent_a.ast, target_a.start, target_a.stop);

  return {build_valid_child(child_a_ast, parent_a, limits),
          build_valid_child(child_b_ast, parent_b, limits)};
}

}  // namespace g3pvm::evo
