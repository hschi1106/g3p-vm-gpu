#include "g3pvm/evolution/crossover.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "genome_meta.hpp"
#include "subtree_utils.hpp"
#include "typed_expr_analysis.hpp"

namespace g3pvm::evo {

ProgramGenome crossover(const ProgramGenome& parent_a,
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

  AstProgram child;
  if (!expr_a.empty() && !expr_b.empty()) {
    const int lhs_idx = std::uniform_int_distribution<int>(0, static_cast<int>(expr_a.size()) - 1)(rng);
    const typed_expr::TypedExprRoot target = expr_a[static_cast<std::size_t>(lhs_idx)];

    std::vector<typed_expr::TypedExprRoot> compatible;
    compatible.reserve(expr_b.size());
    for (const typed_expr::TypedExprRoot& candidate : expr_b) {
      if (candidate.type == target.type) {
        compatible.push_back(candidate);
      }
    }

    if (!compatible.empty()) {
      const int rhs_idx = std::uniform_int_distribution<int>(0, static_cast<int>(compatible.size()) - 1)(rng);
      const typed_expr::TypedExprRoot donor = compatible[static_cast<std::size_t>(rhs_idx)];
      child = subtree::replace_subtree(
          parent_a.ast, target.start, target.stop, parent_b.ast, donor.start, donor.stop);
    }
  }

  if (child.nodes.empty()) {
    return parent_a;
  }

  ProgramGenome out;
  out.ast = std::move(child);
  out.meta = genome_meta::build_meta_fast(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) {
    return parent_a;
  }
  for (const AstNode& node : out.ast.nodes) {
    if (node.kind == NodeKind::FOR_RANGE && (node.i1 < 0 || node.i1 > limits.max_for_k)) {
      return parent_a;
    }
  }
  return out;
}

}  // namespace g3pvm::evo
