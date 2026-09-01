#include "g3pvm/evolution/crossover.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

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
  out.meta = build_genome_meta(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) {
    return fallback_parent;
  }
  if (out.meta.max_depth > limits.max_expr_depth) {
    return fallback_parent;
  }
  return out;
}

}  // namespace

std::pair<ProgramGenome, ProgramGenome> crossover_impl(
    const ProgramGenome& parent_a,
    const VerifiedAst* verified_a,
    const ProgramGenome& parent_b,
    const VerifiedAst* verified_b,
    std::uint64_t seed,
    const Limits& limits) {
  std::mt19937_64 rng(seed);
  std::vector<std::size_t> rebuilt_end_a;
  std::vector<std::size_t> rebuilt_end_b;
  const auto annotation_matches = [](const ProgramGenome& parent,
                                     const VerifiedAst* verified) {
    if (verified == nullptr) return true;
    const std::size_t size = parent.ast.nodes.size();
    return verified->subtree_end.size() == size &&
           verified->expression_types.size() == size &&
           verified->expression_scope_signatures.size() == size &&
           verified->expression_binder_signatures.size() == size &&
           (size == 0 || verified->subtree_end[0] == size);
  };
  if (!annotation_matches(parent_a, verified_a) ||
      !annotation_matches(parent_b, verified_b)) {
    throw std::invalid_argument("crossover VerifiedAst does not match parent shape");
  }
  if (verified_a == nullptr) rebuilt_end_a = subtree::build_subtree_end(parent_a.ast);
  if (verified_b == nullptr) rebuilt_end_b = subtree::build_subtree_end(parent_b.ast);
  const std::vector<std::size_t>& end_a =
      verified_a != nullptr ? verified_a->subtree_end : rebuilt_end_a;
  const std::vector<std::size_t>& end_b =
      verified_b != nullptr ? verified_b->subtree_end : rebuilt_end_b;

  const std::vector<typed_expr::TypedExprRoot> all_expr_a =
      verified_a != nullptr
          ? typed_expr::collect_typed_expr_roots(parent_a.ast, *verified_a)
          : typed_expr::collect_typed_expr_roots(parent_a.ast, end_a);
  const std::vector<typed_expr::TypedExprRoot> all_expr_b =
      verified_b != nullptr
          ? typed_expr::collect_typed_expr_roots(parent_b.ast, *verified_b)
          : typed_expr::collect_typed_expr_roots(parent_b.ast, end_b);
  std::vector<typed_expr::TypedExprRoot> expr_a;
  std::vector<typed_expr::TypedExprRoot> expr_b;
  expr_a.reserve(all_expr_a.size());
  expr_b.reserve(all_expr_b.size());
  for (const typed_expr::TypedExprRoot& root : all_expr_a) {
    if (!typed_expr::is_asgp_phase_body_root(parent_a.ast, end_a, root)) {
      expr_a.push_back(root);
    }
  }
  for (const typed_expr::TypedExprRoot& root : all_expr_b) {
    if (!typed_expr::is_asgp_phase_body_root(parent_b.ast, end_b, root)) {
      expr_b.push_back(root);
    }
  }

  if (expr_a.empty() || expr_b.empty()) {
    return {parent_a, parent_b};
  }

  std::vector<typed_expr::TypedExprRoot> compatible_a;
  compatible_a.reserve(expr_a.size());
  for (const typed_expr::TypedExprRoot& root_a : expr_a) {
    const bool in_b = std::any_of(
        expr_b.begin(), expr_b.end(), [&](const typed_expr::TypedExprRoot& root_b) {
          return typed_expr::typed_subtree_keys_compatible(root_a, root_b);
        });
    if (in_b) {
      compatible_a.push_back(root_a);
    }
  }

  if (compatible_a.empty()) {
    return {parent_a, parent_b};
  }

  const typed_expr::TypedExprRoot& chosen_a = choose_one(rng, compatible_a);
  std::vector<typed_expr::TypedExprRoot> roots_a;
  std::vector<typed_expr::TypedExprRoot> roots_b;
  roots_a.reserve(expr_a.size());
  roots_b.reserve(expr_b.size());
  for (const typed_expr::TypedExprRoot& root : expr_a) {
    if (typed_expr::typed_subtree_keys_compatible(chosen_a, root)) {
      roots_a.push_back(root);
    }
  }
  for (const typed_expr::TypedExprRoot& root : expr_b) {
    if (typed_expr::typed_subtree_keys_compatible(chosen_a, root)) {
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

std::pair<ProgramGenome, ProgramGenome> crossover(const ProgramGenome& parent_a,
                                                  const ProgramGenome& parent_b,
                                                  std::uint64_t seed,
                                                  const Limits& limits) {
  return crossover_impl(parent_a, nullptr, parent_b, nullptr, seed, limits);
}

std::pair<ProgramGenome, ProgramGenome> crossover(const ProgramGenome& parent_a,
                                                  const VerifiedAst& verified_a,
                                                  const ProgramGenome& parent_b,
                                                  const VerifiedAst& verified_b,
                                                  std::uint64_t seed,
                                                  const Limits& limits) {
  return crossover_impl(parent_a, &verified_a, parent_b, &verified_b, seed, limits);
}

}  // namespace g3pvm::evo
