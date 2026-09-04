#include "gagp/evolution/mutation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "gagp/evolution/genome_generation.hpp"
#include "subtree_utils.hpp"
#include "typed_expr_analysis.hpp"

namespace gagp::evo {

namespace {

template <typename T>
const T& choose_one(std::mt19937_64& rng, const std::vector<T>& values) {
  if (values.empty()) {
    throw std::runtime_error("choose_one on empty vector");
  }
  std::uniform_int_distribution<int> dist(0, static_cast<int>(values.size()) - 1);
  const int idx = dist(rng);
  return values[static_cast<std::size_t>(idx)];
}

AstProgram typed_subtree_mutation(const AstProgram& ast,
                                  const VerifiedAst* verified,
                                  std::mt19937_64& rng,
                                  const Limits& limits,
                                  const GrammarConfig& grammar) {
  AstProgram mutated;
  std::vector<std::size_t> rebuilt_end;
  const std::vector<std::size_t>* end = nullptr;
  std::vector<typed_expr::TypedExprRoot> all_roots;
  if (verified != nullptr) {
    if (verified->subtree_end.size() != ast.nodes.size() ||
        verified->expression_types.size() != ast.nodes.size() ||
        verified->expression_scope_signatures.size() != ast.nodes.size() ||
        verified->expression_binder_signatures.size() != ast.nodes.size() ||
        (ast.nodes.size() > 0 && verified->subtree_end[0] != ast.nodes.size())) {
      throw std::invalid_argument("mutation VerifiedAst does not match program shape");
    }
    end = &verified->subtree_end;
    all_roots = typed_expr::collect_typed_expr_roots(ast, *verified);
  } else {
    rebuilt_end = subtree::build_subtree_end(ast);
    end = &rebuilt_end;
    all_roots = typed_expr::collect_typed_expr_roots(ast, *end);
  }
  std::vector<typed_expr::TypedExprRoot> expr_roots;
  expr_roots.reserve(all_roots.size());
  for (const typed_expr::TypedExprRoot& root : all_roots) {
    if (!typed_expr::is_asgp_phase_body_root(ast, *end, root)) {
      expr_roots.push_back(root);
    }
  }
  if (expr_roots.empty()) {
    return mutated;
  }

  const typed_expr::TypedExprRoot target = choose_one(rng, expr_roots);
  AstProgram donor;
  donor.version = k_ast_prefix_version_current;
  const bool allow_asgp_donor = typed_expr::is_statement_value_root(ast, target);
  donor.nodes = subtree::make_random_expr_nodes_for_type(
      rng, donor, target.type, std::max(1, limits.max_expr_depth / 2), grammar, allow_asgp_donor);
  return subtree::replace_subtree(ast, target.start, target.stop, donor, 0, donor.nodes.size());
}

AstProgram constant_perturbation(const AstProgram& ast, std::mt19937_64& rng) {
  AstProgram mutated;
  std::vector<std::size_t> const_nodes;
  for (std::size_t i = 0; i < ast.nodes.size(); ++i) {
    if (ast.nodes[i].kind == NodeKind::CONST) {
      const_nodes.push_back(i);
    }
  }
  if (const_nodes.empty()) {
    return mutated;
  }

  mutated = ast;
  const std::size_t node_index = choose_one(rng, const_nodes);
  const int const_index = mutated.nodes[node_index].i0;
  if (const_index < 0 || static_cast<std::size_t>(const_index) >= mutated.consts.size()) {
    mutated.nodes.clear();
    return mutated;
  }

  Value value = mutated.consts[static_cast<std::size_t>(const_index)];
  if (value.tag == ValueTag::Int) {
    value.i += std::uniform_int_distribution<int>(-2, 2)(rng);
  } else if (value.tag == ValueTag::Float) {
    value.f += std::uniform_real_distribution<double>(-1.0, 1.0)(rng);
  } else if (value.tag == ValueTag::Bool) {
    value.b = !value.b;
  }

  mutated.consts.push_back(value);
  mutated.nodes[node_index].i0 = static_cast<int>(mutated.consts.size() - 1);
  return mutated;
}

}  // namespace

ProgramGenome mutate_impl(const ProgramGenome& genome,
                          const VerifiedAst* verified,
                          std::uint64_t seed,
                          const Limits& limits,
                          double mutation_subtree_prob,
                          const GrammarConfig& grammar) {
  grammar.validate();
  std::mt19937_64 rng(seed);
  if (genome.ast.nodes.empty()) {
    return generate_random_genome(seed, limits, grammar);
  }

  AstProgram mutated;
  const double subtree_prob = std::clamp(mutation_subtree_prob, 0.0, 1.0);
  if (std::bernoulli_distribution(subtree_prob)(rng)) {
    mutated = typed_subtree_mutation(genome.ast, verified, rng, limits, grammar);
  }
  if (mutated.nodes.empty()) {
    mutated = constant_perturbation(genome.ast, rng);
  }
  if (mutated.nodes.empty()) {
    return genome;
  }

  ProgramGenome out;
  out.ast = std::move(mutated);
  out.meta = build_genome_meta(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) {
    return genome;
  }
  if (out.meta.max_depth > limits.max_expr_depth) {
    return genome;
  }
  return out;
}

ProgramGenome mutate(const ProgramGenome& genome,
                     std::uint64_t seed,
                     const Limits& limits,
                     double mutation_subtree_prob,
                     const GrammarConfig& grammar) {
  return mutate_impl(genome, nullptr, seed, limits, mutation_subtree_prob, grammar);
}

ProgramGenome mutate(const ProgramGenome& genome,
                     const VerifiedAst& verified,
                     std::uint64_t seed,
                     const Limits& limits,
                     double mutation_subtree_prob,
                     const GrammarConfig& grammar) {
  return mutate_impl(genome, &verified, seed, limits, mutation_subtree_prob, grammar);
}

}  // namespace gagp::evo
