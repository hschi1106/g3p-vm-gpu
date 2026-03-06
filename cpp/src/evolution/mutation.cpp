#include "g3pvm/evolution/mutation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "genome_meta.hpp"
#include "subtree_utils.hpp"
#include "typed_expr_analysis.hpp"

namespace g3pvm::evo {

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

AstProgram typed_subtree_mutation(const AstProgram& ast, std::mt19937_64& rng, const Limits& limits) {
  AstProgram mutated;
  const std::vector<std::size_t> end = subtree::build_subtree_end(ast);
  const std::vector<typed_expr::TypedExprRoot> expr_roots = typed_expr::collect_typed_expr_roots(ast, end);
  if (expr_roots.empty()) {
    return mutated;
  }

  const typed_expr::TypedExprRoot target = choose_one(rng, expr_roots);
  AstProgram donor;
  donor.version = "ast-prefix-v1";
  donor.nodes = subtree::make_random_expr_nodes_for_type(
      rng, donor, target.type, std::max(1, limits.max_expr_depth / 2));
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

ProgramGenome mutate(const ProgramGenome& genome,
                     std::uint64_t seed,
                     const Limits& limits,
                     double mutation_subtree_prob) {
  std::mt19937_64 rng(seed);
  if (genome.ast.nodes.empty()) {
    return make_random_genome(seed, limits);
  }

  AstProgram mutated;
  const double subtree_prob = std::clamp(mutation_subtree_prob, 0.0, 1.0);
  if (std::bernoulli_distribution(subtree_prob)(rng)) {
    mutated = typed_subtree_mutation(genome.ast, rng, limits);
  }
  if (mutated.nodes.empty()) {
    mutated = constant_perturbation(genome.ast, rng);
  }
  if (mutated.nodes.empty()) {
    return genome;
  }

  ProgramGenome out;
  out.ast = std::move(mutated);
  out.meta = genome_meta::build_meta_fast(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) {
    return genome;
  }
  for (const AstNode& node : out.ast.nodes) {
    if (node.kind == NodeKind::FOR_RANGE && (node.i1 < 0 || node.i1 > limits.max_for_k)) {
      return genome;
    }
  }
  return out;
}

}  // namespace g3pvm::evo
