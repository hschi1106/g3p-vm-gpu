#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo::subtree {

int node_arity(NodeKind kind);
std::vector<std::size_t> build_subtree_end(const AstProgram& program);
std::vector<AstNode> make_random_expr_nodes_for_type(std::mt19937_64& rng,
                                                     AstProgram& target,
                                                     RType type,
                                                     int depth);
AstProgram replace_subtree(const AstProgram& base,
                           std::size_t target_start,
                           std::size_t target_stop,
                           const AstProgram& donor,
                           std::size_t donor_start,
                           std::size_t donor_stop);

}  // namespace g3pvm::evo::subtree
