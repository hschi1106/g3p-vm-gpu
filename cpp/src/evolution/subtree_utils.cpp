#include "subtree_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace g3pvm::evo::subtree {

namespace {

int rand_int(std::mt19937_64& rng, int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(rng);
}

double rand_real(std::mt19937_64& rng, double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(rng);
}

bool rand_prob(std::mt19937_64& rng, double p) {
  std::bernoulli_distribution dist(std::max(0.0, std::min(1.0, p)));
  return dist(rng);
}

bool value_equal(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::None) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int) return a.i == b.i;
  if (a.tag == ValueTag::Float) return a.f == b.f;
  if (a.tag == ValueTag::String || a.tag == ValueTag::List) return a.i == b.i;
  return false;
}

std::size_t fill_subtree_end_at(const AstProgram& program, std::size_t idx, std::vector<std::size_t>& out) {
  if (idx >= program.nodes.size()) throw std::runtime_error("prefix traversal out of range");
  std::size_t cur = idx + 1;
  for (int i = 0; i < node_arity(program.nodes[idx].kind); ++i) {
    cur = fill_subtree_end_at(program, cur, out);
  }
  out[idx] = cur;
  return cur;
}

int find_or_add_name(AstProgram& target, const std::string& name, std::unordered_map<std::string, int>& name2idx) {
  auto it = name2idx.find(name);
  if (it != name2idx.end()) return it->second;
  const int idx = static_cast<int>(target.names.size());
  target.names.push_back(name);
  name2idx[name] = idx;
  return idx;
}

int find_or_add_const(AstProgram& target, const Value& value) {
  for (std::size_t i = 0; i < target.consts.size(); ++i) {
    if (value_equal(target.consts[i], value)) return static_cast<int>(i);
  }
  const int idx = static_cast<int>(target.consts.size());
  target.consts.push_back(value);
  return idx;
}

std::vector<AstNode> map_subtree_nodes_into(AstProgram& target,
                                            const AstProgram& donor,
                                            std::size_t start,
                                            std::size_t stop) {
  std::unordered_map<std::string, int> name2idx;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    name2idx[target.names[i]] = static_cast<int>(i);
  }
  std::unordered_map<int, int> name_map;
  std::unordered_map<int, int> const_map;
  std::vector<AstNode> out;
  out.reserve(stop - start);
  for (std::size_t i = start; i < stop; ++i) {
    AstNode node = donor.nodes[i];
    if (node.kind == NodeKind::CONST) {
      auto it = const_map.find(node.i0);
      if (it == const_map.end()) {
        const int mapped = find_or_add_const(target, donor.consts.at(static_cast<std::size_t>(node.i0)));
        const_map[node.i0] = mapped;
        node.i0 = mapped;
      } else {
        node.i0 = it->second;
      }
    } else if (node.kind == NodeKind::VAR || node.kind == NodeKind::ASSIGN || node.kind == NodeKind::FOR_RANGE) {
      auto it = name_map.find(node.i0);
      if (it == name_map.end()) {
        const int mapped = find_or_add_name(target, donor.names.at(static_cast<std::size_t>(node.i0)), name2idx);
        name_map[node.i0] = mapped;
        node.i0 = mapped;
      } else {
        node.i0 = it->second;
      }
    }
    out.push_back(node);
  }
  return out;
}

}  // namespace

int node_arity(NodeKind kind) {
  switch (kind) {
    case NodeKind::PROGRAM: return 1;
    case NodeKind::BLOCK_NIL: return 0;
    case NodeKind::BLOCK_CONS: return 2;
    case NodeKind::ASSIGN: return 1;
    case NodeKind::IF_STMT: return 3;
    case NodeKind::FOR_RANGE: return 1;
    case NodeKind::RETURN: return 1;
    case NodeKind::CONST: return 0;
    case NodeKind::VAR: return 0;
    case NodeKind::NEG:
    case NodeKind::NOT: return 1;
    case NodeKind::ADD:
    case NodeKind::SUB:
    case NodeKind::MUL:
    case NodeKind::DIV:
    case NodeKind::MOD:
    case NodeKind::LT:
    case NodeKind::LE:
    case NodeKind::GT:
    case NodeKind::GE:
    case NodeKind::EQ:
    case NodeKind::NE:
    case NodeKind::AND:
    case NodeKind::OR: return 2;
    case NodeKind::IF_EXPR: return 3;
    case NodeKind::CALL_ABS: return 1;
    case NodeKind::CALL_MIN:
    case NodeKind::CALL_MAX: return 2;
    case NodeKind::CALL_CLIP: return 3;
    case NodeKind::CALL_LEN: return 1;
    case NodeKind::CALL_CONCAT: return 2;
    case NodeKind::CALL_SLICE: return 3;
    case NodeKind::CALL_INDEX: return 2;
  }
  return 0;
}

std::vector<std::size_t> build_subtree_end(const AstProgram& program) {
  std::vector<std::size_t> out(program.nodes.size(), 0);
  if (!program.nodes.empty()) {
    const std::size_t end = fill_subtree_end_at(program, 0, out);
    if (end != program.nodes.size()) throw std::runtime_error("prefix trailing tokens");
  }
  return out;
}

std::vector<AstNode> make_random_expr_nodes_for_type(std::mt19937_64& rng,
                                                     AstProgram& target,
                                                     RType type,
                                                     int depth) {
  (void)depth;
  AstProgram donor;
  int x_idx = -1;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    if (target.names[i] == "x") {
      x_idx = static_cast<int>(i);
      break;
    }
  }
  if (x_idx >= 0) donor.names.push_back("x");
  if (type == RType::Bool && x_idx >= 0 && rand_prob(rng, 0.6)) {
    donor.nodes.push_back(AstNode{NodeKind::LT, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    donor.consts.push_back(Value::from_int(rand_int(rng, -8, 8)));
    donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if (type == RType::Num && x_idx >= 0 && rand_prob(rng, 0.6)) {
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  Value value = Value::from_int(0);
  if (type == RType::Bool) value = Value::from_bool(rand_prob(rng, 0.5));
  else if (type == RType::NoneType) value = Value::none();
  else if (type == RType::Container) {
    const std::uint32_t n = static_cast<std::uint32_t>(rand_int(rng, 0, 8));
    const std::uint64_t h = static_cast<std::uint64_t>(rng());
    value = rand_prob(rng, 0.5) ? Value::from_string_hash_len(h, n) : Value::from_list_hash_len(h, n);
  } else if (rand_prob(rng, 0.5)) {
    value = Value::from_int(rand_int(rng, -8, 8));
  } else {
    value = Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0);
  }
  donor.consts.push_back(value);
  donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
  return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
}

AstProgram replace_subtree(const AstProgram& base,
                           std::size_t target_start,
                           std::size_t target_stop,
                           const AstProgram& donor,
                           std::size_t donor_start,
                           std::size_t donor_stop) {
  AstProgram out;
  out.version = "ast-prefix-v1";
  out.names = base.names;
  out.consts = base.consts;
  std::vector<AstNode> donor_nodes = map_subtree_nodes_into(out, donor, donor_start, donor_stop);
  out.nodes.reserve(base.nodes.size() - (target_stop - target_start) + donor_nodes.size());
  out.nodes.insert(out.nodes.end(), base.nodes.begin(), base.nodes.begin() + static_cast<std::ptrdiff_t>(target_start));
  out.nodes.insert(out.nodes.end(), donor_nodes.begin(), donor_nodes.end());
  out.nodes.insert(out.nodes.end(), base.nodes.begin() + static_cast<std::ptrdiff_t>(target_stop), base.nodes.end());
  return out;
}

}  // namespace g3pvm::evo::subtree
