#include "subtree_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/runtime/payload/payload.hpp"

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
  if (a.tag == ValueTag::Int || a.tag == ValueTag::FallbackToken) return a.i == b.i;
  if (a.tag == ValueTag::Float) return a.f == b.f;
  if (a.tag == ValueTag::String || a.tag == ValueTag::NumList || a.tag == ValueTag::StringList) return a.i == b.i;
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

RType infer_name_type_from_string(const std::string& name) {
  if (name == "strings" || name == "words" || name == "string_list" ||
      (name.size() >= 8 && name.compare(name.size() - 8, 8, "_strings") == 0) ||
      (name.size() >= 6 && name.compare(name.size() - 6, 6, "_words") == 0)) {
    return RType::StringList;
  }
  if (name == "xs" || name == "items" || name == "list" ||
      (name.size() >= 5 && name.compare(name.size() - 5, 5, "_list") == 0)) {
    return RType::NumList;
  }
  if (name == "s" || name == "str" || name == "text" ||
      (name.size() >= 7 && name.compare(name.size() - 7, 7, "_string") == 0)) {
    return RType::String;
  }
  return RType::Num;
}

std::vector<int> collect_name_ids_for_type(const AstProgram& target, RType type) {
  std::vector<int> out;
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    const RType inferred = infer_name_type_from_string(target.names[i]);
    if (type == RType::Any || inferred == type) {
      out.push_back(static_cast<int>(i));
    }
  }
  return out;
}

int choose_name_id(std::mt19937_64& rng, const std::vector<int>& ids) {
  if (ids.empty()) {
    return -1;
  }
  return ids[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(ids.size()) - 1))];
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
    case NodeKind::FOR_RANGE: return 2;
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
    case NodeKind::CALL_APPEND: return 2;
    case NodeKind::CALL_REVERSE: return 1;
    case NodeKind::CALL_FIND: return 2;
    case NodeKind::CALL_CONTAINS: return 2;
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
  const std::vector<int> num_name_ids = collect_name_ids_for_type(target, RType::Num);
  const std::vector<int> string_name_ids = collect_name_ids_for_type(target, RType::String);
  const std::vector<int> num_list_name_ids = collect_name_ids_for_type(target, RType::NumList);
  const std::vector<int> string_list_name_ids = collect_name_ids_for_type(target, RType::StringList);
  const std::vector<int> container_name_ids = [&]() {
    std::vector<int> out = string_name_ids;
    out.insert(out.end(), num_list_name_ids.begin(), num_list_name_ids.end());
    out.insert(out.end(), string_list_name_ids.begin(), string_list_name_ids.end());
    return out;
  }();
  const int num_name_id = choose_name_id(rng, num_name_ids);
  const int container_name_id = choose_name_id(rng, container_name_ids);
  if (type == RType::Bool && num_name_id >= 0 && rand_prob(rng, 0.6)) {
    donor.names.push_back(target.names[static_cast<std::size_t>(num_name_id)]);
    donor.nodes.push_back(AstNode{NodeKind::LT, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    donor.consts.push_back(Value::from_int(rand_int(rng, -8, 8)));
    donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if (type == RType::Bool && container_name_id >= 0 && rand_prob(rng, 0.4)) {
    donor.names.push_back(target.names[static_cast<std::size_t>(container_name_id)]);
    donor.nodes.push_back(AstNode{NodeKind::LT, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    donor.consts.push_back(Value::from_int(rand_int(rng, -8, 8)));
    donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if (type == RType::Num && num_name_id >= 0 && rand_prob(rng, 0.45)) {
    donor.names.push_back(target.names[static_cast<std::size_t>(num_name_id)]);
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if (type == RType::Num && container_name_id >= 0 && rand_prob(rng, 0.55)) {
    const RType container_type = infer_name_type_from_string(target.names[static_cast<std::size_t>(container_name_id)]);
    donor.names.push_back(target.names[static_cast<std::size_t>(container_name_id)]);
    if (container_type != RType::NumList || rand_prob(rng, 0.5)) {
      donor.nodes.push_back(AstNode{NodeKind::CALL_LEN, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    } else {
      donor.nodes.push_back(AstNode{NodeKind::CALL_INDEX, 0, 0});
      donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
      donor.consts.push_back(Value::from_int(rand_int(rng, -6, 6)));
      donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    }
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  Value value = Value::from_int(0);
  if (type == RType::Bool) value = Value::from_bool(rand_prob(rng, 0.5));
  else if (type == RType::NoneType) value = Value::none();
  else if (type == RType::String) {
    static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz";
    const int len = rand_int(rng, 0, 8);
    std::string s;
    s.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      s.push_back(kAlphabet[rand_int(rng, 0, 25)]);
    }
    value = g3pvm::payload::make_string_value(s);
  } else if (type == RType::NumList) {
    const int len = rand_int(rng, 0, 4);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      if (rand_prob(rng, 0.65)) elems.push_back(Value::from_int(rand_int(rng, -8, 8)));
      else elems.push_back(Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0));
    }
    value = g3pvm::payload::make_num_list_value(elems);
  } else if (type == RType::StringList) {
    static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz";
    const int len = rand_int(rng, 0, 4);
    std::vector<Value> elems;
    elems.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
      const int s_len = rand_int(rng, 0, 5);
      std::string s;
      s.reserve(static_cast<std::size_t>(s_len));
      for (int j = 0; j < s_len; ++j) s.push_back(kAlphabet[rand_int(rng, 0, 25)]);
      elems.push_back(g3pvm::payload::make_string_value(s));
    }
    value = g3pvm::payload::make_string_list_value(elems);
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
