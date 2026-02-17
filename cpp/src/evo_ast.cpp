#include "g3pvm/evo_ast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <utility>

namespace g3pvm::evo {

namespace {

const char* op_name(NodeKind op) {
  switch (op) {
    case NodeKind::ADD:
      return "ADD";
    case NodeKind::SUB:
      return "SUB";
    case NodeKind::MUL:
      return "MUL";
    case NodeKind::DIV:
      return "DIV";
    case NodeKind::MOD:
      return "MOD";
    case NodeKind::LT:
      return "LT";
    case NodeKind::LE:
      return "LE";
    case NodeKind::GT:
      return "GT";
    case NodeKind::GE:
      return "GE";
    case NodeKind::EQ:
      return "EQ";
    case NodeKind::NE:
      return "NE";
    case NodeKind::AND:
      return "AND";
    case NodeKind::OR:
      return "OR";
    case NodeKind::NEG:
      return "NEG";
    case NodeKind::NOT:
      return "NOT";
    }
  return "ADD";
}

int node_arity(NodeKind kind) {
  switch (kind) {
    case NodeKind::PROGRAM:
      return 1;
    case NodeKind::BLOCK_NIL:
      return 0;
    case NodeKind::BLOCK_CONS:
      return 2;
    case NodeKind::ASSIGN:
      return 1;
    case NodeKind::IF_STMT:
      return 3;
    case NodeKind::FOR_RANGE:
      return 1;
    case NodeKind::RETURN:
      return 1;
    case NodeKind::CONST:
      return 0;
    case NodeKind::VAR:
      return 0;
    case NodeKind::NEG:
    case NodeKind::NOT:
      return 1;
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
    case NodeKind::OR:
      return 2;
    case NodeKind::IF_EXPR:
      return 3;
    case NodeKind::CALL_ABS:
      return 1;
    case NodeKind::CALL_MIN:
    case NodeKind::CALL_MAX:
      return 2;
    case NodeKind::CALL_CLIP:
      return 3;
  }
  return 0;
}

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

template <typename T>
const T& choose_one(std::mt19937_64& rng, const std::vector<T>& v) {
  if (v.empty()) {
    throw std::runtime_error("choose_one on empty vector");
  }
  return v[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(v.size()) - 1))];
}

void hash_u8(std::uint64_t& h, std::uint8_t x) {
  h ^= static_cast<std::uint64_t>(x);
  h *= 1099511628211ULL;
}

void hash_bytes(std::uint64_t& h, const void* ptr, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(ptr);
  for (std::size_t i = 0; i < n; ++i) {
    hash_u8(h, p[i]);
  }
}

void hash_i32(std::uint64_t& h, std::int32_t x) { hash_bytes(h, &x, sizeof(x)); }
void hash_u32(std::uint64_t& h, std::uint32_t x) { hash_bytes(h, &x, sizeof(x)); }

std::string canonical_prefix_serialize(const AstProgram& p) {
  std::ostringstream oss;
  oss << "AstPrefix(";
  for (std::size_t i = 0; i < p.nodes.size(); ++i) {
    if (i > 0) oss << ",";
    const AstNode& n = p.nodes[i];
    oss << static_cast<int>(n.kind) << ":" << n.i0 << ":" << n.i1;
  }
  oss << ")";
  return oss.str();
}

GenomeMeta build_meta_fast(const AstProgram& ast) {
  GenomeMeta meta;
  meta.node_count = static_cast<int>(ast.nodes.size());
  meta.max_depth = 0;
  meta.uses_builtins = false;
  for (const AstNode& n : ast.nodes) {
    if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN || n.kind == NodeKind::CALL_MAX ||
        n.kind == NodeKind::CALL_CLIP) {
      meta.uses_builtins = true;
      break;
    }
  }
  std::uint64_t h = 1469598103934665603ULL;
  hash_u32(h, static_cast<std::uint32_t>(ast.nodes.size()));
  hash_u32(h, static_cast<std::uint32_t>(ast.names.size()));
  hash_u32(h, static_cast<std::uint32_t>(ast.consts.size()));
  const std::size_t sample = ast.nodes.size() <= 32 ? 1 : (ast.nodes.size() / 32);
  for (std::size_t i = 0; i < ast.nodes.size(); i += sample) {
    const AstNode& n = ast.nodes[i];
    hash_u32(h, static_cast<std::uint32_t>(n.kind));
    hash_i32(h, static_cast<std::int32_t>(n.i0));
    hash_i32(h, static_cast<std::int32_t>(n.i1));
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  meta.hash_key = oss.str();
  return meta;
}

class Compiler {
 public:
  explicit Compiler(const std::vector<std::string>* preset_locals = nullptr) {
    if (preset_locals != nullptr) {
      for (const std::string& name : *preset_locals) {
        local(name);
      }
    }
  }

  BytecodeProgram finalize() {
    BytecodeProgram out;
    out.consts = consts_;
    out.code = code_;
    out.n_locals = static_cast<int>(var2idx_.size());
    out.var2idx = var2idx_;
    return out;
  }

 private:
  int add_const(const Value& v) {
    consts_.push_back(v);
    return static_cast<int>(consts_.size()) - 1;
  }

  int local(const std::string& name) {
    auto it = var2idx_.find(name);
    if (it != var2idx_.end()) {
      return it->second;
    }
    const int idx = static_cast<int>(var2idx_.size());
    var2idx_[name] = idx;
    return idx;
  }

  std::string new_label(const std::string& prefix) {
    return prefix + "_" + std::to_string(label_counter_++);
  }

  std::string new_temp() { return std::string("\x00for_i_") + std::to_string(tmp_counter_++); }

  void emit(const std::string& op, int a = 0, bool has_a = false, int b = 0, bool has_b = false) {
    code_.push_back(Instr{op, a, b, has_a, has_b});
  }

  void emit_jump(const std::string& op, const std::string& label) {
    emit(op, 0, true, 0, false);
    unresolved_.push_back({static_cast<int>(code_.size()) - 1, label});
  }

  void mark_label(const std::string& name) { labels_[name] = static_cast<int>(code_.size()); }

  void patch_jumps() {
    for (const auto& j : unresolved_) {
      auto it = labels_.find(j.label);
      if (it == labels_.end()) {
        throw std::runtime_error("undefined label");
      }
      code_[static_cast<std::size_t>(j.index)].a = it->second;
      code_[static_cast<std::size_t>(j.index)].has_a = true;
    }
  }

 public:
  const AstNode& node_at(const AstProgram& p, std::size_t idx) const {
    if (idx >= p.nodes.size()) {
      throw std::runtime_error("prefix compile: node index out of range");
    }
    return p.nodes[idx];
  }

  const std::string& name_at(const AstProgram& p, int idx) const {
    if (idx < 0 || static_cast<std::size_t>(idx) >= p.names.size()) {
      throw std::runtime_error("prefix compile: name index out of range");
    }
    return p.names[static_cast<std::size_t>(idx)];
  }

  const Value& const_at(const AstProgram& p, int idx) const {
    if (idx < 0 || static_cast<std::size_t>(idx) >= p.consts.size()) {
      throw std::runtime_error("prefix compile: const index out of range");
    }
    return p.consts[static_cast<std::size_t>(idx)];
  }

  std::size_t compile_expr_prefix(const AstProgram& p, std::size_t idx) {
    const AstNode& n = node_at(p, idx);
    switch (n.kind) {
      case NodeKind::CONST:
        emit("PUSH_CONST", add_const(const_at(p, n.i0)), true);
        return idx + 1;
      case NodeKind::VAR:
        emit("LOAD", local(name_at(p, n.i0)), true);
        return idx + 1;
      case NodeKind::NEG:
      case NodeKind::NOT: {
        std::size_t j = compile_expr_prefix(p, idx + 1);
        emit(n.kind == NodeKind::NEG ? "NEG" : "NOT");
        return j;
      }
      case NodeKind::AND: {
        const std::string false_l = new_label("and_false");
        const std::string end_l = new_label("and_end");
        std::size_t j = compile_expr_prefix(p, idx + 1);
        emit_jump("JMP_IF_FALSE", false_l);
        j = compile_expr_prefix(p, j);
        emit("NOT");
        emit("NOT");
        emit_jump("JMP", end_l);
        mark_label(false_l);
        emit("PUSH_CONST", add_const(Value::from_bool(false)), true);
        mark_label(end_l);
        return j;
      }
      case NodeKind::OR: {
        const std::string true_l = new_label("or_true");
        const std::string end_l = new_label("or_end");
        std::size_t j = compile_expr_prefix(p, idx + 1);
        emit_jump("JMP_IF_TRUE", true_l);
        j = compile_expr_prefix(p, j);
        emit("NOT");
        emit("NOT");
        emit_jump("JMP", end_l);
        mark_label(true_l);
        emit("PUSH_CONST", add_const(Value::from_bool(true)), true);
        mark_label(end_l);
        return j;
      }
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
      case NodeKind::NE: {
        std::size_t j = compile_expr_prefix(p, idx + 1);
        j = compile_expr_prefix(p, j);
        emit(op_name(n.kind));
        return j;
      }
      case NodeKind::IF_EXPR: {
        const std::string else_l = new_label("ifexpr_else");
        const std::string end_l = new_label("ifexpr_end");
        std::size_t j = compile_expr_prefix(p, idx + 1);
        emit_jump("JMP_IF_FALSE", else_l);
        j = compile_expr_prefix(p, j);
        emit_jump("JMP", end_l);
        mark_label(else_l);
        j = compile_expr_prefix(p, j);
        mark_label(end_l);
        return j;
      }
      case NodeKind::CALL_ABS:
      case NodeKind::CALL_MIN:
      case NodeKind::CALL_MAX:
      case NodeKind::CALL_CLIP: {
        std::size_t j = idx + 1;
        const int argc = node_arity(n.kind);
        for (int i = 0; i < argc; ++i) {
          j = compile_expr_prefix(p, j);
        }
        int bid = 0;
        if (n.kind == NodeKind::CALL_ABS) bid = 0;
        else if (n.kind == NodeKind::CALL_MIN) bid = 1;
        else if (n.kind == NodeKind::CALL_MAX) bid = 2;
        else bid = 3;
        emit("CALL_BUILTIN", bid, true, argc, true);
        return j;
      }
      default:
        throw std::runtime_error("prefix compile: expected expr node");
    }
  }

  std::size_t compile_block_prefix(const AstProgram& p, std::size_t idx) {
    const AstNode& n = node_at(p, idx);
    if (n.kind == NodeKind::BLOCK_NIL) {
      return idx + 1;
    }
    if (n.kind != NodeKind::BLOCK_CONS) {
      throw std::runtime_error("prefix compile: expected block node");
    }
    std::size_t j = compile_stmt_prefix(p, idx + 1);
    return compile_block_prefix(p, j);
  }

  std::size_t compile_stmt_prefix(const AstProgram& p, std::size_t idx) {
    const AstNode& n = node_at(p, idx);
    if (n.kind == NodeKind::ASSIGN) {
      std::size_t j = compile_expr_prefix(p, idx + 1);
      emit("STORE", local(name_at(p, n.i0)), true);
      return j;
    }
    if (n.kind == NodeKind::RETURN) {
      std::size_t j = compile_expr_prefix(p, idx + 1);
      emit("RETURN");
      return j;
    }
    if (n.kind == NodeKind::IF_STMT) {
      const std::string else_l = new_label("if_else");
      const std::string end_l = new_label("if_end");
      std::size_t j = compile_expr_prefix(p, idx + 1);
      emit_jump("JMP_IF_FALSE", else_l);
      j = compile_block_prefix(p, j);
      emit_jump("JMP", end_l);
      mark_label(else_l);
      j = compile_block_prefix(p, j);
      mark_label(end_l);
      return j;
    }
    if (n.kind == NodeKind::FOR_RANGE) {
      if (n.i1 < 0) {
        emit("PUSH_CONST", add_const(Value::from_bool(true)), true);
        emit("NEG");
        return compile_block_prefix(p, idx + 1);
      }

      const int idx_k = add_const(Value::from_int(n.i1));
      const int idx_0 = add_const(Value::from_int(0));
      const int idx_1 = add_const(Value::from_int(1));
      const int counter_i = local(new_temp());
      const int user_i = local(name_at(p, n.i0));

      const std::string loop_l = new_label("for_loop");
      const std::string end_l = new_label("for_end");

      emit("PUSH_CONST", idx_0, true);
      emit("STORE", counter_i, true);

      mark_label(loop_l);
      emit("LOAD", counter_i, true);
      emit("PUSH_CONST", idx_k, true);
      emit("LT");
      emit_jump("JMP_IF_FALSE", end_l);

      emit("LOAD", counter_i, true);
      emit("STORE", user_i, true);

      std::size_t j = compile_block_prefix(p, idx + 1);

      emit("LOAD", counter_i, true);
      emit("PUSH_CONST", idx_1, true);
      emit("ADD");
      emit("STORE", counter_i, true);
      emit_jump("JMP", loop_l);
      mark_label(end_l);
      return j;
    }
    throw std::runtime_error("prefix compile: expected stmt node");
  }

  struct UnresolvedJump {
    int index = 0;
    std::string label;
  };

  std::vector<Value> consts_;
  std::vector<Instr> code_;
  std::vector<UnresolvedJump> unresolved_;
  std::unordered_map<std::string, int> labels_;
  std::unordered_map<std::string, int> var2idx_;
  int label_counter_ = 0;
  int tmp_counter_ = 0;

 public:
  BytecodeProgram build(const AstProgram& p) {
    if (p.version != "ast-prefix-v1") {
      throw std::runtime_error("unsupported ast prefix version");
    }
    if (p.nodes.empty() || p.nodes[0].kind != NodeKind::PROGRAM) {
      throw std::runtime_error("prefix compile: bad root");
    }
    std::size_t end = compile_block_prefix(p, 1);
    if (end != p.nodes.size()) {
      throw std::runtime_error("prefix compile: trailing tokens");
    }
    patch_jumps();
    return finalize();
  }
};

bool is_stmt_kind(NodeKind kind) {
  return kind == NodeKind::ASSIGN || kind == NodeKind::IF_STMT || kind == NodeKind::FOR_RANGE || kind == NodeKind::RETURN;
}

bool is_expr_kind(NodeKind kind) {
  return kind == NodeKind::CONST || kind == NodeKind::VAR || kind == NodeKind::NEG || kind == NodeKind::NOT ||
         kind == NodeKind::ADD || kind == NodeKind::SUB || kind == NodeKind::MUL || kind == NodeKind::DIV ||
         kind == NodeKind::MOD || kind == NodeKind::LT || kind == NodeKind::LE || kind == NodeKind::GT ||
         kind == NodeKind::GE || kind == NodeKind::EQ || kind == NodeKind::NE || kind == NodeKind::AND ||
         kind == NodeKind::OR || kind == NodeKind::IF_EXPR || kind == NodeKind::CALL_ABS || kind == NodeKind::CALL_MIN ||
         kind == NodeKind::CALL_MAX || kind == NodeKind::CALL_CLIP;
}

bool value_equal(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::None) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int) return a.i == b.i;
  return a.f == b.f;
}

std::size_t fill_subtree_end_at(const AstProgram& p, std::size_t idx, std::vector<std::size_t>& out) {
  if (idx >= p.nodes.size()) throw std::runtime_error("prefix traversal out of range");
  std::size_t cur = idx + 1;
  for (int i = 0; i < node_arity(p.nodes[idx].kind); ++i) {
    cur = fill_subtree_end_at(p, cur, out);
  }
  out[idx] = cur;
  return cur;
}

std::vector<std::size_t> build_subtree_end(const AstProgram& p) {
  std::vector<std::size_t> out(p.nodes.size(), 0);
  if (!p.nodes.empty()) {
    const std::size_t end = fill_subtree_end_at(p, 0, out);
    if (end != p.nodes.size()) throw std::runtime_error("prefix trailing tokens");
  }
  return out;
}

void collect_roots_walk(const AstProgram& p,
                        const std::vector<std::size_t>& end,
                        std::size_t idx,
                        std::vector<std::size_t>& stmt_roots,
                        std::vector<std::size_t>& expr_roots) {
  const NodeKind kind = p.nodes[idx].kind;
  if (is_stmt_kind(kind)) stmt_roots.push_back(idx);
  if (is_expr_kind(kind)) expr_roots.push_back(idx);
  std::size_t child = idx + 1;
  for (int i = 0; i < node_arity(kind); ++i) {
    collect_roots_walk(p, end, child, stmt_roots, expr_roots);
    child = end[child];
  }
}

std::vector<std::size_t> collect_top_level_stmt_roots(const AstProgram& p, const std::vector<std::size_t>& end) {
  std::vector<std::size_t> out;
  if (p.nodes.empty() || p.nodes[0].kind != NodeKind::PROGRAM) return out;
  std::size_t block = 1;
  while (block < p.nodes.size() && p.nodes[block].kind == NodeKind::BLOCK_CONS) {
    const std::size_t stmt = block + 1;
    out.push_back(stmt);
    block = end[stmt];
  }
  return out;
}

int find_or_add_name(AstProgram& target, const std::string& name, std::unordered_map<std::string, int>& name2idx) {
  auto it = name2idx.find(name);
  if (it != name2idx.end()) return it->second;
  const int idx = static_cast<int>(target.names.size());
  target.names.push_back(name);
  name2idx[name] = idx;
  return idx;
}

int find_or_add_const(AstProgram& target, const Value& v) {
  for (std::size_t i = 0; i < target.consts.size(); ++i) {
    if (value_equal(target.consts[i], v)) return static_cast<int>(i);
  }
  const int idx = static_cast<int>(target.consts.size());
  target.consts.push_back(v);
  return idx;
}

std::vector<AstNode> map_subtree_nodes_into(AstProgram& target,
                                            const AstProgram& donor,
                                            std::size_t start,
                                            std::size_t stop) {
  std::unordered_map<std::string, int> name2idx;
  name2idx.reserve(target.names.size() * 2 + 8);
  for (std::size_t i = 0; i < target.names.size(); ++i) {
    name2idx[target.names[i]] = static_cast<int>(i);
  }
  std::unordered_map<int, int> name_map;
  std::unordered_map<int, int> const_map;
  std::vector<AstNode> out;
  out.reserve(stop - start);
  for (std::size_t i = start; i < stop; ++i) {
    AstNode n = donor.nodes[i];
    if (n.kind == NodeKind::CONST) {
      auto it = const_map.find(n.i0);
      if (it == const_map.end()) {
        const int mapped = find_or_add_const(target, donor.consts.at(static_cast<std::size_t>(n.i0)));
        const_map[n.i0] = mapped;
        n.i0 = mapped;
      } else {
        n.i0 = it->second;
      }
    } else if (n.kind == NodeKind::VAR || n.kind == NodeKind::ASSIGN || n.kind == NodeKind::FOR_RANGE) {
      auto it = name_map.find(n.i0);
      if (it == name_map.end()) {
        const int mapped = find_or_add_name(target, donor.names.at(static_cast<std::size_t>(n.i0)), name2idx);
        name_map[n.i0] = mapped;
        n.i0 = mapped;
      } else {
        n.i0 = it->second;
      }
    }
    out.push_back(n);
  }
  return out;
}

std::vector<AstNode> make_random_expr_nodes_for_type(std::mt19937_64& rng, AstProgram& target, RType t, int depth) {
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
  if (t == RType::Bool && x_idx >= 0 && rand_prob(rng, 0.6)) {
    donor.nodes.push_back(AstNode{NodeKind::LT, 0, 0});
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    donor.consts.push_back(Value::from_int(rand_int(rng, -8, 8)));
    donor.nodes.push_back(AstNode{NodeKind::CONST, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  if (t == RType::Num && x_idx >= 0 && rand_prob(rng, 0.6)) {
    donor.nodes.push_back(AstNode{NodeKind::VAR, 0, 0});
    return map_subtree_nodes_into(target, donor, 0, donor.nodes.size());
  }
  Value v = Value::from_int(0);
  if (t == RType::Bool) v = Value::from_bool(rand_prob(rng, 0.5));
  else if (t == RType::NoneType) v = Value::none();
  else if (rand_prob(rng, 0.5)) v = Value::from_int(rand_int(rng, -8, 8));
  else v = Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0);
  donor.consts.push_back(v);
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

AstProgram top_level_splice_prefix(const AstProgram& a, const AstProgram& b, std::mt19937_64& rng) {
  const std::vector<std::size_t> end_a = build_subtree_end(a);
  const std::vector<std::size_t> end_b = build_subtree_end(b);
  const std::vector<std::size_t> roots_a = collect_top_level_stmt_roots(a, end_a);
  const std::vector<std::size_t> roots_b = collect_top_level_stmt_roots(b, end_b);
  const int cut_a = roots_a.empty() ? 0 : rand_int(rng, 0, static_cast<int>(roots_a.size()));
  const int cut_b = roots_b.empty() ? 0 : rand_int(rng, 0, static_cast<int>(roots_b.size()));

  AstProgram out;
  out.version = "ast-prefix-v1";
  out.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});

  std::vector<std::vector<AstNode>> segs;
  for (int i = 0; i < cut_a; ++i) {
    segs.push_back(map_subtree_nodes_into(out, a, roots_a[static_cast<std::size_t>(i)], end_a[roots_a[static_cast<std::size_t>(i)]]));
  }
  for (std::size_t i = static_cast<std::size_t>(cut_b); i < roots_b.size(); ++i) {
    segs.push_back(map_subtree_nodes_into(out, b, roots_b[i], end_b[roots_b[i]]));
  }
  if (segs.empty()) {
    segs.push_back(map_subtree_nodes_into(out, a, 1, a.nodes.size() > 1 ? 2 : 1));
    segs.back().clear();
    segs.back().push_back(AstNode{NodeKind::RETURN, 0, 0});
    const int cidx = find_or_add_const(out, Value::from_int(0));
    segs.back().push_back(AstNode{NodeKind::CONST, cidx, 0});
  }

  for (const auto& seg : segs) {
    out.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    out.nodes.insert(out.nodes.end(), seg.begin(), seg.end());
  }
  out.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return out;
}

struct PrefixGenCtx {
  std::set<int> num_names;
  std::set<int> bool_names;
  std::set<int> none_names;
  int tmp_idx = 0;
};

int ensure_name(AstProgram& p, const std::string& name) {
  for (std::size_t i = 0; i < p.names.size(); ++i) {
    if (p.names[i] == name) return static_cast<int>(i);
  }
  p.names.push_back(name);
  return static_cast<int>(p.names.size() - 1);
}

int append_const_id(AstProgram& p, const Value& v) {
  p.consts.push_back(v);
  return static_cast<int>(p.consts.size() - 1);
}

int choose_name_for_type(std::mt19937_64& rng, const PrefixGenCtx& ctx, RType t) {
  std::vector<int> v;
  if (t == RType::Num) v.assign(ctx.num_names.begin(), ctx.num_names.end());
  else if (t == RType::Bool) v.assign(ctx.bool_names.begin(), ctx.bool_names.end());
  else if (t == RType::NoneType) v.assign(ctx.none_names.begin(), ctx.none_names.end());
  else {
    v.assign(ctx.num_names.begin(), ctx.num_names.end());
    v.insert(v.end(), ctx.bool_names.begin(), ctx.bool_names.end());
    v.insert(v.end(), ctx.none_names.begin(), ctx.none_names.end());
  }
  if (v.empty()) return -1;
  return choose_one(rng, v);
}

void emit_random_expr(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx, int depth, RType target);

void emit_random_leaf(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx, RType target) {
  const int var_id = choose_name_for_type(rng, ctx, target);
  if (var_id >= 0 && rand_prob(rng, 0.45)) {
    p.nodes.push_back(AstNode{NodeKind::VAR, var_id, 0});
    return;
  }
  if (target == RType::Num) {
    if (rand_prob(rng, 0.5)) p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::from_int(rand_int(rng, -8, 8))), 0});
    else p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::from_float(std::round(rand_real(rng, -8.0, 8.0) * 1000.0) / 1000.0)), 0});
    return;
  }
  if (target == RType::Bool) {
    p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::from_bool(rand_prob(rng, 0.5))), 0});
    return;
  }
  if (target == RType::NoneType) {
    p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::none()), 0});
    return;
  }
  const int mode = rand_int(rng, 0, 2);
  if (mode == 0) p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::none()), 0});
  else if (mode == 1) p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::from_bool(rand_prob(rng, 0.5))), 0});
  else p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::from_int(rand_int(rng, -8, 8))), 0});
}

void emit_random_expr(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx, int depth, RType target) {
  if (target == RType::Any) {
    std::vector<RType> ts = {RType::Num, RType::Bool, RType::NoneType};
    target = choose_one(rng, ts);
  }
  if (depth <= 0) {
    emit_random_leaf(rng, p, ctx, target);
    return;
  }
  if (target == RType::Num) {
    const int c = rand_int(rng, 0, 5);
    if (c == 0) return emit_random_leaf(rng, p, ctx, RType::Num);
    if (c == 1) {
      p.nodes.push_back(AstNode{NodeKind::NEG, 0, 0});
      return emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
    }
    if (c == 2) {
      std::vector<NodeKind> ops = {NodeKind::ADD, NodeKind::SUB, NodeKind::MUL, NodeKind::DIV, NodeKind::MOD};
      p.nodes.push_back(AstNode{choose_one(rng, ops), 0, 0});
      emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
      emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
      return;
    }
    if (c == 3) {
      std::vector<NodeKind> fs = {NodeKind::CALL_ABS, NodeKind::CALL_MIN, NodeKind::CALL_MAX, NodeKind::CALL_CLIP};
      const NodeKind f = choose_one(rng, fs);
      p.nodes.push_back(AstNode{f, 0, 0});
      for (int i = 0; i < node_arity(f); ++i) emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
      return;
    }
    p.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
    emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
    return;
  }
  if (target == RType::Bool) {
    const int c = rand_int(rng, 0, 4);
    if (c == 0) return emit_random_leaf(rng, p, ctx, RType::Bool);
    if (c == 1) {
      p.nodes.push_back(AstNode{NodeKind::NOT, 0, 0});
      return emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
    }
    if (c == 2) {
      std::vector<NodeKind> ops = {NodeKind::LT, NodeKind::LE, NodeKind::GT, NodeKind::GE, NodeKind::EQ, NodeKind::NE};
      p.nodes.push_back(AstNode{choose_one(rng, ops), 0, 0});
      emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
      emit_random_expr(rng, p, ctx, depth - 1, RType::Num);
      return;
    }
    if (c == 3) {
      std::vector<NodeKind> ops = {NodeKind::AND, NodeKind::OR};
      p.nodes.push_back(AstNode{choose_one(rng, ops), 0, 0});
      emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
      emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
      return;
    }
    p.nodes.push_back(AstNode{NodeKind::IF_EXPR, 0, 0});
    emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
    emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
    return;
  }
  emit_random_leaf(rng, p, ctx, RType::NoneType);
}

int choose_or_new_name(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx) {
  std::vector<std::string> base = {"x", "y", "z", "w", "u", "v"};
  std::vector<int> all;
  for (std::size_t i = 0; i < p.names.size(); ++i) all.push_back(static_cast<int>(i));
  if (!all.empty() && !rand_prob(rng, 0.4)) return choose_one(rng, all);
  for (const std::string& n : base) {
    bool found = false;
    for (const std::string& e : p.names) if (e == n) { found = true; break; }
    if (!found) return ensure_name(p, n);
  }
  return ensure_name(p, "t" + std::to_string(ctx.tmp_idx++));
}

void emit_random_stmt(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx, int depth, const Limits& limits);
void emit_random_block(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx, int depth, const Limits& limits, bool force_return, int max_stmts = -1) {
  const int max_n = (max_stmts < 0) ? limits.max_stmts_per_block : max_stmts;
  const int n = rand_int(rng, 1, std::max(1, max_n));
  bool has_ret = false;
  for (int i = 0; i < n; ++i) {
    p.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    const std::size_t before = p.nodes.size();
    emit_random_stmt(rng, p, ctx, depth, limits);
    if (p.nodes[before].kind == NodeKind::RETURN) {
      has_ret = true;
      break;
    }
  }
  if (force_return && !has_ret) {
    p.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
    p.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    emit_random_expr(rng, p, ctx, std::max(0, depth - 1), choose_one(rng, std::vector<RType>{RType::Num, RType::Bool}));
  }
  p.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
}

void emit_random_stmt(std::mt19937_64& rng, AstProgram& p, PrefixGenCtx& ctx, int depth, const Limits& limits) {
  if (depth <= 0) {
    if (rand_prob(rng, 0.75)) {
      const int nid = choose_or_new_name(rng, p, ctx);
      const RType t = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType});
      p.nodes.push_back(AstNode{NodeKind::ASSIGN, nid, 0});
      emit_random_expr(rng, p, ctx, 0, t);
      ctx.num_names.erase(nid);
      ctx.bool_names.erase(nid);
      ctx.none_names.erase(nid);
      if (t == RType::Num) ctx.num_names.insert(nid);
      if (t == RType::Bool) ctx.bool_names.insert(nid);
      if (t == RType::NoneType) ctx.none_names.insert(nid);
      return;
    }
    p.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
    emit_random_expr(rng, p, ctx, 0, RType::Any);
    return;
  }
  const int c = rand_int(rng, 0, 3);
  if (c == 0) {
    const int nid = choose_or_new_name(rng, p, ctx);
    const RType t = choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType});
    p.nodes.push_back(AstNode{NodeKind::ASSIGN, nid, 0});
    emit_random_expr(rng, p, ctx, depth - 1, t);
    ctx.num_names.erase(nid);
    ctx.bool_names.erase(nid);
    ctx.none_names.erase(nid);
    if (t == RType::Num) ctx.num_names.insert(nid);
    if (t == RType::Bool) ctx.bool_names.insert(nid);
    if (t == RType::NoneType) ctx.none_names.insert(nid);
    return;
  }
  if (c == 1) {
    p.nodes.push_back(AstNode{NodeKind::IF_STMT, 0, 0});
    emit_random_expr(rng, p, ctx, depth - 1, RType::Bool);
    PrefixGenCtx tctx = ctx;
    PrefixGenCtx ectx = ctx;
    emit_random_block(rng, p, tctx, depth - 1, limits, false, std::max(1, std::min(2, limits.max_stmts_per_block)));
    emit_random_block(rng, p, ectx, depth - 1, limits, false, std::max(1, std::min(2, limits.max_stmts_per_block)));
    return;
  }
  if (c == 2) {
    int nid = ensure_name(p, choose_one(rng, std::vector<std::string>{"i", "j", "k"}) + std::to_string(rand_int(rng, 0, 9)));
    p.nodes.push_back(AstNode{NodeKind::FOR_RANGE, nid, rand_int(rng, 0, std::max(0, limits.max_for_k))});
    PrefixGenCtx bctx = ctx;
    bctx.num_names.insert(nid);
    emit_random_block(rng, p, bctx, depth - 1, limits, false, std::max(1, std::min(2, limits.max_stmts_per_block)));
    return;
  }
  p.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  emit_random_expr(rng, p, ctx, depth - 1, choose_one(rng, std::vector<RType>{RType::Num, RType::Bool, RType::NoneType}));
}

ProgramGenome as_genome_prefix(const AstProgram& ast) {
  ProgramGenome g;
  g.ast = ast;
  g.meta = build_meta_fast(g.ast);
  return g;
}

struct ExprCheck {
  RType t = RType::Invalid;
  std::size_t next = 0;
};

struct TypedExprRoot {
  std::size_t start = 0;
  std::size_t stop = 0;
  RType type = RType::Invalid;
};

ExprCheck validate_expr_prefix(const AstProgram& p,
                               const std::vector<std::size_t>& end,
                               std::size_t idx,
                               const std::unordered_map<int, RType>& env,
                               int depth,
                               int& max_depth,
                               std::vector<std::string>& errors) {
  if (idx >= p.nodes.size()) {
    errors.push_back("expr out of range");
    return {RType::Invalid, idx};
  }
  max_depth = std::max(max_depth, depth);
  const AstNode& n = p.nodes[idx];
  if (n.kind == NodeKind::CONST) {
    if (n.i0 < 0 || static_cast<std::size_t>(n.i0) >= p.consts.size()) {
      errors.push_back("const index out of range");
      return {RType::Invalid, end[idx]};
    }
    const Value& v = p.consts[static_cast<std::size_t>(n.i0)];
    if (v.tag == ValueTag::None) return {RType::NoneType, end[idx]};
    if (v.tag == ValueTag::Bool) return {RType::Bool, end[idx]};
    return {RType::Num, end[idx]};
  }
  if (n.kind == NodeKind::VAR) {
    auto it = env.find(n.i0);
    return {it == env.end() ? RType::Num : it->second, end[idx]};
  }
  if (n.kind == NodeKind::NEG || n.kind == NodeKind::NOT) {
    ExprCheck e = validate_expr_prefix(p, end, idx + 1, env, depth + 1, max_depth, errors);
    if (n.kind == NodeKind::NEG) return {e.t == RType::Num ? RType::Num : RType::Invalid, e.next};
    return {e.t == RType::Bool ? RType::Bool : RType::Invalid, e.next};
  }
  if (n.kind == NodeKind::IF_EXPR) {
    ExprCheck c = validate_expr_prefix(p, end, idx + 1, env, depth + 1, max_depth, errors);
    ExprCheck t = validate_expr_prefix(p, end, c.next, env, depth + 1, max_depth, errors);
    ExprCheck f = validate_expr_prefix(p, end, t.next, env, depth + 1, max_depth, errors);
    if (c.t != RType::Bool) errors.push_back("IfExpr condition must be Bool");
    if (t.t == f.t && (t.t == RType::Num || t.t == RType::Bool || t.t == RType::NoneType)) return {t.t, f.next};
    errors.push_back("IfExpr branch type mismatch");
    return {RType::Invalid, f.next};
  }
  if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN || n.kind == NodeKind::CALL_MAX || n.kind == NodeKind::CALL_CLIP) {
    std::size_t cur = idx + 1;
    for (int i = 0; i < node_arity(n.kind); ++i) {
      ExprCheck a = validate_expr_prefix(p, end, cur, env, depth + 1, max_depth, errors);
      if (a.t != RType::Num) errors.push_back("builtin args must be Num");
      cur = a.next;
    }
    return {RType::Num, cur};
  }
  if (n.kind == NodeKind::ADD || n.kind == NodeKind::SUB || n.kind == NodeKind::MUL || n.kind == NodeKind::DIV || n.kind == NodeKind::MOD ||
      n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT || n.kind == NodeKind::GE ||
      n.kind == NodeKind::EQ || n.kind == NodeKind::NE || n.kind == NodeKind::AND || n.kind == NodeKind::OR) {
    ExprCheck a = validate_expr_prefix(p, end, idx + 1, env, depth + 1, max_depth, errors);
    ExprCheck b = validate_expr_prefix(p, end, a.next, env, depth + 1, max_depth, errors);
    if (n.kind == NodeKind::AND || n.kind == NodeKind::OR) {
      if (a.t != RType::Bool || b.t != RType::Bool) errors.push_back("AND/OR expects Bool");
      return {RType::Bool, b.next};
    }
    if (n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT || n.kind == NodeKind::GE) {
      if (a.t != RType::Num || b.t != RType::Num) errors.push_back("comparison expects Num");
      return {RType::Bool, b.next};
    }
    if (n.kind == NodeKind::EQ || n.kind == NodeKind::NE) {
      if (!(a.t == b.t && (a.t == RType::Num || a.t == RType::Bool || a.t == RType::NoneType))) {
        errors.push_back("EQ/NE type mismatch");
      }
      return {RType::Bool, b.next};
    }
    if (a.t != RType::Num || b.t != RType::Num) errors.push_back("arith expects Num");
    return {RType::Num, b.next};
  }
  errors.push_back("invalid expr kind");
  return {RType::Invalid, end[idx]};
}

std::size_t validate_block_prefix(const AstProgram& p,
                                  const std::vector<std::size_t>& end,
                                  std::size_t idx,
                                  std::unordered_map<int, RType> env,
                                  const Limits& limits,
                                  bool top_level,
                                  int& max_depth,
                                  bool& has_return,
                                  std::vector<std::string>& errors);

std::size_t validate_stmt_prefix(const AstProgram& p,
                                 const std::vector<std::size_t>& end,
                                 std::size_t idx,
                                 std::unordered_map<int, RType>& env,
                                 const Limits& limits,
                                 int& max_depth,
                                 bool& is_return,
                                 std::vector<std::string>& errors) {
  if (idx >= p.nodes.size()) {
    errors.push_back("stmt out of range");
    return idx;
  }
  const AstNode& n = p.nodes[idx];
  is_return = false;
  if (n.kind == NodeKind::ASSIGN) {
    ExprCheck e = validate_expr_prefix(p, end, idx + 1, env, 1, max_depth, errors);
    if (e.t == RType::Num || e.t == RType::Bool || e.t == RType::NoneType) env[n.i0] = e.t;
    return e.next;
  }
  if (n.kind == NodeKind::RETURN) {
    ExprCheck e = validate_expr_prefix(p, end, idx + 1, env, 1, max_depth, errors);
    is_return = true;
    return e.next;
  }
  if (n.kind == NodeKind::IF_STMT) {
    ExprCheck c = validate_expr_prefix(p, end, idx + 1, env, 1, max_depth, errors);
    if (c.t != RType::Bool) errors.push_back("IfStmt condition must be Bool");
    bool tr = false;
    bool er = false;
    std::size_t n1 = validate_block_prefix(p, end, c.next, env, limits, false, max_depth, tr, errors);
    std::size_t n2 = validate_block_prefix(p, end, n1, env, limits, false, max_depth, er, errors);
    return n2;
  }
  if (n.kind == NodeKind::FOR_RANGE) {
    if (n.i1 < 0) errors.push_back("ForRange.k must be non-negative");
    if (n.i1 > limits.max_for_k) errors.push_back("ForRange.k exceeds max_for_k");
    std::unordered_map<int, RType> env2 = env;
    env2[n.i0] = RType::Num;
    bool br = false;
    return validate_block_prefix(p, end, idx + 1, env2, limits, false, max_depth, br, errors);
  }
  errors.push_back("invalid stmt kind");
  return end[idx];
}

std::size_t validate_block_prefix(const AstProgram& p,
                                  const std::vector<std::size_t>& end,
                                  std::size_t idx,
                                  std::unordered_map<int, RType> env,
                                  const Limits& limits,
                                  bool top_level,
                                  int& max_depth,
                                  bool& has_return,
                                  std::vector<std::string>& errors) {
  has_return = false;
  int count = 0;
  std::size_t cur = idx;
  while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
    ++count;
    bool is_ret = false;
    cur = validate_stmt_prefix(p, end, cur + 1, env, limits, max_depth, is_ret, errors);
    if (is_ret) has_return = true;
  }
  if (cur >= p.nodes.size() || p.nodes[cur].kind != NodeKind::BLOCK_NIL) {
    errors.push_back("block chain must end with BLOCK_NIL");
    return cur;
  }
  if (count > limits.max_stmts_per_block) errors.push_back("block exceeds max_stmts_per_block");
  if (top_level && !has_return) errors.push_back("top-level block must contain at least one Return");
  return cur + 1;
}

ExprCheck infer_expr_prefix(const AstProgram& p,
                           const std::vector<std::size_t>& end,
                           std::size_t idx,
                           const std::unordered_map<int, RType>& env,
                           std::vector<TypedExprRoot>* out) {
  if (idx >= p.nodes.size()) return {RType::Invalid, idx};
  const AstNode& n = p.nodes[idx];
  if (n.kind == NodeKind::CONST) {
    RType t = RType::Num;
    if (n.i0 < 0 || static_cast<std::size_t>(n.i0) >= p.consts.size()) t = RType::Invalid;
    else {
      const Value& v = p.consts[static_cast<std::size_t>(n.i0)];
      if (v.tag == ValueTag::None) t = RType::NoneType;
      else if (v.tag == ValueTag::Bool) t = RType::Bool;
      else t = RType::Num;
    }
    if (out != nullptr && (t == RType::Num || t == RType::Bool || t == RType::NoneType)) {
      out->push_back(TypedExprRoot{idx, end[idx], t});
    }
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::VAR) {
    auto it = env.find(n.i0);
    const RType t = (it == env.end()) ? RType::Num : it->second;
    if (out != nullptr && (t == RType::Num || t == RType::Bool || t == RType::NoneType)) {
      out->push_back(TypedExprRoot{idx, end[idx], t});
    }
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::NEG || n.kind == NodeKind::NOT) {
    ExprCheck e = infer_expr_prefix(p, end, idx + 1, env, out);
    const RType t = (n.kind == NodeKind::NEG) ? (e.t == RType::Num ? RType::Num : RType::Invalid)
                                               : (e.t == RType::Bool ? RType::Bool : RType::Invalid);
    if (out != nullptr && (t == RType::Num || t == RType::Bool || t == RType::NoneType)) {
      out->push_back(TypedExprRoot{idx, e.next, t});
    }
    return {t, e.next};
  }
  if (n.kind == NodeKind::IF_EXPR) {
    ExprCheck c = infer_expr_prefix(p, end, idx + 1, env, out);
    ExprCheck t = infer_expr_prefix(p, end, c.next, env, out);
    ExprCheck f = infer_expr_prefix(p, end, t.next, env, out);
    RType r = RType::Invalid;
    if (c.t == RType::Bool && t.t == f.t && (t.t == RType::Num || t.t == RType::Bool || t.t == RType::NoneType)) {
      r = t.t;
    }
    if (out != nullptr && (r == RType::Num || r == RType::Bool || r == RType::NoneType)) {
      out->push_back(TypedExprRoot{idx, f.next, r});
    }
    return {r, f.next};
  }
  if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN || n.kind == NodeKind::CALL_MAX || n.kind == NodeKind::CALL_CLIP) {
    std::size_t cur = idx + 1;
    bool ok = true;
    for (int i = 0; i < node_arity(n.kind); ++i) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      if (a.t != RType::Num) ok = false;
      cur = a.next;
    }
    const RType r = ok ? RType::Num : RType::Invalid;
    if (out != nullptr && r == RType::Num) {
      out->push_back(TypedExprRoot{idx, cur, r});
    }
    return {r, cur};
  }
  if (n.kind == NodeKind::ADD || n.kind == NodeKind::SUB || n.kind == NodeKind::MUL || n.kind == NodeKind::DIV || n.kind == NodeKind::MOD ||
      n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT || n.kind == NodeKind::GE ||
      n.kind == NodeKind::EQ || n.kind == NodeKind::NE || n.kind == NodeKind::AND || n.kind == NodeKind::OR) {
    ExprCheck a = infer_expr_prefix(p, end, idx + 1, env, out);
    ExprCheck b = infer_expr_prefix(p, end, a.next, env, out);
    RType r = RType::Invalid;
    if (n.kind == NodeKind::AND || n.kind == NodeKind::OR) r = (a.t == RType::Bool && b.t == RType::Bool) ? RType::Bool : RType::Invalid;
    else if (n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT || n.kind == NodeKind::GE)
      r = (a.t == RType::Num && b.t == RType::Num) ? RType::Bool : RType::Invalid;
    else if (n.kind == NodeKind::EQ || n.kind == NodeKind::NE)
      r = (a.t == b.t && (a.t == RType::Num || a.t == RType::Bool || a.t == RType::NoneType)) ? RType::Bool : RType::Invalid;
    else
      r = (a.t == RType::Num && b.t == RType::Num) ? RType::Num : RType::Invalid;
    if (out != nullptr && (r == RType::Num || r == RType::Bool || r == RType::NoneType)) {
      out->push_back(TypedExprRoot{idx, b.next, r});
    }
    return {r, b.next};
  }
  return {RType::Invalid, end[idx]};
}

void collect_typed_exprs_in_stmt(const AstProgram& p,
                                 const std::vector<std::size_t>& end,
                                 std::size_t idx,
                                 std::unordered_map<int, RType>& env,
                                 std::vector<TypedExprRoot>& out) {
  if (idx >= p.nodes.size()) return;
  const AstNode& n = p.nodes[idx];
  if (n.kind == NodeKind::ASSIGN) {
    ExprCheck e = infer_expr_prefix(p, end, idx + 1, env, &out);
    if (e.t == RType::Num || e.t == RType::Bool || e.t == RType::NoneType) env[n.i0] = e.t;
    return;
  }
  if (n.kind == NodeKind::RETURN) {
    (void)infer_expr_prefix(p, end, idx + 1, env, &out);
    return;
  }
  if (n.kind == NodeKind::IF_STMT) {
    ExprCheck c = infer_expr_prefix(p, end, idx + 1, env, &out);
    std::unordered_map<int, RType> env_t = env;
    std::unordered_map<int, RType> env_e = env;
    std::size_t cur = c.next;
    while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(p, end, cur + 1, env_t, out);
      cur = end[cur + 1];
    }
    if (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_NIL) cur += 1;
    while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(p, end, cur + 1, env_e, out);
      cur = end[cur + 1];
    }
    return;
  }
  if (n.kind == NodeKind::FOR_RANGE) {
    std::unordered_map<int, RType> env_b = env;
    env_b[n.i0] = RType::Num;
    std::size_t cur = idx + 1;
    while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(p, end, cur + 1, env_b, out);
      cur = end[cur + 1];
    }
    return;
  }
}

std::vector<TypedExprRoot> collect_typed_expr_roots(const AstProgram& p, const std::vector<std::size_t>& end) {
  std::vector<TypedExprRoot> out;
  if (p.nodes.size() < 2 || p.nodes[0].kind != NodeKind::PROGRAM) return out;
  std::unordered_map<int, RType> env;
  std::size_t cur = 1;
  while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
    collect_typed_exprs_in_stmt(p, end, cur + 1, env, out);
    cur = end[cur + 1];
  }
  return out;
}

}  // namespace

ProgramGenome make_random_genome(std::uint64_t seed, const Limits& limits) {
  std::mt19937_64 rng(seed);
  for (int i = 0; i < 128; ++i) {
    AstProgram p;
    p.version = "ast-prefix-v1";
    p.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
    PrefixGenCtx ctx;
    ctx.num_names.insert(ensure_name(p, "x"));
    emit_random_block(rng, p, ctx, limits.max_expr_depth, limits, true);
    ProgramGenome g = as_genome_prefix(p);
    if (validate_genome(g, limits).is_valid) {
      return g;
    }
  }
  AstProgram p;
  p.version = "ast-prefix-v1";
  p.nodes.push_back(AstNode{NodeKind::PROGRAM, 0, 0});
  p.nodes.push_back(AstNode{NodeKind::BLOCK_CONS, 0, 0});
  p.nodes.push_back(AstNode{NodeKind::RETURN, 0, 0});
  p.nodes.push_back(AstNode{NodeKind::CONST, append_const_id(p, Value::from_int(0)), 0});
  p.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  return as_genome_prefix(p);
}

ProgramGenome mutate(const ProgramGenome& genome, std::uint64_t seed, const Limits& limits) {
  std::mt19937_64 rng(seed);
  const AstProgram& ast = genome.ast;
  if (ast.nodes.empty()) {
    return make_random_genome(seed, limits);
  }
  AstProgram mutated;
  if (rand_prob(rng, 0.8)) {
    const std::vector<std::size_t> end_a = build_subtree_end(ast);
    const std::vector<TypedExprRoot> expr_a = collect_typed_expr_roots(ast, end_a);
    if (!expr_a.empty()) {
      const TypedExprRoot ta = choose_one(rng, expr_a);
      AstProgram donor;
      donor.version = "ast-prefix-v1";
      donor.nodes = make_random_expr_nodes_for_type(rng, donor, ta.type, std::max(1, limits.max_expr_depth / 2));
      mutated = replace_subtree(ast, ta.start, ta.stop, donor, 0, donor.nodes.size());
    }
  }
  if (mutated.nodes.empty()) {
    std::vector<std::size_t> const_nodes;
    for (std::size_t i = 0; i < ast.nodes.size(); ++i) {
      if (ast.nodes[i].kind == NodeKind::CONST) const_nodes.push_back(i);
    }
    if (!const_nodes.empty()) {
      mutated = ast;
      const std::size_t ci = const_nodes[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(const_nodes.size()) - 1))];
      const int old_idx = mutated.nodes[ci].i0;
      if (old_idx >= 0 && static_cast<std::size_t>(old_idx) < mutated.consts.size()) {
        Value v = mutated.consts[static_cast<std::size_t>(old_idx)];
        if (v.tag == ValueTag::Int) {
          v.i += rand_int(rng, -2, 2);
        } else if (v.tag == ValueTag::Float) {
          v.f += rand_real(rng, -1.0, 1.0);
        } else if (v.tag == ValueTag::Bool) {
          v.b = !v.b;
        }
        mutated.consts.push_back(v);
        mutated.nodes[ci].i0 = static_cast<int>(mutated.consts.size() - 1);
      }
    }
  }
  if (mutated.nodes.empty()) return genome;
  ProgramGenome out;
  out.ast = std::move(mutated);
  out.meta = build_meta_fast(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) return genome;
  for (const AstNode& n : out.ast.nodes) {
    if (n.kind == NodeKind::FOR_RANGE && (n.i1 < 0 || n.i1 > limits.max_for_k)) {
      return genome;
    }
  }
  return out;
}

ProgramGenome crossover_top_level(const ProgramGenome& parent_a,
                                  const ProgramGenome& parent_b,
                                  std::uint64_t seed,
                                  const Limits& limits) {
  std::mt19937_64 rng(seed);
  AstProgram child = top_level_splice_prefix(parent_a.ast, parent_b.ast, rng);
  ProgramGenome out;
  out.ast = std::move(child);
  out.meta = build_meta_fast(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) return parent_a;
  for (const AstNode& n : out.ast.nodes) {
    if (n.kind == NodeKind::FOR_RANGE && (n.i1 < 0 || n.i1 > limits.max_for_k)) {
      return parent_a;
    }
  }
  if (!validate_genome(out, limits).is_valid) return parent_a;
  return out;
}

ProgramGenome crossover_typed_subtree(const ProgramGenome& parent_a,
                                      const ProgramGenome& parent_b,
                                      std::uint64_t seed,
                                      const Limits& limits) {
  std::mt19937_64 rng(seed);
  const AstProgram& ast_a = parent_a.ast;
  const AstProgram& ast_b = parent_b.ast;
  const std::vector<std::size_t> end_a = build_subtree_end(ast_a);
  const std::vector<std::size_t> end_b = build_subtree_end(ast_b);

  const std::vector<TypedExprRoot> expr_a = collect_typed_expr_roots(ast_a, end_a);
  const std::vector<TypedExprRoot> expr_b = collect_typed_expr_roots(ast_b, end_b);

  AstProgram child;
  bool child_ready = false;
  if (!expr_a.empty() && !expr_b.empty()) {
    const TypedExprRoot t = expr_a[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(expr_a.size()) - 1))];
    std::vector<TypedExprRoot> pool;
    for (const TypedExprRoot& r : expr_b) {
      if (r.type == t.type) pool.push_back(r);
    }
    if (!pool.empty()) {
      const TypedExprRoot s = pool[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(pool.size()) - 1))];
      child = replace_subtree(ast_a, t.start, t.stop, ast_b, s.start, s.stop);
      child_ready = true;
    }
  }
  if (!child_ready) {
    std::vector<std::size_t> stmt_a;
    std::vector<std::size_t> expr_a_raw;
    std::vector<std::size_t> stmt_b;
    std::vector<std::size_t> expr_b_raw;
    collect_roots_walk(ast_a, end_a, 0, stmt_a, expr_a_raw);
    collect_roots_walk(ast_b, end_b, 0, stmt_b, expr_b_raw);
    if (!stmt_a.empty() && !stmt_b.empty()) {
    const std::size_t t = stmt_a[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(stmt_a.size()) - 1))];
    const NodeKind k = ast_a.nodes[t].kind;
    std::vector<std::size_t> pool;
    for (std::size_t r : stmt_b) {
      if (ast_b.nodes[r].kind == k) pool.push_back(r);
    }
    if (pool.empty()) pool = stmt_b;
    const std::size_t s = pool[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(pool.size()) - 1))];
    child = replace_subtree(ast_a, t, end_a[t], ast_b, s, end_b[s]);
    child_ready = true;
    }
  }
  if (!child_ready) {
    child = top_level_splice_prefix(ast_a, ast_b, rng);
  }
  ProgramGenome out;
  out.ast = std::move(child);
  out.meta = build_meta_fast(out.ast);
  if (out.meta.node_count > limits.max_total_nodes) return parent_a;
  for (const AstNode& n : out.ast.nodes) {
    if (n.kind == NodeKind::FOR_RANGE && (n.i1 < 0 || n.i1 > limits.max_for_k)) {
      return parent_a;
    }
  }
  if (!validate_genome(out, limits).is_valid) return parent_a;
  return out;
}

ProgramGenome crossover(const ProgramGenome& parent_a,
                        const ProgramGenome& parent_b,
                        std::uint64_t seed,
                        CrossoverMethod method,
                        const Limits& limits) {
  if (method == CrossoverMethod::TopLevelSplice) {
    return crossover_top_level(parent_a, parent_b, seed, limits);
  }
  if (method == CrossoverMethod::TypedSubtree) {
    return crossover_typed_subtree(parent_a, parent_b, seed, limits);
  }
  if (method == CrossoverMethod::Hybrid) {
    std::mt19937_64 rng(seed);
    if (rand_prob(rng, 0.7)) {
      return crossover_typed_subtree(parent_a, parent_b, seed, limits);
    }
    return crossover_top_level(parent_a, parent_b, seed, limits);
  }
  throw std::invalid_argument("unknown crossover method");
}

ValidationResult validate_genome(const ProgramGenome& genome, const Limits& limits) {
  ValidationResult out;
  const AstProgram& p = genome.ast;
  if (p.version != "ast-prefix-v1") {
    out.errors.push_back("unsupported ast prefix version");
    out.is_valid = false;
    return out;
  }
  if (p.nodes.empty() || p.nodes[0].kind != NodeKind::PROGRAM) {
    out.errors.push_back("prefix root must be PROGRAM");
    out.is_valid = false;
    return out;
  }
  std::vector<std::size_t> end;
  try {
    end = build_subtree_end(p);
  } catch (const std::exception& e) {
    out.errors.push_back(std::string("invalid prefix structure: ") + e.what());
    out.is_valid = false;
    return out;
  }
  if (static_cast<int>(p.nodes.size()) > limits.max_total_nodes) {
    out.errors.push_back("node count exceeds max_total_nodes");
  }
  int max_depth = 0;
  bool has_return = false;
  const std::size_t final_idx = validate_block_prefix(
      p, end, 1, std::unordered_map<int, RType>{}, limits, true, max_depth, has_return, out.errors);
  if (final_idx != p.nodes.size()) {
    out.errors.push_back("trailing tokens after program block");
  }
  if (max_depth > limits.max_expr_depth) {
    out.errors.push_back("expression depth exceeds max_expr_depth");
  }
  out.is_valid = out.errors.empty();
  return out;
}

BytecodeProgram compile_for_eval(const ProgramGenome& genome) {
  Compiler c;
  return c.build(genome.ast);
}

BytecodeProgram compile_for_eval_with_preset_locals(const ProgramGenome& genome,
                                                    const std::vector<std::string>& preset_locals) {
  Compiler c(&preset_locals);
  return c.build(genome.ast);
}

std::string ast_to_string(const AstProgram& program) { return canonical_prefix_serialize(program); }

}  // namespace g3pvm::evo
