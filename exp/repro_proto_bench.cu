#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "g3pvm/evo_ast.hpp"
#include "g3pvm/vm_gpu.hpp"

namespace {

using g3pvm::InputCase;
using g3pvm::LocalBinding;
using g3pvm::Value;
using g3pvm::evo::AstNode;
using g3pvm::evo::NodeKind;
using g3pvm::evo::ProgramGenome;

enum class RType {
  Num = 0,
  Bool = 1,
  NoneType = 2,
  Any = 3,
  Invalid = 4,
};

struct PlainNode {
  int kind = 0;
  int i0 = 0;
  int i1 = 0;
};

struct CandidateRange {
  int start = -1;
  int stop = -1;
  int tag = 0;
  int aux = -1;
};

enum CandidateTag {
  CAND_EXPR = 0,
  CAND_STMT = 1,
};

struct PackedProgramMeta {
  int used_len = 0;
  int name_count = 0;
  int const_count = 0;
};

struct BenchConfig {
  int population_size = 1024;
  int child_count = 1024;
  int candidates_per_program = 16;
  int donor_pool_size = 256;
  int max_nodes = 96;
  int max_donor_nodes = 24;
  int max_names = 32;
  int max_consts = 32;
  int tournament_k = 3;
  int eval_cases = 1024;
  int repeats = 5;
  std::uint64_t seed = 12345;
  double mutation_ratio = 0.5;
  bool enable_gpu_cheap_validate = true;
  std::string out_json;
};

struct TimingAggregate {
  double cpu_subtree_ms = 0.0;
  double cpu_candidate_ms = 0.0;
  double cpu_donor_ms = 0.0;
  double pack_program_ms = 0.0;
  double pack_candidate_ms = 0.0;
  double pack_donor_ms = 0.0;
  double h2d_program_ms = 0.0;
  double h2d_candidate_ms = 0.0;
  double h2d_donor_ms = 0.0;
  double gpu_eval_ms = 0.0;
  double gpu_selection_ms = 0.0;
  double gpu_variation_ms = 0.0;
  double gpu_cheap_validate_ms = 0.0;
  double d2h_child_ms = 0.0;
  double cpu_validate_fallback_ms = 0.0;
  double gpu_validate_fallback_ms = 0.0;
  double overlap_wall_ms = 0.0;
  double sequential_wall_ms = 0.0;
  double cpu_selection_ms = 0.0;
  double cpu_variation_ms = 0.0;
  std::uint64_t cpu_checksum = 0;
  std::uint64_t gpu_checksum = 0;
  int cpu_fallback_count = 0;
  int gpu_fallback_count = 0;
  int gpu_cheap_reject_count = 0;
  int gpu_full_validate_count = 0;
};

struct PackedHostData {
  std::vector<PlainNode> program_nodes;
  std::vector<PackedProgramMeta> metas;
  std::vector<CandidateRange> candidates;
  std::vector<std::uint64_t> program_name_ids;
  std::vector<Value> program_consts;
  std::vector<PlainNode> donor_nodes;
  std::vector<int> donor_lens;
  std::vector<std::uint64_t> donor_name_ids;
  std::vector<int> donor_name_counts;
  std::vector<Value> donor_consts;
  std::vector<int> donor_const_counts;
};

struct DeviceBuffers {
  PlainNode* d_program_nodes = nullptr;
  PackedProgramMeta* d_metas = nullptr;
  CandidateRange* d_candidates = nullptr;
  std::uint64_t* d_program_name_ids = nullptr;
  Value* d_program_consts = nullptr;
  PlainNode* d_donor_nodes = nullptr;
  int* d_donor_lens = nullptr;
  std::uint64_t* d_donor_name_ids = nullptr;
  int* d_donor_name_counts = nullptr;
  Value* d_donor_consts = nullptr;
  int* d_donor_const_counts = nullptr;
  int* d_fitness = nullptr;
  int* d_parent_a = nullptr;
  int* d_parent_b = nullptr;
  int* d_cand_a = nullptr;
  int* d_cand_b = nullptr;
  int* d_donor_idx = nullptr;
  unsigned char* d_is_mutation = nullptr;
  PlainNode* d_child_nodes = nullptr;
  int* d_child_used_len = nullptr;
  std::uint64_t* d_child_name_ids = nullptr;
  int* d_child_name_counts = nullptr;
  Value* d_child_consts = nullptr;
  int* d_child_const_counts = nullptr;
  unsigned char* d_child_cheap_valid = nullptr;

  ~DeviceBuffers() {
    if (d_program_nodes) cudaFree(d_program_nodes);
    if (d_metas) cudaFree(d_metas);
    if (d_candidates) cudaFree(d_candidates);
    if (d_program_name_ids) cudaFree(d_program_name_ids);
    if (d_program_consts) cudaFree(d_program_consts);
    if (d_donor_nodes) cudaFree(d_donor_nodes);
    if (d_donor_lens) cudaFree(d_donor_lens);
    if (d_donor_name_ids) cudaFree(d_donor_name_ids);
    if (d_donor_name_counts) cudaFree(d_donor_name_counts);
    if (d_donor_consts) cudaFree(d_donor_consts);
    if (d_donor_const_counts) cudaFree(d_donor_const_counts);
    if (d_fitness) cudaFree(d_fitness);
    if (d_parent_a) cudaFree(d_parent_a);
    if (d_parent_b) cudaFree(d_parent_b);
    if (d_cand_a) cudaFree(d_cand_a);
    if (d_cand_b) cudaFree(d_cand_b);
    if (d_donor_idx) cudaFree(d_donor_idx);
    if (d_is_mutation) cudaFree(d_is_mutation);
    if (d_child_nodes) cudaFree(d_child_nodes);
    if (d_child_used_len) cudaFree(d_child_used_len);
    if (d_child_name_ids) cudaFree(d_child_name_ids);
    if (d_child_name_counts) cudaFree(d_child_name_counts);
    if (d_child_consts) cudaFree(d_child_consts);
    if (d_child_const_counts) cudaFree(d_child_const_counts);
    if (d_child_cheap_valid) cudaFree(d_child_cheap_valid);
  }
};

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
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
    case NodeKind::VAR:
      return 0;
    case NodeKind::NEG:
    case NodeKind::NOT:
    case NodeKind::CALL_ABS:
      return 1;
    case NodeKind::CALL_MIN:
    case NodeKind::CALL_MAX:
      return 2;
    case NodeKind::CALL_CLIP:
      return 3;
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
    case NodeKind::IF_EXPR:
      return 2 + (kind == NodeKind::IF_EXPR);
  }
  return 0;
}

__host__ __device__ int node_arity_checked(int kind_raw) {
  const NodeKind kind = static_cast<NodeKind>(kind_raw);
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
    case NodeKind::VAR:
      return 0;
    case NodeKind::NEG:
    case NodeKind::NOT:
    case NodeKind::CALL_ABS:
      return 1;
    case NodeKind::CALL_MIN:
    case NodeKind::CALL_MAX:
      return 2;
    case NodeKind::CALL_CLIP:
      return 3;
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
  }
  return -1;
}

std::size_t fill_subtree_end_at(const std::vector<AstNode>& nodes, std::size_t idx, std::vector<int>& out) {
  if (idx >= nodes.size()) {
    throw std::runtime_error("prefix traversal out of range");
  }
  std::size_t cur = idx + 1;
  for (int i = 0; i < node_arity(nodes[idx].kind); ++i) {
    cur = fill_subtree_end_at(nodes, cur, out);
  }
  out[idx] = static_cast<int>(cur);
  return cur;
}

std::vector<int> build_subtree_end(const std::vector<AstNode>& nodes) {
  std::vector<int> out(nodes.size(), 0);
  if (!nodes.empty()) {
    const std::size_t end = fill_subtree_end_at(nodes, 0, out);
    if (end != nodes.size()) {
      throw std::runtime_error("prefix trailing tokens");
    }
  }
  return out;
}

bool is_candidate_kind(NodeKind kind) {
  return kind != NodeKind::PROGRAM && kind != NodeKind::BLOCK_CONS && kind != NodeKind::BLOCK_NIL;
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

struct FormalCandidateSet {
  std::vector<CandidateRange> expr;
  std::vector<CandidateRange> stmt;
};

__host__ __device__ std::uint64_t hash64(std::uint64_t x);

std::uint64_t hash_name(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

__host__ __device__ bool value_equal_simple(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == g3pvm::ValueTag::None) return true;
  if (a.tag == g3pvm::ValueTag::Bool) return a.b == b.b;
  if (a.tag == g3pvm::ValueTag::Int) return a.i == b.i;
  return a.f == b.f;
}

Value make_const_value(std::uint64_t seed, RType type) {
  const std::uint64_t h = hash64(seed);
  if (type == RType::Bool) {
    return Value::from_bool((h & 1ULL) != 0ULL);
  }
  if (type == RType::NoneType) {
    return Value::none();
  }
  if ((h & 1ULL) == 0ULL) {
    return Value::from_int(static_cast<std::int64_t>(static_cast<int>(h % 17ULL) - 8));
  }
  const double frac = static_cast<double>(h % 16001ULL) / 1000.0;
  return Value::from_float(frac - 8.0);
}

std::vector<PlainNode> make_donor_nodes_for_type(std::uint64_t seed, RType type, int max_donor_nodes) {
  std::vector<PlainNode> out;
  out.reserve(static_cast<std::size_t>(max_donor_nodes));
  const std::uint64_t h = hash64(seed);
  if (type == RType::Bool) {
    if ((h & 1ULL) != 0ULL && max_donor_nodes >= 3) {
      out.push_back(PlainNode{static_cast<int>(NodeKind::LT), 0, 0});
      out.push_back(PlainNode{static_cast<int>(NodeKind::VAR), 0, 0});
      out.push_back(PlainNode{static_cast<int>(NodeKind::CONST), 0, 0});
    } else {
      out.push_back(PlainNode{static_cast<int>(NodeKind::CONST), 0, 0});
    }
    return out;
  }
  if (type == RType::Num) {
    const int mode = static_cast<int>((h >> 3) % 3ULL);
    if (mode == 0) {
      out.push_back(PlainNode{static_cast<int>(NodeKind::VAR), 0, 0});
    } else if (mode == 1 && max_donor_nodes >= 3) {
      out.push_back(PlainNode{static_cast<int>(NodeKind::ADD), 0, 0});
      out.push_back(PlainNode{static_cast<int>(NodeKind::VAR), 0, 0});
      out.push_back(PlainNode{static_cast<int>(NodeKind::CONST), 0, 0});
    } else if (max_donor_nodes >= 2) {
      out.push_back(PlainNode{static_cast<int>(NodeKind::NEG), 0, 0});
      out.push_back(PlainNode{static_cast<int>(NodeKind::VAR), 0, 0});
    } else {
      out.push_back(PlainNode{static_cast<int>(NodeKind::CONST), 0, 0});
    }
    return out;
  }
  out.push_back(PlainNode{static_cast<int>(NodeKind::CONST), 0, 0});
  return out;
}

ExprCheck infer_expr_prefix(const std::vector<AstNode>& nodes,
                            const std::vector<int>& end,
                            std::size_t idx,
                            const std::unordered_map<int, RType>& env,
                            std::vector<TypedExprRoot>* out) {
  if (idx >= nodes.size()) {
    return {RType::Invalid, idx};
  }
  const AstNode& n = nodes[idx];
  if (n.kind == NodeKind::CONST) {
    RType t = RType::Num;
    if (out != nullptr) {
      out->push_back(TypedExprRoot{idx, static_cast<std::size_t>(end[idx]), t});
    }
    return {t, static_cast<std::size_t>(end[idx])};
  }
  if (n.kind == NodeKind::VAR) {
    auto it = env.find(n.i0);
    const RType t = (it == env.end()) ? RType::Num : it->second;
    if (out != nullptr && t != RType::Invalid) {
      out->push_back(TypedExprRoot{idx, static_cast<std::size_t>(end[idx]), t});
    }
    return {t, static_cast<std::size_t>(end[idx])};
  }
  if (n.kind == NodeKind::NEG || n.kind == NodeKind::NOT) {
    ExprCheck e = infer_expr_prefix(nodes, end, idx + 1, env, out);
    const RType t = (n.kind == NodeKind::NEG) ? (e.t == RType::Num ? RType::Num : RType::Invalid)
                                              : (e.t == RType::Bool ? RType::Bool : RType::Invalid);
    if (out != nullptr && t != RType::Invalid) {
      out->push_back(TypedExprRoot{idx, e.next, t});
    }
    return {t, e.next};
  }
  if (n.kind == NodeKind::IF_EXPR) {
    ExprCheck c = infer_expr_prefix(nodes, end, idx + 1, env, out);
    ExprCheck t = infer_expr_prefix(nodes, end, c.next, env, out);
    ExprCheck f = infer_expr_prefix(nodes, end, t.next, env, out);
    RType r = RType::Invalid;
    if (c.t == RType::Bool && t.t == f.t && (t.t == RType::Num || t.t == RType::Bool || t.t == RType::NoneType)) {
      r = t.t;
    }
    if (out != nullptr && r != RType::Invalid) {
      out->push_back(TypedExprRoot{idx, f.next, r});
    }
    return {r, f.next};
  }
  if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN || n.kind == NodeKind::CALL_MAX ||
      n.kind == NodeKind::CALL_CLIP) {
    std::size_t cur = idx + 1;
    bool ok = true;
    for (int i = 0; i < node_arity(n.kind); ++i) {
      ExprCheck a = infer_expr_prefix(nodes, end, cur, env, out);
      if (a.t != RType::Num) {
        ok = false;
      }
      cur = a.next;
    }
    const RType r = ok ? RType::Num : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) {
      out->push_back(TypedExprRoot{idx, cur, r});
    }
    return {r, cur};
  }
  if (n.kind == NodeKind::ADD || n.kind == NodeKind::SUB || n.kind == NodeKind::MUL || n.kind == NodeKind::DIV ||
      n.kind == NodeKind::MOD || n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT ||
      n.kind == NodeKind::GE || n.kind == NodeKind::EQ || n.kind == NodeKind::NE || n.kind == NodeKind::AND ||
      n.kind == NodeKind::OR) {
    ExprCheck a = infer_expr_prefix(nodes, end, idx + 1, env, out);
    ExprCheck b = infer_expr_prefix(nodes, end, a.next, env, out);
    RType r = RType::Invalid;
    if (n.kind == NodeKind::AND || n.kind == NodeKind::OR) {
      r = (a.t == RType::Bool && b.t == RType::Bool) ? RType::Bool : RType::Invalid;
    } else if (n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT ||
               n.kind == NodeKind::GE) {
      r = (a.t == RType::Num && b.t == RType::Num) ? RType::Bool : RType::Invalid;
    } else if (n.kind == NodeKind::EQ || n.kind == NodeKind::NE) {
      r = (a.t == b.t && (a.t == RType::Num || a.t == RType::Bool || a.t == RType::NoneType)) ? RType::Bool
                                                                                                 : RType::Invalid;
    } else {
      r = (a.t == RType::Num && b.t == RType::Num) ? RType::Num : RType::Invalid;
    }
    if (out != nullptr && r != RType::Invalid) {
      out->push_back(TypedExprRoot{idx, b.next, r});
    }
    return {r, b.next};
  }
  return {RType::Invalid, static_cast<std::size_t>(end[idx])};
}

void collect_typed_exprs_in_stmt(const std::vector<AstNode>& nodes,
                                 const std::vector<int>& end,
                                 std::size_t idx,
                                 std::unordered_map<int, RType>& env,
                                 std::vector<TypedExprRoot>& out) {
  if (idx >= nodes.size()) {
    return;
  }
  const AstNode& n = nodes[idx];
  if (n.kind == NodeKind::ASSIGN) {
    ExprCheck e = infer_expr_prefix(nodes, end, idx + 1, env, &out);
    if (e.t == RType::Num || e.t == RType::Bool || e.t == RType::NoneType) {
      env[n.i0] = e.t;
    }
    return;
  }
  if (n.kind == NodeKind::RETURN) {
    (void)infer_expr_prefix(nodes, end, idx + 1, env, &out);
    return;
  }
  if (n.kind == NodeKind::IF_STMT) {
    ExprCheck c = infer_expr_prefix(nodes, end, idx + 1, env, &out);
    std::unordered_map<int, RType> env_t = env;
    std::unordered_map<int, RType> env_e = env;
    std::size_t cur = c.next;
    while (cur < nodes.size() && nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(nodes, end, cur + 1, env_t, out);
      cur = static_cast<std::size_t>(end[cur + 1]);
    }
    if (cur < nodes.size() && nodes[cur].kind == NodeKind::BLOCK_NIL) {
      cur += 1;
    }
    while (cur < nodes.size() && nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(nodes, end, cur + 1, env_e, out);
      cur = static_cast<std::size_t>(end[cur + 1]);
    }
    return;
  }
  if (n.kind == NodeKind::FOR_RANGE) {
    std::unordered_map<int, RType> env_b = env;
    env_b[n.i0] = RType::Num;
    std::size_t cur = idx + 1;
    while (cur < nodes.size() && nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(nodes, end, cur + 1, env_b, out);
      cur = static_cast<std::size_t>(end[cur + 1]);
    }
  }
}

std::vector<TypedExprRoot> collect_typed_expr_roots(const std::vector<AstNode>& nodes, const std::vector<int>& end) {
  std::vector<TypedExprRoot> out;
  if (nodes.size() < 2 || nodes[0].kind != NodeKind::PROGRAM) {
    return out;
  }
  std::unordered_map<int, RType> env;
  std::size_t cur = 1;
  while (cur < nodes.size() && nodes[cur].kind == NodeKind::BLOCK_CONS) {
    collect_typed_exprs_in_stmt(nodes, end, cur + 1, env, out);
    cur = static_cast<std::size_t>(end[cur + 1]);
  }
  return out;
}

bool is_stmt_kind(NodeKind kind) {
  return kind == NodeKind::ASSIGN || kind == NodeKind::IF_STMT || kind == NodeKind::FOR_RANGE ||
         kind == NodeKind::RETURN;
}

bool is_expr_kind(NodeKind kind) {
  return kind == NodeKind::CONST || kind == NodeKind::VAR || kind == NodeKind::NEG || kind == NodeKind::NOT ||
         kind == NodeKind::ADD || kind == NodeKind::SUB || kind == NodeKind::MUL || kind == NodeKind::DIV ||
         kind == NodeKind::MOD || kind == NodeKind::LT || kind == NodeKind::LE || kind == NodeKind::GT ||
         kind == NodeKind::GE || kind == NodeKind::EQ || kind == NodeKind::NE || kind == NodeKind::AND ||
         kind == NodeKind::OR || kind == NodeKind::IF_EXPR || kind == NodeKind::CALL_ABS ||
         kind == NodeKind::CALL_MIN || kind == NodeKind::CALL_MAX || kind == NodeKind::CALL_CLIP;
}

void collect_roots_walk(const std::vector<AstNode>& nodes,
                        const std::vector<int>& end,
                        std::size_t idx,
                        std::vector<std::size_t>& stmt_roots,
                        std::vector<std::size_t>& expr_roots) {
  const NodeKind kind = nodes[idx].kind;
  if (is_stmt_kind(kind)) {
    stmt_roots.push_back(idx);
  }
  if (is_expr_kind(kind)) {
    expr_roots.push_back(idx);
  }
  std::size_t child = idx + 1;
  for (int i = 0; i < node_arity(kind); ++i) {
    collect_roots_walk(nodes, end, child, stmt_roots, expr_roots);
    child = static_cast<std::size_t>(end[child]);
  }
}

FormalCandidateSet collect_formal_candidates(const std::vector<AstNode>& nodes,
                                             const std::vector<int>& subtree_end,
                                             int limit) {
  FormalCandidateSet out;
  const std::vector<TypedExprRoot> expr_roots = collect_typed_expr_roots(nodes, subtree_end);
  out.expr.reserve(std::min<int>(static_cast<int>(expr_roots.size()), limit));
  if (!expr_roots.empty()) {
    const int used = std::min<int>(static_cast<int>(expr_roots.size()), limit);
    const double step = static_cast<double>(expr_roots.size()) / static_cast<double>(used);
    for (int i = 0; i < used; ++i) {
      std::size_t pick = static_cast<std::size_t>(i * step);
      if (pick >= expr_roots.size()) {
        pick = expr_roots.size() - 1;
      }
      const TypedExprRoot& root = expr_roots[pick];
      out.expr.push_back(CandidateRange{static_cast<int>(root.start), static_cast<int>(root.stop), CAND_EXPR,
                                        static_cast<int>(root.type)});
    }
  }

  std::vector<std::size_t> stmt_roots;
  std::vector<std::size_t> expr_raw;
  if (!nodes.empty()) {
    collect_roots_walk(nodes, subtree_end, 0, stmt_roots, expr_raw);
  }
  out.stmt.reserve(std::min<int>(static_cast<int>(stmt_roots.size()), limit));
  if (!stmt_roots.empty()) {
    const int used = std::min<int>(static_cast<int>(stmt_roots.size()), limit);
    const double step = static_cast<double>(stmt_roots.size()) / static_cast<double>(used);
    for (int i = 0; i < used; ++i) {
      std::size_t pick = static_cast<std::size_t>(i * step);
      if (pick >= stmt_roots.size()) {
        pick = stmt_roots.size() - 1;
      }
      const std::size_t root = stmt_roots[pick];
      out.stmt.push_back(CandidateRange{static_cast<int>(root), subtree_end[root], CAND_STMT,
                                        static_cast<int>(nodes[root].kind)});
    }
  }

  if (out.expr.empty() && out.stmt.empty()) {
    out.stmt.push_back(CandidateRange{1, std::min<int>(2, static_cast<int>(nodes.size())), CAND_STMT,
                                      static_cast<int>(NodeKind::RETURN)});
  }
  return out;
}

std::uint64_t mix_seed(std::uint64_t seed, std::uint64_t salt) {
  std::uint64_t x = seed ^ (salt + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

ProgramGenome make_genome_for_seed(std::uint64_t seed, int max_nodes) {
  g3pvm::evo::Limits limits;
  limits.max_total_nodes = max_nodes;
  limits.max_expr_depth = 5;
  limits.max_stmts_per_block = 6;
  return g3pvm::evo::make_random_genome(seed, limits);
}

std::vector<ProgramGenome> make_population(const BenchConfig& cfg) {
  std::vector<ProgramGenome> out;
  out.reserve(cfg.population_size);
  for (int i = 0; i < cfg.population_size; ++i) {
    out.push_back(make_genome_for_seed(cfg.seed + static_cast<std::uint64_t>(i), cfg.max_nodes));
  }
  return out;
}

std::vector<g3pvm::BytecodeProgram> compile_population(const std::vector<ProgramGenome>& genomes) {
  std::vector<g3pvm::BytecodeProgram> out;
  out.reserve(genomes.size());
  const std::vector<std::string> preset_locals = {"x"};
  for (const ProgramGenome& genome : genomes) {
    out.push_back(g3pvm::evo::compile_for_eval_with_preset_locals(genome, preset_locals));
  }
  return out;
}

void prepare_eval_data(int eval_cases, std::vector<InputCase>* shared_cases, std::vector<Value>* shared_answer) {
  shared_cases->clear();
  shared_answer->clear();
  shared_cases->reserve(static_cast<std::size_t>(eval_cases));
  shared_answer->reserve(static_cast<std::size_t>(eval_cases));
  const int half = eval_cases / 2;
  for (int i = 0; i < eval_cases; ++i) {
    const int x = i - half;
    shared_cases->push_back(InputCase{LocalBinding{0, Value::from_int(x)}});
    shared_answer->push_back(Value::from_int(x + 1));
  }
}

struct PreprocessData {
  std::vector<std::vector<int>> subtree_ends;
  std::vector<FormalCandidateSet> candidates;
  std::vector<std::vector<PlainNode>> donor_pool;
  std::vector<std::vector<std::uint64_t>> donor_name_ids;
  std::vector<std::vector<Value>> donor_consts;
};

PreprocessData run_preprocess(const std::vector<ProgramGenome>& genomes,
                              const BenchConfig& cfg,
                              TimingAggregate* agg) {
  PreprocessData out;
  out.subtree_ends.resize(genomes.size());
  out.candidates.resize(genomes.size());

  const auto subtree_t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < genomes.size(); ++i) {
    out.subtree_ends[i] = build_subtree_end(genomes[i].ast.nodes);
  }
  const auto subtree_t1 = std::chrono::steady_clock::now();

  const auto cand_t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < genomes.size(); ++i) {
    out.candidates[i] = collect_formal_candidates(genomes[i].ast.nodes, out.subtree_ends[i], cfg.candidates_per_program);
  }
  const auto cand_t1 = std::chrono::steady_clock::now();

  const auto donor_t0 = std::chrono::steady_clock::now();
  out.donor_pool.reserve(static_cast<std::size_t>(cfg.donor_pool_size));
  out.donor_name_ids.reserve(static_cast<std::size_t>(cfg.donor_pool_size));
  out.donor_consts.reserve(static_cast<std::size_t>(cfg.donor_pool_size));
  for (int i = 0; i < cfg.donor_pool_size; ++i) {
    const std::uint64_t s = mix_seed(cfg.seed, static_cast<std::uint64_t>(i + 1));
    const int prog_idx = static_cast<int>(s % static_cast<std::uint64_t>(genomes.size()));
    const FormalCandidateSet& cand = out.candidates[static_cast<std::size_t>(prog_idx)];
    RType donor_type = RType::Num;
    if (!cand.expr.empty()) {
      const CandidateRange pick =
          cand.expr[static_cast<std::size_t>((s >> 8) % static_cast<std::uint64_t>(cand.expr.size()))];
      donor_type = static_cast<RType>(pick.aux);
    }
    out.donor_pool.push_back(make_donor_nodes_for_type(s, donor_type, cfg.max_donor_nodes));
    std::vector<std::uint64_t> donor_names;
    std::vector<Value> donor_consts;
    donor_names.push_back(hash_name("x"));
    donor_consts.push_back(make_const_value(s ^ 0xabcddcbaULL, donor_type));
    out.donor_name_ids.push_back(std::move(donor_names));
    out.donor_consts.push_back(std::move(donor_consts));
  }
  const auto donor_t1 = std::chrono::steady_clock::now();

  agg->cpu_subtree_ms += ms_between(subtree_t0, subtree_t1);
  agg->cpu_candidate_ms += ms_between(cand_t0, cand_t1);
  agg->cpu_donor_ms += ms_between(donor_t0, donor_t1);
  return out;
}

PackedHostData pack_host_data(const std::vector<ProgramGenome>& genomes,
                              const PreprocessData& prep,
                              const BenchConfig& cfg,
                              TimingAggregate* agg) {
  PackedHostData out;
  out.program_nodes.resize(static_cast<std::size_t>(cfg.population_size * cfg.max_nodes));
  out.metas.resize(static_cast<std::size_t>(cfg.population_size));
  out.candidates.resize(static_cast<std::size_t>(cfg.population_size * cfg.candidates_per_program),
                        CandidateRange{-1, -1, -1, -1});
  out.program_name_ids.resize(static_cast<std::size_t>(cfg.population_size * cfg.max_names), 0ULL);
  out.program_consts.resize(static_cast<std::size_t>(cfg.population_size * cfg.max_consts), Value::none());
  out.donor_nodes.resize(static_cast<std::size_t>(cfg.donor_pool_size * cfg.max_donor_nodes));
  out.donor_lens.resize(static_cast<std::size_t>(cfg.donor_pool_size), 0);
  out.donor_name_ids.resize(static_cast<std::size_t>(cfg.donor_pool_size * cfg.max_names), 0ULL);
  out.donor_name_counts.resize(static_cast<std::size_t>(cfg.donor_pool_size), 0);
  out.donor_consts.resize(static_cast<std::size_t>(cfg.donor_pool_size * cfg.max_consts), Value::none());
  out.donor_const_counts.resize(static_cast<std::size_t>(cfg.donor_pool_size), 0);

  const auto prog_t0 = std::chrono::steady_clock::now();
  for (int p = 0; p < cfg.population_size; ++p) {
    const auto& nodes = genomes[static_cast<std::size_t>(p)].ast.nodes;
    const int used = std::min<int>(static_cast<int>(nodes.size()), cfg.max_nodes);
    out.metas[static_cast<std::size_t>(p)].used_len = used;
    out.metas[static_cast<std::size_t>(p)].name_count =
        std::min<int>(static_cast<int>(genomes[static_cast<std::size_t>(p)].ast.names.size()), cfg.max_names);
    out.metas[static_cast<std::size_t>(p)].const_count =
        std::min<int>(static_cast<int>(genomes[static_cast<std::size_t>(p)].ast.consts.size()), cfg.max_consts);
    const std::size_t base = static_cast<std::size_t>(p * cfg.max_nodes);
    for (int i = 0; i < used; ++i) {
      const AstNode& n = nodes[static_cast<std::size_t>(i)];
      out.program_nodes[base + static_cast<std::size_t>(i)] = PlainNode{static_cast<int>(n.kind), n.i0, n.i1};
    }
    const std::size_t name_base = static_cast<std::size_t>(p * cfg.max_names);
    for (int i = 0; i < out.metas[static_cast<std::size_t>(p)].name_count; ++i) {
      out.program_name_ids[name_base + static_cast<std::size_t>(i)] =
          hash_name(genomes[static_cast<std::size_t>(p)].ast.names[static_cast<std::size_t>(i)]);
    }
    const std::size_t const_base = static_cast<std::size_t>(p * cfg.max_consts);
    for (int i = 0; i < out.metas[static_cast<std::size_t>(p)].const_count; ++i) {
      out.program_consts[const_base + static_cast<std::size_t>(i)] =
          genomes[static_cast<std::size_t>(p)].ast.consts[static_cast<std::size_t>(i)];
    }
  }
  const auto prog_t1 = std::chrono::steady_clock::now();

  const auto cand_t0 = std::chrono::steady_clock::now();
  for (int p = 0; p < cfg.population_size; ++p) {
    const auto& set = prep.candidates[static_cast<std::size_t>(p)];
    std::vector<CandidateRange> cand;
    cand.reserve(static_cast<std::size_t>(cfg.candidates_per_program));
    for (const CandidateRange& one : set.expr) {
      cand.push_back(one);
    }
    for (const CandidateRange& one : set.stmt) {
      if (static_cast<int>(cand.size()) >= cfg.candidates_per_program) {
        break;
      }
      cand.push_back(one);
    }
    const int used = std::min<int>(static_cast<int>(cand.size()), cfg.candidates_per_program);
    const std::size_t base = static_cast<std::size_t>(p * cfg.candidates_per_program);
    for (int i = 0; i < used; ++i) {
      out.candidates[base + static_cast<std::size_t>(i)] = cand[static_cast<std::size_t>(i)];
    }
  }
  const auto cand_t1 = std::chrono::steady_clock::now();

  const auto donor_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.donor_pool_size; ++i) {
    const auto& donor = prep.donor_pool[static_cast<std::size_t>(i)];
    const int used = std::min<int>(static_cast<int>(donor.size()), cfg.max_donor_nodes);
    out.donor_lens[static_cast<std::size_t>(i)] = used;
    const std::size_t base = static_cast<std::size_t>(i * cfg.max_donor_nodes);
    for (int j = 0; j < used; ++j) {
      out.donor_nodes[base + static_cast<std::size_t>(j)] = donor[static_cast<std::size_t>(j)];
    }
    const auto& donor_names = prep.donor_name_ids[static_cast<std::size_t>(i)];
    out.donor_name_counts[static_cast<std::size_t>(i)] =
        std::min<int>(static_cast<int>(donor_names.size()), cfg.max_names);
    for (int j = 0; j < out.donor_name_counts[static_cast<std::size_t>(i)]; ++j) {
      out.donor_name_ids[static_cast<std::size_t>(i * cfg.max_names + j)] = donor_names[static_cast<std::size_t>(j)];
    }
    const auto& donor_consts = prep.donor_consts[static_cast<std::size_t>(i)];
    out.donor_const_counts[static_cast<std::size_t>(i)] =
        std::min<int>(static_cast<int>(donor_consts.size()), cfg.max_consts);
    for (int j = 0; j < out.donor_const_counts[static_cast<std::size_t>(i)]; ++j) {
      out.donor_consts[static_cast<std::size_t>(i * cfg.max_consts + j)] = donor_consts[static_cast<std::size_t>(j)];
    }
  }
  const auto donor_t1 = std::chrono::steady_clock::now();

  agg->pack_program_ms += ms_between(prog_t0, prog_t1);
  agg->pack_candidate_ms += ms_between(cand_t0, cand_t1);
  agg->pack_donor_ms += ms_between(donor_t0, donor_t1);
  return out;
}

void ensure_cuda(cudaError_t code, const char* what) {
  if (code != cudaSuccess) {
    std::ostringstream oss;
    oss << what << ": " << cudaGetErrorString(code);
    throw std::runtime_error(oss.str());
  }
}

void allocate_device_buffers(DeviceBuffers* dev, const BenchConfig& cfg) {
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_program_nodes),
                         sizeof(PlainNode) * static_cast<std::size_t>(cfg.population_size * cfg.max_nodes)),
              "cudaMalloc d_program_nodes");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_metas),
                         sizeof(PackedProgramMeta) * static_cast<std::size_t>(cfg.population_size)),
              "cudaMalloc d_metas");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_candidates),
                         sizeof(CandidateRange) *
                             static_cast<std::size_t>(cfg.population_size * cfg.candidates_per_program)),
              "cudaMalloc d_candidates");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_program_name_ids),
                         sizeof(std::uint64_t) * static_cast<std::size_t>(cfg.population_size * cfg.max_names)),
              "cudaMalloc d_program_name_ids");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_program_consts),
                         sizeof(Value) * static_cast<std::size_t>(cfg.population_size * cfg.max_consts)),
              "cudaMalloc d_program_consts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_nodes),
                         sizeof(PlainNode) * static_cast<std::size_t>(cfg.donor_pool_size * cfg.max_donor_nodes)),
              "cudaMalloc d_donor_nodes");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_lens),
                         sizeof(int) * static_cast<std::size_t>(cfg.donor_pool_size)),
              "cudaMalloc d_donor_lens");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_name_ids),
                         sizeof(std::uint64_t) * static_cast<std::size_t>(cfg.donor_pool_size * cfg.max_names)),
              "cudaMalloc d_donor_name_ids");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_name_counts),
                         sizeof(int) * static_cast<std::size_t>(cfg.donor_pool_size)),
              "cudaMalloc d_donor_name_counts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_consts),
                         sizeof(Value) * static_cast<std::size_t>(cfg.donor_pool_size * cfg.max_consts)),
              "cudaMalloc d_donor_consts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_const_counts),
                         sizeof(int) * static_cast<std::size_t>(cfg.donor_pool_size)),
              "cudaMalloc d_donor_const_counts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_fitness),
                         sizeof(int) * static_cast<std::size_t>(cfg.population_size)),
              "cudaMalloc d_fitness");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_parent_a),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count)),
              "cudaMalloc d_parent_a");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_parent_b),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count)),
              "cudaMalloc d_parent_b");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_cand_a),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count)),
              "cudaMalloc d_cand_a");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_cand_b),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count)),
              "cudaMalloc d_cand_b");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_donor_idx),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count)),
              "cudaMalloc d_donor_idx");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_is_mutation),
                         sizeof(unsigned char) * static_cast<std::size_t>(cfg.child_count)),
              "cudaMalloc d_is_mutation");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_nodes),
                         sizeof(PlainNode) * static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_nodes)),
              "cudaMalloc d_child_nodes");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_used_len),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count * 2)),
              "cudaMalloc d_child_used_len");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_name_ids),
                         sizeof(std::uint64_t) * static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_names)),
              "cudaMalloc d_child_name_ids");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_name_counts),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count * 2)),
              "cudaMalloc d_child_name_counts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_consts),
                         sizeof(Value) * static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_consts)),
              "cudaMalloc d_child_consts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_const_counts),
                         sizeof(int) * static_cast<std::size_t>(cfg.child_count * 2)),
              "cudaMalloc d_child_const_counts");
  ensure_cuda(cudaMalloc(reinterpret_cast<void**>(&dev->d_child_cheap_valid),
                         sizeof(unsigned char) * static_cast<std::size_t>(cfg.child_count * 2)),
              "cudaMalloc d_child_cheap_valid");
}

void copy_host_to_device(const PackedHostData& packed, DeviceBuffers* dev, const BenchConfig& cfg, TimingAggregate* agg) {
  const auto prog_t0 = std::chrono::steady_clock::now();
  ensure_cuda(cudaMemcpy(dev->d_program_nodes, packed.program_nodes.data(),
                         sizeof(PlainNode) * packed.program_nodes.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy programs");
  ensure_cuda(cudaMemcpy(dev->d_metas, packed.metas.data(),
                         sizeof(PackedProgramMeta) * packed.metas.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy metas");
  ensure_cuda(cudaMemcpy(dev->d_program_name_ids, packed.program_name_ids.data(),
                         sizeof(std::uint64_t) * packed.program_name_ids.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy program name ids");
  ensure_cuda(cudaMemcpy(dev->d_program_consts, packed.program_consts.data(),
                         sizeof(Value) * packed.program_consts.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy program consts");
  ensure_cuda(cudaDeviceSynchronize(), "cuda sync after programs copy");
  const auto prog_t1 = std::chrono::steady_clock::now();

  const auto cand_t0 = std::chrono::steady_clock::now();
  ensure_cuda(cudaMemcpy(dev->d_candidates, packed.candidates.data(),
                         sizeof(CandidateRange) * packed.candidates.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy candidates");
  ensure_cuda(cudaDeviceSynchronize(), "cuda sync after candidates copy");
  const auto cand_t1 = std::chrono::steady_clock::now();

  const auto donor_t0 = std::chrono::steady_clock::now();
  ensure_cuda(cudaMemcpy(dev->d_donor_nodes, packed.donor_nodes.data(),
                         sizeof(PlainNode) * packed.donor_nodes.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy donor nodes");
  ensure_cuda(cudaMemcpy(dev->d_donor_lens, packed.donor_lens.data(),
                         sizeof(int) * packed.donor_lens.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy donor lens");
  ensure_cuda(cudaMemcpy(dev->d_donor_name_ids, packed.donor_name_ids.data(),
                         sizeof(std::uint64_t) * packed.donor_name_ids.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy donor name ids");
  ensure_cuda(cudaMemcpy(dev->d_donor_name_counts, packed.donor_name_counts.data(),
                         sizeof(int) * packed.donor_name_counts.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy donor name counts");
  ensure_cuda(cudaMemcpy(dev->d_donor_consts, packed.donor_consts.data(),
                         sizeof(Value) * packed.donor_consts.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy donor consts");
  ensure_cuda(cudaMemcpy(dev->d_donor_const_counts, packed.donor_const_counts.data(),
                         sizeof(int) * packed.donor_const_counts.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy donor const counts");
  ensure_cuda(cudaDeviceSynchronize(), "cuda sync after donors copy");
  const auto donor_t1 = std::chrono::steady_clock::now();

  agg->h2d_program_ms += ms_between(prog_t0, prog_t1);
  agg->h2d_candidate_ms += ms_between(cand_t0, cand_t1);
  agg->h2d_donor_ms += ms_between(donor_t0, donor_t1);
}

void copy_host_to_device_async(const PackedHostData& packed, DeviceBuffers* dev, const BenchConfig& cfg, cudaStream_t stream) {
  ensure_cuda(cudaMemcpyAsync(dev->d_program_nodes, packed.program_nodes.data(),
                              sizeof(PlainNode) * packed.program_nodes.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync programs");
  ensure_cuda(cudaMemcpyAsync(dev->d_metas, packed.metas.data(),
                              sizeof(PackedProgramMeta) * packed.metas.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync metas");
  ensure_cuda(cudaMemcpyAsync(dev->d_program_name_ids, packed.program_name_ids.data(),
                              sizeof(std::uint64_t) * packed.program_name_ids.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync program name ids");
  ensure_cuda(cudaMemcpyAsync(dev->d_program_consts, packed.program_consts.data(),
                              sizeof(Value) * packed.program_consts.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync program consts");
  ensure_cuda(cudaMemcpyAsync(dev->d_candidates, packed.candidates.data(),
                              sizeof(CandidateRange) * packed.candidates.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync candidates");
  ensure_cuda(cudaMemcpyAsync(dev->d_donor_nodes, packed.donor_nodes.data(),
                              sizeof(PlainNode) * packed.donor_nodes.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync donor nodes");
  ensure_cuda(cudaMemcpyAsync(dev->d_donor_lens, packed.donor_lens.data(),
                              sizeof(int) * packed.donor_lens.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync donor lens");
  ensure_cuda(cudaMemcpyAsync(dev->d_donor_name_ids, packed.donor_name_ids.data(),
                              sizeof(std::uint64_t) * packed.donor_name_ids.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync donor name ids");
  ensure_cuda(cudaMemcpyAsync(dev->d_donor_name_counts, packed.donor_name_counts.data(),
                              sizeof(int) * packed.donor_name_counts.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync donor name counts");
  ensure_cuda(cudaMemcpyAsync(dev->d_donor_consts, packed.donor_consts.data(),
                              sizeof(Value) * packed.donor_consts.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync donor consts");
  ensure_cuda(cudaMemcpyAsync(dev->d_donor_const_counts, packed.donor_const_counts.data(),
                              sizeof(int) * packed.donor_const_counts.size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync donor const counts");
}

__host__ __device__ std::uint64_t hash64(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

__global__ void tournament_select_kernel(const int* fitness,
                                         int population_size,
                                         int child_count,
                                         int candidates_per_program,
                                         int donor_pool_size,
                                         int tournament_k,
                                         double mutation_ratio,
                                         std::uint64_t seed,
                                         int* parent_a,
                                         int* parent_b,
                                         int* cand_a,
                                         int* cand_b,
                                         int* donor_idx,
                                         unsigned char* is_mutation) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= child_count) {
    return;
  }
  const std::uint64_t base = hash64(seed + static_cast<std::uint64_t>(idx) * 0x9e3779b97f4a7c15ULL);
  auto run_tournament = [&](std::uint64_t local_seed) {
    int best = static_cast<int>(local_seed % static_cast<std::uint64_t>(population_size));
    int best_fit = fitness[best];
    for (int i = 1; i < tournament_k; ++i) {
      local_seed = hash64(local_seed + static_cast<std::uint64_t>(i));
      const int cand = static_cast<int>(local_seed % static_cast<std::uint64_t>(population_size));
      const int cand_fit = fitness[cand];
      if (cand_fit > best_fit) {
        best = cand;
        best_fit = cand_fit;
      }
    }
    return best;
  };

  const int pa = run_tournament(base);
  const int pb = run_tournament(hash64(base ^ 0x123456789abcdefULL));
  parent_a[idx] = pa;
  parent_b[idx] = pb;

  const std::uint64_t cseed = hash64(base ^ 0xfeedbeefULL);
  cand_a[idx] = static_cast<int>(cseed % static_cast<std::uint64_t>(candidates_per_program));
  cand_b[idx] = static_cast<int>((cseed >> 7) % static_cast<std::uint64_t>(candidates_per_program));
  donor_idx[idx] = static_cast<int>((cseed >> 17) % static_cast<std::uint64_t>(donor_pool_size));

  const double pick = static_cast<double>(cseed & 0xffffULL) / 65535.0;
  is_mutation[idx] = (pick < mutation_ratio) ? 1 : 0;
}

__device__ void copy_range(const PlainNode* src, int src_start, int src_stop, PlainNode* dst, int dst_start) {
  for (int i = src_start; i < src_stop; ++i) {
    dst[dst_start + (i - src_start)] = src[i];
  }
}

__device__ int remap_name_id(std::uint64_t name_id, std::uint64_t* child_names, int* child_name_count, int max_names) {
  for (int i = 0; i < *child_name_count; ++i) {
    if (child_names[i] == name_id) {
      return i;
    }
  }
  if (*child_name_count < max_names) {
    const int idx = *child_name_count;
    child_names[idx] = name_id;
    *child_name_count += 1;
    return idx;
  }
  return 0;
}

__device__ int remap_const_value(const Value& v, Value* child_consts, int* child_const_count, int max_consts) {
  for (int i = 0; i < *child_const_count; ++i) {
    if (value_equal_simple(child_consts[i], v)) {
      return i;
    }
  }
  if (*child_const_count < max_consts) {
    const int idx = *child_const_count;
    child_consts[idx] = v;
    *child_const_count += 1;
    return idx;
  }
  return 0;
}

__device__ PlainNode remap_node_for_child(const PlainNode& in,
                                          const std::uint64_t* source_names,
                                          const Value* source_consts,
                                          std::uint64_t* child_names,
                                          int* child_name_count,
                                          Value* child_consts,
                                          int* child_const_count,
                                          int max_names,
                                          int max_consts) {
  PlainNode out = in;
  const NodeKind kind = static_cast<NodeKind>(in.kind);
  if (kind == NodeKind::CONST) {
    out.i0 = remap_const_value(source_consts[in.i0], child_consts, child_const_count, max_consts);
  } else if (kind == NodeKind::VAR || kind == NodeKind::ASSIGN || kind == NodeKind::FOR_RANGE) {
    out.i0 = remap_name_id(source_names[in.i0], child_names, child_name_count, max_names);
  }
  return out;
}

__global__ void cheap_validate_kernel(const PlainNode* child_nodes,
                                      const int* child_used_len,
                                      const int* child_name_counts,
                                      const int* child_const_counts,
                                      int max_nodes,
                                      int max_for_k,
                                      int max_names,
                                      int max_consts,
                                      int total_children,
                                      unsigned char* child_cheap_valid) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_children) {
    return;
  }

  const int used_len = child_used_len[idx];
  const int name_count = child_name_counts[idx];
  const int const_count = child_const_counts[idx];
  bool ok = true;

  if (used_len <= 0 || used_len > max_nodes) ok = false;
  if (name_count < 0 || name_count > max_names) ok = false;
  if (const_count < 0 || const_count > max_consts) ok = false;

  if (ok) {
    const PlainNode* nodes = child_nodes + idx * max_nodes;
    if (nodes[0].kind != static_cast<int>(NodeKind::PROGRAM)) {
      ok = false;
    }
    int pending = 1;
    for (int i = 0; ok && i < used_len; ++i) {
      if (pending <= 0) {
        ok = false;
        break;
      }
      const PlainNode& n = nodes[i];
      const int arity = node_arity_checked(n.kind);
      if (arity < 0) {
        ok = false;
        break;
      }
      if ((n.kind == static_cast<int>(NodeKind::CONST) && (n.i0 < 0 || n.i0 >= const_count)) ||
          ((n.kind == static_cast<int>(NodeKind::VAR) || n.kind == static_cast<int>(NodeKind::ASSIGN) ||
            n.kind == static_cast<int>(NodeKind::FOR_RANGE)) &&
           (n.i0 < 0 || n.i0 >= name_count))) {
        ok = false;
        break;
      }
      if (n.kind == static_cast<int>(NodeKind::FOR_RANGE) && (n.i1 < 0 || n.i1 > max_for_k)) {
        ok = false;
        break;
      }
      pending -= 1;
      pending += arity;
    }
    if (ok && pending != 0) {
      ok = false;
    }
  }

  child_cheap_valid[idx] = ok ? 1 : 0;
}

__global__ void variation_kernel(const PlainNode* program_nodes,
                                 const PackedProgramMeta* metas,
                                 const CandidateRange* candidates,
                                 const std::uint64_t* program_name_ids,
                                 const Value* program_consts,
                                 const PlainNode* donor_nodes,
                                 const int* donor_lens,
                                 const std::uint64_t* donor_name_ids,
                                 const int* donor_name_counts,
                                 const Value* donor_consts,
                                 const int* donor_const_counts,
                                 int max_nodes,
                                 int candidates_per_program,
                                 int max_donor_nodes,
                                 int max_names,
                                 int max_consts,
                                 int child_count,
                                 const int* parent_a,
                                 const int* parent_b,
                                 const int* cand_a,
                                 const int* cand_b,
                                 const int* donor_idx,
                                 const unsigned char* is_mutation,
                                 PlainNode* child_nodes,
                                 int* child_used_len_out,
                                 std::uint64_t* child_name_ids_out,
                                 int* child_name_counts_out,
                                 Value* child_consts_out,
                                 int* child_const_counts_out) {
  const int pair_idx = static_cast<int>(blockIdx.x);
  if (pair_idx >= child_count) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);

  const int pa = parent_a[pair_idx];
  const int pb = parent_b[pair_idx];
  const int base_a = pa * max_nodes;
  const int base_b = pb * max_nodes;
  const CandidateRange ra = candidates[pa * candidates_per_program + cand_a[pair_idx]];
  const int child_base_ab = (pair_idx * 2) * max_nodes;
  const int child_base_ba = (pair_idx * 2 + 1) * max_nodes;
  const int len_a = metas[pa].used_len;
  const int len_b = metas[pb].used_len;
  const int name_count_a = metas[pa].name_count;
  const int const_count_a = metas[pa].const_count;
  const int name_count_b = metas[pb].name_count;
  const int const_count_b = metas[pb].const_count;

  int prefix_a = 0;
  int replace_a = 0;
  int donor_len_for_a = 0;
  const PlainNode* donor_ptr_for_a = nullptr;
  int prefix_b = 0;
  int replace_b = 0;
  int donor_len_for_b = 0;
  const PlainNode* donor_ptr_for_b = nullptr;

  if (is_mutation[pair_idx]) {
    prefix_a = max(0, ra.start);
    replace_a = max(0, ra.stop - ra.start);
    const int did = donor_idx[pair_idx];
    donor_len_for_a = donor_lens[did];
    donor_ptr_for_a = donor_nodes + did * max_donor_nodes;

    prefix_b = 0;
    replace_b = donor_len_for_a;
    donor_len_for_b = replace_a;
    donor_ptr_for_b = program_nodes + base_a + prefix_a;
  } else {
    const CandidateRange rb = candidates[pb * candidates_per_program + cand_b[pair_idx]];
    prefix_a = max(0, ra.start);
    replace_a = max(0, ra.stop - ra.start);
    donor_len_for_a = max(0, rb.stop - rb.start);
    donor_ptr_for_a = program_nodes + base_b + rb.start;

    prefix_b = max(0, rb.start);
    replace_b = max(0, rb.stop - rb.start);
    donor_len_for_b = replace_a;
    donor_ptr_for_b = program_nodes + base_a + prefix_a;
  }

  __shared__ std::uint64_t child_a_names[32];
  __shared__ std::uint64_t child_b_names[32];
  __shared__ Value child_a_consts[32];
  __shared__ Value child_b_consts[32];
  __shared__ int child_a_name_count;
  __shared__ int child_b_name_count;
  __shared__ int child_a_const_count;
  __shared__ int child_b_const_count;
  if (tid == 0) {
    child_a_name_count = min(name_count_a, max_names);
    child_a_const_count = min(const_count_a, max_consts);
    child_b_name_count = is_mutation[pair_idx] ? min(donor_name_counts[donor_idx[pair_idx]], max_names)
                                               : min(name_count_b, max_names);
    child_b_const_count = is_mutation[pair_idx] ? min(donor_const_counts[donor_idx[pair_idx]], max_consts)
                                                : min(const_count_b, max_consts);
    for (int i = 0; i < child_a_name_count; ++i) child_a_names[i] = program_name_ids[pa * max_names + i];
    for (int i = 0; i < child_a_const_count; ++i) child_a_consts[i] = program_consts[pa * max_consts + i];
    if (is_mutation[pair_idx]) {
      for (int i = 0; i < child_b_name_count; ++i) child_b_names[i] = donor_name_ids[donor_idx[pair_idx] * max_names + i];
      for (int i = 0; i < child_b_const_count; ++i) child_b_consts[i] = donor_consts[donor_idx[pair_idx] * max_consts + i];
    } else {
      for (int i = 0; i < child_b_name_count; ++i) child_b_names[i] = program_name_ids[pb * max_names + i];
      for (int i = 0; i < child_b_const_count; ++i) child_b_consts[i] = program_consts[pb * max_consts + i];
    }
  }
  __syncthreads();

  const int suffix_a_start = prefix_a + replace_a;
  const int suffix_a_len = max(0, len_a - suffix_a_start);
  const int out_a_len = prefix_a + donor_len_for_a + suffix_a_len;

  if (out_a_len > max_nodes || prefix_a >= len_a || replace_a <= 0) {
    for (int i = tid; i < len_a; i += blockDim.x) {
      child_nodes[child_base_ab + i] = program_nodes[base_a + i];
    }
    if (tid == 0) {
      child_used_len_out[pair_idx * 2] = len_a;
    }
  } else {
    for (int i = tid; i < prefix_a; i += blockDim.x) {
      child_nodes[child_base_ab + i] = program_nodes[base_a + i];
    }
    for (int i = tid; i < donor_len_for_a; i += blockDim.x) {
      const std::uint64_t* src_names =
          is_mutation[pair_idx] ? (donor_name_ids + donor_idx[pair_idx] * max_names) : (program_name_ids + pb * max_names);
      const Value* src_consts =
          is_mutation[pair_idx] ? (donor_consts + donor_idx[pair_idx] * max_consts) : (program_consts + pb * max_consts);
      child_nodes[child_base_ab + prefix_a + i] =
          remap_node_for_child(donor_ptr_for_a[i], src_names, src_consts, child_a_names, &child_a_name_count,
                               child_a_consts, &child_a_const_count, max_names, max_consts);
    }
    for (int i = tid; i < suffix_a_len; i += blockDim.x) {
      child_nodes[child_base_ab + prefix_a + donor_len_for_a + i] = program_nodes[base_a + suffix_a_start + i];
    }
  }

  const int source_b_len = is_mutation[pair_idx] ? max(0, replace_b) : len_b;
  const int suffix_b_start = prefix_b + replace_b;
  const int suffix_b_len = max(0, source_b_len - suffix_b_start);
  const int out_b_len = prefix_b + donor_len_for_b + suffix_b_len;

  if (out_b_len > max_nodes || prefix_b > source_b_len || replace_b < 0) {
    const PlainNode* source_b = is_mutation[pair_idx] ? donor_ptr_for_a : (program_nodes + base_b);
    for (int i = tid; i < source_b_len; i += blockDim.x) {
      child_nodes[child_base_ba + i] = source_b[i];
    }
    if (tid == 0) {
      child_used_len_out[pair_idx * 2 + 1] = source_b_len;
    }
    return;
  }

  const PlainNode* source_b = is_mutation[pair_idx] ? donor_ptr_for_a : (program_nodes + base_b);
  for (int i = tid; i < prefix_b; i += blockDim.x) {
    child_nodes[child_base_ba + i] = source_b[i];
  }
  for (int i = tid; i < donor_len_for_b; i += blockDim.x) {
    const std::uint64_t* src_names = program_name_ids + pa * max_names;
    const Value* src_consts = program_consts + pa * max_consts;
    child_nodes[child_base_ba + prefix_b + i] =
        remap_node_for_child(donor_ptr_for_b[i], src_names, src_consts, child_b_names, &child_b_name_count,
                             child_b_consts, &child_b_const_count, max_names, max_consts);
  }
  for (int i = tid; i < suffix_b_len; i += blockDim.x) {
    child_nodes[child_base_ba + prefix_b + donor_len_for_b + i] = source_b[suffix_b_start + i];
  }
  __syncthreads();
  if (tid == 0) {
    const int out_idx_a = pair_idx * 2;
    const int out_idx_b = pair_idx * 2 + 1;
    child_used_len_out[out_idx_a] = (out_a_len > max_nodes || prefix_a >= len_a || replace_a <= 0) ? len_a : out_a_len;
    child_used_len_out[out_idx_b] = out_b_len;
    child_name_counts_out[out_idx_a] = child_a_name_count;
    child_const_counts_out[out_idx_a] = child_a_const_count;
    child_name_counts_out[out_idx_b] = child_b_name_count;
    child_const_counts_out[out_idx_b] = child_b_const_count;
    for (int i = 0; i < child_a_name_count; ++i) {
      child_name_ids_out[out_idx_a * max_names + i] = child_a_names[i];
    }
    for (int i = 0; i < child_a_const_count; ++i) {
      child_consts_out[out_idx_a * max_consts + i] = child_a_consts[i];
    }
    for (int i = 0; i < child_b_name_count; ++i) {
      child_name_ids_out[out_idx_b * max_names + i] = child_b_names[i];
    }
    for (int i = 0; i < child_b_const_count; ++i) {
      child_consts_out[out_idx_b * max_consts + i] = child_b_consts[i];
    }
  }
}

void cpu_tournament_select(const std::vector<int>& fitness,
                           const BenchConfig& cfg,
                           std::vector<int>* parent_a,
                           std::vector<int>* parent_b,
                           std::vector<int>* cand_a,
                           std::vector<int>* cand_b,
                           std::vector<int>* donor_idx,
                           std::vector<unsigned char>* is_mutation) {
  parent_a->resize(static_cast<std::size_t>(cfg.child_count));
  parent_b->resize(static_cast<std::size_t>(cfg.child_count));
  cand_a->resize(static_cast<std::size_t>(cfg.child_count));
  cand_b->resize(static_cast<std::size_t>(cfg.child_count));
  donor_idx->resize(static_cast<std::size_t>(cfg.child_count));
  is_mutation->resize(static_cast<std::size_t>(cfg.child_count));
  for (int idx = 0; idx < cfg.child_count; ++idx) {
    const std::uint64_t base = hash64(cfg.seed + static_cast<std::uint64_t>(idx) * 0x9e3779b97f4a7c15ULL);
    auto run_tournament = [&](std::uint64_t local_seed) {
      int best = static_cast<int>(local_seed % static_cast<std::uint64_t>(cfg.population_size));
      int best_fit = fitness[static_cast<std::size_t>(best)];
      for (int i = 1; i < cfg.tournament_k; ++i) {
        local_seed = hash64(local_seed + static_cast<std::uint64_t>(i));
        const int cand = static_cast<int>(local_seed % static_cast<std::uint64_t>(cfg.population_size));
        if (fitness[static_cast<std::size_t>(cand)] > best_fit) {
          best = cand;
          best_fit = fitness[static_cast<std::size_t>(cand)];
        }
      }
      return best;
    };
    (*parent_a)[static_cast<std::size_t>(idx)] = run_tournament(base);
    (*parent_b)[static_cast<std::size_t>(idx)] = run_tournament(hash64(base ^ 0x123456789abcdefULL));
    const std::uint64_t cseed = hash64(base ^ 0xfeedbeefULL);
    (*cand_a)[static_cast<std::size_t>(idx)] =
        static_cast<int>(cseed % static_cast<std::uint64_t>(cfg.candidates_per_program));
    (*cand_b)[static_cast<std::size_t>(idx)] =
        static_cast<int>((cseed >> 7) % static_cast<std::uint64_t>(cfg.candidates_per_program));
    (*donor_idx)[static_cast<std::size_t>(idx)] =
        static_cast<int>((cseed >> 17) % static_cast<std::uint64_t>(cfg.donor_pool_size));
    const double pick = static_cast<double>(cseed & 0xffffULL) / 65535.0;
    (*is_mutation)[static_cast<std::size_t>(idx)] = (pick < cfg.mutation_ratio) ? 1 : 0;
  }
}

int cpu_remap_name_id(std::uint64_t name_id, std::vector<std::uint64_t>* child_names, int max_names) {
  for (std::size_t i = 0; i < child_names->size(); ++i) {
    if ((*child_names)[i] == name_id) return static_cast<int>(i);
  }
  if (static_cast<int>(child_names->size()) < max_names) {
    child_names->push_back(name_id);
    return static_cast<int>(child_names->size() - 1);
  }
  return 0;
}

int cpu_remap_const_value(const Value& v, std::vector<Value>* child_consts, int max_consts) {
  for (std::size_t i = 0; i < child_consts->size(); ++i) {
    if (value_equal_simple((*child_consts)[i], v)) return static_cast<int>(i);
  }
  if (static_cast<int>(child_consts->size()) < max_consts) {
    child_consts->push_back(v);
    return static_cast<int>(child_consts->size() - 1);
  }
  return 0;
}

PlainNode cpu_remap_node_for_child(const PlainNode& in,
                                   const std::uint64_t* source_names,
                                   const Value* source_consts,
                                   std::vector<std::uint64_t>* child_names,
                                   std::vector<Value>* child_consts,
                                   int max_names,
                                   int max_consts) {
  PlainNode out = in;
  const NodeKind kind = static_cast<NodeKind>(in.kind);
  if (kind == NodeKind::CONST) {
    out.i0 = cpu_remap_const_value(source_consts[in.i0], child_consts, max_consts);
  } else if (kind == NodeKind::VAR || kind == NodeKind::ASSIGN || kind == NodeKind::FOR_RANGE) {
    out.i0 = cpu_remap_name_id(source_names[in.i0], child_names, max_names);
  }
  return out;
}

ProgramGenome build_genome_from_flat(const PlainNode* nodes,
                                     int node_count,
                                     const std::uint64_t* name_ids,
                                     int name_count,
                                     const Value* consts,
                                     int const_count) {
  ProgramGenome genome;
  genome.ast.version = "ast-prefix-v1";
  genome.ast.nodes.reserve(static_cast<std::size_t>(node_count));
  genome.ast.names.reserve(static_cast<std::size_t>(name_count));
  genome.ast.consts.reserve(static_cast<std::size_t>(const_count));
  for (int i = 0; i < node_count; ++i) {
    const PlainNode& n = nodes[i];
    genome.ast.nodes.push_back(AstNode{static_cast<NodeKind>(n.kind), n.i0, n.i1});
  }
  for (int i = 0; i < name_count; ++i) {
    std::ostringstream oss;
    oss << "n_" << std::hex << name_ids[i];
    genome.ast.names.push_back(oss.str());
  }
  for (int i = 0; i < const_count; ++i) {
    genome.ast.consts.push_back(consts[i]);
  }
  genome.meta.node_count = node_count;
  return genome;
}

std::uint64_t checksum_genome_nodes(const ProgramGenome& genome) {
  std::uint64_t checksum = 0;
  for (const AstNode& n : genome.ast.nodes) {
    checksum ^= static_cast<std::uint64_t>(static_cast<int>(n.kind) + 131 * n.i0 + 977 * n.i1);
    checksum = hash64(checksum + 0x9e3779b97f4a7c15ULL);
  }
  return checksum;
}

std::uint64_t validate_and_fallback_children_cpu(const std::vector<ProgramGenome>& genomes,
                                                 const BenchConfig& cfg,
                                                 const std::vector<int>& parent_a,
                                                 const std::vector<int>& parent_b,
                                                 const std::vector<unsigned char>& is_mutation,
                                                 const std::vector<PlainNode>& child_nodes,
                                                 const std::vector<int>& child_used_len,
                                                 const std::vector<std::uint64_t>& child_name_ids,
                                                 const std::vector<int>& child_name_counts,
                                                 const std::vector<Value>& child_consts,
                                                 const std::vector<int>& child_const_counts,
                                                 const std::vector<unsigned char>* cheap_valid,
                                                 TimingAggregate* agg,
                                                 bool gpu_side) {
  g3pvm::evo::Limits limits;
  limits.max_total_nodes = cfg.max_nodes;
  limits.max_expr_depth = 5;
  limits.max_stmts_per_block = 6;
  limits.max_for_k = 16;

  const auto t0 = std::chrono::steady_clock::now();
  std::uint64_t checksum = 0;
  int fallback_count = 0;
  int cheap_reject_count = 0;
  int full_validate_count = 0;
  const int total_children = cfg.child_count * 2;
  for (int c = 0; c < total_children; ++c) {
    if (cheap_valid != nullptr && (*cheap_valid)[static_cast<std::size_t>(c)] == 0) {
      fallback_count += 1;
      cheap_reject_count += 1;
      const int pair_idx = c / 2;
      const bool use_a = (c % 2 == 0) || is_mutation[static_cast<std::size_t>(pair_idx)];
      const ProgramGenome& parent = genomes[static_cast<std::size_t>(use_a ? parent_a[static_cast<std::size_t>(pair_idx)]
                                                                            : parent_b[static_cast<std::size_t>(pair_idx)])];
      checksum ^= checksum_genome_nodes(parent);
      continue;
    }
    const int node_count = child_used_len[static_cast<std::size_t>(c)];
    full_validate_count += 1;
    const ProgramGenome built = build_genome_from_flat(
        &child_nodes[static_cast<std::size_t>(c * cfg.max_nodes)], node_count,
        &child_name_ids[static_cast<std::size_t>(c * cfg.max_names)], child_name_counts[static_cast<std::size_t>(c)],
        &child_consts[static_cast<std::size_t>(c * cfg.max_consts)], child_const_counts[static_cast<std::size_t>(c)]);
    const g3pvm::evo::ValidationResult vr = g3pvm::evo::validate_genome(built, limits);
    if (vr.is_valid) {
      checksum ^= checksum_genome_nodes(built);
    } else {
      fallback_count += 1;
      const int pair_idx = c / 2;
      const bool use_a = (c % 2 == 0) || is_mutation[static_cast<std::size_t>(pair_idx)];
      const ProgramGenome& parent = genomes[static_cast<std::size_t>(use_a ? parent_a[static_cast<std::size_t>(pair_idx)]
                                                                            : parent_b[static_cast<std::size_t>(pair_idx)])];
      checksum ^= checksum_genome_nodes(parent);
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (gpu_side) {
    agg->gpu_validate_fallback_ms += ms_between(t0, t1);
    agg->gpu_fallback_count += fallback_count;
    agg->gpu_cheap_reject_count += cheap_reject_count;
    agg->gpu_full_validate_count += full_validate_count;
  } else {
    agg->cpu_validate_fallback_ms += ms_between(t0, t1);
    agg->cpu_fallback_count += fallback_count;
  }
  return checksum;
}

std::uint64_t cpu_variation(const PackedHostData& packed,
                            const BenchConfig& cfg,
                            const std::vector<int>& parent_a,
                            const std::vector<int>& parent_b,
                            const std::vector<int>& cand_a,
                            const std::vector<int>& cand_b,
                            const std::vector<int>& donor_idx,
                            const std::vector<unsigned char>& is_mutation,
                            std::vector<PlainNode>* out_child_nodes,
                            std::vector<int>* out_child_used_len,
                            std::vector<std::uint64_t>* out_child_name_ids,
                            std::vector<int>* out_child_name_counts,
                            std::vector<Value>* out_child_consts,
                            std::vector<int>* out_child_const_counts) {
  std::vector<PlainNode> child_nodes(static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_nodes));
  std::vector<int> child_used_len(static_cast<std::size_t>(cfg.child_count * 2), 0);
  std::vector<std::uint64_t> child_name_ids(static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_names), 0ULL);
  std::vector<int> child_name_counts(static_cast<std::size_t>(cfg.child_count * 2), 0);
  std::vector<Value> child_consts(static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_consts), Value::none());
  std::vector<int> child_const_counts(static_cast<std::size_t>(cfg.child_count * 2), 0);
  for (int pair_idx = 0; pair_idx < cfg.child_count; ++pair_idx) {
    const int pa = parent_a[static_cast<std::size_t>(pair_idx)];
    const int pb = parent_b[static_cast<std::size_t>(pair_idx)];
    const CandidateRange ra = packed.candidates[static_cast<std::size_t>(pa * cfg.candidates_per_program +
                                                                         cand_a[static_cast<std::size_t>(pair_idx)])];
    const int len_a = packed.metas[static_cast<std::size_t>(pa)].used_len;
    const int len_b = packed.metas[static_cast<std::size_t>(pb)].used_len;
    const std::size_t out_base_ab = static_cast<std::size_t>((pair_idx * 2) * cfg.max_nodes);
    const std::size_t out_base_ba = static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_nodes);
    const std::size_t base_a = static_cast<std::size_t>(pa * cfg.max_nodes);
    const std::size_t base_b = static_cast<std::size_t>(pb * cfg.max_nodes);
    int prefix_a = std::max(0, ra.start);
    int replace_a = std::max(0, ra.stop - ra.start);
    int donor_len_for_a = 0;
    const PlainNode* donor_ptr_for_a = nullptr;
    int prefix_b = 0;
    int replace_b = 0;
    int donor_len_for_b = 0;
    const PlainNode* donor_ptr_for_b = nullptr;
    if (is_mutation[static_cast<std::size_t>(pair_idx)]) {
      const int did = donor_idx[static_cast<std::size_t>(pair_idx)];
      donor_len_for_a = packed.donor_lens[static_cast<std::size_t>(did)];
      donor_ptr_for_a = &packed.donor_nodes[static_cast<std::size_t>(did * cfg.max_donor_nodes)];
      prefix_b = 0;
      replace_b = donor_len_for_a;
      donor_len_for_b = replace_a;
      donor_ptr_for_b = &packed.program_nodes[base_a + static_cast<std::size_t>(prefix_a)];
    } else {
      const CandidateRange rb = packed.candidates[static_cast<std::size_t>(pb * cfg.candidates_per_program +
                                                                           cand_b[static_cast<std::size_t>(pair_idx)])];
      donor_len_for_a = std::max(0, rb.stop - rb.start);
      donor_ptr_for_a = &packed.program_nodes[base_b + static_cast<std::size_t>(rb.start)];
      prefix_b = std::max(0, rb.start);
      replace_b = std::max(0, rb.stop - rb.start);
      donor_len_for_b = replace_a;
      donor_ptr_for_b = &packed.program_nodes[base_a + static_cast<std::size_t>(prefix_a)];
    }

    const int suffix_a_start = prefix_a + replace_a;
    const int suffix_a_len = std::max(0, len_a - suffix_a_start);
    const int out_a_len = prefix_a + donor_len_for_a + suffix_a_len;
    if (out_a_len > cfg.max_nodes || prefix_a >= len_a || replace_a <= 0) {
      std::memcpy(&child_nodes[out_base_ab], &packed.program_nodes[base_a],
                  sizeof(PlainNode) * static_cast<std::size_t>(len_a));
      child_used_len[static_cast<std::size_t>(pair_idx * 2)] = len_a;
      child_name_counts[static_cast<std::size_t>(pair_idx * 2)] = packed.metas[static_cast<std::size_t>(pa)].name_count;
      child_const_counts[static_cast<std::size_t>(pair_idx * 2)] = packed.metas[static_cast<std::size_t>(pa)].const_count;
      for (int i = 0; i < packed.metas[static_cast<std::size_t>(pa)].name_count; ++i) {
        child_name_ids[static_cast<std::size_t>((pair_idx * 2) * cfg.max_names + i)] =
            packed.program_name_ids[static_cast<std::size_t>(pa * cfg.max_names + i)];
      }
      for (int i = 0; i < packed.metas[static_cast<std::size_t>(pa)].const_count; ++i) {
        child_consts[static_cast<std::size_t>((pair_idx * 2) * cfg.max_consts + i)] =
            packed.program_consts[static_cast<std::size_t>(pa * cfg.max_consts + i)];
      }
    } else {
      child_used_len[static_cast<std::size_t>(pair_idx * 2)] = out_a_len;
      std::vector<std::uint64_t> child_a_names;
      std::vector<Value> child_a_consts;
      for (int i = 0; i < packed.metas[static_cast<std::size_t>(pa)].name_count; ++i) {
        child_a_names.push_back(packed.program_name_ids[static_cast<std::size_t>(pa * cfg.max_names + i)]);
      }
      for (int i = 0; i < packed.metas[static_cast<std::size_t>(pa)].const_count; ++i) {
        child_a_consts.push_back(packed.program_consts[static_cast<std::size_t>(pa * cfg.max_consts + i)]);
      }
      std::memcpy(&child_nodes[out_base_ab], &packed.program_nodes[base_a],
                  sizeof(PlainNode) * static_cast<std::size_t>(prefix_a));
      const std::uint64_t* src_names_a =
          is_mutation[static_cast<std::size_t>(pair_idx)]
              ? &packed.donor_name_ids[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)] * cfg.max_names)]
              : &packed.program_name_ids[static_cast<std::size_t>(pb * cfg.max_names)];
      const Value* src_consts_a =
          is_mutation[static_cast<std::size_t>(pair_idx)]
              ? &packed.donor_consts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)] * cfg.max_consts)]
              : &packed.program_consts[static_cast<std::size_t>(pb * cfg.max_consts)];
      for (int i = 0; i < donor_len_for_a; ++i) {
        child_nodes[out_base_ab + static_cast<std::size_t>(prefix_a + i)] =
            cpu_remap_node_for_child(donor_ptr_for_a[i], src_names_a, src_consts_a, &child_a_names, &child_a_consts,
                                     cfg.max_names, cfg.max_consts);
      }
      child_name_counts[static_cast<std::size_t>(pair_idx * 2)] = static_cast<int>(child_a_names.size());
      child_const_counts[static_cast<std::size_t>(pair_idx * 2)] = static_cast<int>(child_a_consts.size());
      for (std::size_t i = 0; i < child_a_names.size(); ++i) {
        child_name_ids[static_cast<std::size_t>((pair_idx * 2) * cfg.max_names) + i] = child_a_names[i];
      }
      for (std::size_t i = 0; i < child_a_consts.size(); ++i) {
        child_consts[static_cast<std::size_t>((pair_idx * 2) * cfg.max_consts) + i] = child_a_consts[i];
      }
      std::memcpy(&child_nodes[out_base_ab + static_cast<std::size_t>(prefix_a + donor_len_for_a)],
                  &packed.program_nodes[base_a + static_cast<std::size_t>(suffix_a_start)],
                  sizeof(PlainNode) * static_cast<std::size_t>(suffix_a_len));
    }

    const PlainNode* source_b =
        is_mutation[static_cast<std::size_t>(pair_idx)] ? donor_ptr_for_a : &packed.program_nodes[base_b];
    const int source_b_len = is_mutation[static_cast<std::size_t>(pair_idx)] ? replace_b : len_b;
    const int suffix_b_start = prefix_b + replace_b;
    const int suffix_b_len = std::max(0, source_b_len - suffix_b_start);
    const int out_b_len = prefix_b + donor_len_for_b + suffix_b_len;
    if (out_b_len > cfg.max_nodes || prefix_b > source_b_len || replace_b < 0) {
      std::memcpy(&child_nodes[out_base_ba], source_b, sizeof(PlainNode) * static_cast<std::size_t>(source_b_len));
      child_used_len[static_cast<std::size_t>(pair_idx * 2 + 1)] = source_b_len;
      if (is_mutation[static_cast<std::size_t>(pair_idx)]) {
        child_name_counts[static_cast<std::size_t>(pair_idx * 2 + 1)] =
            packed.donor_name_counts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)])];
        child_const_counts[static_cast<std::size_t>(pair_idx * 2 + 1)] =
            packed.donor_const_counts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)])];
        for (int i = 0; i < child_name_counts[static_cast<std::size_t>(pair_idx * 2 + 1)]; ++i) {
          child_name_ids[static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_names + i)] =
              packed.donor_name_ids[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)] * cfg.max_names + i)];
        }
        for (int i = 0; i < child_const_counts[static_cast<std::size_t>(pair_idx * 2 + 1)]; ++i) {
          child_consts[static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_consts + i)] =
              packed.donor_consts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)] * cfg.max_consts + i)];
        }
      } else {
        child_name_counts[static_cast<std::size_t>(pair_idx * 2 + 1)] = packed.metas[static_cast<std::size_t>(pb)].name_count;
        child_const_counts[static_cast<std::size_t>(pair_idx * 2 + 1)] = packed.metas[static_cast<std::size_t>(pb)].const_count;
        for (int i = 0; i < child_name_counts[static_cast<std::size_t>(pair_idx * 2 + 1)]; ++i) {
          child_name_ids[static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_names + i)] =
              packed.program_name_ids[static_cast<std::size_t>(pb * cfg.max_names + i)];
        }
        for (int i = 0; i < child_const_counts[static_cast<std::size_t>(pair_idx * 2 + 1)]; ++i) {
          child_consts[static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_consts + i)] =
              packed.program_consts[static_cast<std::size_t>(pb * cfg.max_consts + i)];
        }
      }
      continue;
    }
    child_used_len[static_cast<std::size_t>(pair_idx * 2 + 1)] = out_b_len;
    std::vector<std::uint64_t> child_b_names;
    std::vector<Value> child_b_consts;
    if (is_mutation[static_cast<std::size_t>(pair_idx)]) {
      for (int i = 0; i < packed.donor_name_counts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)])]; ++i) {
        child_b_names.push_back(packed.donor_name_ids[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)] * cfg.max_names + i)]);
      }
      for (int i = 0; i < packed.donor_const_counts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)])]; ++i) {
        child_b_consts.push_back(packed.donor_consts[static_cast<std::size_t>(donor_idx[static_cast<std::size_t>(pair_idx)] * cfg.max_consts + i)]);
      }
    } else {
      for (int i = 0; i < packed.metas[static_cast<std::size_t>(pb)].name_count; ++i) {
        child_b_names.push_back(packed.program_name_ids[static_cast<std::size_t>(pb * cfg.max_names + i)]);
      }
      for (int i = 0; i < packed.metas[static_cast<std::size_t>(pb)].const_count; ++i) {
        child_b_consts.push_back(packed.program_consts[static_cast<std::size_t>(pb * cfg.max_consts + i)]);
      }
    }
    std::memcpy(&child_nodes[out_base_ba], source_b, sizeof(PlainNode) * static_cast<std::size_t>(prefix_b));
    const std::uint64_t* src_names_b = &packed.program_name_ids[static_cast<std::size_t>(pa * cfg.max_names)];
    const Value* src_consts_b = &packed.program_consts[static_cast<std::size_t>(pa * cfg.max_consts)];
    for (int i = 0; i < donor_len_for_b; ++i) {
      child_nodes[out_base_ba + static_cast<std::size_t>(prefix_b + i)] =
          cpu_remap_node_for_child(donor_ptr_for_b[i], src_names_b, src_consts_b, &child_b_names, &child_b_consts,
                                   cfg.max_names, cfg.max_consts);
    }
    child_name_counts[static_cast<std::size_t>(pair_idx * 2 + 1)] = static_cast<int>(child_b_names.size());
    child_const_counts[static_cast<std::size_t>(pair_idx * 2 + 1)] = static_cast<int>(child_b_consts.size());
    for (std::size_t i = 0; i < child_b_names.size(); ++i) {
      child_name_ids[static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_names) + i] = child_b_names[i];
    }
    for (std::size_t i = 0; i < child_b_consts.size(); ++i) {
      child_consts[static_cast<std::size_t>((pair_idx * 2 + 1) * cfg.max_consts) + i] = child_b_consts[i];
    }
    std::memcpy(&child_nodes[out_base_ba + static_cast<std::size_t>(prefix_b + donor_len_for_b)],
                source_b + suffix_b_start, sizeof(PlainNode) * static_cast<std::size_t>(suffix_b_len));
  }
  *out_child_nodes = std::move(child_nodes);
  *out_child_used_len = std::move(child_used_len);
  *out_child_name_ids = std::move(child_name_ids);
  *out_child_name_counts = std::move(child_name_counts);
  *out_child_consts = std::move(child_consts);
  *out_child_const_counts = std::move(child_const_counts);
  return 0;
}

std::uint64_t checksum_nodes(const std::vector<PlainNode>& nodes) {
  std::uint64_t checksum = 0;
  for (const PlainNode& n : nodes) {
    checksum ^= static_cast<std::uint64_t>(n.kind + 131 * n.i0 + 977 * n.i1);
    checksum = hash64(checksum + 0x9e3779b97f4a7c15ULL);
  }
  return checksum;
}

BenchConfig parse_args(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--population-size") cfg.population_size = std::stoi(need("--population-size"));
    else if (arg == "--child-count") cfg.child_count = std::stoi(need("--child-count"));
    else if (arg == "--candidates-per-program") cfg.candidates_per_program = std::stoi(need("--candidates-per-program"));
    else if (arg == "--donor-pool-size") cfg.donor_pool_size = std::stoi(need("--donor-pool-size"));
    else if (arg == "--max-nodes") cfg.max_nodes = std::stoi(need("--max-nodes"));
    else if (arg == "--max-donor-nodes") cfg.max_donor_nodes = std::stoi(need("--max-donor-nodes"));
    else if (arg == "--tournament-k") cfg.tournament_k = std::stoi(need("--tournament-k"));
    else if (arg == "--eval-cases") cfg.eval_cases = std::stoi(need("--eval-cases"));
    else if (arg == "--repeats") cfg.repeats = std::stoi(need("--repeats"));
    else if (arg == "--seed") cfg.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
    else if (arg == "--mutation-ratio") cfg.mutation_ratio = std::stod(need("--mutation-ratio"));
    else if (arg == "--disable-gpu-cheap-validate") cfg.enable_gpu_cheap_validate = false;
    else if (arg == "--out-json") cfg.out_json = need("--out-json");
    else if (arg == "--help") {
      std::cout
          << "Usage: g3pvm_repro_proto_bench [options]\n"
          << "  --population-size N\n"
          << "  --child-count N\n"
          << "  --candidates-per-program N\n"
          << "  --donor-pool-size N\n"
          << "  --max-nodes N\n"
          << "  --max-donor-nodes N\n"
          << "  --tournament-k N\n"
          << "  --eval-cases N\n"
          << "  --repeats N\n"
          << "  --seed N\n"
          << "  --mutation-ratio X\n"
          << "  --disable-gpu-cheap-validate\n"
          << "  --out-json PATH\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return cfg;
}

std::string to_json(const BenchConfig& cfg, const TimingAggregate& agg) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  const double cpu_pre_ms = agg.cpu_subtree_ms + agg.cpu_candidate_ms + agg.cpu_donor_ms;
  const double pack_ms = agg.pack_program_ms + agg.pack_candidate_ms + agg.pack_donor_ms;
  const double h2d_ms = agg.h2d_program_ms + agg.h2d_candidate_ms + agg.h2d_donor_ms;
  const double gpu_total_ms = agg.gpu_selection_ms + agg.gpu_variation_ms;
  const double gpu_total_with_d2h_ms = gpu_total_ms + agg.d2h_child_ms;
  const double cpu_total_ms = agg.cpu_selection_ms + agg.cpu_variation_ms;
  oss << "{\n";
  oss << "  \"config\": {\n";
  oss << "    \"population_size\": " << cfg.population_size << ",\n";
  oss << "    \"child_count\": " << cfg.child_count << ",\n";
  oss << "    \"candidates_per_program\": " << cfg.candidates_per_program << ",\n";
  oss << "    \"donor_pool_size\": " << cfg.donor_pool_size << ",\n";
  oss << "    \"max_nodes\": " << cfg.max_nodes << ",\n";
  oss << "    \"max_donor_nodes\": " << cfg.max_donor_nodes << ",\n";
  oss << "    \"tournament_k\": " << cfg.tournament_k << ",\n";
  oss << "    \"eval_cases\": " << cfg.eval_cases << ",\n";
  oss << "    \"repeats\": " << cfg.repeats << ",\n";
  oss << "    \"mutation_ratio\": " << cfg.mutation_ratio << ",\n";
  oss << "    \"enable_gpu_cheap_validate\": " << (cfg.enable_gpu_cheap_validate ? "true" : "false") << "\n";
  oss << "  },\n";
  oss << "  \"timings_ms\": {\n";
  oss << "    \"cpu_subtree\": " << agg.cpu_subtree_ms << ",\n";
  oss << "    \"cpu_candidate\": " << agg.cpu_candidate_ms << ",\n";
  oss << "    \"cpu_donor\": " << agg.cpu_donor_ms << ",\n";
  oss << "    \"cpu_preprocess_total\": " << cpu_pre_ms << ",\n";
  oss << "    \"pack_program\": " << agg.pack_program_ms << ",\n";
  oss << "    \"pack_candidate\": " << agg.pack_candidate_ms << ",\n";
  oss << "    \"pack_donor\": " << agg.pack_donor_ms << ",\n";
  oss << "    \"pack_total\": " << pack_ms << ",\n";
  oss << "    \"h2d_program\": " << agg.h2d_program_ms << ",\n";
  oss << "    \"h2d_candidate\": " << agg.h2d_candidate_ms << ",\n";
  oss << "    \"h2d_donor\": " << agg.h2d_donor_ms << ",\n";
  oss << "    \"h2d_total\": " << h2d_ms << ",\n";
  oss << "    \"gpu_eval\": " << agg.gpu_eval_ms << ",\n";
  oss << "    \"gpu_selection\": " << agg.gpu_selection_ms << ",\n";
  oss << "    \"gpu_variation\": " << agg.gpu_variation_ms << ",\n";
  oss << "    \"gpu_cheap_validate\": " << agg.gpu_cheap_validate_ms << ",\n";
  oss << "    \"d2h_child\": " << agg.d2h_child_ms << ",\n";
  oss << "    \"gpu_validate_fallback\": " << agg.gpu_validate_fallback_ms << ",\n";
  oss << "    \"gpu_selection_variation_total\": " << gpu_total_ms << ",\n";
  oss << "    \"gpu_selection_variation_d2h_total\": " << gpu_total_with_d2h_ms << ",\n";
  oss << "    \"sequential_wall\": " << agg.sequential_wall_ms << ",\n";
  oss << "    \"overlap_wall\": " << agg.overlap_wall_ms << ",\n";
  oss << "    \"cpu_selection\": " << agg.cpu_selection_ms << ",\n";
  oss << "    \"cpu_variation\": " << agg.cpu_variation_ms << ",\n";
  oss << "    \"cpu_validate_fallback\": " << agg.cpu_validate_fallback_ms << ",\n";
  oss << "    \"cpu_selection_variation_total\": " << cpu_total_ms << "\n";
  oss << "  },\n";
  oss << "  \"overlap\": {\n";
  oss << "    \"cpu_preprocess_hidden_by_eval\": " << (cpu_pre_ms <= agg.gpu_eval_ms ? "true" : "false") << ",\n";
  oss << "    \"pack_plus_h2d_hidden_by_eval\": " << ((pack_ms + h2d_ms) <= agg.gpu_eval_ms ? "true" : "false")
      << ",\n";
  oss << "    \"cpu_preprocess_plus_pack_plus_h2d_hidden_by_eval\": "
      << ((cpu_pre_ms + pack_ms + h2d_ms) <= agg.gpu_eval_ms ? "true" : "false") << ",\n";
  oss << "    \"gpu_selection_variation_speedup_over_cpu\": "
      << (gpu_total_ms > 0.0 ? cpu_total_ms / gpu_total_ms : 0.0) << ",\n";
  oss << "    \"gpu_selection_variation_d2h_speedup_over_cpu\": "
      << (gpu_total_with_d2h_ms > 0.0 ? cpu_total_ms / gpu_total_with_d2h_ms : 0.0) << ",\n";
  oss << "    \"overlap_speedup_over_sequential\": "
      << (agg.overlap_wall_ms > 0.0 ? agg.sequential_wall_ms / agg.overlap_wall_ms : 0.0) << "\n";
  oss << "  },\n";
  oss << "  \"checksums\": {\n";
  oss << "    \"cpu\": " << agg.cpu_checksum << ",\n";
  oss << "    \"gpu\": " << agg.gpu_checksum << "\n";
  oss << "  },\n";
  oss << "  \"fallback_counts\": {\n";
  oss << "    \"cpu\": " << agg.cpu_fallback_count << ",\n";
  oss << "    \"gpu\": " << agg.gpu_fallback_count << "\n";
  oss << "  },\n";
  oss << "  \"gpu_cheap_validate\": {\n";
  oss << "    \"cheap_reject_count\": " << agg.gpu_cheap_reject_count << ",\n";
  oss << "    \"full_validate_count\": " << agg.gpu_full_validate_count << "\n";
  oss << "  }\n";
  oss << "}\n";
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const BenchConfig cfg = parse_args(argc, argv);
    DeviceBuffers dev;
    allocate_device_buffers(&dev, cfg);

    const std::vector<ProgramGenome> genomes = make_population(cfg);
    const std::vector<g3pvm::BytecodeProgram> bytecode = compile_population(genomes);

    std::vector<InputCase> shared_cases;
    std::vector<Value> shared_answer;
    prepare_eval_data(cfg.eval_cases, &shared_cases, &shared_answer);

    g3pvm::FitnessSessionGpu session;
    const g3pvm::FitnessEvalResult init_result = session.init(shared_cases, shared_answer, 20000, 256);
    if (!init_result.ok) {
      std::cerr << "gpu session init failed: " << init_result.err.message << "\n";
      return 2;
    }

    TimingAggregate agg;
    std::vector<int> last_fitness(static_cast<std::size_t>(cfg.population_size), 0);

    for (int rep = 0; rep < cfg.repeats; ++rep) {
      const g3pvm::FitnessEvalResult eval_result = session.eval_programs(bytecode);
      if (!eval_result.ok) {
        std::cerr << "gpu eval failed: " << eval_result.err.message << "\n";
        return 2;
      }
      agg.gpu_eval_ms += eval_result.total_ms;
      last_fitness = eval_result.fitness;
      ensure_cuda(cudaMemcpy(dev.d_fitness, last_fitness.data(),
                             sizeof(int) * static_cast<std::size_t>(cfg.population_size), cudaMemcpyHostToDevice),
                  "cudaMemcpy fitness");

      const PreprocessData prep = run_preprocess(genomes, cfg, &agg);
      const PackedHostData packed = pack_host_data(genomes, prep, cfg, &agg);
      copy_host_to_device(packed, &dev, cfg, &agg);

      std::vector<int> cpu_pa;
      std::vector<int> cpu_pb;
      std::vector<int> cpu_ca;
      std::vector<int> cpu_cb;
      std::vector<int> cpu_di;
      std::vector<unsigned char> cpu_mut;

      const auto cpu_sel_t0 = std::chrono::steady_clock::now();
      cpu_tournament_select(last_fitness, cfg, &cpu_pa, &cpu_pb, &cpu_ca, &cpu_cb, &cpu_di, &cpu_mut);
      const auto cpu_sel_t1 = std::chrono::steady_clock::now();
      agg.cpu_selection_ms += ms_between(cpu_sel_t0, cpu_sel_t1);

      std::vector<PlainNode> cpu_child_nodes;
      std::vector<int> cpu_child_used_len;
      std::vector<std::uint64_t> cpu_child_name_ids;
      std::vector<int> cpu_child_name_counts;
      std::vector<Value> cpu_child_consts;
      std::vector<int> cpu_child_const_counts;
      const auto cpu_var_t0 = std::chrono::steady_clock::now();
      (void)cpu_variation(packed, cfg, cpu_pa, cpu_pb, cpu_ca, cpu_cb, cpu_di, cpu_mut, &cpu_child_nodes,
                          &cpu_child_used_len,
                          &cpu_child_name_ids, &cpu_child_name_counts, &cpu_child_consts, &cpu_child_const_counts);
      const auto cpu_var_t1 = std::chrono::steady_clock::now();
      agg.cpu_variation_ms += ms_between(cpu_var_t0, cpu_var_t1);
      agg.cpu_checksum ^= validate_and_fallback_children_cpu(genomes, cfg, cpu_pa, cpu_pb, cpu_mut, cpu_child_nodes,
                                                             cpu_child_used_len,
                                                             cpu_child_name_ids, cpu_child_name_counts, cpu_child_consts,
                                                             cpu_child_const_counts, nullptr, &agg, false);

      const int threads = 256;
      const int blocks = (cfg.child_count + threads - 1) / threads;
      const auto gpu_sel_t0 = std::chrono::steady_clock::now();
      tournament_select_kernel<<<blocks, threads>>>(
          dev.d_fitness, cfg.population_size, cfg.child_count, cfg.candidates_per_program, cfg.donor_pool_size,
          cfg.tournament_k, cfg.mutation_ratio, cfg.seed + static_cast<std::uint64_t>(rep), dev.d_parent_a,
          dev.d_parent_b, dev.d_cand_a, dev.d_cand_b, dev.d_donor_idx, dev.d_is_mutation);
      ensure_cuda(cudaGetLastError(), "launch tournament_select_kernel");
      ensure_cuda(cudaDeviceSynchronize(), "sync tournament_select_kernel");
      const auto gpu_sel_t1 = std::chrono::steady_clock::now();
      agg.gpu_selection_ms += ms_between(gpu_sel_t0, gpu_sel_t1);

      const auto gpu_var_t0 = std::chrono::steady_clock::now();
      variation_kernel<<<static_cast<unsigned int>(cfg.child_count), 128>>>(
          dev.d_program_nodes, dev.d_metas, dev.d_candidates, dev.d_program_name_ids, dev.d_program_consts,
          dev.d_donor_nodes, dev.d_donor_lens, dev.d_donor_name_ids, dev.d_donor_name_counts, dev.d_donor_consts,
          dev.d_donor_const_counts, cfg.max_nodes, cfg.candidates_per_program, cfg.max_donor_nodes, cfg.max_names,
          cfg.max_consts, cfg.child_count, dev.d_parent_a, dev.d_parent_b, dev.d_cand_a, dev.d_cand_b,
          dev.d_donor_idx, dev.d_is_mutation, dev.d_child_nodes, dev.d_child_used_len, dev.d_child_name_ids, dev.d_child_name_counts,
          dev.d_child_consts, dev.d_child_const_counts);
      ensure_cuda(cudaGetLastError(), "launch variation_kernel");
      ensure_cuda(cudaDeviceSynchronize(), "sync variation_kernel");
      const auto gpu_var_t1 = std::chrono::steady_clock::now();
      agg.gpu_variation_ms += ms_between(gpu_var_t0, gpu_var_t1);

      const int total_children = cfg.child_count * 2;
      if (cfg.enable_gpu_cheap_validate) {
        const auto gpu_cheap_t0 = std::chrono::steady_clock::now();
        cheap_validate_kernel<<<(total_children + threads - 1) / threads, threads>>>(
            dev.d_child_nodes, dev.d_child_used_len, dev.d_child_name_counts, dev.d_child_const_counts, cfg.max_nodes, 16,
            cfg.max_names, cfg.max_consts, total_children, dev.d_child_cheap_valid);
        ensure_cuda(cudaGetLastError(), "launch cheap_validate_kernel");
        ensure_cuda(cudaDeviceSynchronize(), "sync cheap_validate_kernel");
        const auto gpu_cheap_t1 = std::chrono::steady_clock::now();
        agg.gpu_cheap_validate_ms += ms_between(gpu_cheap_t0, gpu_cheap_t1);
      } else {
        ensure_cuda(cudaMemset(dev.d_child_cheap_valid, 1, sizeof(unsigned char) * static_cast<std::size_t>(total_children)),
                    "cudaMemset child_cheap_valid");
      }

      std::vector<PlainNode> child_copy(static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_nodes));
      std::vector<int> child_used_len(static_cast<std::size_t>(cfg.child_count * 2));
      std::vector<std::uint64_t> child_name_ids(static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_names));
      std::vector<int> child_name_counts(static_cast<std::size_t>(cfg.child_count * 2));
      std::vector<Value> child_consts(static_cast<std::size_t>(cfg.child_count * 2 * cfg.max_consts));
      std::vector<int> child_const_counts(static_cast<std::size_t>(cfg.child_count * 2));
      std::vector<unsigned char> child_cheap_valid(static_cast<std::size_t>(cfg.child_count * 2));
      const auto d2h_t0 = std::chrono::steady_clock::now();
      ensure_cuda(cudaMemcpy(child_copy.data(), dev.d_child_nodes, sizeof(PlainNode) * child_copy.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy child copy");
      ensure_cuda(cudaMemcpy(child_used_len.data(), dev.d_child_used_len, sizeof(int) * child_used_len.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy child_used_len");
      ensure_cuda(cudaMemcpy(child_name_ids.data(), dev.d_child_name_ids,
                             sizeof(std::uint64_t) * child_name_ids.size(), cudaMemcpyDeviceToHost),
                  "cudaMemcpy child_name_ids");
      ensure_cuda(cudaMemcpy(child_name_counts.data(), dev.d_child_name_counts,
                             sizeof(int) * child_name_counts.size(), cudaMemcpyDeviceToHost),
                  "cudaMemcpy child_name_counts");
      ensure_cuda(cudaMemcpy(child_consts.data(), dev.d_child_consts, sizeof(Value) * child_consts.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy child_consts");
      ensure_cuda(cudaMemcpy(child_const_counts.data(), dev.d_child_const_counts,
                             sizeof(int) * child_const_counts.size(), cudaMemcpyDeviceToHost),
                  "cudaMemcpy child_const_counts");
      ensure_cuda(cudaMemcpy(child_cheap_valid.data(), dev.d_child_cheap_valid,
                             sizeof(unsigned char) * child_cheap_valid.size(), cudaMemcpyDeviceToHost),
                  "cudaMemcpy child_cheap_valid");
      ensure_cuda(cudaDeviceSynchronize(), "sync child copy");
      const auto d2h_t1 = std::chrono::steady_clock::now();
      agg.d2h_child_ms += ms_between(d2h_t0, d2h_t1);
      agg.gpu_checksum ^= validate_and_fallback_children_cpu(genomes, cfg, cpu_pa, cpu_pb, cpu_mut, child_copy,
                                                             child_used_len,
                                                             child_name_ids, child_name_counts, child_consts,
                                                             child_const_counts, &child_cheap_valid, &agg, true);
    }

    const double inv = 1.0 / static_cast<double>(cfg.repeats);
    agg.cpu_subtree_ms *= inv;
    agg.cpu_candidate_ms *= inv;
    agg.cpu_donor_ms *= inv;
    agg.pack_program_ms *= inv;
    agg.pack_candidate_ms *= inv;
    agg.pack_donor_ms *= inv;
    agg.h2d_program_ms *= inv;
    agg.h2d_candidate_ms *= inv;
    agg.h2d_donor_ms *= inv;
    agg.gpu_eval_ms *= inv;
    agg.gpu_selection_ms *= inv;
    agg.gpu_variation_ms *= inv;
    agg.gpu_cheap_validate_ms *= inv;
    agg.d2h_child_ms *= inv;
    agg.cpu_selection_ms *= inv;
    agg.cpu_variation_ms *= inv;

    {
      TimingAggregate tmp_seq;
      const auto seq_t0 = std::chrono::steady_clock::now();
      const g3pvm::FitnessEvalResult seq_eval = session.eval_programs(bytecode);
      if (!seq_eval.ok) {
        throw std::runtime_error("gpu eval failed during sequential overlap test: " + seq_eval.err.message);
      }
      const PreprocessData seq_prep = run_preprocess(genomes, cfg, &tmp_seq);
      const PackedHostData seq_packed = pack_host_data(genomes, seq_prep, cfg, &tmp_seq);
      copy_host_to_device(seq_packed, &dev, cfg, &tmp_seq);
      const auto seq_t1 = std::chrono::steady_clock::now();
      agg.sequential_wall_ms = ms_between(seq_t0, seq_t1);
    }

    {
      DeviceBuffers overlap_dev;
      allocate_device_buffers(&overlap_dev, cfg);
      cudaStream_t overlap_stream = nullptr;
      ensure_cuda(cudaStreamCreate(&overlap_stream), "cudaStreamCreate overlap_stream");
      g3pvm::FitnessEvalResult overlap_eval;
      std::string overlap_err;

      const auto overlap_t0 = std::chrono::steady_clock::now();
      std::thread eval_thread([&]() {
        overlap_eval = session.eval_programs(bytecode);
        if (!overlap_eval.ok) {
          overlap_err = overlap_eval.err.message;
        }
      });

      TimingAggregate tmp_overlap;
      const PreprocessData overlap_prep = run_preprocess(genomes, cfg, &tmp_overlap);
      const PackedHostData overlap_packed = pack_host_data(genomes, overlap_prep, cfg, &tmp_overlap);
      copy_host_to_device_async(overlap_packed, &overlap_dev, cfg, overlap_stream);
      ensure_cuda(cudaStreamSynchronize(overlap_stream), "cudaStreamSynchronize overlap_stream");
      eval_thread.join();
      const auto overlap_t1 = std::chrono::steady_clock::now();
      ensure_cuda(cudaStreamDestroy(overlap_stream), "cudaStreamDestroy overlap_stream");
      if (!overlap_err.empty()) {
        throw std::runtime_error("gpu eval failed during overlap test: " + overlap_err);
      }
      agg.overlap_wall_ms = ms_between(overlap_t0, overlap_t1);
    }

    const std::string json = to_json(cfg, agg);
    std::cout << json;
    if (!cfg.out_json.empty()) {
      std::ofstream out(cfg.out_json);
      out << json;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
