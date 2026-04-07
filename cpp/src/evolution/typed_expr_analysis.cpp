#include "typed_expr_analysis.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "subtree_utils.hpp"

namespace g3pvm::evo::typed_expr {

namespace {

RType infer_unbound_var_type(const AstProgram& p, int name_id) {
  if (name_id < 0 || static_cast<std::size_t>(name_id) >= p.names.size()) {
    return RType::Num;
  }
  const std::string& name = p.names[static_cast<std::size_t>(name_id)];
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

bool is_stmt_kind(NodeKind kind) {
  return kind == NodeKind::ASSIGN || kind == NodeKind::IF_STMT || kind == NodeKind::FOR_RANGE || kind == NodeKind::RETURN;
}

bool is_expr_kind(NodeKind kind) {
  return kind == NodeKind::CONST || kind == NodeKind::VAR || kind == NodeKind::NEG || kind == NodeKind::NOT ||
         kind == NodeKind::ADD || kind == NodeKind::SUB || kind == NodeKind::MUL || kind == NodeKind::DIV ||
         kind == NodeKind::MOD || kind == NodeKind::LT || kind == NodeKind::LE || kind == NodeKind::GT ||
         kind == NodeKind::GE || kind == NodeKind::EQ || kind == NodeKind::NE || kind == NodeKind::AND ||
         kind == NodeKind::OR || kind == NodeKind::IF_EXPR || kind == NodeKind::CALL_ABS || kind == NodeKind::CALL_MIN ||
         kind == NodeKind::CALL_MAX || kind == NodeKind::CALL_CLIP || kind == NodeKind::CALL_LEN ||
         kind == NodeKind::CALL_CONCAT || kind == NodeKind::CALL_SLICE || kind == NodeKind::CALL_INDEX ||
         kind == NodeKind::CALL_APPEND || kind == NodeKind::CALL_REVERSE || kind == NodeKind::CALL_FIND ||
         kind == NodeKind::CALL_CONTAINS;
}

bool is_sequence_type(RType type) {
  return type == RType::String || type == RType::NumList || type == RType::StringList;
}

bool is_value_type(RType type) {
  return type == RType::Num || type == RType::Bool || type == RType::NoneType || is_sequence_type(type);
}

struct ExprCheck {
  RType t = RType::Invalid;
  std::size_t next = 0;
};

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
      else if (v.tag == ValueTag::String) t = RType::String;
      else if (v.tag == ValueTag::NumList) t = RType::NumList;
      else if (v.tag == ValueTag::StringList) t = RType::StringList;
      else t = RType::Num;
    }
    if (out != nullptr && t != RType::Invalid) out->push_back(TypedExprRoot{idx, end[idx], t});
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::VAR) {
    auto it = env.find(n.i0);
    const RType t = (it == env.end()) ? infer_unbound_var_type(p, n.i0) : it->second;
    if (out != nullptr) out->push_back(TypedExprRoot{idx, end[idx], t});
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::NEG || n.kind == NodeKind::NOT) {
    ExprCheck e = infer_expr_prefix(p, end, idx + 1, env, out);
    const RType t = (n.kind == NodeKind::NEG) ? (e.t == RType::Num ? RType::Num : RType::Invalid)
                                               : (e.t == RType::Bool ? RType::Bool : RType::Invalid);
    if (out != nullptr && t != RType::Invalid) out->push_back(TypedExprRoot{idx, e.next, t});
    return {t, e.next};
  }
  if (n.kind == NodeKind::IF_EXPR) {
    ExprCheck c = infer_expr_prefix(p, end, idx + 1, env, out);
    ExprCheck t = infer_expr_prefix(p, end, c.next, env, out);
    ExprCheck f = infer_expr_prefix(p, end, t.next, env, out);
    RType r = RType::Invalid;
    if (c.t == RType::Bool && t.t == f.t && is_value_type(t.t)) {
      r = t.t;
    }
    if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, f.next, r});
    return {r, f.next};
  }
  if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN || n.kind == NodeKind::CALL_MAX ||
      n.kind == NodeKind::CALL_CLIP || n.kind == NodeKind::CALL_LEN || n.kind == NodeKind::CALL_CONCAT ||
      n.kind == NodeKind::CALL_SLICE || n.kind == NodeKind::CALL_INDEX || n.kind == NodeKind::CALL_APPEND ||
      n.kind == NodeKind::CALL_REVERSE || n.kind == NodeKind::CALL_FIND || n.kind == NodeKind::CALL_CONTAINS) {
    std::size_t cur = idx + 1;
    bool ok = true;
    if (n.kind == NodeKind::CALL_LEN) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      if (!is_sequence_type(a.t)) ok = false;
      cur = a.next;
      const RType r = ok ? RType::Num : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, cur, r});
      return {r, cur};
    }
    if (n.kind == NodeKind::CALL_CONCAT) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck b = infer_expr_prefix(p, end, a.next, env, out);
      const RType r = (a.t == b.t && is_sequence_type(a.t)) ? a.t : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, b.next, r});
      return {r, b.next};
    }
    if (n.kind == NodeKind::CALL_SLICE) {
      ExprCheck x = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck lo = infer_expr_prefix(p, end, x.next, env, out);
      ExprCheck hi = infer_expr_prefix(p, end, lo.next, env, out);
      const RType r = (is_sequence_type(x.t) && lo.t == RType::Num && hi.t == RType::Num)
                          ? x.t
                          : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, hi.next, r});
      return {r, hi.next};
    }
    if (n.kind == NodeKind::CALL_INDEX) {
      ExprCheck x = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck i = infer_expr_prefix(p, end, x.next, env, out);
      RType r = RType::Invalid;
      if (i.t == RType::Num) {
        if (x.t == RType::String) r = RType::String;
        else if (x.t == RType::NumList) r = RType::Num;
        else if (x.t == RType::StringList) r = RType::String;
      }
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, i.next, r});
      return {r, i.next};
    }
    if (n.kind == NodeKind::CALL_APPEND) {
      ExprCheck xs = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck elem = infer_expr_prefix(p, end, xs.next, env, out);
      const RType r = ((xs.t == RType::NumList && elem.t == RType::Num) ||
                       (xs.t == RType::StringList && elem.t == RType::String))
                          ? xs.t
                          : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, elem.next, r});
      return {r, elem.next};
    }
    if (n.kind == NodeKind::CALL_REVERSE) {
      ExprCheck x = infer_expr_prefix(p, end, cur, env, out);
      const RType r = is_sequence_type(x.t) ? x.t : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, x.next, r});
      return {r, x.next};
    }
    if (n.kind == NodeKind::CALL_FIND || n.kind == NodeKind::CALL_CONTAINS) {
      ExprCheck haystack = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck needle = infer_expr_prefix(p, end, haystack.next, env, out);
      const RType r = (haystack.t == RType::String && needle.t == RType::String)
                          ? (n.kind == NodeKind::CALL_FIND ? RType::Num : RType::Bool)
                          : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, needle.next, r});
      return {r, needle.next};
    }
    for (int i = 0; i < subtree::node_arity(n.kind); ++i) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      if (a.t != RType::Num) ok = false;
      cur = a.next;
    }
    const RType r = ok ? RType::Num : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, cur, r});
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
      r = (a.t == b.t && is_value_type(a.t))
              ? RType::Bool
              : RType::Invalid;
    else
      r = (a.t == RType::Num && b.t == RType::Num) ? RType::Num : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) out->push_back(TypedExprRoot{idx, b.next, r});
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
    if (e.t != RType::Invalid) env[n.i0] = e.t;
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
    const ExprCheck bound = infer_expr_prefix(p, end, idx + 1, env, nullptr);
    std::unordered_map<int, RType> env_b = env;
    env_b[n.i0] = RType::Num;
    std::size_t cur = bound.next;
    while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(p, end, cur + 1, env_b, out);
      cur = end[cur + 1];
    }
  }
}

}  // namespace

std::vector<TypedExprRoot> collect_typed_expr_roots(const AstProgram& program,
                                                    const std::vector<std::size_t>& subtree_end) {
  std::vector<TypedExprRoot> out;
  if (program.nodes.size() < 2 || program.nodes[0].kind != NodeKind::PROGRAM) return out;
  std::unordered_map<int, RType> env;
  std::size_t cur = 1;
  while (cur < program.nodes.size() && program.nodes[cur].kind == NodeKind::BLOCK_CONS) {
    collect_typed_exprs_in_stmt(program, subtree_end, cur + 1, env, out);
    cur = subtree_end[cur + 1];
  }
  return out;
}

}  // namespace g3pvm::evo::typed_expr
