#include "typed_expr_analysis.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gagp/evolution/node_descriptor.hpp"
#include "subtree_utils.hpp"

namespace gagp::evo::typed_expr {

namespace {

RType infer_unbound_var_type(const AstProgram& p, int name_id) {
  if (name_id < 0 || static_cast<std::size_t>(name_id) >= p.names.size()) {
    return RType::Int;
  }
  const std::string& name = p.names[static_cast<std::size_t>(name_id)];
  if (name == "strings" || name == "words" || name == "string_list" ||
      (name.size() >= 8 && name.compare(name.size() - 8, 8, "_strings") == 0) ||
      (name.size() >= 6 && name.compare(name.size() - 6, 6, "_words") == 0)) {
    return RType::StringList;
  }
  if (name == "fs" || name == "floats" || name == "float_list" ||
      (name.size() >= 6 && name.compare(name.size() - 6, 6, "_float") == 0) ||
      (name.size() >= 7 && name.compare(name.size() - 7, 7, "_floats") == 0)) {
    return RType::FloatList;
  }
  if (name == "xs" || name == "items" || name == "list" ||
      (name.size() >= 5 && name.compare(name.size() - 5, 5, "_list") == 0)) {
    return RType::IntList;
  }
  if (name == "s" || name == "str" || name == "text" ||
      (name.size() >= 7 && name.compare(name.size() - 7, 7, "_string") == 0)) {
    return RType::String;
  }
  if (name == "f" || name == "flt" || name == "float" ||
      (name.size() >= 2 && name.compare(name.size() - 2, 2, "_f") == 0)) {
    return RType::Float;
  }
  return RType::Int;
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
         kind == NodeKind::CALL_MAX || kind == NodeKind::CALL_CLIP || kind == NodeKind::CALL_IDIV0 ||
         kind == NodeKind::CALL_IMOD0 || kind == NodeKind::CALL_LEN ||
         kind == NodeKind::CALL_CONCAT || kind == NodeKind::CALL_SLICE || kind == NodeKind::CALL_INDEX ||
         kind == NodeKind::CALL_APPEND || kind == NodeKind::CALL_PREPEND || kind == NodeKind::CALL_REVERSE ||
         kind == NodeKind::CALL_FIND || kind == NodeKind::CALL_CONTAINS ||
         kind == NodeKind::CALL_CHAR_TO_STRING || kind == NodeKind::CALL_STRING_TO_CHAR ||
         kind == NodeKind::CALL_ORD || kind == NodeKind::CALL_CHR ||
         kind == NodeKind::CALL_IS_LETTER || kind == NodeKind::CALL_IS_DIGIT ||
         kind == NodeKind::CALL_IS_SPACE || kind == NodeKind::CALL_IS_VOWEL ||
         kind == NodeKind::CALL_TO_LOWER || kind == NodeKind::CALL_TO_UPPER ||
         kind == NodeKind::CALL_TO_STRING || kind == NodeKind::CALL_SINGLETON ||
         kind == NodeKind::BOUND_VAR || kind == NodeKind::MAP_LIST ||
         kind == NodeKind::FILTER_LIST || kind == NodeKind::LINEAR_REC || kind == NodeKind::ASGP_DC ||
         kind == NodeKind::ASGP_DP1D || kind == NodeKind::ASGP_DP2D;
}

bool is_structured_root_kind(NodeKind kind) {
  return kind == NodeKind::MAP_LIST || kind == NodeKind::FILTER_LIST || kind == NodeKind::LINEAR_REC ||
         kind == NodeKind::ASGP_DC || kind == NodeKind::ASGP_DP1D || kind == NodeKind::ASGP_DP2D;
}

bool range_contains(std::size_t start, std::size_t stop, std::size_t idx) {
  return idx >= start && idx < stop;
}

bool subtree_contains_bound_var(const AstProgram& p, std::size_t start, std::size_t stop) {
  stop = std::min(stop, p.nodes.size());
  for (std::size_t i = start; i < stop; ++i) {
    if (p.nodes[i].kind == NodeKind::BOUND_VAR) return true;
  }
  return false;
}

bool is_sequence_type(RType type) {
  return type == RType::String || type == RType::IntList || type == RType::FloatList ||
         type == RType::StringList;
}

bool is_value_type(RType type) {
  return type == RType::Int || type == RType::Float || type == RType::Bool ||
         type == RType::Char || is_sequence_type(type);
}

bool is_numeric_type(RType type) {
  return type == RType::Int || type == RType::Float;
}

std::uint64_t mix_u64(std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

std::uint64_t node_signature(NodeKind kind) {
  return mix_u64(1469598103934665603ULL, static_cast<std::uint64_t>(kind));
}

std::uint64_t env_signature(const std::unordered_map<int, RType>& env) {
  std::vector<std::pair<int, RType>> entries(env.begin(), env.end());
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  std::uint64_t h = 1469598103934665603ULL;
  for (const auto& entry : entries) {
    h = mix_u64(h, static_cast<std::uint64_t>(entry.first + 1));
    h = mix_u64(h, static_cast<std::uint64_t>(entry.second));
  }
  return h;
}

void append_root(std::vector<TypedExprRoot>* out,
                 const AstProgram& p,
                 std::size_t start,
                 std::size_t stop,
                 RType type,
                 const std::unordered_map<int, RType>& env) {
  if (out == nullptr || type == RType::Invalid) {
    return;
  }
  const std::uint64_t visible = env_signature(env);
  TypedExprRoot root;
  root.start = start;
  root.stop = stop;
  root.type = type;
  root.scope_signature = visible;
  root.visible_env_signature = visible;
  out->push_back(root);
}

RType list_elem_type(RType type) {
  if (type == RType::IntList) return RType::Int;
  if (type == RType::FloatList) return RType::Float;
  if (type == RType::StringList) return RType::String;
  return RType::Invalid;
}

RType list_type_for_elem(RType type) {
  if (type == RType::Int) return RType::IntList;
  if (type == RType::Float) return RType::FloatList;
  if (type == RType::String) return RType::StringList;
  return RType::Invalid;
}

RType list_type_for_tag(int tag) {
  if (tag == static_cast<int>(ListTypeTag::Int)) return RType::IntList;
  if (tag == static_cast<int>(ListTypeTag::Float)) return RType::FloatList;
  if (tag == static_cast<int>(ListTypeTag::String)) return RType::StringList;
  return RType::Invalid;
}

const LinearRecBinders* linear_rec_binders_for_node(const AstProgram& p, std::size_t idx) {
  for (const LinearRecBinders& binders : p.linear_rec_binders) {
    if (binders.node_index == idx) return &binders;
  }
  return nullptr;
}

const AsgpDcBinders* asgp_dc_binders_for_node(const AstProgram& p, std::size_t idx) {
  for (const AsgpDcBinders& binders : p.asgp_dc_binders) {
    if (binders.node_index == idx) return &binders;
  }
  return nullptr;
}

const AsgpDp1dSpec* asgp_dp1d_spec_for_node(const AstProgram& p, std::size_t idx) {
  for (const AsgpDp1dSpec& spec : p.asgp_dp1d_specs) {
    if (spec.node_index == idx) return &spec;
  }
  return nullptr;
}

const AsgpDp2dSpec* asgp_dp2d_spec_for_node(const AstProgram& p, std::size_t idx) {
  for (const AsgpDp2dSpec& spec : p.asgp_dp2d_specs) {
    if (spec.node_index == idx) return &spec;
  }
  return nullptr;
}

RType const_type_for_index(const AstProgram& p, int const_index) {
  if (const_index < 0 || static_cast<std::size_t>(const_index) >= p.consts.size()) {
    return RType::Invalid;
  }
  const Value& v = p.consts[static_cast<std::size_t>(const_index)];
  if (v.tag == ValueTag::Bool) return RType::Bool;
  if (v.tag == ValueTag::Char) return RType::Char;
  if (v.tag == ValueTag::String) return RType::String;
  if (v.tag == ValueTag::IntList) return RType::IntList;
  if (v.tag == ValueTag::FloatList) return RType::FloatList;
  if (v.tag == ValueTag::StringList) return RType::StringList;
  if (v.tag == ValueTag::Int) return RType::Int;
  if (v.tag == ValueTag::Float) return RType::Float;
  return RType::Invalid;
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
    RType t = RType::Int;
    if (n.i0 < 0 || static_cast<std::size_t>(n.i0) >= p.consts.size()) t = RType::Invalid;
    else {
      const Value& v = p.consts[static_cast<std::size_t>(n.i0)];
      if (v.tag == ValueTag::Invalid) t = RType::Invalid;
      else if (v.tag == ValueTag::Bool) t = RType::Bool;
      else if (v.tag == ValueTag::Char) t = RType::Char;
      else if (v.tag == ValueTag::String) t = RType::String;
      else if (v.tag == ValueTag::IntList) t = RType::IntList;
      else if (v.tag == ValueTag::FloatList) t = RType::FloatList;
      else if (v.tag == ValueTag::StringList) t = RType::StringList;
      else if (v.tag == ValueTag::Int) t = RType::Int;
      else if (v.tag == ValueTag::Float) t = RType::Float;
      else t = RType::Invalid;
    }
    if (out != nullptr && t != RType::Invalid) append_root(out, p, idx, end[idx], t, env);
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::VAR) {
    auto it = env.find(n.i0);
    const RType t = (it == env.end()) ? infer_unbound_var_type(p, n.i0) : it->second;
    if (out != nullptr) append_root(out, p, idx, end[idx], t, env);
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::BOUND_VAR) {
    auto it = env.find(n.i0);
    const RType t = (it == env.end()) ? RType::Invalid : it->second;
    return {t, end[idx]};
  }
  if (n.kind == NodeKind::NEG || n.kind == NodeKind::NOT) {
    ExprCheck e = infer_expr_prefix(p, end, idx + 1, env, out);
    const RType t = (n.kind == NodeKind::NEG) ? (is_numeric_type(e.t) ? e.t : RType::Invalid)
                                               : (e.t == RType::Bool ? RType::Bool : RType::Invalid);
    if (out != nullptr && t != RType::Invalid) append_root(out, p, idx, e.next, t, env);
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
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, f.next, r, env);
    return {r, f.next};
  }
  if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN || n.kind == NodeKind::CALL_MAX ||
      n.kind == NodeKind::CALL_CLIP || n.kind == NodeKind::CALL_IDIV0 || n.kind == NodeKind::CALL_IMOD0 ||
      n.kind == NodeKind::CALL_LEN || n.kind == NodeKind::CALL_CONCAT ||
      n.kind == NodeKind::CALL_SLICE || n.kind == NodeKind::CALL_INDEX || n.kind == NodeKind::CALL_APPEND ||
      n.kind == NodeKind::CALL_PREPEND || n.kind == NodeKind::CALL_REVERSE ||
      n.kind == NodeKind::CALL_FIND || n.kind == NodeKind::CALL_CONTAINS ||
      n.kind == NodeKind::CALL_CHAR_TO_STRING || n.kind == NodeKind::CALL_STRING_TO_CHAR ||
      n.kind == NodeKind::CALL_ORD || n.kind == NodeKind::CALL_CHR ||
      n.kind == NodeKind::CALL_IS_LETTER || n.kind == NodeKind::CALL_IS_DIGIT ||
      n.kind == NodeKind::CALL_IS_SPACE || n.kind == NodeKind::CALL_IS_VOWEL ||
      n.kind == NodeKind::CALL_TO_LOWER || n.kind == NodeKind::CALL_TO_UPPER ||
      n.kind == NodeKind::CALL_TO_STRING || n.kind == NodeKind::CALL_SINGLETON) {
    std::size_t cur = idx + 1;
    bool ok = true;
    if (n.kind == NodeKind::CALL_LEN) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      if (!is_sequence_type(a.t)) ok = false;
      cur = a.next;
      const RType r = ok ? RType::Int : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, cur, r, env);
      return {r, cur};
    }
    if (n.kind == NodeKind::CALL_CONCAT) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck b = infer_expr_prefix(p, end, a.next, env, out);
      const RType r = (a.t == b.t && is_sequence_type(a.t)) ? a.t : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, b.next, r, env);
      return {r, b.next};
    }
    if (n.kind == NodeKind::CALL_SLICE) {
      ExprCheck x = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck lo = infer_expr_prefix(p, end, x.next, env, out);
      ExprCheck hi = infer_expr_prefix(p, end, lo.next, env, out);
      const RType r = (is_sequence_type(x.t) && lo.t == RType::Int && hi.t == RType::Int)
                          ? x.t
                          : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, hi.next, r, env);
      return {r, hi.next};
    }
    if (n.kind == NodeKind::CALL_INDEX) {
      ExprCheck x = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck i = infer_expr_prefix(p, end, x.next, env, out);
      RType r = RType::Invalid;
      if (i.t == RType::Int) {
        if (x.t == RType::String) r = RType::Char;
        else if (x.t == RType::IntList) r = RType::Int;
        else if (x.t == RType::FloatList) r = RType::Float;
        else if (x.t == RType::StringList) r = RType::String;
      }
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, i.next, r, env);
      return {r, i.next};
    }
    if (n.kind == NodeKind::CALL_APPEND || n.kind == NodeKind::CALL_PREPEND) {
      ExprCheck xs = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck elem = infer_expr_prefix(p, end, xs.next, env, out);
      const RType r = ((xs.t == RType::IntList && elem.t == RType::Int) ||
                       (xs.t == RType::FloatList && elem.t == RType::Float) ||
                       (xs.t == RType::StringList && elem.t == RType::String))
                          ? xs.t
                          : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, elem.next, r, env);
      return {r, elem.next};
    }
    if (n.kind == NodeKind::CALL_REVERSE) {
      ExprCheck x = infer_expr_prefix(p, end, cur, env, out);
      const RType r = is_sequence_type(x.t) ? x.t : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, x.next, r, env);
      return {r, x.next};
    }
    if (n.kind == NodeKind::CALL_FIND || n.kind == NodeKind::CALL_CONTAINS) {
      ExprCheck haystack = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck needle = infer_expr_prefix(p, end, haystack.next, env, out);
      const RType r = (haystack.t == RType::String && needle.t == RType::String)
                          ? (n.kind == NodeKind::CALL_FIND ? RType::Int : RType::Bool)
                          : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, needle.next, r, env);
      return {r, needle.next};
    }
    if (n.kind == NodeKind::CALL_IDIV0 || n.kind == NodeKind::CALL_IMOD0) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      ExprCheck b = infer_expr_prefix(p, end, a.next, env, out);
      const RType r = (a.t == RType::Int && b.t == RType::Int) ? RType::Int : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, b.next, r, env);
      return {r, b.next};
    }
    if (n.kind == NodeKind::CALL_CHAR_TO_STRING || n.kind == NodeKind::CALL_ORD ||
        n.kind == NodeKind::CALL_IS_LETTER || n.kind == NodeKind::CALL_IS_DIGIT ||
        n.kind == NodeKind::CALL_IS_SPACE || n.kind == NodeKind::CALL_IS_VOWEL ||
        n.kind == NodeKind::CALL_TO_LOWER || n.kind == NodeKind::CALL_TO_UPPER) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      RType r = RType::Invalid;
      if (a.t == RType::Char) {
        if (n.kind == NodeKind::CALL_CHAR_TO_STRING) r = RType::String;
        else if (n.kind == NodeKind::CALL_ORD) r = RType::Int;
        else if (n.kind == NodeKind::CALL_TO_LOWER || n.kind == NodeKind::CALL_TO_UPPER) r = RType::Char;
        else r = RType::Bool;
      }
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, a.next, r, env);
      return {r, a.next};
    }
    if (n.kind == NodeKind::CALL_STRING_TO_CHAR) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      const RType r = (a.t == RType::String) ? RType::Char : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, a.next, r, env);
      return {r, a.next};
    }
    if (n.kind == NodeKind::CALL_CHR) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      const RType r = (a.t == RType::Int) ? RType::Char : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, a.next, r, env);
      return {r, a.next};
    }
    if (n.kind == NodeKind::CALL_TO_STRING) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      const RType r = is_numeric_type(a.t) ? RType::String : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, a.next, r, env);
      return {r, a.next};
    }
    if (n.kind == NodeKind::CALL_SINGLETON) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      RType r = RType::Invalid;
      if (a.t == RType::Char) r = RType::String;
      else if (a.t == RType::Int) r = RType::IntList;
      else if (a.t == RType::Float) r = RType::FloatList;
      else if (a.t == RType::String) r = RType::StringList;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, a.next, r, env);
      return {r, a.next};
    }
    if (n.kind == NodeKind::CALL_ABS || n.kind == NodeKind::CALL_MIN ||
        n.kind == NodeKind::CALL_MAX || n.kind == NodeKind::CALL_CLIP) {
      RType numeric_type = RType::Invalid;
      for (int i = 0; i < subtree::node_arity(n.kind); ++i) {
        ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
        if (!is_numeric_type(a.t)) {
          ok = false;
        } else if (numeric_type == RType::Invalid) {
          numeric_type = a.t;
        } else if (numeric_type != a.t) {
          ok = false;
        }
        cur = a.next;
      }
      const RType r = ok ? numeric_type : RType::Invalid;
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, cur, r, env);
      return {r, cur};
    }
    for (int i = 0; i < subtree::node_arity(n.kind); ++i) {
      ExprCheck a = infer_expr_prefix(p, end, cur, env, out);
      if (a.t != RType::Int) ok = false;
      cur = a.next;
    }
    const RType r = ok ? RType::Int : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, cur, r, env);
    return {r, cur};
  }
  if (n.kind == NodeKind::MAP_LIST) {
    ExprCheck source = infer_expr_prefix(p, end, idx + 1, env, out);
    const RType elem_t = list_elem_type(source.t);
    std::unordered_map<int, RType> body_env = env;
    if (elem_t != RType::Invalid) {
      body_env[n.i0] = elem_t;
    }
    ExprCheck body = infer_expr_prefix(p, end, source.next, body_env, out);
    const RType result_t = list_type_for_tag(n.i1);
    const RType expected_body_t = list_elem_type(result_t);
    const RType r = (elem_t != RType::Invalid && body.t == expected_body_t) ? result_t : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, body.next, r, env);
    return {r, body.next};
  }
  if (n.kind == NodeKind::FILTER_LIST) {
    ExprCheck source = infer_expr_prefix(p, end, idx + 1, env, out);
    const RType elem_t = list_elem_type(source.t);
    std::unordered_map<int, RType> pred_env = env;
    if (elem_t != RType::Invalid) {
      pred_env[n.i0] = elem_t;
    }
    ExprCheck pred = infer_expr_prefix(p, end, source.next, pred_env, out);
    const RType r = (elem_t != RType::Invalid && pred.t == RType::Bool) ? source.t : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, pred.next, r, env);
    return {r, pred.next};
  }
  if (n.kind == NodeKind::LINEAR_REC) {
    const LinearRecBinders* binders = linear_rec_binders_for_node(p, idx);
    ExprCheck source = infer_expr_prefix(p, end, idx + 1, env, out);
    ExprCheck start_idx = infer_expr_prefix(p, end, source.next, env, out);
    ExprCheck empty_case = infer_expr_prefix(p, end, start_idx.next, env, out);
    std::size_t step_idx = empty_case.next;
    std::size_t last_idx = end[step_idx];
    const RType elem_t = list_elem_type(source.t);
    RType r = RType::Invalid;
    if (binders != nullptr && elem_t != RType::Invalid && start_idx.t == RType::Int && is_value_type(empty_case.t)) {
      std::unordered_map<int, RType> step_env = env;
      step_env[binders->elem_name] = elem_t;
      step_env[binders->accum_name] = empty_case.t;
      step_env[binders->index_name] = RType::Int;
      ExprCheck step = infer_expr_prefix(p, end, step_idx, step_env, out);

      std::unordered_map<int, RType> last_env = env;
      last_env[binders->elem_name] = elem_t;
      last_env[binders->index_name] = RType::Int;
      ExprCheck last = infer_expr_prefix(p, end, last_idx, last_env, out);
      if (step.t == empty_case.t && last.t == empty_case.t) {
        r = empty_case.t;
      }
      if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, last.next, r, env);
      return {r, last.next};
    }
    ExprCheck step = infer_expr_prefix(p, end, step_idx, env, out);
    ExprCheck last = infer_expr_prefix(p, end, step.next, env, out);
    return {RType::Invalid, last.next};
  }
  if (n.kind == NodeKind::ASGP_DC) {
    const AsgpDcBinders* binders = asgp_dc_binders_for_node(p, idx);
    ExprCheck source = infer_expr_prefix(p, end, idx + 1, env, out);
    const std::size_t solve_idx = source.next;
    const std::size_t divide_idx = solve_idx < end.size() ? end[solve_idx] : solve_idx;
    const std::size_t combine_idx = divide_idx < end.size() ? end[divide_idx] : divide_idx;
    const std::size_t next = combine_idx < end.size() ? end[combine_idx] : end[idx];
    RType r = RType::Invalid;
    if (binders != nullptr && is_sequence_type(source.t) && solve_idx < p.nodes.size() &&
        divide_idx < p.nodes.size() && combine_idx < p.nodes.size()) {
      std::unordered_map<int, RType> solve_env;
      solve_env[binders->solve_xs_name] = source.t;
      solve_env[binders->solve_n_name] = RType::Int;
      solve_env[binders->solve_lo_name] = RType::Int;
      ExprCheck solve = infer_expr_prefix(p, end, solve_idx, solve_env, out);

      std::unordered_map<int, RType> divide_env;
      divide_env[binders->divide_n_name] = RType::Int;
      ExprCheck divide = infer_expr_prefix(p, end, divide_idx, divide_env, out);

      std::unordered_map<int, RType> combine_env;
      combine_env[binders->combine_left_name] = solve.t;
      combine_env[binders->combine_right_name] = solve.t;
      ExprCheck combine = infer_expr_prefix(p, end, combine_idx, combine_env, out);
      if (is_value_type(solve.t) && divide.t == RType::Int && combine.t == solve.t) {
        r = solve.t;
      }
    }
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, next, r, env);
    return {r, next};
  }
  if (n.kind == NodeKind::ASGP_DP1D) {
    const AsgpDp1dSpec* spec = asgp_dp1d_spec_for_node(p, idx);
    ExprCheck state = infer_expr_prefix(p, end, idx + 1, env, out);
    const std::size_t solve_idx = state.next;
    const std::size_t transition_idx = solve_idx < end.size() ? end[solve_idx] : solve_idx;
    const std::size_t next = transition_idx < end.size() ? end[transition_idx] : end[idx];
    RType r = RType::Invalid;
    if (spec != nullptr && state.t == RType::Int && solve_idx < p.nodes.size() &&
        transition_idx < p.nodes.size()) {
      std::unordered_map<int, RType> solve_env;
      solve_env[spec->solve_state_name] = RType::Int;
      ExprCheck solve = infer_expr_prefix(p, end, solve_idx, solve_env, out);

      std::unordered_map<int, RType> transition_env;
      transition_env[spec->transition_state_name] = RType::Int;
      for (int name_id : spec->transition_dep_names) {
        transition_env[name_id] = solve.t;
      }
      ExprCheck transition = infer_expr_prefix(p, end, transition_idx, transition_env, out);
      if (is_value_type(solve.t) && const_type_for_index(p, spec->boundary_const) == solve.t &&
          transition.t == solve.t) {
        r = solve.t;
      }
    }
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, next, r, env);
    return {r, next};
  }
  if (n.kind == NodeKind::ASGP_DP2D) {
    const AsgpDp2dSpec* spec = asgp_dp2d_spec_for_node(p, idx);
    ExprCheck state_i = infer_expr_prefix(p, end, idx + 1, env, out);
    ExprCheck state_j = infer_expr_prefix(p, end, state_i.next, env, out);
    const std::size_t solve_idx = state_j.next;
    const std::size_t transition_idx = solve_idx < end.size() ? end[solve_idx] : solve_idx;
    const std::size_t next = transition_idx < end.size() ? end[transition_idx] : end[idx];
    RType r = RType::Invalid;
    if (spec != nullptr && state_i.t == RType::Int && state_j.t == RType::Int &&
        solve_idx < p.nodes.size() && transition_idx < p.nodes.size()) {
      std::unordered_map<int, RType> solve_env;
      solve_env[spec->solve_i_name] = RType::Int;
      solve_env[spec->solve_j_name] = RType::Int;
      ExprCheck solve = infer_expr_prefix(p, end, solve_idx, solve_env, out);

      std::unordered_map<int, RType> transition_env;
      transition_env[spec->transition_i_name] = RType::Int;
      transition_env[spec->transition_j_name] = RType::Int;
      for (int name_id : spec->transition_dep_names) {
        transition_env[name_id] = solve.t;
      }
      ExprCheck transition = infer_expr_prefix(p, end, transition_idx, transition_env, out);
      if (is_value_type(solve.t) && const_type_for_index(p, spec->boundary_const) == solve.t &&
          transition.t == solve.t) {
        r = solve.t;
      }
    }
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, next, r, env);
    return {r, next};
  }
  if (n.kind == NodeKind::ADD || n.kind == NodeKind::SUB || n.kind == NodeKind::MUL || n.kind == NodeKind::DIV || n.kind == NodeKind::MOD ||
      n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT || n.kind == NodeKind::GE ||
      n.kind == NodeKind::EQ || n.kind == NodeKind::NE || n.kind == NodeKind::AND || n.kind == NodeKind::OR) {
    ExprCheck a = infer_expr_prefix(p, end, idx + 1, env, out);
    ExprCheck b = infer_expr_prefix(p, end, a.next, env, out);
    RType r = RType::Invalid;
    if (n.kind == NodeKind::AND || n.kind == NodeKind::OR) r = (a.t == RType::Bool && b.t == RType::Bool) ? RType::Bool : RType::Invalid;
    else if (n.kind == NodeKind::LT || n.kind == NodeKind::LE || n.kind == NodeKind::GT || n.kind == NodeKind::GE)
      r = (a.t == b.t && is_numeric_type(a.t)) ? RType::Bool : RType::Invalid;
    else if (n.kind == NodeKind::EQ || n.kind == NodeKind::NE)
      r = (a.t == b.t && is_value_type(a.t))
              ? RType::Bool
              : RType::Invalid;
    else
      r = (a.t == b.t && is_numeric_type(a.t))
              ? ((n.kind == NodeKind::DIV) ? RType::Float : a.t)
              : RType::Invalid;
    if (out != nullptr && r != RType::Invalid) append_root(out, p, idx, b.next, r, env);
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
    const ExprCheck bound = infer_expr_prefix(p, end, idx + 1, env, &out);
    std::unordered_map<int, RType> env_b = env;
    env_b[n.i0] = RType::Int;
    std::size_t cur = bound.next;
    while (cur < p.nodes.size() && p.nodes[cur].kind == NodeKind::BLOCK_CONS) {
      collect_typed_exprs_in_stmt(p, end, cur + 1, env_b, out);
      cur = end[cur + 1];
    }
  }
}

void annotate_typed_root(const AstProgram& program,
                         const std::vector<std::size_t>& subtree_end,
                         TypedExprRoot& root) {
  if (root.start >= program.nodes.size() || subtree_end.size() != program.nodes.size()) {
    return;
  }
  const NodeKind root_kind = program.nodes[root.start].kind;
  if (is_structured_root_kind(root_kind)) {
    root.scheme_kind = static_cast<int>(root_kind);
    root.binder_signature = node_signature(root_kind);
    if (root_kind == NodeKind::MAP_LIST || root_kind == NodeKind::FILTER_LIST) {
      root.binder_signature = mix_u64(root.binder_signature, static_cast<std::uint64_t>(program.nodes[root.start].i1));
    } else if (root_kind == NodeKind::LINEAR_REC) {
      const LinearRecBinders* binders = linear_rec_binders_for_node(program, root.start);
      if (binders != nullptr) {
        root.binder_signature = mix_u64(root.binder_signature, 3);
      }
    } else if (root_kind == NodeKind::ASGP_DP1D) {
      const AsgpDp1dSpec* spec = asgp_dp1d_spec_for_node(program, root.start);
      if (spec != nullptr) {
        root.dp_dependency_arity = static_cast<int>(spec->transition_dep_names.size());
        root.binder_signature = mix_u64(root.binder_signature, static_cast<std::uint64_t>(root.dp_dependency_arity));
      }
    } else if (root_kind == NodeKind::ASGP_DP2D) {
      const AsgpDp2dSpec* spec = asgp_dp2d_spec_for_node(program, root.start);
      if (spec != nullptr) {
        root.dp_dependency_arity = static_cast<int>(spec->transition_dep_names.size());
        root.binder_signature = mix_u64(root.binder_signature, static_cast<std::uint64_t>(root.dp_dependency_arity));
      }
    }
  }

  for (std::size_t idx = 0; idx < program.nodes.size(); ++idx) {
    const NodeKind kind = program.nodes[idx].kind;
    if (kind == NodeKind::ASGP_DC) {
      const std::size_t source = idx + 1;
      if (source >= program.nodes.size()) continue;
      const std::size_t solve = subtree_end[source];
      if (solve >= program.nodes.size()) continue;
      const std::size_t divide = subtree_end[solve];
      if (divide >= program.nodes.size()) continue;
      const std::size_t combine = subtree_end[divide];
      if (combine >= program.nodes.size()) continue;
      if (range_contains(solve, subtree_end[solve], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DC);
        root.phase_name = 1;
        root.binder_signature = node_signature(NodeKind::ASGP_DC);
      } else if (range_contains(divide, subtree_end[divide], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DC);
        root.phase_name = 2;
        root.binder_signature = node_signature(NodeKind::ASGP_DC);
      } else if (range_contains(combine, subtree_end[combine], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DC);
        root.phase_name = 3;
        root.binder_signature = node_signature(NodeKind::ASGP_DC);
      }
    } else if (kind == NodeKind::ASGP_DP1D) {
      const std::size_t state = idx + 1;
      if (state >= program.nodes.size()) continue;
      const std::size_t solve = subtree_end[state];
      if (solve >= program.nodes.size()) continue;
      const std::size_t transition = subtree_end[solve];
      if (transition >= program.nodes.size()) continue;
      const AsgpDp1dSpec* spec = asgp_dp1d_spec_for_node(program, idx);
      const int dep_arity = spec == nullptr ? -1 : static_cast<int>(spec->transition_dep_names.size());
      if (range_contains(solve, subtree_end[solve], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DP1D);
        root.phase_name = 1;
        root.dp_dependency_arity = dep_arity;
        root.binder_signature = mix_u64(node_signature(NodeKind::ASGP_DP1D), static_cast<std::uint64_t>(dep_arity + 1));
      } else if (range_contains(transition, subtree_end[transition], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DP1D);
        root.phase_name = 2;
        root.dp_dependency_arity = dep_arity;
        root.binder_signature = mix_u64(node_signature(NodeKind::ASGP_DP1D), static_cast<std::uint64_t>(dep_arity + 1));
      }
    } else if (kind == NodeKind::ASGP_DP2D) {
      const std::size_t state_i = idx + 1;
      if (state_i >= program.nodes.size()) continue;
      const std::size_t state_j = subtree_end[state_i];
      if (state_j >= program.nodes.size()) continue;
      const std::size_t solve = subtree_end[state_j];
      if (solve >= program.nodes.size()) continue;
      const std::size_t transition = subtree_end[solve];
      if (transition >= program.nodes.size()) continue;
      const AsgpDp2dSpec* spec = asgp_dp2d_spec_for_node(program, idx);
      const int dep_arity = spec == nullptr ? -1 : static_cast<int>(spec->transition_dep_names.size());
      if (range_contains(solve, subtree_end[solve], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DP2D);
        root.phase_name = 1;
        root.dp_dependency_arity = dep_arity;
        root.binder_signature = mix_u64(node_signature(NodeKind::ASGP_DP2D), static_cast<std::uint64_t>(dep_arity + 1));
      } else if (range_contains(transition, subtree_end[transition], root.start)) {
        root.scheme_kind = static_cast<int>(NodeKind::ASGP_DP2D);
        root.phase_name = 2;
        root.dp_dependency_arity = dep_arity;
        root.binder_signature = mix_u64(node_signature(NodeKind::ASGP_DP2D), static_cast<std::uint64_t>(dep_arity + 1));
      }
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
  out.erase(std::remove_if(out.begin(), out.end(), [&](const TypedExprRoot& root) {
              if (root.start >= program.nodes.size()) return true;
              if (is_structured_root_kind(program.nodes[root.start].kind)) return false;
              return subtree_contains_bound_var(program, root.start, root.stop);
            }),
            out.end());
  for (TypedExprRoot& root : out) {
    annotate_typed_root(program, subtree_end, root);
  }
  return out;
}

std::vector<TypedExprRoot> collect_typed_expr_roots(const AstProgram& program,
                                                    const VerifiedAst& verified) {
  std::vector<TypedExprRoot> out;
  if (program.nodes.size() != verified.subtree_end.size() ||
      program.nodes.size() != verified.expression_types.size() ||
      program.nodes.size() != verified.expression_scope_signatures.size() ||
      program.nodes.size() != verified.expression_binder_signatures.size()) {
    return out;
  }
  for (std::size_t index = 0; index < program.nodes.size(); ++index) {
    const NodeDescriptor& descriptor = node_descriptor(program.nodes[index].kind);
    if (!descriptor.typed_subtree_eligible ||
        verified.expression_types[index] == RType::Invalid) {
      continue;
    }
    const std::size_t stop = verified.subtree_end[index];
    if (!is_structured_root_kind(program.nodes[index].kind) &&
        subtree_contains_bound_var(program, index, stop)) {
      continue;
    }
    TypedExprRoot root;
    root.start = index;
    root.stop = stop;
    root.type = verified.expression_types[index];
    root.scope_signature = verified.expression_scope_signatures[index];
    root.visible_env_signature = verified.expression_scope_signatures[index];
    root.binder_signature = verified.expression_binder_signatures[index];
    annotate_typed_root(program, verified.subtree_end, root);
    out.push_back(root);
  }
  return out;
}

bool is_asgp_phase_body_root(const AstProgram& program,
                             const std::vector<std::size_t>& subtree_end,
                             const TypedExprRoot& root) {
  if (root.start >= program.nodes.size() || subtree_end.size() != program.nodes.size()) {
    return false;
  }
  for (std::size_t idx = 0; idx < program.nodes.size(); ++idx) {
    const NodeKind kind = program.nodes[idx].kind;
    if (kind == NodeKind::ASGP_DC) {
      const std::size_t source = idx + 1;
      if (source >= program.nodes.size()) continue;
      const std::size_t solve = subtree_end[source];
      if (solve >= program.nodes.size()) continue;
      const std::size_t divide = subtree_end[solve];
      if (divide >= program.nodes.size()) continue;
      const std::size_t combine = subtree_end[divide];
      if (range_contains(solve, subtree_end[solve], root.start) ||
          range_contains(divide, subtree_end[divide], root.start) ||
          range_contains(combine, subtree_end[combine], root.start)) {
        return true;
      }
    } else if (kind == NodeKind::ASGP_DP1D) {
      const std::size_t state = idx + 1;
      if (state >= program.nodes.size()) continue;
      const std::size_t solve = subtree_end[state];
      if (solve >= program.nodes.size()) continue;
      const std::size_t transition = subtree_end[solve];
      if (transition >= program.nodes.size()) continue;
      if (range_contains(solve, subtree_end[solve], root.start) ||
          range_contains(transition, subtree_end[transition], root.start)) {
        return true;
      }
    } else if (kind == NodeKind::ASGP_DP2D) {
      const std::size_t state_i = idx + 1;
      if (state_i >= program.nodes.size()) continue;
      const std::size_t state_j = subtree_end[state_i];
      if (state_j >= program.nodes.size()) continue;
      const std::size_t solve = subtree_end[state_j];
      if (solve >= program.nodes.size()) continue;
      const std::size_t transition = subtree_end[solve];
      if (transition >= program.nodes.size()) continue;
      if (range_contains(solve, subtree_end[solve], root.start) ||
          range_contains(transition, subtree_end[transition], root.start)) {
        return true;
      }
    }
  }
  return false;
}

bool is_statement_value_root(const AstProgram& program, const TypedExprRoot& root) {
  if (root.start >= program.nodes.size()) {
    return false;
  }
  for (std::size_t idx = 0; idx + 1 < program.nodes.size(); ++idx) {
    const NodeKind kind = program.nodes[idx].kind;
    if ((kind == NodeKind::ASSIGN || kind == NodeKind::RETURN) && idx + 1 == root.start) {
      return true;
    }
  }
  return false;
}

bool typed_subtree_keys_compatible(const TypedExprRoot& a, const TypedExprRoot& b) {
  return a.type == b.type &&
         a.scope_signature == b.scope_signature &&
         a.binder_signature == b.binder_signature &&
         a.scheme_kind == b.scheme_kind &&
         a.phase_name == b.phase_name &&
         a.visible_env_signature == b.visible_env_signature &&
         a.dp_dependency_arity == b.dp_dependency_arity;
}

}  // namespace gagp::evo::typed_expr
