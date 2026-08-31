#include "g3pvm/evolution/ast_verify.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "g3pvm/evolution/node_descriptor.hpp"

namespace g3pvm::evo {
namespace {

using TypeEnv = std::unordered_map<int, RType>;

std::string node_path(std::size_t index) {
  return "$.nodes[" + std::to_string(index) + "]";
}

bool is_value_type(RType type) {
  return type == RType::Int || type == RType::Float || type == RType::Bool ||
         type == RType::Char || type == RType::String || type == RType::IntList ||
         type == RType::FloatList || type == RType::StringList;
}

bool is_numeric_type(RType type) {
  return type == RType::Int || type == RType::Float;
}

bool is_sequence_type(RType type) {
  return type == RType::String || type == RType::IntList ||
         type == RType::FloatList || type == RType::StringList;
}

RType list_element_type(RType type) {
  if (type == RType::IntList) return RType::Int;
  if (type == RType::FloatList) return RType::Float;
  if (type == RType::StringList) return RType::String;
  return RType::Invalid;
}

RType list_type_from_tag(int tag) {
  if (tag == static_cast<int>(ListTypeTag::Int)) return RType::IntList;
  if (tag == static_cast<int>(ListTypeTag::Float)) return RType::FloatList;
  if (tag == static_cast<int>(ListTypeTag::String)) return RType::StringList;
  return RType::Invalid;
}

RType value_type(const Value& value) {
  switch (value.tag) {
    case ValueTag::Int: return RType::Int;
    case ValueTag::Float: return RType::Float;
    case ValueTag::Bool: return RType::Bool;
    case ValueTag::Char: return RType::Char;
    case ValueTag::String: return RType::String;
    case ValueTag::IntList: return RType::IntList;
    case ValueTag::FloatList: return RType::FloatList;
    case ValueTag::StringList: return RType::StringList;
    case ValueTag::FallbackToken:
    case ValueTag::Invalid: return RType::Invalid;
  }
  return RType::Invalid;
}

const char* type_name(RType type) {
  switch (type) {
    case RType::Int: return "Int";
    case RType::Float: return "Float";
    case RType::Bool: return "Bool";
    case RType::Char: return "Char";
    case RType::String: return "String";
    case RType::IntList: return "IntList";
    case RType::FloatList: return "FloatList";
    case RType::StringList: return "StringList";
    case RType::Any: return "Any";
    case RType::Invalid: return "Invalid";
  }
  return "Invalid";
}

bool is_asgp_kind(NodeKind kind) {
  return kind == NodeKind::ASGP_DC || kind == NodeKind::ASGP_DP1D ||
         kind == NodeKind::ASGP_DP2D;
}

std::uint64_t type_env_signature(const TypeEnv& env) {
  std::vector<std::pair<int, RType>> entries(env.begin(), env.end());
  std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
    return left.first < right.first;
  });
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto& entry : entries) {
    hash ^= static_cast<std::uint64_t>(entry.first + 1);
    hash *= 1099511628211ULL;
    hash ^= static_cast<std::uint64_t>(entry.second) + 1ULL;
    hash *= 1099511628211ULL;
  }
  return hash;
}

class TypedVerifier {
 public:
  TypedVerifier(const AstProgram& ast, const std::vector<InputSpec>& inputs,
                const VerifyOptions& options, AstVerifyResult structural)
      : ast_(ast), inputs_(inputs), options_(options), result_(std::move(structural)) {}

  AstVerifyResult run() {
    if (!result_) return result_;
    if (!prepare_names_and_inputs()) return result_;
    if (!check_grammar_config()) return result_;

    TypeEnv locals;
    for (const InputSpec& input : inputs_) {
      const auto it = name_to_id_.find(input.name);
      if (it != name_to_id_.end()) locals[it->second] = input.type;
    }
    TypeEnv binders;
    if (!verify_block(1, &locals, binders)) return result_;
    if (!saw_return_) {
      return fail(VerifyCode::MissingReturn, 0, "$.nodes",
                  "a well-typed program must contain at least one Return");
    }
    result_.verified.return_type = return_type_;
    result_.ok = true;
    result_.diagnostic = VerifyDiagnostic{};
    return result_;
  }

 private:
  struct ExprResult {
    RType type = RType::Invalid;
    std::size_t next = 0;
  };

  AstVerifyResult fail(VerifyCode code, std::size_t node_index, std::string path,
                       std::string message) {
    result_.ok = false;
    result_.diagnostic = VerifyDiagnostic{code, node_index, std::move(path), std::move(message)};
    return result_;
  }

  ExprResult fail_expr(VerifyCode code, std::size_t node_index, std::string message) {
    (void)fail(code, node_index, node_path(node_index), std::move(message));
    return ExprResult{RType::Invalid, result_.verified.subtree_end[node_index]};
  }

  bool fail_bool(VerifyCode code, std::size_t node_index, std::string path,
                 std::string message) {
    (void)fail(code, node_index, std::move(path), std::move(message));
    return false;
  }

  bool prepare_names_and_inputs() {
    for (std::size_t i = 0; i < ast_.names.size(); ++i) {
      if (!name_to_id_.emplace(ast_.names[i], static_cast<int>(i)).second) {
        return fail_bool(VerifyCode::DuplicateName, 0,
                         "$.names[" + std::to_string(i) + "]",
                         "AST names must be unique because runtime locals are keyed by name");
      }
    }
    std::set<std::string> input_names;
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      const InputSpec& input = inputs_[i];
      if (!input_names.insert(input.name).second) {
        return fail_bool(VerifyCode::DuplicateInput, 0,
                         "$.inputs[" + std::to_string(i) + "]",
                         "input specifications must have unique names");
      }
      if (input.name.empty() || !is_value_type(input.type)) {
        return fail_bool(VerifyCode::InvalidInputType, 0,
                         "$.inputs[" + std::to_string(i) + "]",
                         "input specifications require a name and one exact public value type");
      }
    }
    return true;
  }

  bool check_grammar_config() {
    if (options_.grammar_config == nullptr) return true;
    for (std::size_t i = 0; i < ast_.nodes.size(); ++i) {
      if (!options_.grammar_config->allows_node_kind(ast_.nodes[i].kind)) {
        return fail_bool(VerifyCode::GrammarConfigDisallowed, i, node_path(i),
                         "node is valid in the language but disabled by the supplied grammar config");
      }
    }
    for (std::size_t i = 0; i < ast_.consts.size(); ++i) {
      if (!options_.grammar_config->allows_type(value_type(ast_.consts[i]))) {
        return fail_bool(VerifyCode::GrammarConfigDisallowed, 0,
                         "$.consts[" + std::to_string(i) + "]",
                         "constant type is disabled by the supplied grammar config");
      }
    }
    return true;
  }

  const LinearRecBinders& linear_metadata(std::size_t node_index) const {
    for (const LinearRecBinders& row : ast_.linear_rec_binders) {
      if (row.node_index == node_index) return row;
    }
    return ast_.linear_rec_binders.front();
  }

  const AsgpDcBinders& dc_metadata(std::size_t node_index) const {
    for (const AsgpDcBinders& row : ast_.asgp_dc_binders) {
      if (row.node_index == node_index) return row;
    }
    return ast_.asgp_dc_binders.front();
  }

  const AsgpDp1dSpec& dp1_metadata(std::size_t node_index) const {
    for (const AsgpDp1dSpec& row : ast_.asgp_dp1d_specs) {
      if (row.node_index == node_index) return row;
    }
    return ast_.asgp_dp1d_specs.front();
  }

  const AsgpDp2dSpec& dp2_metadata(std::size_t node_index) const {
    for (const AsgpDp2dSpec& row : ast_.asgp_dp2d_specs) {
      if (row.node_index == node_index) return row;
    }
    return ast_.asgp_dp2d_specs.front();
  }

  bool distinct_binders(std::size_t node_index, const std::vector<int>& names,
                        const std::string& context) {
    std::set<int> unique(names.begin(), names.end());
    if (unique.size() != names.size()) {
      return fail_bool(VerifyCode::DuplicateBinder, node_index, node_path(node_index),
                       context + " binder roles must use distinct names");
    }
    return true;
  }

  std::vector<std::size_t> children(std::size_t node_index) const {
    std::vector<std::size_t> out;
    const int arity = node_descriptor(ast_.nodes[node_index].kind).prefix_arity;
    out.reserve(static_cast<std::size_t>(arity));
    std::size_t next = node_index + 1;
    for (int i = 0; i < arity; ++i) {
      out.push_back(next);
      next = result_.verified.subtree_end[next];
    }
    return out;
  }

  ExprResult typed(std::size_t node_index, RType type) {
    result_.verified.expression_types[node_index] = type;
    return ExprResult{type, result_.verified.subtree_end[node_index]};
  }

  ExprResult require_type(std::size_t node_index, RType actual, RType expected,
                          const std::string& context) {
    if (actual != expected) {
      return fail_expr(VerifyCode::TypeMismatch, node_index,
                       context + " requires " + type_name(expected) + ", got " +
                           type_name(actual));
    }
    return typed(node_index, expected);
  }

  ExprResult verify_expression(std::size_t node_index, const TypeEnv& locals,
                               const TypeEnv& binders, bool asgp_phase) {
    const AstNode& node = ast_.nodes[node_index];
    const NodeKind kind = node.kind;
    result_.verified.expression_scope_signatures[node_index] = type_env_signature(locals);
    result_.verified.expression_binder_signatures[node_index] = type_env_signature(binders);
    if (asgp_phase && is_asgp_kind(kind)) {
      return fail_expr(VerifyCode::NestedAsgp, node_index,
                       "ASGP phase bodies cannot contain ASGP source forms");
    }

    if (kind == NodeKind::CONST) {
      return typed(node_index, value_type(ast_.consts[static_cast<std::size_t>(node.i0)]));
    }
    if (kind == NodeKind::VAR) {
      const auto it = locals.find(node.i0);
      if (it == locals.end()) {
        return fail_expr(VerifyCode::UndefinedLocal, node_index,
                         "Var reads an ordinary local that is not defined in this scope");
      }
      return typed(node_index, it->second);
    }
    if (kind == NodeKind::BOUND_VAR) {
      const auto it = binders.find(node.i0);
      if (it == binders.end()) {
        return fail_expr(VerifyCode::UndefinedBinder, node_index,
                         "BoundVar reads a binder that is not visible in this lexical phase");
      }
      return typed(node_index, it->second);
    }

    const std::vector<std::size_t> child = children(node_index);
    if (kind == NodeKind::MAP_LIST) return verify_map(node_index, child, locals, binders, asgp_phase);
    if (kind == NodeKind::FILTER_LIST) return verify_filter(node_index, child, locals, binders, asgp_phase);
    if (kind == NodeKind::LINEAR_REC) return verify_linear(node_index, child, locals, binders, asgp_phase);
    if (kind == NodeKind::ASGP_DC) return verify_dc(node_index, child, locals, binders);
    if (kind == NodeKind::ASGP_DP1D) return verify_dp1(node_index, child, locals, binders);
    if (kind == NodeKind::ASGP_DP2D) return verify_dp2(node_index, child, locals, binders);

    std::vector<RType> args;
    args.reserve(child.size());
    for (std::size_t index : child) {
      const ExprResult value = verify_expression(index, locals, binders, asgp_phase);
      if (!result_) return value;
      args.push_back(value.type);
    }

    if (kind == NodeKind::NEG) {
      if (!is_numeric_type(args[0])) return fail_expr(VerifyCode::TypeMismatch, node_index, "NEG requires Int or Float");
      return typed(node_index, args[0]);
    }
    if (kind == NodeKind::NOT) return require_type(node_index, args[0], RType::Bool, "NOT");
    if (kind == NodeKind::IF_EXPR) {
      if (args[0] != RType::Bool || args[1] != args[2] || !is_value_type(args[1])) {
        return fail_expr(VerifyCode::TypeMismatch, node_index,
                         "IfExpr requires Bool condition and exactly matching branch types");
      }
      return typed(node_index, args[1]);
    }
    if (kind == NodeKind::AND || kind == NodeKind::OR) {
      if (args[0] != RType::Bool || args[1] != RType::Bool) {
        return fail_expr(VerifyCode::TypeMismatch, node_index, "boolean operator requires Bool operands");
      }
      return typed(node_index, RType::Bool);
    }
    if (kind == NodeKind::LT || kind == NodeKind::LE || kind == NodeKind::GT || kind == NodeKind::GE) {
      if (args[0] != args[1] || !is_numeric_type(args[0])) {
        return fail_expr(VerifyCode::TypeMismatch, node_index,
                         "ordered comparison requires identical numeric operand types");
      }
      return typed(node_index, RType::Bool);
    }
    if (kind == NodeKind::EQ || kind == NodeKind::NE) {
      if (args[0] != args[1] || !is_value_type(args[0])) {
        return fail_expr(VerifyCode::TypeMismatch, node_index,
                         "equality requires identical public value types");
      }
      return typed(node_index, RType::Bool);
    }
    if (kind == NodeKind::ADD || kind == NodeKind::SUB || kind == NodeKind::MUL ||
        kind == NodeKind::DIV || kind == NodeKind::MOD) {
      if (args[0] != args[1] || !is_numeric_type(args[0])) {
        return fail_expr(VerifyCode::TypeMismatch, node_index,
                         "arithmetic requires identical numeric operand types");
      }
      return typed(node_index, kind == NodeKind::DIV ? RType::Float : args[0]);
    }
    if (node_descriptor(kind).is_builtin()) return verify_builtin(node_index, args);
    return fail_expr(VerifyCode::TypeMismatch, node_index, "expression has no typing rule");
  }

  ExprResult verify_builtin(std::size_t node_index, const std::vector<RType>& args) {
    const NodeKind kind = ast_.nodes[node_index].kind;
    auto same_numeric = [&]() {
      return !args.empty() && is_numeric_type(args[0]) &&
             std::all_of(args.begin(), args.end(), [&](RType type) { return type == args[0]; });
    };
    if (kind == NodeKind::CALL_ABS || kind == NodeKind::CALL_MIN ||
        kind == NodeKind::CALL_MAX || kind == NodeKind::CALL_CLIP) {
      if (!same_numeric()) return fail_expr(VerifyCode::TypeMismatch, node_index, "numeric builtin requires identical numeric argument types");
      return typed(node_index, args[0]);
    }
    if (kind == NodeKind::CALL_IDIV0 || kind == NodeKind::CALL_IMOD0) {
      if (args[0] != RType::Int || args[1] != RType::Int) return fail_expr(VerifyCode::TypeMismatch, node_index, "protected integer builtin requires Int arguments");
      return typed(node_index, RType::Int);
    }
    if (kind == NodeKind::CALL_LEN) {
      if (!is_sequence_type(args[0])) return fail_expr(VerifyCode::TypeMismatch, node_index, "len requires a sequence");
      return typed(node_index, RType::Int);
    }
    if (kind == NodeKind::CALL_CONCAT) {
      if (args[0] != args[1] || !is_sequence_type(args[0])) return fail_expr(VerifyCode::TypeMismatch, node_index, "concat requires matching sequence types");
      return typed(node_index, args[0]);
    }
    if (kind == NodeKind::CALL_SLICE) {
      if (!is_sequence_type(args[0]) || args[1] != RType::Int || args[2] != RType::Int) return fail_expr(VerifyCode::TypeMismatch, node_index, "slice requires (Seq, Int, Int)");
      return typed(node_index, args[0]);
    }
    if (kind == NodeKind::CALL_INDEX) {
      if (args[1] != RType::Int || !is_sequence_type(args[0])) return fail_expr(VerifyCode::TypeMismatch, node_index, "index requires (Seq, Int)");
      if (args[0] == RType::String) return typed(node_index, RType::Char);
      return typed(node_index, list_element_type(args[0]));
    }
    if (kind == NodeKind::CALL_APPEND || kind == NodeKind::CALL_PREPEND) {
      if (list_element_type(args[0]) != args[1] || args[1] == RType::Invalid) return fail_expr(VerifyCode::TypeMismatch, node_index, "list insertion requires an exact element type");
      return typed(node_index, args[0]);
    }
    if (kind == NodeKind::CALL_REVERSE) {
      if (!is_sequence_type(args[0])) return fail_expr(VerifyCode::TypeMismatch, node_index, "reverse requires a sequence");
      return typed(node_index, args[0]);
    }
    if (kind == NodeKind::CALL_FIND || kind == NodeKind::CALL_CONTAINS) {
      if (args[0] != RType::String || args[1] != RType::String) return fail_expr(VerifyCode::TypeMismatch, node_index, "find/contains require String arguments");
      return typed(node_index, kind == NodeKind::CALL_FIND ? RType::Int : RType::Bool);
    }
    if (kind == NodeKind::CALL_CHAR_TO_STRING) return unary_builtin(node_index, args[0], RType::Char, RType::String);
    if (kind == NodeKind::CALL_STRING_TO_CHAR) return unary_builtin(node_index, args[0], RType::String, RType::Char);
    if (kind == NodeKind::CALL_ORD) return unary_builtin(node_index, args[0], RType::Char, RType::Int);
    if (kind == NodeKind::CALL_CHR) return unary_builtin(node_index, args[0], RType::Int, RType::Char);
    if (kind == NodeKind::CALL_IS_LETTER || kind == NodeKind::CALL_IS_DIGIT ||
        kind == NodeKind::CALL_IS_SPACE || kind == NodeKind::CALL_IS_VOWEL) {
      return unary_builtin(node_index, args[0], RType::Char, RType::Bool);
    }
    if (kind == NodeKind::CALL_TO_LOWER || kind == NodeKind::CALL_TO_UPPER) {
      return unary_builtin(node_index, args[0], RType::Char, RType::Char);
    }
    if (kind == NodeKind::CALL_TO_STRING) {
      if (!is_numeric_type(args[0])) return fail_expr(VerifyCode::TypeMismatch, node_index, "to_string requires Int or Float");
      return typed(node_index, RType::String);
    }
    if (kind == NodeKind::CALL_SINGLETON) {
      if (args[0] == RType::Char) return typed(node_index, RType::String);
      if (args[0] == RType::Int) return typed(node_index, RType::IntList);
      if (args[0] == RType::Float) return typed(node_index, RType::FloatList);
      if (args[0] == RType::String) return typed(node_index, RType::StringList);
      return fail_expr(VerifyCode::TypeMismatch, node_index, "singleton requires Char, Int, Float, or String");
    }
    return fail_expr(VerifyCode::TypeMismatch, node_index, "builtin has no typing rule");
  }

  ExprResult unary_builtin(std::size_t node_index, RType actual, RType input, RType output) {
    if (actual != input) return fail_expr(VerifyCode::TypeMismatch, node_index, "builtin argument has the wrong exact type");
    return typed(node_index, output);
  }

  ExprResult verify_map(std::size_t node_index, const std::vector<std::size_t>& child,
                        const TypeEnv& locals, const TypeEnv& binders, bool asgp_phase) {
    const ExprResult source = verify_expression(child[0], locals, binders, asgp_phase);
    if (!result_) return source;
    const RType element = list_element_type(source.type);
    const RType result_type = list_type_from_tag(ast_.nodes[node_index].i1);
    const RType expected_body = list_element_type(result_type);
    if (element == RType::Invalid || expected_body == RType::Invalid) return fail_expr(VerifyCode::TypeMismatch, node_index, "MapList requires a typed list source and result");
    TypeEnv body_binders = binders;
    body_binders[ast_.nodes[node_index].i0] = element;
    const ExprResult body = verify_expression(child[1], locals, body_binders, asgp_phase);
    if (!result_) return body;
    if (body.type != expected_body) return fail_expr(VerifyCode::TypeMismatch, node_index, "MapList body type does not match its result-list tag");
    return typed(node_index, result_type);
  }

  ExprResult verify_filter(std::size_t node_index, const std::vector<std::size_t>& child,
                           const TypeEnv& locals, const TypeEnv& binders, bool asgp_phase) {
    const ExprResult source = verify_expression(child[0], locals, binders, asgp_phase);
    if (!result_) return source;
    const RType element = list_element_type(source.type);
    if (element == RType::Invalid) return fail_expr(VerifyCode::TypeMismatch, node_index, "FilterList requires a typed list source");
    TypeEnv predicate_binders = binders;
    predicate_binders[ast_.nodes[node_index].i0] = element;
    const ExprResult predicate = verify_expression(child[1], locals, predicate_binders, asgp_phase);
    if (!result_) return predicate;
    if (predicate.type != RType::Bool) return fail_expr(VerifyCode::TypeMismatch, node_index, "FilterList predicate must return Bool");
    return typed(node_index, source.type);
  }

  ExprResult verify_linear(std::size_t node_index, const std::vector<std::size_t>& child,
                           const TypeEnv& locals, const TypeEnv& binders, bool asgp_phase) {
    const LinearRecBinders& metadata = linear_metadata(node_index);
    if (!distinct_binders(node_index, {metadata.elem_name, metadata.accum_name, metadata.index_name}, "LinearRec")) return ExprResult{};
    const ExprResult source = verify_expression(child[0], locals, binders, asgp_phase);
    if (!result_) return source;
    const ExprResult start = verify_expression(child[1], locals, binders, asgp_phase);
    if (!result_) return start;
    const ExprResult empty = verify_expression(child[2], locals, binders, asgp_phase);
    if (!result_) return empty;
    const RType element = list_element_type(source.type);
    if (element == RType::Invalid || start.type != RType::Int || !is_value_type(empty.type)) return fail_expr(VerifyCode::TypeMismatch, node_index, "LinearRec requires (List, Int, R) before its phase bodies");
    TypeEnv step_binders = binders;
    step_binders[metadata.elem_name] = element;
    step_binders[metadata.accum_name] = empty.type;
    step_binders[metadata.index_name] = RType::Int;
    const ExprResult step = verify_expression(child[3], locals, step_binders, asgp_phase);
    if (!result_) return step;
    TypeEnv last_binders = binders;
    last_binders[metadata.elem_name] = element;
    last_binders[metadata.index_name] = RType::Int;
    const ExprResult last = verify_expression(child[4], locals, last_binders, asgp_phase);
    if (!result_) return last;
    if (step.type != empty.type || last.type != empty.type) return fail_expr(VerifyCode::TypeMismatch, node_index, "LinearRec empty, step, and last results must have one exact type");
    return typed(node_index, empty.type);
  }

  ExprResult verify_dc(std::size_t node_index, const std::vector<std::size_t>& child,
                       const TypeEnv& locals, const TypeEnv& binders) {
    const AsgpDcBinders& metadata = dc_metadata(node_index);
    if (!distinct_binders(node_index, {metadata.solve_xs_name, metadata.solve_n_name, metadata.solve_lo_name}, "ASGP-DC solve") ||
        !distinct_binders(node_index, {metadata.combine_left_name, metadata.combine_right_name}, "ASGP-DC combine")) return ExprResult{};
    const ExprResult source = verify_expression(child[0], locals, binders, false);
    if (!result_) return source;
    if (!is_sequence_type(source.type)) return fail_expr(VerifyCode::TypeMismatch, node_index, "ASGP-DC source must be a sequence");
    TypeEnv solve_binders{{metadata.solve_xs_name, source.type}, {metadata.solve_n_name, RType::Int}, {metadata.solve_lo_name, RType::Int}};
    const ExprResult solve = verify_expression(child[1], TypeEnv{}, solve_binders, true);
    if (!result_) return solve;
    TypeEnv divide_binders{{metadata.divide_n_name, RType::Int}};
    const ExprResult divide = verify_expression(child[2], TypeEnv{}, divide_binders, true);
    if (!result_) return divide;
    TypeEnv combine_binders{{metadata.combine_left_name, solve.type}, {metadata.combine_right_name, solve.type}};
    const ExprResult combine = verify_expression(child[3], TypeEnv{}, combine_binders, true);
    if (!result_) return combine;
    if (!is_value_type(solve.type) || divide.type != RType::Int || combine.type != solve.type) return fail_expr(VerifyCode::TypeMismatch, node_index, "ASGP-DC solve/combine types must match and divide must return Int");
    return typed(node_index, solve.type);
  }

  ExprResult verify_dp1(std::size_t node_index, const std::vector<std::size_t>& child,
                        const TypeEnv& locals, const TypeEnv& binders) {
    const AsgpDp1dSpec& metadata = dp1_metadata(node_index);
    std::vector<int> transition_names{metadata.transition_state_name};
    transition_names.insert(transition_names.end(), metadata.transition_dep_names.begin(), metadata.transition_dep_names.end());
    if (!distinct_binders(node_index, transition_names, "ASGP-DP1D transition")) return ExprResult{};
    const ExprResult state = verify_expression(child[0], locals, binders, false);
    if (!result_) return state;
    TypeEnv solve_binders{{metadata.solve_state_name, RType::Int}};
    const ExprResult solve = verify_expression(child[1], TypeEnv{}, solve_binders, true);
    if (!result_) return solve;
    TypeEnv transition_binders{{metadata.transition_state_name, RType::Int}};
    for (int name : metadata.transition_dep_names) transition_binders[name] = solve.type;
    const ExprResult transition = verify_expression(child[2], TypeEnv{}, transition_binders, true);
    if (!result_) return transition;
    if (state.type != RType::Int || !is_value_type(solve.type) || transition.type != solve.type ||
        value_type(ast_.consts[static_cast<std::size_t>(metadata.boundary_const)]) != solve.type) {
      return fail_expr(VerifyCode::TypeMismatch, node_index, "ASGP-DP1D requires Int state and identical solve, transition, and boundary types");
    }
    return typed(node_index, solve.type);
  }

  ExprResult verify_dp2(std::size_t node_index, const std::vector<std::size_t>& child,
                        const TypeEnv& locals, const TypeEnv& binders) {
    const AsgpDp2dSpec& metadata = dp2_metadata(node_index);
    if (!distinct_binders(node_index, {metadata.solve_i_name, metadata.solve_j_name}, "ASGP-DP2D solve")) return ExprResult{};
    std::vector<int> transition_names{metadata.transition_i_name, metadata.transition_j_name};
    transition_names.insert(transition_names.end(), metadata.transition_dep_names.begin(), metadata.transition_dep_names.end());
    if (!distinct_binders(node_index, transition_names, "ASGP-DP2D transition")) return ExprResult{};
    const ExprResult state_i = verify_expression(child[0], locals, binders, false);
    if (!result_) return state_i;
    const ExprResult state_j = verify_expression(child[1], locals, binders, false);
    if (!result_) return state_j;
    TypeEnv solve_binders{{metadata.solve_i_name, RType::Int}, {metadata.solve_j_name, RType::Int}};
    const ExprResult solve = verify_expression(child[2], TypeEnv{}, solve_binders, true);
    if (!result_) return solve;
    TypeEnv transition_binders{{metadata.transition_i_name, RType::Int}, {metadata.transition_j_name, RType::Int}};
    for (int name : metadata.transition_dep_names) transition_binders[name] = solve.type;
    const ExprResult transition = verify_expression(child[3], TypeEnv{}, transition_binders, true);
    if (!result_) return transition;
    if (state_i.type != RType::Int || state_j.type != RType::Int || !is_value_type(solve.type) ||
        transition.type != solve.type ||
        value_type(ast_.consts[static_cast<std::size_t>(metadata.boundary_const)]) != solve.type) {
      return fail_expr(VerifyCode::TypeMismatch, node_index, "ASGP-DP2D requires Int states and identical solve, transition, and boundary types");
    }
    return typed(node_index, solve.type);
  }

  bool verify_block(std::size_t block_index, TypeEnv* locals, const TypeEnv& binders) {
    std::size_t current = block_index;
    while (ast_.nodes[current].kind == NodeKind::BLOCK_CONS) {
      const std::size_t statement_index = current + 1;
      if (!verify_statement(statement_index, locals, binders)) return false;
      current = result_.verified.subtree_end[statement_index];
    }
    return true;
  }

  bool verify_statement(std::size_t node_index, TypeEnv* locals, const TypeEnv& binders) {
    const AstNode& node = ast_.nodes[node_index];
    if (node.kind == NodeKind::ASSIGN) {
      const ExprResult value = verify_expression(node_index + 1, *locals, binders, false);
      if (!result_) return false;
      (*locals)[node.i0] = value.type;
      return true;
    }
    if (node.kind == NodeKind::RETURN) {
      const ExprResult value = verify_expression(node_index + 1, *locals, binders, false);
      if (!result_) return false;
      if (!saw_return_) {
        return_type_ = value.type;
        saw_return_ = true;
      } else if (return_type_ != value.type) {
        return fail_bool(VerifyCode::InconsistentReturnType, node_index, node_path(node_index),
                         "all Return statements must have one exact result type");
      }
      return true;
    }
    if (node.kind == NodeKind::IF_STMT) {
      const std::size_t condition_index = node_index + 1;
      const ExprResult condition = verify_expression(condition_index, *locals, binders, false);
      if (!result_) return false;
      if (condition.type != RType::Bool) {
        return fail_bool(VerifyCode::TypeMismatch, condition_index, node_path(condition_index),
                         "IfStmt condition must have type Bool");
      }
      const std::size_t then_index = condition.next;
      const std::size_t else_index = result_.verified.subtree_end[then_index];
      TypeEnv then_locals = *locals;
      TypeEnv else_locals = *locals;
      if (!verify_block(then_index, &then_locals, binders) ||
          !verify_block(else_index, &else_locals, binders)) return false;
      TypeEnv merged;
      for (const auto& entry : then_locals) {
        const auto other = else_locals.find(entry.first);
        if (other != else_locals.end() && other->second == entry.second) merged.insert(entry);
      }
      *locals = std::move(merged);
      return true;
    }
    if (node.kind == NodeKind::FOR_RANGE) {
      const std::size_t bound_index = node_index + 1;
      const ExprResult bound = verify_expression(bound_index, *locals, binders, false);
      if (!result_) return false;
      if (bound.type != RType::Int) {
        return fail_bool(VerifyCode::TypeMismatch, bound_index, node_path(bound_index),
                         "ForRange bound must have type Int");
      }
      TypeEnv body_locals = *locals;
      body_locals[node.i0] = RType::Int;
      return verify_block(bound.next, &body_locals, binders);
    }
    return fail_bool(VerifyCode::TypeMismatch, node_index, node_path(node_index),
                     "statement has no typing rule");
  }

  const AstProgram& ast_;
  const std::vector<InputSpec>& inputs_;
  const VerifyOptions& options_;
  AstVerifyResult result_;
  std::unordered_map<std::string, int> name_to_id_;
  bool saw_return_ = false;
  RType return_type_ = RType::Invalid;
};

}  // namespace

AstVerifyResult verify_ast(const AstProgram& ast, const std::vector<InputSpec>& inputs,
                           const VerifyOptions& options) {
  AstVerifyResult structural = verify_ast_structure(ast, options);
  if (!structural) return structural;
  return TypedVerifier(ast, inputs, options, std::move(structural)).run();
}

}  // namespace g3pvm::evo
