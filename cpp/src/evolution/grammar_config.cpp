#include "g3pvm/evolution/grammar_config.hpp"

#include <stdexcept>

namespace g3pvm::evo {

GrammarConfig GrammarConfig::all_enabled() {
  return GrammarConfig{};
}

GrammarConfig GrammarConfig::scalar() {
  GrammarConfig cfg;
  cfg.builtin_len = false;
  cfg.builtin_concat = false;
  cfg.builtin_slice = false;
  cfg.builtin_index = false;
  cfg.builtin_append = false;
  cfg.builtin_reverse = false;
  cfg.builtin_find = false;
  cfg.builtin_contains = false;
  cfg.value_string = false;
  cfg.value_num_list = false;
  cfg.value_string_list = false;
  return cfg;
}

void GrammarConfig::validate() const {
  if (!statement_return) {
    throw std::invalid_argument("grammar config must enable statements.return");
  }
  if (!expression_const) {
    throw std::invalid_argument("grammar config must enable expressions.const");
  }
  if (!value_int && !value_float) {
    throw std::invalid_argument("grammar config must enable values.int or values.float");
  }
  if (statement_for_range && !value_int) {
    throw std::invalid_argument("statements.for_range requires values.int");
  }
}

bool GrammarConfig::is_all_enabled() const {
  return statement_assign && statement_if_stmt && statement_for_range && statement_return &&
         expression_const && expression_var && expression_if_expr &&
         unary_neg && unary_not &&
         binary_add && binary_sub && binary_mul && binary_div && binary_mod &&
         binary_lt && binary_le && binary_gt && binary_ge && binary_eq && binary_ne &&
         binary_and && binary_or &&
         builtin_abs && builtin_min && builtin_max && builtin_clip &&
         builtin_len && builtin_concat && builtin_slice && builtin_index &&
         builtin_append && builtin_reverse && builtin_find && builtin_contains &&
         value_int && value_float && value_bool && value_none &&
         value_string && value_num_list && value_string_list;
}

bool GrammarConfig::allows_type(RType type) const {
  switch (type) {
    case RType::Num:
      return value_int || value_float;
    case RType::Bool:
      return value_bool;
    case RType::NoneType:
      return value_none;
    case RType::String:
      return value_string;
    case RType::NumList:
      return value_num_list;
    case RType::StringList:
      return value_string_list;
    case RType::Any:
      return value_int || value_float || value_bool || value_none ||
             value_string || value_num_list || value_string_list;
    case RType::Invalid:
      return false;
  }
  return false;
}

bool GrammarConfig::allows_node_kind(NodeKind kind) const {
  switch (kind) {
    case NodeKind::PROGRAM:
    case NodeKind::BLOCK_NIL:
    case NodeKind::BLOCK_CONS:
      return true;
    case NodeKind::ASSIGN:
      return statement_assign;
    case NodeKind::IF_STMT:
      return statement_if_stmt;
    case NodeKind::FOR_RANGE:
      return statement_for_range;
    case NodeKind::RETURN:
      return statement_return;
    case NodeKind::CONST:
      return expression_const;
    case NodeKind::VAR:
      return expression_var;
    case NodeKind::NEG:
      return unary_neg;
    case NodeKind::NOT:
      return unary_not;
    case NodeKind::ADD:
      return binary_add;
    case NodeKind::SUB:
      return binary_sub;
    case NodeKind::MUL:
      return binary_mul;
    case NodeKind::DIV:
      return binary_div;
    case NodeKind::MOD:
      return binary_mod;
    case NodeKind::LT:
      return binary_lt;
    case NodeKind::LE:
      return binary_le;
    case NodeKind::GT:
      return binary_gt;
    case NodeKind::GE:
      return binary_ge;
    case NodeKind::EQ:
      return binary_eq;
    case NodeKind::NE:
      return binary_ne;
    case NodeKind::AND:
      return binary_and;
    case NodeKind::OR:
      return binary_or;
    case NodeKind::IF_EXPR:
      return expression_if_expr;
    case NodeKind::CALL_ABS:
      return builtin_abs;
    case NodeKind::CALL_MIN:
      return builtin_min;
    case NodeKind::CALL_MAX:
      return builtin_max;
    case NodeKind::CALL_CLIP:
      return builtin_clip;
    case NodeKind::CALL_LEN:
      return builtin_len;
    case NodeKind::CALL_CONCAT:
      return builtin_concat;
    case NodeKind::CALL_SLICE:
      return builtin_slice;
    case NodeKind::CALL_INDEX:
      return builtin_index;
    case NodeKind::CALL_APPEND:
      return builtin_append;
    case NodeKind::CALL_REVERSE:
      return builtin_reverse;
    case NodeKind::CALL_FIND:
      return builtin_find;
    case NodeKind::CALL_CONTAINS:
      return builtin_contains;
  }
  return false;
}

}  // namespace g3pvm::evo
