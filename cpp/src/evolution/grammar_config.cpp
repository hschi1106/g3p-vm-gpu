#include "gagp/evolution/grammar_config.hpp"

#include <stdexcept>

namespace gagp::evo {

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
  cfg.builtin_prepend = false;
  cfg.builtin_reverse = false;
  cfg.builtin_find = false;
  cfg.builtin_contains = false;
  cfg.builtin_singleton = false;
  cfg.builtin_char_to_string = false;
  cfg.builtin_string_to_char = false;
  cfg.builtin_ord = false;
  cfg.builtin_chr = false;
  cfg.builtin_is_letter = false;
  cfg.builtin_is_digit = false;
  cfg.builtin_is_space = false;
  cfg.builtin_is_vowel = false;
  cfg.builtin_to_lower = false;
  cfg.builtin_to_upper = false;
  cfg.builtin_to_string = false;
  cfg.expression_map_list = false;
  cfg.expression_filter_list = false;
  cfg.expression_linear_rec = false;
  cfg.expression_asgp_dc = false;
  cfg.expression_asgp_dp1d = false;
  cfg.expression_asgp_dp2d = false;
  cfg.value_char = false;
  cfg.value_string = false;
  cfg.value_int_list = false;
  cfg.value_float_list = false;
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
         expression_map_list && expression_filter_list && expression_linear_rec &&
         expression_asgp_dc && expression_asgp_dp1d && expression_asgp_dp2d &&
         unary_neg && unary_not &&
         binary_add && binary_sub && binary_mul && binary_div && binary_mod &&
         binary_lt && binary_le && binary_gt && binary_ge && binary_eq && binary_ne &&
         binary_and && binary_or &&
         builtin_abs && builtin_min && builtin_max && builtin_clip && builtin_idiv0 && builtin_imod0 &&
         builtin_len && builtin_concat && builtin_slice && builtin_index &&
         builtin_append && builtin_prepend && builtin_reverse && builtin_find && builtin_contains &&
         builtin_singleton && builtin_char_to_string && builtin_string_to_char && builtin_ord &&
         builtin_chr && builtin_is_letter && builtin_is_digit && builtin_is_space && builtin_is_vowel &&
         builtin_to_lower && builtin_to_upper && builtin_to_string &&
         value_int && value_float && value_bool && value_char &&
         value_string && value_int_list && value_float_list && value_string_list &&
         !compat_legacy_num_list_inputs_as_any;
}

bool GrammarConfig::allows_type(RType type) const {
  switch (type) {
    case RType::Int:
      return value_int;
    case RType::Float:
      return value_float;
    case RType::Bool:
      return value_bool;
    case RType::Char:
      return value_char;
    case RType::String:
      return value_string;
    case RType::IntList:
      return value_int_list;
    case RType::FloatList:
      return value_float_list;
    case RType::StringList:
      return value_string_list;
    case RType::Any:
      return value_int || value_float || value_bool || value_char ||
             value_string || value_int_list || value_float_list || value_string_list;
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
    case NodeKind::BOUND_VAR:
      return expression_var;
    case NodeKind::MAP_LIST:
      return expression_map_list;
    case NodeKind::FILTER_LIST:
      return expression_filter_list;
    case NodeKind::LINEAR_REC:
      return expression_linear_rec;
    case NodeKind::ASGP_DC:
      return expression_asgp_dc;
    case NodeKind::ASGP_DP1D:
      return expression_asgp_dp1d;
    case NodeKind::ASGP_DP2D:
      return expression_asgp_dp2d;
    case NodeKind::DP1_BACKWARD1:
    case NodeKind::DP1_BACKWARD2:
    case NodeKind::DP1_BACKWARD3:
    case NodeKind::DP1_FORWARD1:
    case NodeKind::DP1_FORWARD2:
    case NodeKind::DP1_FORWARD3:
    case NodeKind::DP2_CROSS_BACKWARD:
    case NodeKind::DP2_CROSS_FORWARD:
    case NodeKind::DP2_DIAGONAL_BACKWARD:
    case NodeKind::DP2_DIAGONAL_FORWARD:
    case NodeKind::DP2_NEIGHBORHOOD_BACKWARD3:
    case NodeKind::DP2_NEIGHBORHOOD_FORWARD3:
      return expression_asgp_dp1d || expression_asgp_dp2d;
    case NodeKind::CALL_ABS:
      return builtin_abs;
    case NodeKind::CALL_MIN:
      return builtin_min;
    case NodeKind::CALL_MAX:
      return builtin_max;
    case NodeKind::CALL_CLIP:
      return builtin_clip;
    case NodeKind::CALL_IDIV0:
      return builtin_idiv0;
    case NodeKind::CALL_IMOD0:
      return builtin_imod0;
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
    case NodeKind::CALL_PREPEND:
      return builtin_prepend;
    case NodeKind::CALL_REVERSE:
      return builtin_reverse;
    case NodeKind::CALL_FIND:
      return builtin_find;
    case NodeKind::CALL_CONTAINS:
      return builtin_contains;
    case NodeKind::CALL_SINGLETON:
      return builtin_singleton;
    case NodeKind::CALL_CHAR_TO_STRING:
      return builtin_char_to_string;
    case NodeKind::CALL_STRING_TO_CHAR:
      return builtin_string_to_char;
    case NodeKind::CALL_ORD:
      return builtin_ord;
    case NodeKind::CALL_CHR:
      return builtin_chr;
    case NodeKind::CALL_IS_LETTER:
      return builtin_is_letter;
    case NodeKind::CALL_IS_DIGIT:
      return builtin_is_digit;
    case NodeKind::CALL_IS_SPACE:
      return builtin_is_space;
    case NodeKind::CALL_IS_VOWEL:
      return builtin_is_vowel;
    case NodeKind::CALL_TO_LOWER:
      return builtin_to_lower;
    case NodeKind::CALL_TO_UPPER:
      return builtin_to_upper;
    case NodeKind::CALL_TO_STRING:
      return builtin_to_string;
    case NodeKind::COUNT:
      return false;
  }
  return false;
}

}  // namespace gagp::evo
