#pragma once

#include "g3pvm/evolution/ast_program.hpp"

namespace g3pvm::evo {

struct GrammarConfig {
  bool statement_assign = true;
  bool statement_if_stmt = true;
  bool statement_for_range = true;
  bool statement_return = true;

  bool expression_const = true;
  bool expression_var = true;
  bool expression_if_expr = true;
  bool expression_map_list = true;
  bool expression_filter_list = true;
  bool expression_linear_rec = true;
  bool expression_asgp_dc = true;
  bool expression_asgp_dp1d = true;
  bool expression_asgp_dp2d = true;

  bool unary_neg = true;
  bool unary_not = true;

  bool binary_add = true;
  bool binary_sub = true;
  bool binary_mul = true;
  bool binary_div = true;
  bool binary_mod = true;
  bool binary_lt = true;
  bool binary_le = true;
  bool binary_gt = true;
  bool binary_ge = true;
  bool binary_eq = true;
  bool binary_ne = true;
  bool binary_and = true;
  bool binary_or = true;

  bool builtin_abs = true;
  bool builtin_min = true;
  bool builtin_max = true;
  bool builtin_clip = true;
  bool builtin_idiv0 = true;
  bool builtin_imod0 = true;
  bool builtin_len = true;
  bool builtin_concat = true;
  bool builtin_slice = true;
  bool builtin_index = true;
  bool builtin_append = true;
  bool builtin_prepend = true;
  bool builtin_reverse = true;
  bool builtin_find = true;
  bool builtin_contains = true;
  bool builtin_singleton = true;
  bool builtin_char_to_string = true;
  bool builtin_string_to_char = true;
  bool builtin_ord = true;
  bool builtin_chr = true;
  bool builtin_is_letter = true;
  bool builtin_is_digit = true;
  bool builtin_is_space = true;
  bool builtin_is_vowel = true;
  bool builtin_to_lower = true;
  bool builtin_to_upper = true;
  bool builtin_to_string = true;

  bool value_int = true;
  bool value_float = true;
  bool value_bool = true;
  bool value_char = true;
  bool value_string = true;
  bool value_int_list = true;
  bool value_float_list = true;
  bool value_string_list = true;

  bool compat_legacy_num_list_inputs_as_any = false;

  static GrammarConfig all_enabled();
  static GrammarConfig scalar();

  void validate() const;
  bool is_all_enabled() const;
  bool allows_type(RType type) const;
  bool allows_node_kind(NodeKind kind) const;
};

}  // namespace g3pvm::evo
