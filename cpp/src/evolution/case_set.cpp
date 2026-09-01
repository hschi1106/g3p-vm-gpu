#include "g3pvm/evolution/case_set.hpp"

#include <set>

namespace g3pvm::evo {
namespace {

RType value_rtype(const Value& value, bool allow_fallback) {
  switch (value.tag) {
    case ValueTag::Int: return RType::Int;
    case ValueTag::Float: return RType::Float;
    case ValueTag::Bool: return RType::Bool;
    case ValueTag::Char: return RType::Char;
    case ValueTag::String: return RType::String;
    case ValueTag::IntList: return RType::IntList;
    case ValueTag::FloatList: return RType::FloatList;
    case ValueTag::StringList: return RType::StringList;
    case ValueTag::FallbackToken: return allow_fallback ? RType::Any : RType::Invalid;
    case ValueTag::Invalid: return RType::Invalid;
  }
  return RType::Invalid;
}

RType merge_input_type(RType current, RType next) {
  if (current == RType::Invalid) return next;
  if (next == RType::Invalid || current == next) return current;
  return RType::Any;
}

RType infer_expected_return_type(const std::vector<EvalCase>& cases) {
  RType inferred = RType::Invalid;
  bool seen = false;
  for (const EvalCase& one_case : cases) {
    const RType one_type = value_rtype(one_case.expected, false);
    if (one_type == RType::Invalid) return RType::Invalid;
    if (!seen) {
      inferred = one_type;
      seen = true;
    } else if (inferred != one_type) {
      return RType::Invalid;
    }
  }
  return seen ? inferred : RType::Invalid;
}

}  // namespace

RType generation_input_type_for_grammar(RType inferred, const GrammarConfig& grammar) {
  if (grammar.compat_legacy_num_list_inputs_as_any &&
      (inferred == RType::IntList || inferred == RType::FloatList)) {
    return RType::Any;
  }
  return inferred;
}

CaseSet prepare_case_set(const std::vector<EvalCase>& cases, const GrammarConfig& grammar) {
  CaseSet out;
  std::set<std::string> names;
  for (const EvalCase& one_case : cases) {
    for (const auto& entry : one_case.inputs) names.insert(entry.first);
  }
  out.input_names.assign(names.begin(), names.end());

  out.input_specs.reserve(out.input_names.size());
  for (const std::string& name : out.input_names) {
    RType type = RType::Invalid;
    for (const EvalCase& one_case : cases) {
      const auto found = one_case.inputs.find(name);
      if (found != one_case.inputs.end()) {
        type = merge_input_type(type, value_rtype(found->second, true));
      }
    }
    out.input_specs.push_back(
        InputSpec{name, generation_input_type_for_grammar(type, grammar)});
  }

  out.bindings.reserve(cases.size());
  out.expected_values.reserve(cases.size());
  for (const EvalCase& one_case : cases) {
    CaseBindings bindings;
    bindings.reserve(out.input_names.size());
    for (std::size_t i = 0; i < out.input_names.size(); ++i) {
      const auto found = one_case.inputs.find(out.input_names[i]);
      if (found != one_case.inputs.end()) {
        bindings.push_back(InputBinding{static_cast<int>(i), found->second});
      }
    }
    out.bindings.push_back(std::move(bindings));
    out.expected_values.push_back(one_case.expected);
  }
  out.expected_return_type = infer_expected_return_type(cases);
  return out;
}

std::vector<InputSpec> canonical_input_specs(const std::vector<EvalCase>& cases,
                                             const GrammarConfig& grammar) {
  return prepare_case_set(cases, grammar).input_specs;
}

}  // namespace g3pvm::evo
