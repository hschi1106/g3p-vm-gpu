#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "g3pvm/evolution/ast_program.hpp"

namespace g3pvm::evo {

enum class NodeCategory {
  Program,
  Block,
  Statement,
  Expression,
  DependencyMarker,
};

enum class NodeIndexRole {
  Unused,
  Name,
  Constant,
  ListTypeTag,
};

enum class NodeMetadataKind {
  None,
  LinearRecBinders,
  AsgpDcBinders,
  AsgpDp1dSpec,
  AsgpDp2dSpec,
};

enum class DependencyFamily {
  None,
  Dp1d,
  Dp2d,
};

enum class GrammarFeature {
  Always,
  StatementAssign,
  StatementIf,
  StatementForRange,
  StatementReturn,
  ExpressionConst,
  ExpressionVar,
  ExpressionIf,
  ExpressionMapList,
  ExpressionFilterList,
  ExpressionLinearRec,
  ExpressionAsgpDc,
  ExpressionAsgpDp1d,
  ExpressionAsgpDp2d,
  AsgpDependency,
  UnaryNeg,
  UnaryNot,
  BinaryAdd,
  BinarySub,
  BinaryMul,
  BinaryDiv,
  BinaryMod,
  BinaryLt,
  BinaryLe,
  BinaryGt,
  BinaryGe,
  BinaryEq,
  BinaryNe,
  BinaryAnd,
  BinaryOr,
  BuiltinAbs,
  BuiltinMin,
  BuiltinMax,
  BuiltinClip,
  BuiltinIdiv0,
  BuiltinImod0,
  BuiltinLen,
  BuiltinConcat,
  BuiltinSlice,
  BuiltinIndex,
  BuiltinAppend,
  BuiltinPrepend,
  BuiltinReverse,
  BuiltinFind,
  BuiltinContains,
  BuiltinCharToString,
  BuiltinStringToChar,
  BuiltinOrd,
  BuiltinChr,
  BuiltinIsLetter,
  BuiltinIsDigit,
  BuiltinIsSpace,
  BuiltinIsVowel,
  BuiltinToLower,
  BuiltinToUpper,
  BuiltinToString,
  BuiltinSingleton,
};

enum class NodeTypingRule {
  Program,
  Block,
  Assign,
  IfStatement,
  ForRange,
  Return,
  Constant,
  Variable,
  UnaryNumeric,
  UnaryBool,
  NumericBinary,
  OrderedComparison,
  Equality,
  BooleanBinary,
  Conditional,
  Builtin,
  BoundVariable,
  MapList,
  FilterList,
  LinearRec,
  AsgpDc,
  AsgpDp1d,
  AsgpDp2d,
  DependencyMarker,
};

struct NodeDescriptor {
  NodeKind kind;
  std::string_view source_name;
  std::string_view serialized_name;
  NodeCategory category;
  int prefix_arity;
  NodeIndexRole i0_role;
  NodeIndexRole i1_role;
  int builtin_id;
  int builtin_arity;
  int dependency_arity;
  DependencyFamily dependency_family;
  NodeMetadataKind metadata;
  GrammarFeature grammar_feature;
  NodeTypingRule typing_rule;
  bool typed_subtree_eligible;

  constexpr bool is_builtin() const noexcept { return builtin_id >= 0; }
};

inline constexpr std::size_t k_node_kind_count = static_cast<std::size_t>(NodeKind::COUNT);

const std::array<NodeDescriptor, k_node_kind_count>& all_node_descriptors() noexcept;
const NodeDescriptor& node_descriptor(NodeKind kind);
bool is_known_node_kind(int value) noexcept;

}  // namespace g3pvm::evo
