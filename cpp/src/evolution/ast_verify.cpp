#include "g3pvm/evolution/ast_verify.hpp"

#include <algorithm>
#include <set>
#include <string>

#include "g3pvm/evolution/node_descriptor.hpp"

namespace g3pvm::evo {

const char* verify_code_name(VerifyCode code) noexcept {
  switch (code) {
    case VerifyCode::Ok: return "ok";
    case VerifyCode::UnsupportedVersion: return "unsupported_version";
    case VerifyCode::EmptyProgram: return "empty_program";
    case VerifyCode::UnknownNodeKind: return "unknown_node_kind";
    case VerifyCode::InvalidRoot: return "invalid_root";
    case VerifyCode::UnexpectedNodeCategory: return "unexpected_node_category";
    case VerifyCode::TruncatedPrefix: return "truncated_prefix";
    case VerifyCode::TrailingNodes: return "trailing_nodes";
    case VerifyCode::NameIndexOutOfRange: return "name_index_out_of_range";
    case VerifyCode::ConstantIndexOutOfRange: return "constant_index_out_of_range";
    case VerifyCode::InvalidConstantTag: return "invalid_constant_tag";
    case VerifyCode::InvalidIndexField: return "invalid_index_field";
    case VerifyCode::InvalidListTypeTag: return "invalid_list_type_tag";
    case VerifyCode::MissingMetadata: return "missing_metadata";
    case VerifyCode::DuplicateMetadata: return "duplicate_metadata";
    case VerifyCode::MetadataNodeMismatch: return "metadata_node_mismatch";
    case VerifyCode::InvalidDependencyKind: return "invalid_dependency_kind";
    case VerifyCode::DependencyArityMismatch: return "dependency_arity_mismatch";
    case VerifyCode::InvalidBounds: return "invalid_bounds";
    case VerifyCode::ResourceLimit: return "resource_limit";
  }
  return "unknown_verify_code";
}

namespace {

std::string node_path(std::size_t index) {
  return "$.nodes[" + std::to_string(index) + "]";
}

bool is_public_value_tag(ValueTag tag) {
  switch (tag) {
    case ValueTag::Int:
    case ValueTag::Float:
    case ValueTag::Bool:
    case ValueTag::Char:
    case ValueTag::String:
    case ValueTag::IntList:
    case ValueTag::FloatList:
    case ValueTag::StringList:
      return true;
    case ValueTag::FallbackToken:
    case ValueTag::Invalid:
      return false;
  }
  return false;
}

class StructuralVerifier {
 public:
  StructuralVerifier(const AstProgram& ast, const VerifyOptions& options)
      : ast_(ast), options_(options) {
    result_.verified.subtree_end.assign(ast.nodes.size(), 0);
    result_.verified.expression_types.assign(ast.nodes.size(), RType::Invalid);
  }

  AstVerifyResult run() {
    if (ast_.version != k_ast_prefix_version_current) {
      return fail(VerifyCode::UnsupportedVersion, 0, "$.version",
                  "expected AST version ast-prefix");
    }
    if (ast_.nodes.empty()) {
      return fail(VerifyCode::EmptyProgram, 0, "$.nodes", "program has no prefix nodes");
    }
    if (options_.max_nodes != 0 && ast_.nodes.size() > options_.max_nodes) {
      return fail(VerifyCode::ResourceLimit, 0, "$.nodes", "node count exceeds configured limit");
    }
    const std::size_t metadata_count = ast_.linear_rec_binders.size() + ast_.asgp_dc_binders.size() +
                                       ast_.asgp_dp1d_specs.size() + ast_.asgp_dp2d_specs.size();
    if (options_.max_metadata_entries != 0 && metadata_count > options_.max_metadata_entries) {
      return fail(VerifyCode::ResourceLimit, 0, "$", "metadata count exceeds configured limit");
    }
    if (!validate_nodes_and_tables()) return result_;
    if (ast_.nodes.front().kind != NodeKind::PROGRAM) {
      return fail(VerifyCode::InvalidRoot, 0, node_path(0), "root node must be PROGRAM");
    }

    std::size_t end = 0;
    if (!parse_program(0, &end)) return result_;
    if (end != ast_.nodes.size()) {
      return fail(VerifyCode::TrailingNodes, end, node_path(end),
                  "prefix program has trailing nodes");
    }
    if (!validate_side_tables()) return result_;

    result_.ok = true;
    result_.diagnostic = VerifyDiagnostic{};
    return result_;
  }

 private:
  AstVerifyResult fail(VerifyCode code, std::size_t node_index, std::string path,
                       std::string message) {
    result_.ok = false;
    result_.diagnostic = VerifyDiagnostic{code, node_index, std::move(path), std::move(message)};
    return result_;
  }

  bool fail_bool(VerifyCode code, std::size_t node_index, std::string path,
                 std::string message) {
    (void)fail(code, node_index, std::move(path), std::move(message));
    return false;
  }

  bool valid_name(int value, std::size_t node_index, const std::string& path) {
    if (value < 0 || static_cast<std::size_t>(value) >= ast_.names.size()) {
      return fail_bool(VerifyCode::NameIndexOutOfRange, node_index, path,
                       "name index is outside the names table");
    }
    return true;
  }

  bool valid_const(int value, std::size_t node_index, const std::string& path) {
    if (value < 0 || static_cast<std::size_t>(value) >= ast_.consts.size()) {
      return fail_bool(VerifyCode::ConstantIndexOutOfRange, node_index, path,
                       "constant index is outside the consts table");
    }
    return true;
  }

  bool validate_index_role(NodeIndexRole role, int value, std::size_t node_index,
                           const std::string& path) {
    switch (role) {
      case NodeIndexRole::Unused:
        if (value != 0) {
          return fail_bool(VerifyCode::InvalidIndexField, node_index, path,
                           "unused node index field must be zero");
        }
        return true;
      case NodeIndexRole::Name:
        return valid_name(value, node_index, path);
      case NodeIndexRole::Constant:
        return valid_const(value, node_index, path);
      case NodeIndexRole::ListTypeTag:
        if (value < static_cast<int>(ListTypeTag::Int) ||
            value > static_cast<int>(ListTypeTag::String)) {
          return fail_bool(VerifyCode::InvalidListTypeTag, node_index, path,
                           "list type tag must be Int, Float, or String");
        }
        return true;
    }
    return false;
  }

  bool validate_nodes_and_tables() {
    for (std::size_t i = 0; i < ast_.nodes.size(); ++i) {
      const AstNode& node = ast_.nodes[i];
      const int kind_value = static_cast<int>(node.kind);
      if (!is_known_node_kind(kind_value)) {
        return fail_bool(VerifyCode::UnknownNodeKind, i, node_path(i) + ".kind",
                         "node kind is outside the current descriptor table");
      }
      const NodeDescriptor& descriptor = node_descriptor(node.kind);
      if (!validate_index_role(descriptor.i0_role, node.i0, i, node_path(i) + ".i0")) return false;
      if (!validate_index_role(descriptor.i1_role, node.i1, i, node_path(i) + ".i1")) return false;
    }
    for (std::size_t i = 0; i < ast_.consts.size(); ++i) {
      if (!is_public_value_tag(ast_.consts[i].tag)) {
        return fail_bool(VerifyCode::InvalidConstantTag, 0,
                         "$.consts[" + std::to_string(i) + "]",
                         "constant uses a private or invalid value tag");
      }
    }
    return true;
  }

  bool node_at(std::size_t index, NodeCategory category, const AstNode** out) {
    if (index >= ast_.nodes.size()) {
      return fail_bool(VerifyCode::TruncatedPrefix, index, node_path(index),
                       "prefix program ended before the expected node");
    }
    const NodeDescriptor& descriptor = node_descriptor(ast_.nodes[index].kind);
    if (descriptor.category != category) {
      return fail_bool(VerifyCode::UnexpectedNodeCategory, index, node_path(index),
                       "node appears in an invalid prefix grammar position");
    }
    *out = &ast_.nodes[index];
    return true;
  }

  bool parse_program(std::size_t index, std::size_t* end) {
    const AstNode* node = nullptr;
    if (!node_at(index, NodeCategory::Program, &node)) return false;
    if (node->kind != NodeKind::PROGRAM) {
      return fail_bool(VerifyCode::InvalidRoot, index, node_path(index),
                       "program position requires PROGRAM");
    }
    std::size_t next = 0;
    if (!parse_block(index + 1, &next)) return false;
    result_.verified.subtree_end[index] = next;
    *end = next;
    return true;
  }

  bool parse_block(std::size_t index, std::size_t* end) {
    const AstNode* node = nullptr;
    if (!node_at(index, NodeCategory::Block, &node)) return false;
    if (node->kind == NodeKind::BLOCK_NIL) {
      result_.verified.subtree_end[index] = index + 1;
      *end = index + 1;
      return true;
    }
    if (node->kind != NodeKind::BLOCK_CONS) {
      return fail_bool(VerifyCode::UnexpectedNodeCategory, index, node_path(index),
                       "block position requires BLOCK_NIL or BLOCK_CONS");
    }
    std::size_t next = 0;
    if (!parse_statement(index + 1, &next)) return false;
    if (!parse_block(next, &next)) return false;
    result_.verified.subtree_end[index] = next;
    *end = next;
    return true;
  }

  bool parse_statement(std::size_t index, std::size_t* end) {
    const AstNode* node = nullptr;
    if (!node_at(index, NodeCategory::Statement, &node)) return false;
    ++result_.verified.statement_count;
    if (options_.max_statements != 0 &&
        result_.verified.statement_count > options_.max_statements) {
      return fail_bool(VerifyCode::ResourceLimit, index, node_path(index),
                       "statement count exceeds configured limit");
    }

    std::size_t next = index + 1;
    switch (node->kind) {
      case NodeKind::ASSIGN:
      case NodeKind::RETURN:
        if (!parse_expression(next, 1, &next)) return false;
        break;
      case NodeKind::IF_STMT:
        if (!parse_expression(next, 1, &next) || !parse_block(next, &next) ||
            !parse_block(next, &next)) return false;
        break;
      case NodeKind::FOR_RANGE:
        if (!parse_expression(next, 1, &next) || !parse_block(next, &next)) return false;
        break;
      default:
        return fail_bool(VerifyCode::UnexpectedNodeCategory, index, node_path(index),
                         "unknown statement form");
    }
    result_.verified.subtree_end[index] = next;
    *end = next;
    return true;
  }

  bool parse_expression(std::size_t index, std::size_t depth, std::size_t* end) {
    const AstNode* node = nullptr;
    if (!node_at(index, NodeCategory::Expression, &node)) return false;
    result_.verified.max_expression_depth =
        std::max(result_.verified.max_expression_depth, depth);
    if (options_.max_expression_depth != 0 && depth > options_.max_expression_depth) {
      return fail_bool(VerifyCode::ResourceLimit, index, node_path(index),
                       "expression depth exceeds configured limit");
    }
    const NodeDescriptor& descriptor = node_descriptor(node->kind);
    std::size_t next = index + 1;
    for (int child = 0; child < descriptor.prefix_arity; ++child) {
      if (!parse_expression(next, depth + 1, &next)) return false;
    }
    result_.verified.subtree_end[index] = next;
    *end = next;
    return true;
  }

  bool check_metadata_node(std::size_t node_index, NodeKind kind,
                           const std::string& path) {
    if (node_index >= ast_.nodes.size() || ast_.nodes[node_index].kind != kind) {
      return fail_bool(VerifyCode::MetadataNodeMismatch, node_index, path + ".node_index",
                       "metadata node_index does not identify the matching structured node");
    }
    return true;
  }

  bool check_unique(std::set<std::size_t>* seen, std::size_t node_index,
                    const std::string& path) {
    if (!seen->insert(node_index).second) {
      return fail_bool(VerifyCode::DuplicateMetadata, node_index, path + ".node_index",
                       "structured node has duplicate side-table metadata");
    }
    return true;
  }

  bool validate_side_tables() {
    std::set<std::size_t> expected_linear;
    std::set<std::size_t> expected_dc;
    std::set<std::size_t> expected_dp1;
    std::set<std::size_t> expected_dp2;
    for (std::size_t i = 0; i < ast_.nodes.size(); ++i) {
      switch (node_descriptor(ast_.nodes[i].kind).metadata) {
        case NodeMetadataKind::LinearRecBinders: expected_linear.insert(i); break;
        case NodeMetadataKind::AsgpDcBinders: expected_dc.insert(i); break;
        case NodeMetadataKind::AsgpDp1dSpec: expected_dp1.insert(i); break;
        case NodeMetadataKind::AsgpDp2dSpec: expected_dp2.insert(i); break;
        case NodeMetadataKind::None: break;
      }
    }

    std::set<std::size_t> seen_linear;
    for (std::size_t i = 0; i < ast_.linear_rec_binders.size(); ++i) {
      const LinearRecBinders& row = ast_.linear_rec_binders[i];
      const std::string path = "$.linear_rec_binders[" + std::to_string(i) + "]";
      if (!check_metadata_node(row.node_index, NodeKind::LINEAR_REC, path) ||
          !check_unique(&seen_linear, row.node_index, path) ||
          !valid_name(row.elem_name, row.node_index, path + ".elem_name") ||
          !valid_name(row.accum_name, row.node_index, path + ".accum_name") ||
          !valid_name(row.index_name, row.node_index, path + ".index_name")) return false;
    }

    std::set<std::size_t> seen_dc;
    for (std::size_t i = 0; i < ast_.asgp_dc_binders.size(); ++i) {
      const AsgpDcBinders& row = ast_.asgp_dc_binders[i];
      const std::string path = "$.asgp_dc_binders[" + std::to_string(i) + "]";
      if (!check_metadata_node(row.node_index, NodeKind::ASGP_DC, path) ||
          !check_unique(&seen_dc, row.node_index, path) ||
          !valid_name(row.solve_xs_name, row.node_index, path + ".solve_xs_name") ||
          !valid_name(row.solve_n_name, row.node_index, path + ".solve_n_name") ||
          !valid_name(row.solve_lo_name, row.node_index, path + ".solve_lo_name") ||
          !valid_name(row.divide_n_name, row.node_index, path + ".divide_n_name") ||
          !valid_name(row.combine_left_name, row.node_index, path + ".combine_left_name") ||
          !valid_name(row.combine_right_name, row.node_index, path + ".combine_right_name")) return false;
    }

    std::set<std::size_t> seen_dp1;
    for (std::size_t i = 0; i < ast_.asgp_dp1d_specs.size(); ++i) {
      const AsgpDp1dSpec& row = ast_.asgp_dp1d_specs[i];
      const std::string path = "$.asgp_dp1d_specs[" + std::to_string(i) + "]";
      if (!check_metadata_node(row.node_index, NodeKind::ASGP_DP1D, path) ||
          !check_unique(&seen_dp1, row.node_index, path)) return false;
      if (row.lo >= row.hi || row.base_state < row.lo || row.base_state >= row.hi) {
        return fail_bool(VerifyCode::InvalidBounds, row.node_index, path,
                         "DP1D bounds must be non-empty and contain base_state");
      }
      if (!valid_const(row.boundary_const, row.node_index, path + ".boundary_const")) return false;
      if (!is_known_node_kind(static_cast<int>(row.dep_kind)) ||
          node_descriptor(row.dep_kind).dependency_family != DependencyFamily::Dp1d) {
        return fail_bool(VerifyCode::InvalidDependencyKind, row.node_index, path + ".dep_kind",
                         "DP1D metadata requires a DP1D dependency kind");
      }
      const int arity = node_descriptor(row.dep_kind).dependency_arity;
      if (row.dep_offsets.size() != static_cast<std::size_t>(arity) ||
          row.transition_dep_names.size() != static_cast<std::size_t>(arity)) {
        return fail_bool(VerifyCode::DependencyArityMismatch, row.node_index, path,
                         "DP1D dependency arrays do not match the dependency kind arity");
      }
      for (std::size_t j = 0; j < row.dep_offsets.size(); ++j) {
        if (row.dep_offsets[j] <= 0) {
          return fail_bool(VerifyCode::InvalidBounds, row.node_index,
                           path + ".dep_offsets[" + std::to_string(j) + "]",
                           "DP1D dependency offsets must be positive");
        }
      }
      if (!valid_name(row.solve_state_name, row.node_index, path + ".solve_state_name") ||
          !valid_name(row.transition_state_name, row.node_index, path + ".transition_state_name")) return false;
      for (std::size_t j = 0; j < row.transition_dep_names.size(); ++j) {
        if (!valid_name(row.transition_dep_names[j], row.node_index,
                        path + ".transition_dep_names[" + std::to_string(j) + "]")) return false;
      }
    }

    std::set<std::size_t> seen_dp2;
    for (std::size_t i = 0; i < ast_.asgp_dp2d_specs.size(); ++i) {
      const AsgpDp2dSpec& row = ast_.asgp_dp2d_specs[i];
      const std::string path = "$.asgp_dp2d_specs[" + std::to_string(i) + "]";
      if (!check_metadata_node(row.node_index, NodeKind::ASGP_DP2D, path) ||
          !check_unique(&seen_dp2, row.node_index, path)) return false;
      if (row.i_lo >= row.i_hi || row.j_lo >= row.j_hi ||
          row.base_i < row.i_lo || row.base_i >= row.i_hi ||
          row.base_j < row.j_lo || row.base_j >= row.j_hi) {
        return fail_bool(VerifyCode::InvalidBounds, row.node_index, path,
                         "DP2D bounds must be non-empty and contain the base cell");
      }
      if (!valid_const(row.boundary_const, row.node_index, path + ".boundary_const")) return false;
      if (!is_known_node_kind(static_cast<int>(row.dep_kind)) ||
          node_descriptor(row.dep_kind).dependency_family != DependencyFamily::Dp2d) {
        return fail_bool(VerifyCode::InvalidDependencyKind, row.node_index, path + ".dep_kind",
                         "DP2D metadata requires a DP2D dependency kind");
      }
      const int arity = node_descriptor(row.dep_kind).dependency_arity;
      if (row.transition_dep_names.size() != static_cast<std::size_t>(arity)) {
        return fail_bool(VerifyCode::DependencyArityMismatch, row.node_index, path,
                         "DP2D dependency names do not match the dependency kind arity");
      }
      if (!valid_name(row.solve_i_name, row.node_index, path + ".solve_i_name") ||
          !valid_name(row.solve_j_name, row.node_index, path + ".solve_j_name") ||
          !valid_name(row.transition_i_name, row.node_index, path + ".transition_i_name") ||
          !valid_name(row.transition_j_name, row.node_index, path + ".transition_j_name")) return false;
      for (std::size_t j = 0; j < row.transition_dep_names.size(); ++j) {
        if (!valid_name(row.transition_dep_names[j], row.node_index,
                        path + ".transition_dep_names[" + std::to_string(j) + "]")) return false;
      }
    }

    if (!check_missing(expected_linear, seen_linear, "$.linear_rec_binders") ||
        !check_missing(expected_dc, seen_dc, "$.asgp_dc_binders") ||
        !check_missing(expected_dp1, seen_dp1, "$.asgp_dp1d_specs") ||
        !check_missing(expected_dp2, seen_dp2, "$.asgp_dp2d_specs")) return false;
    return true;
  }

  bool check_missing(const std::set<std::size_t>& expected,
                     const std::set<std::size_t>& seen,
                     const std::string& path) {
    for (std::size_t node_index : expected) {
      if (seen.count(node_index) == 0U) {
        return fail_bool(VerifyCode::MissingMetadata, node_index, path,
                         "structured node is missing required side-table metadata");
      }
    }
    return true;
  }

  const AstProgram& ast_;
  const VerifyOptions& options_;
  AstVerifyResult result_;
};

}  // namespace

AstVerifyResult verify_ast_structure(const AstProgram& ast, const VerifyOptions& options) {
  return StructuralVerifier(ast, options).run();
}

}  // namespace g3pvm::evo
