#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "g3pvm/evolution/ast_program.hpp"

namespace g3pvm::evo {

enum class VerifyCode {
  Ok,
  UnsupportedVersion,
  EmptyProgram,
  UnknownNodeKind,
  InvalidRoot,
  UnexpectedNodeCategory,
  TruncatedPrefix,
  TrailingNodes,
  NameIndexOutOfRange,
  ConstantIndexOutOfRange,
  InvalidConstantTag,
  InvalidIndexField,
  InvalidListTypeTag,
  MissingMetadata,
  DuplicateMetadata,
  MetadataNodeMismatch,
  InvalidDependencyKind,
  DependencyArityMismatch,
  InvalidBounds,
  ResourceLimit,
};

const char* verify_code_name(VerifyCode code) noexcept;

struct VerifyDiagnostic {
  VerifyCode code = VerifyCode::Ok;
  std::size_t node_index = 0;
  std::string path;
  std::string message;
};

struct VerifyOptions {
  std::size_t max_nodes = 0;
  std::size_t max_expression_depth = 0;
  std::size_t max_statements = 0;
  std::size_t max_metadata_entries = 0;
};

struct VerifiedAst {
  RType return_type = RType::Invalid;
  std::vector<std::size_t> subtree_end;
  std::vector<RType> expression_types;
  std::size_t max_expression_depth = 0;
  std::size_t statement_count = 0;
};

struct AstVerifyResult {
  bool ok = false;
  VerifiedAst verified;
  VerifyDiagnostic diagnostic;

  explicit operator bool() const noexcept { return ok; }
};

AstVerifyResult verify_ast_structure(const AstProgram& ast,
                                     const VerifyOptions& options = VerifyOptions{});

}  // namespace g3pvm::evo
