#include <cstdint>
#include <iostream>
#include <string>

#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/genome_generation.hpp"

namespace {

using namespace g3pvm::evo;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

AstProgram simple_program() {
  AstProgram ast;
  ast.consts = {g3pvm::Value::from_int(7)};
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  return ast;
}

AstProgram linear_program() {
  AstProgram ast;
  ast.names = {"u", "v", "idx"};
  ast.consts = {
      g3pvm::Value::from_int(1),
      g3pvm::Value::from_int(0),
      g3pvm::Value::from_int(2),
      g3pvm::Value::from_int(3),
      g3pvm::Value::from_int(4),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::CONST, 4, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  return ast;
}

AstProgram dp1_program() {
  AstProgram ast;
  ast.names = {"solve_s", "transition_s", "dep"};
  ast.consts = {
      g3pvm::Value::from_int(2),
      g3pvm::Value::from_int(1),
      g3pvm::Value::from_int(3),
      g3pvm::Value::from_int(0),
  };
  ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP1D, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ast.asgp_dp1d_specs = {AsgpDp1dSpec{
      3, 0, 5, 0, 3, NodeKind::DP1_BACKWARD1, {1}, 0, 1, {2},
  }};
  return ast;
}

bool expect_code(const AstProgram& ast, VerifyCode expected, const std::string& label,
                 const VerifyOptions& options = VerifyOptions{}) {
  const AstVerifyResult result = verify_ast_structure(ast, options);
  return check(!result, label + " should fail") &&
         check(result.diagnostic.code == expected,
               label + " expected " + verify_code_name(expected) + " but got " +
                   verify_code_name(result.diagnostic.code)) &&
         check(!result.diagnostic.path.empty(), label + " should report a path") &&
         check(!result.diagnostic.message.empty(), label + " should report a message");
}

}  // namespace

int main() {
  using namespace g3pvm::evo;

  const AstVerifyResult valid = verify_ast_structure(simple_program());
  if (!check(valid.ok, "simple program should verify") ||
      !check(valid.verified.subtree_end == std::vector<std::size_t>({5, 5, 4, 4, 5}),
             "simple subtree boundaries") ||
      !check(valid.verified.max_expression_depth == 1, "simple expression depth") ||
      !check(valid.verified.statement_count == 1, "simple statement count")) return 1;

  AstProgram ast = simple_program();
  ast.version = "ast-prefix-old";
  if (!expect_code(ast, VerifyCode::UnsupportedVersion, "old version")) return 1;

  ast = simple_program();
  ast.nodes.clear();
  if (!expect_code(ast, VerifyCode::EmptyProgram, "empty program")) return 1;

  ast = simple_program();
  ast.nodes[0].kind = static_cast<NodeKind>(999);
  if (!expect_code(ast, VerifyCode::UnknownNodeKind, "unknown kind")) return 1;

  ast = simple_program();
  ast.nodes[0].kind = NodeKind::BLOCK_NIL;
  if (!expect_code(ast, VerifyCode::InvalidRoot, "invalid root")) return 1;

  ast = simple_program();
  ast.nodes[2].kind = NodeKind::CONST;
  if (!expect_code(ast, VerifyCode::UnexpectedNodeCategory, "expression in statement slot")) return 1;

  ast = simple_program();
  ast.nodes.pop_back();
  if (!expect_code(ast, VerifyCode::TruncatedPrefix, "truncated block")) return 1;

  ast = simple_program();
  ast.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  if (!expect_code(ast, VerifyCode::TrailingNodes, "trailing node")) return 1;

  ast = simple_program();
  ast.nodes[3].i0 = 1;
  if (!expect_code(ast, VerifyCode::ConstantIndexOutOfRange, "constant index")) return 1;

  ast = simple_program();
  ast.nodes[3] = AstNode{NodeKind::VAR, 0, 0};
  if (!expect_code(ast, VerifyCode::NameIndexOutOfRange, "name index")) return 1;

  ast = simple_program();
  ast.nodes[0].i1 = 1;
  if (!expect_code(ast, VerifyCode::InvalidIndexField, "unused index")) return 1;

  ast = simple_program();
  ast.consts[0] = g3pvm::Value::invalid();
  if (!expect_code(ast, VerifyCode::InvalidConstantTag, "invalid constant tag")) return 1;

  ast = simple_program();
  ast.names = {"u"};
  ast.nodes[3] = AstNode{NodeKind::MAP_LIST, 0, 0};
  ast.nodes.insert(ast.nodes.begin() + 4, AstNode{NodeKind::CONST, 0, 0});
  ast.nodes.insert(ast.nodes.begin() + 5, AstNode{NodeKind::CONST, 0, 0});
  if (!expect_code(ast, VerifyCode::InvalidListTypeTag, "map result list tag")) return 1;

  ast = linear_program();
  ast.linear_rec_binders.clear();
  if (!expect_code(ast, VerifyCode::MissingMetadata, "missing linear metadata")) return 1;

  ast = linear_program();
  ast.linear_rec_binders.push_back(ast.linear_rec_binders.front());
  if (!expect_code(ast, VerifyCode::DuplicateMetadata, "duplicate linear metadata")) return 1;

  ast = simple_program();
  ast.names = {"u", "v", "idx"};
  ast.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  if (!expect_code(ast, VerifyCode::MetadataNodeMismatch, "misplaced linear metadata")) return 1;

  ast = dp1_program();
  if (!check(verify_ast_structure(ast).ok, "valid DP1D structure")) return 1;
  ast.asgp_dp1d_specs[0].dep_offsets.push_back(2);
  if (!expect_code(ast, VerifyCode::DependencyArityMismatch, "DP1D dependency arity")) return 1;

  ast = dp1_program();
  ast.asgp_dp1d_specs[0].dep_kind = NodeKind::ADD;
  if (!expect_code(ast, VerifyCode::InvalidDependencyKind, "DP1D dependency kind")) return 1;

  ast = dp1_program();
  ast.asgp_dp1d_specs[0].base_state = 5;
  if (!expect_code(ast, VerifyCode::InvalidBounds, "DP1D base state")) return 1;

  VerifyOptions limits;
  limits.max_nodes = 4;
  if (!expect_code(simple_program(), VerifyCode::ResourceLimit, "node limit", limits)) return 1;

  limits = VerifyOptions{};
  limits.max_expression_depth = 1;
  ast = simple_program();
  ast.nodes[3] = AstNode{NodeKind::NEG, 0, 0};
  ast.nodes.insert(ast.nodes.begin() + 4, AstNode{NodeKind::CONST, 0, 0});
  if (!expect_code(ast, VerifyCode::ResourceLimit, "expression depth limit", limits)) return 1;

  for (std::uint64_t seed = 0; seed < 128; ++seed) {
    const ProgramGenome genome = generate_random_genome(seed);
    const AstVerifyResult generated = verify_ast_structure(genome.ast);
    if (!check(generated.ok,
               "generated seed " + std::to_string(seed) + " should verify structurally: " +
                   generated.diagnostic.message)) return 1;
  }

  return 0;
}
