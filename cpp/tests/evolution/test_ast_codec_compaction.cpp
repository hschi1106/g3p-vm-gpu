#include <iostream>
#include <string>

#include "gagp/cli/commands.hpp"
#include "gagp/cli/json.hpp"
#include "gagp/evolution/ast_verify.hpp"
#include "gagp/evolution/compiler.hpp"
#include "gagp/evolution/genome.hpp"
#include "gagp/evolution/repro/pack.hpp"
#include "gagp/runtime/cpu/execute_bytecode_cpu.hpp"

namespace {

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

gagp::evo::ProgramGenome bloated_genome() {
  using gagp::evo::AstNode;
  using gagp::evo::NodeKind;
  gagp::evo::ProgramGenome genome;
  genome.ast.names = {"unused", "also_unused"};
  genome.ast.consts = {
      gagp::Value::from_int(111),
      gagp::Value::from_int(42),
      gagp::Value::from_int(222),
  };
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  genome.meta = gagp::evo::build_genome_meta(genome.ast);
  return genome;
}

long long execute_int(const gagp::evo::ProgramGenome& genome) {
  const auto verified = gagp::evo::verify_ast(genome.ast, {});
  if (!verified.ok) return -1;
  const auto result = gagp::execute_bytecode_cpu(
      gagp::evo::compile_for_eval(genome, verified.verified), {}, 100);
  return (!result.is_error && result.value.tag == gagp::ValueTag::Int)
             ? result.value.i
             : -1;
}

}  // namespace

int main() {
  const auto original = bloated_genome();
  const std::string encoded = gagp::cli_detail::encode_ast_json(original.ast);
  const auto decoded = gagp::cli_detail::decode_ast_json(
      gagp::cli_detail::JsonParser(encoded).parse());
  if (!check(gagp::evo::ast_cache_key(decoded) ==
                 gagp::evo::ast_cache_key(original.ast),
             "AST JSON round trip must preserve canonical key")) return 1;
  if (!check(gagp::cli_detail::encode_ast_json(decoded) == encoded,
             "AST JSON codec must be canonical")) return 1;

  const auto compacted = gagp::evo::repro::compact_genome_tables(original);
  const auto compacted_twice = gagp::evo::repro::compact_genome_tables(compacted);
  if (!check(compacted.ast.names.empty() && compacted.ast.consts.size() == 1 &&
                 compacted.ast.nodes[3].i0 == 0,
             "table compaction must discard and remap dead entries")) return 1;
  if (!check(execute_int(original) == 42 && execute_int(compacted) == 42,
             "table compaction must preserve execution")) return 1;
  if (!check(compacted.meta.program_key == compacted_twice.meta.program_key &&
                 gagp::cli_detail::encode_ast_json(compacted.ast) ==
                     gagp::cli_detail::encode_ast_json(compacted_twice.ast),
             "table compaction must be idempotent in representation and key")) return 1;

  std::cout << "gagp_test_ast_codec_compaction: OK\n";
  return 0;
}
