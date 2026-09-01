#include <iostream>
#include <string>

#include "g3pvm/cli/commands.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/repro/pack.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"

namespace {

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

g3pvm::evo::ProgramGenome bloated_genome() {
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;
  g3pvm::evo::ProgramGenome genome;
  genome.ast.names = {"unused", "also_unused"};
  genome.ast.consts = {
      g3pvm::Value::from_int(111),
      g3pvm::Value::from_int(42),
      g3pvm::Value::from_int(222),
  };
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
  return genome;
}

long long execute_int(const g3pvm::evo::ProgramGenome& genome) {
  const auto verified = g3pvm::evo::verify_ast(genome.ast, {});
  if (!verified.ok) return -1;
  const auto result = g3pvm::execute_bytecode_cpu(
      g3pvm::evo::compile_for_eval(genome, verified.verified), {}, 100);
  return (!result.is_error && result.value.tag == g3pvm::ValueTag::Int)
             ? result.value.i
             : -1;
}

}  // namespace

int main() {
  const auto original = bloated_genome();
  const std::string encoded = g3pvm::cli_detail::encode_ast_json(original.ast);
  const auto decoded = g3pvm::cli_detail::decode_ast_json(
      g3pvm::cli_detail::JsonParser(encoded).parse());
  if (!check(g3pvm::evo::ast_cache_key(decoded) ==
                 g3pvm::evo::ast_cache_key(original.ast),
             "AST JSON round trip must preserve canonical key")) return 1;
  if (!check(g3pvm::cli_detail::encode_ast_json(decoded) == encoded,
             "AST JSON codec must be canonical")) return 1;

  const auto compacted = g3pvm::evo::repro::compact_genome_tables(original);
  const auto compacted_twice = g3pvm::evo::repro::compact_genome_tables(compacted);
  if (!check(compacted.ast.names.empty() && compacted.ast.consts.size() == 1 &&
                 compacted.ast.nodes[3].i0 == 0,
             "table compaction must discard and remap dead entries")) return 1;
  if (!check(execute_int(original) == 42 && execute_int(compacted) == 42,
             "table compaction must preserve execution")) return 1;
  if (!check(compacted.meta.program_key == compacted_twice.meta.program_key &&
                 g3pvm::cli_detail::encode_ast_json(compacted.ast) ==
                     g3pvm::cli_detail::encode_ast_json(compacted_twice.ast),
             "table compaction must be idempotent in representation and key")) return 1;

  std::cout << "g3pvm_test_ast_codec_compaction: OK\n";
  return 0;
}
