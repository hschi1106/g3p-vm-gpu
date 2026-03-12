#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo {

GenomeMeta build_genome_meta(const AstProgram& ast) {
  GenomeMeta meta;
  meta.node_count = static_cast<int>(ast.nodes.size());
  meta.max_depth = 0;
  meta.uses_builtins = false;
  for (const AstNode& node : ast.nodes) {
    if (node.kind == NodeKind::CALL_ABS || node.kind == NodeKind::CALL_MIN || node.kind == NodeKind::CALL_MAX ||
        node.kind == NodeKind::CALL_CLIP || node.kind == NodeKind::CALL_LEN || node.kind == NodeKind::CALL_CONCAT ||
        node.kind == NodeKind::CALL_SLICE || node.kind == NodeKind::CALL_INDEX) {
      meta.uses_builtins = true;
      break;
    }
  }
  meta.program_key = ast_cache_key(ast);
  return meta;
}

}  // namespace g3pvm::evo
