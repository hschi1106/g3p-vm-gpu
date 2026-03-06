#include "genome_meta.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace g3pvm::evo {

namespace {

void hash_u8(std::uint64_t& h, std::uint8_t x) {
  h ^= static_cast<std::uint64_t>(x);
  h *= 1099511628211ULL;
}

void hash_bytes(std::uint64_t& h, const void* ptr, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(ptr);
  for (std::size_t i = 0; i < n; ++i) {
    hash_u8(h, p[i]);
  }
}

void hash_i32(std::uint64_t& h, std::int32_t x) { hash_bytes(h, &x, sizeof(x)); }
void hash_u32(std::uint64_t& h, std::uint32_t x) { hash_bytes(h, &x, sizeof(x)); }

std::string canonical_prefix_serialize(const AstProgram& program) {
  std::ostringstream oss;
  oss << "AstPrefix(";
  for (std::size_t i = 0; i < program.nodes.size(); ++i) {
    if (i > 0) oss << ",";
    const AstNode& node = program.nodes[i];
    oss << static_cast<int>(node.kind) << ":" << node.i0 << ":" << node.i1;
  }
  oss << ")";
  return oss.str();
}

}  // namespace

namespace genome_meta {

GenomeMeta build_meta_fast(const AstProgram& ast) {
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
  std::uint64_t h = 1469598103934665603ULL;
  hash_u32(h, static_cast<std::uint32_t>(ast.nodes.size()));
  hash_u32(h, static_cast<std::uint32_t>(ast.names.size()));
  hash_u32(h, static_cast<std::uint32_t>(ast.consts.size()));
  const std::size_t sample = ast.nodes.size() <= 32 ? 1 : (ast.nodes.size() / 32);
  for (std::size_t i = 0; i < ast.nodes.size(); i += sample) {
    const AstNode& node = ast.nodes[i];
    hash_u32(h, static_cast<std::uint32_t>(node.kind));
    hash_i32(h, static_cast<std::int32_t>(node.i0));
    hash_i32(h, static_cast<std::int32_t>(node.i1));
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  meta.hash_key = oss.str();
  return meta;
}

}  // namespace genome_meta

std::string ast_to_string(const AstProgram& program) { return canonical_prefix_serialize(program); }

}  // namespace g3pvm::evo
