#include "g3pvm/evolution/genome.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace g3pvm::evo {

namespace {

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

std::string encode_value_for_cache_key(const Value& value) {
  std::ostringstream oss;
  oss << static_cast<int>(value.tag) << ":";
  if (value.tag == ValueTag::Int || value.tag == ValueTag::String || value.tag == ValueTag::List) {
    oss << value.i;
    return oss.str();
  }
  if (value.tag == ValueTag::Float) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value.f, sizeof(bits));
    oss << std::hex << std::setfill('0') << std::setw(16) << bits;
    return oss.str();
  }
  if (value.tag == ValueTag::Bool) {
    oss << (value.b ? 1 : 0);
    return oss.str();
  }
  oss << "none";
  return oss.str();
}

std::string canonical_cache_key_serialize(const AstProgram& program) {
  std::ostringstream oss;
  oss << "AstCache(";
  oss << "version:" << program.version.size() << ":" << program.version;
  oss << ";names:" << program.names.size();
  for (const std::string& name : program.names) {
    oss << "|" << name.size() << ":" << name;
  }
  oss << ";consts:" << program.consts.size();
  for (const Value& value : program.consts) {
    const std::string encoded = encode_value_for_cache_key(value);
    oss << "|" << encoded.size() << ":" << encoded;
  }
  oss << ";nodes:" << program.nodes.size();
  for (const AstNode& node : program.nodes) {
    oss << "|" << static_cast<int>(node.kind) << ":" << node.i0 << ":" << node.i1;
  }
  oss << ")";
  return oss.str();
}

}  // namespace

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
  meta.program_key = canonical_cache_key_serialize(ast);
  return meta;
}

std::string ast_to_string(const AstProgram& program) { return canonical_prefix_serialize(program); }

std::string ast_cache_key(const AstProgram& program) { return canonical_cache_key_serialize(program); }

}  // namespace g3pvm::evo
