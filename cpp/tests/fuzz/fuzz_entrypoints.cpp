#include "fuzz_entrypoints.hpp"

#include <algorithm>
#include <string>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/evolution/ast_verify.hpp"

void fuzz_bytecode_json(const std::uint8_t* data, std::size_t size) noexcept {
  try {
    std::string text;
    if (size > 0) {
      text.assign(reinterpret_cast<const char*>(data), size);
    }
    const g3pvm::cli_detail::JsonValue root =
        g3pvm::cli_detail::JsonParser(text).parse();
    if (root.kind != g3pvm::cli_detail::JsonValue::Kind::Object) return;
    if (root.object_v.find("programs") != root.object_v.end()) {
      (void)g3pvm::cli_detail::decode_programs(
          g3pvm::cli_detail::require_object_field(root, "programs"));
    } else if (root.object_v.find("n_locals") != root.object_v.end()) {
      (void)g3pvm::cli_detail::decode_program(root);
    } else if (root.object_v.find("type") != root.object_v.end()) {
      (void)g3pvm::cli_detail::decode_typed_value(root);
    }
  } catch (...) {
  }
}

void fuzz_ast_verify(const std::uint8_t* data, std::size_t size) noexcept {
  try {
    auto byte_at = [&](std::size_t index) -> std::uint8_t {
      return index < size ? data[index] : 0U;
    };
    g3pvm::evo::AstProgram ast;
    const std::size_t name_count = std::min<std::size_t>(byte_at(0) % 8U, 4U);
    for (std::size_t i = 0; i < name_count; ++i) {
      ast.names.push_back("n" + std::to_string(i));
    }
    const std::size_t const_count = std::min<std::size_t>(byte_at(1) % 12U, 6U);
    for (std::size_t i = 0; i < const_count; ++i) {
      const std::uint8_t raw = byte_at(2 + i);
      switch (raw % 4U) {
        case 0: ast.consts.push_back(g3pvm::Value::from_int(raw)); break;
        case 1: ast.consts.push_back(g3pvm::Value::from_float(raw / 3.0)); break;
        case 2: ast.consts.push_back(g3pvm::Value::from_bool((raw & 1U) != 0U)); break;
        default: ast.consts.push_back(g3pvm::Value::from_char(raw)); break;
      }
    }
    const std::size_t offset = 2 + const_count;
    const std::size_t available = size > offset ? size - offset : 0;
    const std::size_t node_count = std::min<std::size_t>(available / 3U, 128U);
    for (std::size_t i = 0; i < node_count; ++i) {
      const std::size_t at = offset + i * 3U;
      ast.nodes.push_back({static_cast<g3pvm::evo::NodeKind>(byte_at(at)),
                           static_cast<int>(static_cast<std::int8_t>(byte_at(at + 1))),
                           static_cast<int>(static_cast<std::int8_t>(byte_at(at + 2)))});
    }
    (void)g3pvm::evo::verify_ast_structure(ast);
    (void)g3pvm::evo::verify_ast(ast, {});
  } catch (...) {
  }
}
