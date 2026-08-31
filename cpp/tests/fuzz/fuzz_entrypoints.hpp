#pragma once

#include <cstddef>
#include <cstdint>

void fuzz_bytecode_json(const std::uint8_t* data, std::size_t size) noexcept;
void fuzz_ast_verify(const std::uint8_t* data, std::size_t size) noexcept;
