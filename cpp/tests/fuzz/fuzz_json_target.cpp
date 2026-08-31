#include <cstddef>
#include <cstdint>

#include "fuzz_entrypoints.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  fuzz_bytecode_json(data, size);
  return 0;
}
