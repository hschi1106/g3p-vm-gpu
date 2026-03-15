#pragma once

#include <cstdint>

namespace g3pvm::gpu_detail {

enum class DPayloadFlavor : std::uint8_t {
  None = 0,
  StringOnly = 1,
  ListOnly = 2,
  Mixed = 3,
};

}  // namespace g3pvm::gpu_detail
