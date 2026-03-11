#include "opcode_map_gpu.hpp"

#include "g3pvm/core/opcode.hpp"
#include "g3pvm/runtime/gpu/constants_gpu.hpp"

namespace g3pvm::gpu_detail {

int host_opcode(const Opcode op) { return static_cast<int>(op); }

}  // namespace g3pvm::gpu_detail
