#include "opcode_map_gpu.hpp"

#include "gagp/core/opcode.hpp"
#include "gagp/runtime/gpu/constants_gpu.hpp"

namespace gagp::gpu_detail {

int host_opcode(const Opcode op) { return static_cast<int>(op); }

}  // namespace gagp::gpu_detail
