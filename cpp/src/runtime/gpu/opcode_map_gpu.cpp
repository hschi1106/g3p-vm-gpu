#include "opcode_map_gpu.hpp"

#include "g3pvm/core/opcode.hpp"
#include "g3pvm/runtime/gpu/constants_gpu.hpp"

namespace g3pvm::gpu_detail {

int host_opcode(const std::string& op) {
  Opcode opcode = Opcode::PushConst;
  if (!opcode_from_name(op, opcode)) {
    return -1;
  }
  return static_cast<int>(opcode);
}

}  // namespace g3pvm::gpu_detail
