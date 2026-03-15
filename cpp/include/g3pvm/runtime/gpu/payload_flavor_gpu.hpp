#pragma once

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/runtime/gpu/payload_flavor_types.hpp"

namespace g3pvm::gpu_detail {

DPayloadFlavor classify_payload_flavor_for_program(const BytecodeProgram& prog,
                                                   unsigned shared_input_payload_mask);

}  // namespace g3pvm::gpu_detail
