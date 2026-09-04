#pragma once

#include "gagp/core/bytecode.hpp"
#include "gagp/runtime/gpu/payload_flavor_types.hpp"

namespace gagp::gpu_detail {

DPayloadFlavor classify_payload_flavor_for_program(const BytecodeProgram& prog,
                                                   unsigned shared_input_payload_mask);

}  // namespace gagp::gpu_detail
