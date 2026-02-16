#pragma once

#include <string>

#include "g3pvm/errors.hpp"

namespace g3pvm::gpu_detail {

int host_opcode(const std::string& op);
ErrCode from_device_err(int code);
const char* device_err_message(int code);

}  // namespace g3pvm::gpu_detail
