#pragma once

#include "g3pvm/value_semantics.hpp"
#include "types.hpp"

namespace g3pvm::gpu_detail {

__device__ inline void d_fail(DResult& out, DeviceErrCode code) {
  out.is_error = 1;
  out.err_code = static_cast<int>(code);
}

__device__ inline bool d_value_equal_for_fitness(const Value& a, const Value& b) {
  return vm_semantics::values_equal_for_fitness(a, b);
}

}  // namespace g3pvm::gpu_detail
