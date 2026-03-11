#pragma once

#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/runtime/gpu/device_types_gpu.hpp"

namespace g3pvm::gpu_detail {

__device__ inline void d_fail(DResult& out, DeviceErrCode code) {
  out.is_error = 1;
  out.err_code = static_cast<int>(code);
}

__device__ inline bool d_fitness_score_for_values(const Value& actual,
                                                  const Value& expected,
                                                  double penalty,
                                                  double& out_score) {
  return vm_semantics::fitness_score_for_values(actual, expected, penalty, out_score);
}

}  // namespace g3pvm::gpu_detail
