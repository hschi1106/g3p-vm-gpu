#pragma once

#include "g3pvm/core/value_semantics.hpp"
#include "types.hpp"

namespace g3pvm::gpu_detail {

__device__ inline void d_fail(DResult& out, DeviceErrCode code) {
  out.is_error = 1;
  out.err_code = static_cast<int>(code);
}

__device__ inline bool d_fitness_score_for_values(const Value& actual,
                                                  const Value& expected,
                                                  double numeric_type_penalty,
                                                  double& out_score) {
  return vm_semantics::fitness_score_for_values(actual, expected, numeric_type_penalty, out_score);
}

}  // namespace g3pvm::gpu_detail
