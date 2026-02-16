#pragma once

#include <cmath>

#include "types.hpp"

namespace g3pvm::gpu_detail {

__device__ inline void d_fail(DResult& out, DeviceErrCode code) {
  out.is_error = 1;
  out.err_code = static_cast<int>(code);
}

__device__ inline bool d_value_equal_for_fitness(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::None) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int) return a.i == b.i;
  if (a.tag == ValueTag::Float) return fabs(a.f - b.f) <= 1e-12;
  return false;
}

}  // namespace g3pvm::gpu_detail
