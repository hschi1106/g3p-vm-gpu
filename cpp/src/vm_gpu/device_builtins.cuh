#pragma once

#include "device_arith.cuh"

namespace g3pvm::gpu_detail {

__device__ inline bool d_builtin_call(int bid, const Value* args, int argc, Value& out, DeviceErrCode& err) {
  if (bid == 0) {
    if (argc != 1) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    if (!d_is_num(x)) {
      err = DERR_TYPE;
      return false;
    }
    out = (x.tag == ValueTag::Float) ? Value::from_float(x.f < 0 ? -x.f : x.f)
                                      : Value::from_int(x.i < 0 ? -x.i : x.i);
    return true;
  }

  if (bid == 1 || bid == 2) {
    if (argc != 2) {
      err = DERR_TYPE;
      return false;
    }
    double a = 0.0;
    double b = 0.0;
    bool any_float = false;
    if (!d_to_numeric_pair(args[0], args[1], a, b, any_float)) {
      err = DERR_TYPE;
      return false;
    }
    const double pick = (bid == 1) ? ((a <= b) ? a : b) : ((a >= b) ? a : b);
    out = any_float ? Value::from_float(pick) : Value::from_int(static_cast<long long>(pick));
    return true;
  }

  if (bid == 3) {
    if (argc != 3) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!d_is_num(x) || !d_is_num(lo) || !d_is_num(hi)) {
      err = DERR_TYPE;
      return false;
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      if (lo2 > hi2) {
        err = DERR_VALUE;
        return false;
      }
      out = (x2 < lo2) ? Value::from_float(lo2)
                       : ((x2 > hi2) ? Value::from_float(hi2) : Value::from_float(x2));
      return true;
    }
    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    if (lo2 > hi2) {
      err = DERR_VALUE;
      return false;
    }
    out = (x2 < lo2) ? Value::from_int(lo2)
                     : ((x2 > hi2) ? Value::from_int(hi2) : Value::from_int(x2));
    return true;
  }

  err = DERR_NAME;
  return false;
}

}  // namespace g3pvm::gpu_detail
