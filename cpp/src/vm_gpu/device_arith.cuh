#pragma once

#include <cmath>

#include "types.hpp"

namespace g3pvm::gpu_detail {

__device__ inline bool d_is_num(const Value& v) {
  return v.tag == ValueTag::Int || v.tag == ValueTag::Float;
}

__device__ inline bool d_to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                                         bool& any_float) {
  if (!d_is_num(a) || !d_is_num(b)) return false;
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

__device__ inline double d_floor(double x) {
  long long i = static_cast<long long>(x);
  if (static_cast<double>(i) > x) return static_cast<double>(i - 1);
  return static_cast<double>(i);
}

__device__ inline double d_float_mod(double a, double b) {
  return a - d_floor(a / b) * b;
}

__device__ inline long long d_int_mod(long long a, long long b) {
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

__device__ inline bool d_compare(const int op, const Value& a, const Value& b, bool& out_bool,
                                 DeviceErrCode& err) {
  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (d_to_numeric_pair(a, b, a_num, b_num, any_float)) {
    if (op == OP_LT) out_bool = a_num < b_num;
    else if (op == OP_LE) out_bool = a_num <= b_num;
    else if (op == OP_GT) out_bool = a_num > b_num;
    else if (op == OP_GE) out_bool = a_num >= b_num;
    else if (op == OP_EQ) out_bool = a_num == b_num;
    else if (op == OP_NE) out_bool = a_num != b_num;
    else {
      err = DERR_TYPE;
      return false;
    }
    return true;
  }

  if (a.tag == ValueTag::Bool && b.tag == ValueTag::Bool) {
    if (op == OP_EQ) {
      out_bool = (a.b == b.b);
      return true;
    }
    if (op == OP_NE) {
      out_bool = (a.b != b.b);
      return true;
    }
    err = DERR_TYPE;
    return false;
  }

  if (a.tag == ValueTag::None || b.tag == ValueTag::None) {
    if (op == OP_EQ) {
      out_bool = (a.tag == ValueTag::None && b.tag == ValueTag::None);
      return true;
    }
    if (op == OP_NE) {
      out_bool = !(a.tag == ValueTag::None && b.tag == ValueTag::None);
      return true;
    }
    err = DERR_TYPE;
    return false;
  }

  err = DERR_TYPE;
  return false;
}

}  // namespace g3pvm::gpu_detail
