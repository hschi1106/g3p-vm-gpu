#pragma once

#include "g3pvm/value_semantics.hpp"
#include "types.hpp"

namespace g3pvm::gpu_detail {

__device__ inline bool d_is_num(const Value& v) {
  return is_numeric(v);
}

__device__ inline bool d_to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                                         bool& any_float) {
  return vm_semantics::to_numeric_pair(a, b, a_out, b_out, any_float);
}

__device__ inline double d_floor(double x) {
  return vm_semantics::py_floor(x);
}

__device__ inline double d_float_mod(double a, double b) {
  return vm_semantics::py_float_mod(a, b);
}

__device__ inline long long d_int_mod(long long a, long long b) {
  return vm_semantics::py_int_mod(a, b);
}

__device__ inline bool d_compare(const int op, const Value& a, const Value& b, bool& out_bool,
                                 DeviceErrCode& err) {
  vm_semantics::CmpOp cmp_op = vm_semantics::CmpOp::EQ;
  if (op == OP_LT) cmp_op = vm_semantics::CmpOp::LT;
  else if (op == OP_LE) cmp_op = vm_semantics::CmpOp::LE;
  else if (op == OP_GT) cmp_op = vm_semantics::CmpOp::GT;
  else if (op == OP_GE) cmp_op = vm_semantics::CmpOp::GE;
  else if (op == OP_EQ) cmp_op = vm_semantics::CmpOp::EQ;
  else if (op == OP_NE) cmp_op = vm_semantics::CmpOp::NE;
  else {
    err = DERR_TYPE;
    return false;
  }

  const vm_semantics::CompareStatus status = vm_semantics::compare_values(cmp_op, a, b, out_bool);
  if (status == vm_semantics::CompareStatus::Ok) {
    return true;
  }
  err = DERR_TYPE;
  return false;
}

}  // namespace g3pvm::gpu_detail
