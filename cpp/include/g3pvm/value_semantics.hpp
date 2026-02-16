#pragma once

#include <cmath>
#include <cstdint>

#include "g3pvm/value.hpp"

namespace g3pvm::vm_semantics {

#if defined(__CUDACC__)
#define G3PVM_VM_HD __host__ __device__
#else
#define G3PVM_VM_HD
#endif

enum class CmpOp : std::uint8_t {
  LT,
  LE,
  GT,
  GE,
  EQ,
  NE,
};

enum class CompareStatus : std::uint8_t {
  Ok,
  InvalidOp,
  BoolOrderingNotSupported,
  NoneOrderingNotSupported,
  UnsupportedTypes,
};

G3PVM_VM_HD inline bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                                        bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
    return false;
  }
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

G3PVM_VM_HD inline double py_floor(double x) {
  long long i = static_cast<long long>(x);
  if (static_cast<double>(i) > x) {
    return static_cast<double>(i - 1);
  }
  return static_cast<double>(i);
}

G3PVM_VM_HD inline double py_float_mod(double a, double b) {
  return a - py_floor(a / b) * b;
}

G3PVM_VM_HD inline long long py_int_mod(long long a, long long b) {
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}

G3PVM_VM_HD inline CompareStatus compare_values(CmpOp op, const Value& a, const Value& b, bool& out_bool) {
  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (to_numeric_pair(a, b, a_num, b_num, any_float)) {
    if (op == CmpOp::LT) out_bool = a_num < b_num;
    else if (op == CmpOp::LE) out_bool = a_num <= b_num;
    else if (op == CmpOp::GT) out_bool = a_num > b_num;
    else if (op == CmpOp::GE) out_bool = a_num >= b_num;
    else if (op == CmpOp::EQ) out_bool = a_num == b_num;
    else if (op == CmpOp::NE) out_bool = a_num != b_num;
    else return CompareStatus::InvalidOp;
    return CompareStatus::Ok;
  }

  if (a.tag == ValueTag::Bool && b.tag == ValueTag::Bool) {
    if (op == CmpOp::EQ) {
      out_bool = (a.b == b.b);
      return CompareStatus::Ok;
    }
    if (op == CmpOp::NE) {
      out_bool = (a.b != b.b);
      return CompareStatus::Ok;
    }
    return CompareStatus::BoolOrderingNotSupported;
  }

  if (a.tag == ValueTag::None || b.tag == ValueTag::None) {
    if (op == CmpOp::EQ) {
      out_bool = (a.tag == ValueTag::None && b.tag == ValueTag::None);
      return CompareStatus::Ok;
    }
    if (op == CmpOp::NE) {
      out_bool = !(a.tag == ValueTag::None && b.tag == ValueTag::None);
      return CompareStatus::Ok;
    }
    return CompareStatus::NoneOrderingNotSupported;
  }

  return CompareStatus::UnsupportedTypes;
}

G3PVM_VM_HD inline bool values_equal_for_fitness(const Value& a, const Value& b) {
  if (a.tag != b.tag) {
    return false;
  }
  if (a.tag == ValueTag::None) {
    return true;
  }
  if (a.tag == ValueTag::Bool) {
    return a.b == b.b;
  }
  if (a.tag == ValueTag::Int) {
    return a.i == b.i;
  }
  if (a.tag == ValueTag::Float) {
    return fabs(a.f - b.f) <= 1e-12;
  }
  return false;
}

#undef G3PVM_VM_HD

}  // namespace g3pvm::vm_semantics
