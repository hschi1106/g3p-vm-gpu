#pragma once

#include <cmath>
#include <cstdint>

#include "g3pvm/core/value.hpp"

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
  double r = std::fmod(a, b);
  if (r == 0.0) {
    return std::copysign(0.0, b);
  }
  if (std::signbit(r) != std::signbit(b)) {
    r += b;
  }
  return r;
}

G3PVM_VM_HD inline double canonicalize_vm_float(double value) {
  if (!std::isfinite(value) || value == 0.0) {
    return value == 0.0 ? 0.0 : value;
  }
  int exponent = 0;
  const double mantissa = std::frexp(value, &exponent);
  constexpr int kMantissaBits = 32;
  const long long quantized_mantissa = static_cast<long long>(std::llround(std::ldexp(mantissa, kMantissaBits)));
  return std::ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

G3PVM_VM_HD inline long long py_int_mod(long long a, long long b) {
  if (b == -1) {
    return 0;
  }
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}

G3PVM_VM_HD inline long long wrap_int_neg(long long x) {
  return static_cast<long long>(0ULL - static_cast<std::uint64_t>(x));
}

G3PVM_VM_HD inline long long wrap_int_add(long long a, long long b) {
  return static_cast<long long>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}

G3PVM_VM_HD inline long long wrap_int_sub(long long a, long long b) {
  return static_cast<long long>(static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b));
}

G3PVM_VM_HD inline long long wrap_int_mul(long long a, long long b) {
  return static_cast<long long>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
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

  if ((a.tag == ValueTag::String && b.tag == ValueTag::String) ||
      (a.tag == ValueTag::NumList && b.tag == ValueTag::NumList) ||
      (a.tag == ValueTag::StringList && b.tag == ValueTag::StringList)) {
    if (op == CmpOp::EQ) {
      out_bool = (a.i == b.i);
      return CompareStatus::Ok;
    }
    if (op == CmpOp::NE) {
      out_bool = (a.i != b.i);
      return CompareStatus::Ok;
    }
    return CompareStatus::UnsupportedTypes;
  }

  return CompareStatus::UnsupportedTypes;
}

G3PVM_VM_HD inline bool fitness_score_for_values(const Value& actual,
                                                 const Value& expected,
                                                 double penalty,
                                                 double& out_score) {
  const double penalty_mag = fabs(penalty);
  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (to_numeric_pair(actual, expected, a_num, b_num, any_float)) {
    const double diff = a_num - b_num;
    if (!std::isfinite(a_num) || !std::isfinite(b_num) || !std::isfinite(diff)) {
      out_score = -penalty_mag;
      return true;
    }
    out_score = -fmin(fabs(diff), penalty_mag);
    return true;
  }

  if (is_numeric(expected)) {
    out_score = -penalty_mag;
    return true;
  }

  if (actual.tag != expected.tag) {
    out_score = -penalty_mag;
    return true;
  }

  if (actual.tag == ValueTag::None) {
    out_score = 1.0;
    return true;
  }
  if (actual.tag == ValueTag::Bool) {
    out_score = (actual.b == expected.b) ? 1.0 : 0.0;
    return true;
  }
  if (actual.tag == ValueTag::String || actual.tag == ValueTag::NumList || actual.tag == ValueTag::StringList) {
    out_score = (actual.i == expected.i) ? 1.0 : 0.0;
    return true;
  }

  out_score = -penalty_mag;
  return true;
}

#undef G3PVM_VM_HD

}  // namespace g3pvm::vm_semantics
