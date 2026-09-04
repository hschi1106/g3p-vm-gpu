#pragma once

#include <cmath>
#include <cstdint>

#include "gagp/core/value.hpp"

namespace gagp::vm_semantics {

#if defined(__CUDACC__)
#define GAGP_VM_HD __host__ __device__
#else
#define GAGP_VM_HD
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
  InvalidOrderingNotSupported,
  UnsupportedTypes,
};

GAGP_VM_HD inline bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                                        bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
    return false;
  }
  if (a.tag != b.tag) {
    return false;
  }
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

GAGP_VM_HD inline bool to_numeric_pair_for_fitness(const Value& a,
                                                    const Value& b,
                                                    double& a_out,
                                                    double& b_out,
                                                    bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
    return false;
  }
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

GAGP_VM_HD inline double py_floor(double x) {
  long long i = static_cast<long long>(x);
  if (static_cast<double>(i) > x) {
    return static_cast<double>(i - 1);
  }
  return static_cast<double>(i);
}

GAGP_VM_HD inline double py_float_mod(double a, double b) {
  double r = std::fmod(a, b);
  if (r == 0.0) {
    return std::copysign(0.0, b);
  }
  if (std::signbit(r) != std::signbit(b)) {
    r += b;
  }
  return r;
}

GAGP_VM_HD inline double canonicalize_vm_float(double value) {
  if (!std::isfinite(value) || value == 0.0) {
    return value == 0.0 ? 0.0 : value;
  }
  int exponent = 0;
  const double mantissa = std::frexp(value, &exponent);
  constexpr int kMantissaBits = 32;
  const long long quantized_mantissa = static_cast<long long>(std::llround(std::ldexp(mantissa, kMantissaBits)));
  return std::ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

GAGP_VM_HD inline long long py_int_mod(long long a, long long b) {
  if (b == -1) {
    return 0;
  }
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}

GAGP_VM_HD inline long long wrap_int_neg(long long x) {
  return static_cast<long long>(0ULL - static_cast<std::uint64_t>(x));
}

GAGP_VM_HD inline long long wrap_int_add(long long a, long long b) {
  return static_cast<long long>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}

GAGP_VM_HD inline long long wrap_int_sub(long long a, long long b) {
  return static_cast<long long>(static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b));
}

GAGP_VM_HD inline long long wrap_int_mul(long long a, long long b) {
  return static_cast<long long>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
}

GAGP_VM_HD inline CompareStatus compare_values(CmpOp op, const Value& a, const Value& b, bool& out_bool) {
  if (op == CmpOp::EQ || op == CmpOp::NE) {
    if (a.tag != b.tag) {
      return CompareStatus::UnsupportedTypes;
    }
    bool equal = false;
    if (a.tag == ValueTag::Bool) {
      equal = (a.b == b.b);
    } else if (a.tag == ValueTag::Float) {
      equal = (a.f == b.f);
    } else if (a.tag == ValueTag::Int || a.tag == ValueTag::Char ||
               a.tag == ValueTag::String || a.tag == ValueTag::IntList ||
               a.tag == ValueTag::FloatList || a.tag == ValueTag::StringList) {
      equal = (a.i == b.i);
    } else {
      return CompareStatus::UnsupportedTypes;
    }
    out_bool = (op == CmpOp::EQ) ? equal : !equal;
    return CompareStatus::Ok;
  }

  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (to_numeric_pair(a, b, a_num, b_num, any_float)) {
    if (op == CmpOp::LT) out_bool = a_num < b_num;
    else if (op == CmpOp::LE) out_bool = a_num <= b_num;
    else if (op == CmpOp::GT) out_bool = a_num > b_num;
    else if (op == CmpOp::GE) out_bool = a_num >= b_num;
    else return CompareStatus::InvalidOp;
    return CompareStatus::Ok;
  }

  if (a.tag == ValueTag::Bool && b.tag == ValueTag::Bool) {
    return CompareStatus::BoolOrderingNotSupported;
  }

  if (a.tag == ValueTag::Invalid || b.tag == ValueTag::Invalid) {
    return CompareStatus::InvalidOrderingNotSupported;
  }

  if ((a.tag == ValueTag::Char && b.tag == ValueTag::Char) ||
      (a.tag == ValueTag::String && b.tag == ValueTag::String) ||
      (a.tag == ValueTag::IntList && b.tag == ValueTag::IntList) ||
      (a.tag == ValueTag::FloatList && b.tag == ValueTag::FloatList) ||
      (a.tag == ValueTag::StringList && b.tag == ValueTag::StringList)) {
    return CompareStatus::UnsupportedTypes;
  }

  return CompareStatus::UnsupportedTypes;
}

GAGP_VM_HD inline bool fitness_score_for_values(const Value& actual,
                                                 const Value& expected,
                                                 double penalty,
                                                 double& out_score) {
  const double penalty_mag = fabs(penalty);
  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (to_numeric_pair_for_fitness(actual, expected, a_num, b_num, any_float)) {
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

  if (actual.tag == ValueTag::Bool) {
    out_score = (actual.b == expected.b) ? 1.0 : 0.0;
    return true;
  }
  if (actual.tag == ValueTag::Char) {
    out_score = (actual.i == expected.i) ? 1.0 : 0.0;
    return true;
  }
  if (actual.tag == ValueTag::String || actual.tag == ValueTag::IntList ||
      actual.tag == ValueTag::FloatList || actual.tag == ValueTag::StringList) {
    out_score = (actual.i == expected.i) ? 1.0 : 0.0;
    return true;
  }

  out_score = -penalty_mag;
  return true;
}

#undef GAGP_VM_HD

}  // namespace gagp::vm_semantics
