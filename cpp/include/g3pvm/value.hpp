#pragma once

#include <cstdint>

namespace g3pvm {

#ifdef __CUDACC__
#define G3PVM_HD __host__ __device__
#else
#define G3PVM_HD
#endif

enum class ValueTag {
  Int,
  Float,
  Bool,
  None,
};

struct Value {
  ValueTag tag = ValueTag::None;
  std::int64_t i = 0;
  double f = 0.0;
  bool b = false;

  G3PVM_HD static Value from_int(std::int64_t v) {
    Value out;
    out.tag = ValueTag::Int;
    out.i = v;
    return out;
  }

  G3PVM_HD static Value from_float(double v) {
    Value out;
    out.tag = ValueTag::Float;
    out.f = v;
    return out;
  }

  G3PVM_HD static Value from_bool(bool v) {
    Value out;
    out.tag = ValueTag::Bool;
    out.b = v;
    return out;
  }

  G3PVM_HD static Value none() { return Value{}; }
};

G3PVM_HD inline bool is_numeric(const Value& v) {
  return v.tag == ValueTag::Int || v.tag == ValueTag::Float;
}

#undef G3PVM_HD

}  // namespace g3pvm
