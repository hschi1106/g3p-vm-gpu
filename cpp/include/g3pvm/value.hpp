#pragma once

#include <cstdint>
#include <type_traits>

namespace g3pvm {

#ifdef __CUDACC__
#define G3PVM_HD __host__ __device__
#else
#define G3PVM_HD
#endif

enum class ValueTag : std::uint8_t {
  Int,
  Float,
  Bool,
  None,
};

struct Value {
  union {
    std::int64_t i;
    double f;
  };
  bool b = false;
  ValueTag tag = ValueTag::None;

  G3PVM_HD static Value from_int(std::int64_t v) {
    Value out;
    out.i = 0;
    out.tag = ValueTag::Int;
    out.i = v;
    return out;
  }

  G3PVM_HD static Value from_float(double v) {
    Value out;
    out.i = 0;
    out.tag = ValueTag::Float;
    out.f = v;
    return out;
  }

  G3PVM_HD static Value from_bool(bool v) {
    Value out;
    out.i = 0;
    out.tag = ValueTag::Bool;
    out.b = v;
    return out;
  }

  G3PVM_HD static Value none() {
    Value out;
    out.i = 0;
    out.b = false;
    out.tag = ValueTag::None;
    return out;
  }
};

G3PVM_HD inline bool is_numeric(const Value& v) {
  return v.tag == ValueTag::Int || v.tag == ValueTag::Float;
}

static_assert(std::is_trivially_copyable<Value>::value, "Value must be trivially copyable");
static_assert(sizeof(Value) <= 16, "Value should remain compact");

#undef G3PVM_HD

}  // namespace g3pvm
