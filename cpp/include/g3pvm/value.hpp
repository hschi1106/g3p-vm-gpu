#pragma once

#include <cstdint>

namespace g3pvm {

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

  static Value from_int(std::int64_t v) {
    Value out;
    out.tag = ValueTag::Int;
    out.i = v;
    return out;
  }

  static Value from_float(double v) {
    Value out;
    out.tag = ValueTag::Float;
    out.f = v;
    return out;
  }

  static Value from_bool(bool v) {
    Value out;
    out.tag = ValueTag::Bool;
    out.b = v;
    return out;
  }

  static Value none() { return Value{}; }
};

inline bool is_numeric(const Value& v) {
  return v.tag == ValueTag::Int || v.tag == ValueTag::Float;
}

}  // namespace g3pvm
