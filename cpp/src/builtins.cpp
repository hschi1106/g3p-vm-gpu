#include "g3pvm/builtins.hpp"

namespace g3pvm {

namespace {

bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                     bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
    return false;
  }
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

BuiltinResult fail(ErrCode code, const std::string& message) {
  BuiltinResult out;
  out.is_error = true;
  out.err = Err{code, message};
  return out;
}

}  // namespace

BuiltinResult builtin_call(const std::string& name, const std::vector<Value>& args) {
  if (name == "abs") {
    if (args.size() != 1) {
      return fail(ErrCode::Type, "abs expects 1 argument");
    }
    const Value& x = args[0];
    if (!is_numeric(x)) {
      return fail(ErrCode::Type, "abs expects a numeric argument");
    }
    BuiltinResult out;
    out.value = (x.tag == ValueTag::Float) ? Value::from_float(x.f < 0 ? -x.f : x.f)
                                            : Value::from_int(x.i < 0 ? -x.i : x.i);
    return out;
  }

  if (name == "min" || name == "max") {
    if (args.size() != 2) {
      return fail(ErrCode::Type, name + " expects 2 arguments");
    }
    double a = 0.0;
    double b = 0.0;
    bool any_float = false;
    if (!to_numeric_pair(args[0], args[1], a, b, any_float)) {
      return fail(ErrCode::Type, name + " expects numeric arguments");
    }
    const double pick = (name == "min") ? (a <= b ? a : b) : (a >= b ? a : b);
    BuiltinResult out;
    out.value = any_float ? Value::from_float(pick)
                          : Value::from_int(static_cast<long long>(pick));
    return out;
  }

  if (name == "clip") {
    if (args.size() != 3) {
      return fail(ErrCode::Type, "clip expects 3 arguments: clip(x, lo, hi)");
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!is_numeric(x) || !is_numeric(lo) || !is_numeric(hi)) {
      return fail(ErrCode::Type, "clip expects numeric arguments");
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      if (lo2 > hi2) {
        return fail(ErrCode::Value, "clip requires lo <= hi");
      }
      BuiltinResult out;
      if (x2 < lo2) {
        out.value = Value::from_float(lo2);
      } else if (x2 > hi2) {
        out.value = Value::from_float(hi2);
      } else {
        out.value = Value::from_float(x2);
      }
      return out;
    }

    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    if (lo2 > hi2) {
      return fail(ErrCode::Value, "clip requires lo <= hi");
    }
    BuiltinResult out;
    if (x2 < lo2) {
      out.value = Value::from_int(lo2);
    } else if (x2 > hi2) {
      out.value = Value::from_int(hi2);
    } else {
      out.value = Value::from_int(x2);
    }
    return out;
  }

  return fail(ErrCode::Name, "unknown builtin: " + name);
}

}  // namespace g3pvm
