#pragma once

#include <string>
#include <vector>

#include "gagp/core/builtin.hpp"
#include "gagp/core/errors.hpp"
#include "gagp/core/value.hpp"

namespace gagp {

struct BuiltinResult {
  bool is_error = false;
  Value value = Value::invalid();
  Err err{ErrCode::Value, ""};
};

BuiltinResult builtin_call(BuiltinId id, const std::vector<Value>& args);

}  // namespace gagp
