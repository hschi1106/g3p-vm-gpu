#pragma once

#include <string>
#include <vector>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"

namespace g3pvm {

struct BuiltinResult {
  bool is_error = false;
  Value value = Value::none();
  Err err{ErrCode::Value, ""};
};

BuiltinResult builtin_call(BuiltinId id, const std::vector<Value>& args);

}  // namespace g3pvm
