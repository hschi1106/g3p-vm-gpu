#pragma once

#include <string>
#include <vector>

#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"

namespace g3pvm {

struct BuiltinResult {
  bool is_error = false;
  Value value = Value::none();
  Err err{ErrCode::Value, ""};
};

BuiltinResult builtin_call(const std::string& name, const std::vector<Value>& args);

}  // namespace g3pvm
