#pragma once

#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"

namespace g3pvm {

struct VMResult {
  bool is_error = false;
  Value value = Value::none();
  Err err{ErrCode::Value, ""};
};

VMResult run_bytecode(const BytecodeProgram& program,
                      const std::vector<std::pair<int, Value>>& inputs,
                      int fuel = 10000);

}  // namespace g3pvm
