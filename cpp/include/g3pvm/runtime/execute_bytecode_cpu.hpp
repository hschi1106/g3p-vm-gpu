#pragma once

#include <utility>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"

namespace g3pvm {

struct ExecResult {
  bool is_error = false;
  Value value = Value::none();
  Err err{ErrCode::Value, ""};
};

ExecResult execute_bytecode_cpu(const BytecodeProgram& program,
                                const std::vector<std::pair<int, Value>>& inputs,
                                int fuel = 10000);

}  // namespace g3pvm
