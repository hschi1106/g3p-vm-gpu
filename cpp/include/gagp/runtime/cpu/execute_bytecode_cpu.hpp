#pragma once

#include <utility>
#include <vector>

#include "gagp/core/bytecode.hpp"
#include "gagp/core/errors.hpp"
#include "gagp/core/value.hpp"

namespace gagp {

struct ExecResult {
  bool is_error = false;
  Value value = Value::invalid();
  Err err{ErrCode::Value, ""};
};

ExecResult execute_bytecode_cpu(const BytecodeProgram& program,
                                const std::vector<std::pair<int, Value>>& inputs,
                                int fuel = 10000);

}  // namespace gagp
