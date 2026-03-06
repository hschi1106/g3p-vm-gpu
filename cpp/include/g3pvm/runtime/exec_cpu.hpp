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

struct InputBinding {
  int idx = 0;
  Value value = Value::none();
};

using CaseInputs = std::vector<InputBinding>;

ExecResult exec_cpu(const BytecodeProgram& program,
                      const std::vector<std::pair<int, Value>>& inputs,
                      int fuel = 10000);

// Returns one fitness score per program.
// Per-case scoring:
//   if expected is numeric:
//     actual numeric => -abs(actual - expected)
//     actual non-numeric => -penalty
//   Bool/None/String/List => exact match ? 1 : 0
//     type mismatch => -penalty
//   runtime error => -penalty
// All programs share one case set.
std::vector<double> eval_fitness_cpu(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    double penalty = 1.0);

}  // namespace g3pvm
