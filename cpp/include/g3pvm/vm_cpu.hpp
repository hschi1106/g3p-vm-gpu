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

struct LocalBinding {
  int idx = 0;
  Value value = Value::none();
};

using InputCase = std::vector<LocalBinding>;

VMResult run_bytecode(const BytecodeProgram& program,
                      const std::vector<std::pair<int, Value>>& inputs,
                      int fuel = 10000);

// Returns one fitness score per program.
// Scoring:
//   score = exact_match_count - round(mean_abs_error) + runtime_error_count * (-10)
// where mean_abs_error is over numeric predictions/targets across all cases.
// Non-numeric non-error mismatches contribute 0 to mean_abs_error and exact bonus.
// All programs share one case set.
std::vector<int> run_bytecode_cpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000);

}  // namespace g3pvm
