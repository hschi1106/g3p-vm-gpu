#pragma once

#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/value.hpp"

namespace g3pvm {

struct InputBinding {
  int idx = 0;
  Value value = Value::none();
};

using CaseBindings = std::vector<InputBinding>;

// Returns one fitness score per program.
// Per-case scoring:
//   if expected is numeric:
//     actual numeric => -abs(actual - expected)
//     actual non-numeric => -penalty
//   Bool/None/String/NumList/StringList => exact match ? 1 : 0
//     type mismatch => -penalty
//   runtime error => -penalty
// All programs share one case set.
std::vector<double> eval_fitness_cpu(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<CaseBindings>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    double penalty = 1.0,
    int reduction_lanes = 1);

}  // namespace g3pvm
