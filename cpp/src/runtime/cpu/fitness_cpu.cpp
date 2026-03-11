#include "g3pvm/runtime/fitness_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/runtime/execute_bytecode_cpu.hpp"

namespace g3pvm {

namespace {

double canonicalize_fitness_accumulator(double value) {
  if (!std::isfinite(value) || value == 0.0) {
    return value == 0.0 ? 0.0 : value;
  }
  int exponent = 0;
  const double mantissa = std::frexp(value, &exponent);
  constexpr int kMantissaBits = 48;
  const long long quantized_mantissa = std::llround(std::ldexp(mantissa, kMantissaBits));
  return std::ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

}  // namespace

std::vector<double> eval_fitness_cpu(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    double penalty,
    int reduction_lanes) {
  if (programs.empty() || shared_cases.empty()) {
    return {};
  }
  if (shared_answer.size() != shared_cases.size()) {
    return {};
  }

  std::vector<double> fitness(programs.size(), 0.0);
  const int lanes = std::max(1, reduction_lanes);
  for (std::size_t p = 0; p < programs.size(); ++p) {
    std::vector<double> partial_scores(static_cast<std::size_t>(lanes), 0.0);
    for (int lane = 0; lane < lanes; ++lane) {
      const std::size_t chunk_start =
          (shared_cases.size() * static_cast<std::size_t>(lane)) / static_cast<std::size_t>(lanes);
      const std::size_t chunk_end =
          (shared_cases.size() * static_cast<std::size_t>(lane + 1)) / static_cast<std::size_t>(lanes);
      double local_score = 0.0;
      for (std::size_t c = chunk_start; c < chunk_end; ++c) {
        std::vector<std::pair<int, Value>> inputs;
        inputs.reserve(shared_cases[c].size());
        for (const InputBinding& binding : shared_cases[c]) {
          inputs.push_back({binding.idx, binding.value});
        }
        const ExecResult out = execute_bytecode_cpu(programs[p], inputs, fuel);
        if (out.is_error) {
          local_score = canonicalize_fitness_accumulator(local_score - std::fabs(penalty));
          continue;
        }

        double case_score = 0.0;
        if (vm_semantics::fitness_score_for_values(out.value, shared_answer[c], penalty, case_score)) {
          local_score = canonicalize_fitness_accumulator(local_score + case_score);
        }
      }
      partial_scores[static_cast<std::size_t>(lane)] = local_score;
    }

    double score = 0.0;
    for (double partial_score : partial_scores) {
      score = canonicalize_fitness_accumulator(score + partial_score);
    }
    fitness[p] = score;
  }

  return fitness;
}

}  // namespace g3pvm
