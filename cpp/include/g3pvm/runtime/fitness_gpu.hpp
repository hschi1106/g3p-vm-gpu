#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/fitness_cpu.hpp"

namespace g3pvm {

// Returns one fitness score per program.
// Per-case scoring:
//   if expected is numeric:
//     actual numeric => -abs(actual - expected)
//     actual non-numeric => -penalty
//   Bool/None/String/List => exact match ? 1 : 0
//     type mismatch => -penalty
//   runtime error => -penalty
// All programs share one case set.
// Returns empty vector when input shapes are invalid or device is unavailable.
std::vector<double> eval_fitness_gpu(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    int blocksize = 256,
    double penalty = 1.0);

// Debug variant for fitness evaluation with observable CUDA failure reason.
struct FitnessEvalResult {
  bool ok = false;
  std::vector<double> fitness;
  double pack_programs_ms = 0.0;
  double upload_programs_ms = 0.0;
  double kernel_exec_ms = 0.0;
  double copyback_ms = 0.0;
  double total_ms = 0.0;
  Err err{ErrCode::Value, ""};
};

FitnessEvalResult eval_fitness_gpu_profiled(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    int blocksize = 256,
    double penalty = 1.0);

class FitnessSessionGpu {
 public:
  FitnessSessionGpu();
  ~FitnessSessionGpu();
  FitnessSessionGpu(const FitnessSessionGpu&) = delete;
  FitnessSessionGpu& operator=(const FitnessSessionGpu&) = delete;
  FitnessSessionGpu(FitnessSessionGpu&&) noexcept;
  FitnessSessionGpu& operator=(FitnessSessionGpu&&) noexcept;

  FitnessEvalResult init(const std::vector<CaseInputs>& shared_cases,
                            const std::vector<Value>& shared_answer,
                            int fuel = 10000,
                            int blocksize = 256,
                            double penalty = 1.0);
  FitnessEvalResult eval_programs(const std::vector<BytecodeProgram>& programs) const;
  bool is_ready() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g3pvm
