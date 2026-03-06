#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/exec_cpu.hpp"

namespace g3pvm {

// Returns one fitness score per program.
// Per-case scoring:
//   numeric expected/actual => -abs(actual - expected)
//   Bool/None/String/List => exact match ? 1 : 0
//   runtime error => 0
// All programs share one case set.
// Returns empty vector when input shapes are invalid or device is unavailable.
std::vector<double> eval_fitness_gpu(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<CaseInputs>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    int blocksize = 256);

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
    int blocksize = 256);

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
                            int blocksize = 256);
  FitnessEvalResult eval_programs(const std::vector<BytecodeProgram>& programs) const;
  bool is_ready() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g3pvm
