#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace g3pvm {

// Returns nested results in program-major order: out[program_idx][case_idx].
// All programs share one case set.
// Returns ValueError when CUDA runtime/device is unavailable.
std::vector<std::vector<VMResult>> run_bytecode_gpu_multi_batch(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    int fuel = 10000,
    int blocksize = 256);

// Returns one fitness score per program.
// Scoring:
//   score = exact_match_count - round(mean_abs_error) + runtime_error_count * (-10)
// where mean_abs_error is over numeric predictions/targets across all cases.
// Non-numeric non-error mismatches contribute 0 to mean_abs_error and exact bonus.
// All programs share one case set.
// Returns empty vector when input shapes are invalid or device is unavailable.
std::vector<int> run_bytecode_gpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    int blocksize = 256);

// Debug variant for fitness evaluation with observable CUDA failure reason.
struct GPUFitnessEvalResult {
  bool ok = false;
  std::vector<int> fitness;
  double pack_programs_ms = 0.0;
  double upload_programs_ms = 0.0;
  double kernel_exec_ms = 0.0;
  double copyback_ms = 0.0;
  double total_ms = 0.0;
  Err err{ErrCode::Value, ""};
};

GPUFitnessEvalResult run_bytecode_gpu_multi_fitness_shared_cases_debug(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    int blocksize = 256);

class GPUFitnessSession {
 public:
  GPUFitnessSession();
  ~GPUFitnessSession();
  GPUFitnessSession(const GPUFitnessSession&) = delete;
  GPUFitnessSession& operator=(const GPUFitnessSession&) = delete;
  GPUFitnessSession(GPUFitnessSession&&) noexcept;
  GPUFitnessSession& operator=(GPUFitnessSession&&) noexcept;

  GPUFitnessEvalResult init(const std::vector<InputCase>& shared_cases,
                            const std::vector<Value>& shared_answer,
                            int fuel = 10000,
                            int blocksize = 256);
  GPUFitnessEvalResult eval_programs(const std::vector<BytecodeProgram>& programs) const;
  bool is_ready() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g3pvm
