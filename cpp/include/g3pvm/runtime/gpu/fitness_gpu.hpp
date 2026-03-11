#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"

namespace g3pvm {

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
