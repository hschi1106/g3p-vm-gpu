#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"

namespace g3pvm {

struct FitnessSessionInitTiming {
  double device_select_ms = 0.0;
  double shared_case_pack_ms = 0.0;
  double payload_cache_warm_ms = 0.0;
  double upload_ms = 0.0;
  double total_ms = 0.0;
};

struct FitnessSessionInitResult {
  bool ok = false;
  FitnessSessionInitTiming timing;
  Err err{ErrCode::Value, ""};
};

struct FitnessEvalTiming {
  double pack_ms = 0.0;
  double launch_prep_ms = 0.0;
  double upload_ms = 0.0;
  double kernel_ms = 0.0;
  double copyback_ms = 0.0;
  double teardown_ms = 0.0;
  double total_ms = 0.0;
};

struct FitnessEvalResult {
  bool ok = false;
  std::vector<double> fitness;
  FitnessEvalTiming timing;
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

  FitnessSessionInitResult init(const std::vector<CaseBindings>& shared_cases,
                                const std::vector<Value>& shared_answer,
                                int fuel = 10000,
                                int blocksize = 1024,
                                double penalty = 1.0);
  FitnessEvalResult eval_programs(const std::vector<BytecodeProgram>& programs) const;
  bool is_ready() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g3pvm
