#pragma once

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
// Scoring per case: error=-10, wrong=0, correct=1.
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
  Err err{ErrCode::Value, ""};
};

GPUFitnessEvalResult run_bytecode_gpu_multi_fitness_shared_cases_debug(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel = 10000,
    int blocksize = 256);

}  // namespace g3pvm
