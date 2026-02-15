#pragma once

#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace g3pvm {

// Returns ValueError when CUDA runtime/device is unavailable.
std::vector<VMResult> run_bytecode_gpu_batch(const BytecodeProgram& program,
                                             const std::vector<InputCase>& cases,
                                             int fuel = 10000,
                                             int blocksize = 256);

// Returns nested results in program-major order: out[program_idx][case_idx].
// Returns ValueError when CUDA runtime/device is unavailable.
std::vector<std::vector<VMResult>> run_bytecode_gpu_multi_batch(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<std::vector<InputCase>>& cases_by_program,
    int fuel = 10000,
    int blocksize = 256);

// Returns one fitness score per program.
// Scoring per case: error=-10, wrong=0, correct=1.
// Returns empty vector when input shapes are invalid or device is unavailable.
std::vector<int> run_bytecode_gpu_multi_fitness(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<std::vector<InputCase>>& cases_by_program,
    const std::vector<std::vector<Value>>& expected_by_program,
    int fuel = 10000,
    int blocksize = 256);

}  // namespace g3pvm
