#pragma once

#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace g3pvm {

struct LocalBinding {
  int idx = 0;
  Value value = Value::none();
};

using InputCase = std::vector<LocalBinding>;

// Returns ValueError when CUDA runtime/device is unavailable.
std::vector<VMResult> run_bytecode_gpu_batch(const BytecodeProgram& program,
                                             const std::vector<InputCase>& cases,
                                             int fuel = 10000,
                                             int blocksize = 256);

}  // namespace g3pvm
