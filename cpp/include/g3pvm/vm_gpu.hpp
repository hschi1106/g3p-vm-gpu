#pragma once

#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace g3pvm {

// Returns ValueError when CUDA runtime/device is unavailable.
VMResult run_bytecode_gpu(const BytecodeProgram& program,
                          const std::vector<std::pair<int, Value>>& inputs,
                          int fuel = 10000);

}  // namespace g3pvm
