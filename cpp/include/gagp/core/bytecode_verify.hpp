#pragma once

#include <cstddef>
#include <string>

#include "gagp/core/bytecode.hpp"

namespace gagp {

enum class BytecodeVerifyCode {
  Ok,
  InvalidLocalCount,
  InvalidConstant,
  UnknownOpcode,
  MissingOperand,
  InvalidConstantIndex,
  InvalidLocalIndex,
  InvalidBuiltinId,
  InvalidBuiltinArity,
  InvalidJumpTarget,
  InvalidPrivateOpcode,
  InvalidSegmentIndex,
  StackUnderflow,
  StackJoinMismatch,
  InvalidFallthrough,
  InvalidVarMapping,
  InvalidBinderLocal,
  InvalidSegmentMetadata,
  ResourceLimit,
};

const char* bytecode_verify_code_name(BytecodeVerifyCode code) noexcept;

struct BytecodeVerifyDiagnostic {
  BytecodeVerifyCode code = BytecodeVerifyCode::Ok;
  std::size_t instruction_index = 0;
  std::string path;
  std::string message;
};

struct BytecodeVerifyOptions {
  bool allow_private_opcodes = true;
  std::size_t max_instructions_per_code = 0;
  std::size_t max_constants_per_code = 0;
  std::size_t max_locals_per_code = 0;
  std::size_t max_segments = 0;
  std::size_t max_stack_depth = 0;
};

struct VerifiedBytecode {
  std::size_t max_stack_depth = 0;
  std::size_t phase_program_count = 0;
  bool has_reachable_return = false;
};

struct BytecodeVerifyResult {
  bool ok = false;
  VerifiedBytecode verified;
  BytecodeVerifyDiagnostic diagnostic;

  explicit operator bool() const noexcept { return ok; }
};

BytecodeVerifyResult verify_bytecode(
    const BytecodeProgram& program,
    const BytecodeVerifyOptions& options = BytecodeVerifyOptions{});

}  // namespace gagp
