#pragma once

#include <cstdint>

#include "g3pvm/value.hpp"
#include "constants.hpp"

namespace g3pvm::gpu_detail {

struct DInstr {
  std::uint8_t op = 0;
  std::uint8_t flags = 0;
  std::int32_t a = 0;
  std::int32_t b = 0;
};

struct DResult {
  int is_error = 0;
  int err_code = DERR_VALUE;
  Value value = Value::none();
};

struct DProgramMeta {
  int code_offset = 0;
  int code_len = 0;
  int const_offset = 0;
  int const_len = 0;
  int n_locals = 0;
  int case_offset = 0;
  int case_count = 0;
  int case_local_offset = 0;
  int is_valid = 0;
  int err_code = DERR_VALUE;
};

__host__ __device__ inline bool d_has_a(const DInstr& ins) { return (ins.flags & DINSTR_HAS_A) != 0; }
__host__ __device__ inline bool d_has_b(const DInstr& ins) { return (ins.flags & DINSTR_HAS_B) != 0; }

}  // namespace g3pvm::gpu_detail
