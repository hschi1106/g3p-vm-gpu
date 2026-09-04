#pragma once

#include <cstdint>

#include "gagp/core/errors.hpp"
#include "gagp/core/value.hpp"
#include "gagp/runtime/gpu/constants_gpu.hpp"
#include "gagp/runtime/gpu/payload_flavor_types.hpp"

namespace gagp::gpu_detail {

struct DInstr {
  std::uint8_t op = 0;
  std::uint8_t flags = 0;
  std::int32_t a = 0;
  std::int32_t b = 0;
};

struct DResult {
  int is_error = 0;
  ErrCode err_code = ErrCode::Value;
  Value value = Value::invalid();
};

struct DPhaseMeta {
  int code_offset = 0;
  int code_len = 0;
  int const_offset = 0;
  int const_len = 0;
  int n_locals = 0;
};

struct DAsgpDcSegment {
  int solve_xs_local = -1;
  int solve_n_local = -1;
  int solve_lo_local = -1;
  int divide_n_local = -1;
  int combine_left_local = -1;
  int combine_right_local = -1;
  DPhaseMeta solve;
  DPhaseMeta divide;
  DPhaseMeta combine;
};

struct DAsgpDp1dSegment {
  int lo = 0;
  int hi = 0;
  int base_state = 0;
  Value boundary_value = Value::invalid();
  int dep_kind = 0;
  int dep_offset_count = 0;
  int dep_offsets[DMAX_ASGP_DP_DEPS] = {};
  int solve_state_local = -1;
  int transition_state_local = -1;
  int transition_dep_count = 0;
  int transition_dep_locals[DMAX_ASGP_DP_DEPS] = {};
  DPhaseMeta solve;
  DPhaseMeta transition;
};

struct DAsgpDp2dSegment {
  int i_lo = 0;
  int i_hi = 0;
  int j_lo = 0;
  int j_hi = 0;
  int base_i = 0;
  int base_j = 0;
  Value boundary_value = Value::invalid();
  int dep_kind = 0;
  int solve_i_local = -1;
  int solve_j_local = -1;
  int transition_i_local = -1;
  int transition_j_local = -1;
  int transition_dep_count = 0;
  int transition_dep_locals[DMAX_ASGP_DP_DEPS] = {};
  DPhaseMeta solve;
  DPhaseMeta transition;
};

struct DProgramMeta {
  int code_offset = 0;
  int code_len = 0;
  int const_offset = 0;
  int const_len = 0;
  int n_locals = 0;
  int asgp_dc_offset = 0;
  int asgp_dc_count = 0;
  int asgp_dp1d_offset = 0;
  int asgp_dp1d_count = 0;
  int asgp_dp2d_offset = 0;
  int asgp_dp2d_count = 0;
  int case_offset = 0;
  int case_count = 0;
  int case_local_offset = 0;
  int is_valid = 0;
  DPayloadFlavor payload_flavor = DPayloadFlavor::None;
  ErrCode err_code = ErrCode::Value;
};

struct DStringPayloadEntry {
  std::int64_t packed = 0;
  int offset = 0;
  int len = 0;
};

struct DListPayloadEntry {
  ValueTag tag = ValueTag::Invalid;
  std::int64_t packed = 0;
  int offset = 0;
  int len = 0;
};

__host__ __device__ inline bool d_has_a(const DInstr& ins) { return (ins.flags & DINSTR_HAS_A) != 0; }
__host__ __device__ inline bool d_has_b(const DInstr& ins) { return (ins.flags & DINSTR_HAS_B) != 0; }

}  // namespace gagp::gpu_detail
