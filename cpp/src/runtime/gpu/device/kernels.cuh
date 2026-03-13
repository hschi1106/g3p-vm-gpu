#pragma once

#include <cmath>
#include "execute_bytecode_device.cuh"

namespace g3pvm::gpu_detail {

__device__ inline double d_canonicalize_fitness_accumulator(double value) {
  if (!isfinite(value) || value == 0.0) {
    return value == 0.0 ? 0.0 : value;
  }
  int exponent = 0;
  const double mantissa = frexp(value, &exponent);
  constexpr int kMantissaBits = 48;
  const long long quantized_mantissa = llround(ldexp(mantissa, kMantissaBits));
  return ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

template <bool EnablePayload>
__global__ void evaluate_fitness_subset(
    const int* program_indices, int n_program_indices,
    const Value* all_consts, const DInstr* all_code, const DProgramMeta* metas,
    const Value* shared_case_local_vals, const unsigned char* shared_case_local_set,
    const Value* shared_answer,
    const DStringPayloadEntry* string_payload_entries, int string_payload_entry_count,
    const char* string_payload_bytes,
    const DListPayloadEntry* list_payload_entries, int list_payload_entry_count,
    const Value* list_payload_values,
    int fuel, double penalty, double* fitness_out) {
  const int program_slot = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (program_slot < 0 || program_slot >= n_program_indices) return;

  const int prog_idx = program_indices[program_slot];

  const DProgramMeta meta = metas[prog_idx];
  const DPayloadTables payload_tables{
      string_payload_entries,
      string_payload_entry_count,
      string_payload_bytes,
      list_payload_entries,
      list_payload_entry_count,
      list_payload_values,
  };

  extern __shared__ DInstr shared_code[];
  if (meta.is_valid && meta.code_len > 0) {
    for (int i = tid; i < meta.code_len; i += static_cast<int>(blockDim.x)) {
      shared_code[i] = all_code[meta.code_offset + i];
    }
  }
  __syncthreads();

  extern __shared__ unsigned char shared_bytes[];
  const std::size_t code_bytes = sizeof(DInstr) * static_cast<std::size_t>(meta.code_len);
  const std::size_t partial_offset =
      (code_bytes + alignof(double) - 1u) & ~static_cast<std::size_t>(alignof(double) - 1u);
  double* partial_scores = reinterpret_cast<double*>(shared_bytes + partial_offset);

  double local_score = 0.0;
  const int chunk_start = (meta.case_count * tid) / static_cast<int>(blockDim.x);
  const int chunk_end = (meta.case_count * (tid + 1)) / static_cast<int>(blockDim.x);
  for (int local_case = chunk_start; local_case < chunk_end; ++local_case) {
    const DResult result = d_execute_bytecode<EnablePayload>(
        meta, shared_code, all_consts, shared_case_local_vals, shared_case_local_set, payload_tables, local_case, fuel);
    if (result.is_error) {
      local_score = d_canonicalize_fitness_accumulator(local_score - fabs(penalty));
      continue;
    }

    double case_score = 0.0;
    if (vm_semantics::fitness_score_for_values(result.value, shared_answer[local_case], penalty, case_score)) {
      local_score = d_canonicalize_fitness_accumulator(local_score + case_score);
    }
  }

  partial_scores[tid] = local_score;
  __syncthreads();

  if (tid == 0) {
    double total_score = 0.0;
    for (int i = 0; i < static_cast<int>(blockDim.x); ++i) {
      total_score = d_canonicalize_fitness_accumulator(total_score + partial_scores[i]);
    }
    fitness_out[prog_idx] = total_score;
  }
}

}  // namespace g3pvm::gpu_detail
