#pragma once

#include <cmath>
#include <cstdint>

#include "device_builtins.cuh"
#include "device_exec.cuh"

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

__device__ inline DResult d_exec_one_case(const DProgramMeta& meta,
                                          const DInstr* shared_code,
                                          const Value* all_consts,
                                          const Value* shared_case_local_vals,
                                          const unsigned char* shared_case_local_set,
                                          const DPayloadTables& payload_tables,
                                          int local_case,
                                          int fuel) {
  DResult result;
  result.is_error = 0;
  result.err_code = DERR_VALUE;
  result.value = Value::none();

  if (!meta.is_valid) {
    d_fail(result, static_cast<DeviceErrCode>(meta.err_code));
    return result;
  }

  Value stack[MAX_STACK];
  Value locals[MAX_LOCALS];
  static_assert(MAX_LOCALS <= 64, "local_set_mask requires MAX_LOCALS <= 64");
  std::uint64_t local_set_mask = 0;

  const int base = local_case * MAX_LOCALS;
  for (int i = 0; i < meta.n_locals; ++i) {
    locals[i] = shared_case_local_vals[base + i];
    if (shared_case_local_set[base + i]) {
      local_set_mask |= (std::uint64_t{1} << i);
    }
  }

  int sp = 0;
  int ip = 0;
  int fuel_left = fuel;
  bool returned = false;
  DThreadPayloadState payload_state;

  while (ip < meta.code_len) {
    if (fuel_left <= 0) {
      d_fail(result, DERR_TIMEOUT);
      break;
    }
    fuel_left -= 1;

    const DInstr ins = shared_code[ip];
    ip += 1;

    if (ins.op == OP_PUSH_CONST) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a >= meta.const_len || sp >= MAX_STACK) {
        d_fail(result, DERR_VALUE);
        break;
      }
      stack[sp++] = all_consts[meta.const_offset + ins.a];
      continue;
    }

    if (ins.op == OP_LOAD) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a >= meta.n_locals || sp >= MAX_STACK) {
        d_fail(result, DERR_NAME);
        break;
      }
      if ((local_set_mask & (std::uint64_t{1} << ins.a)) == 0) {
        d_fail(result, DERR_NAME);
        break;
      }
      stack[sp++] = locals[ins.a];
      continue;
    }

    if (ins.op == OP_STORE) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a >= meta.n_locals) {
        d_fail(result, DERR_NAME);
        break;
      }
      if (sp < 1) {
        d_fail(result, DERR_VALUE);
        break;
      }
      locals[ins.a] = stack[--sp];
      local_set_mask |= (std::uint64_t{1} << ins.a);
      continue;
    }

    if (ins.op == OP_NEG || ins.op == OP_NOT) {
      if (sp < 1) {
        d_fail(result, DERR_VALUE);
        break;
      }
      Value x = stack[--sp];
      if (ins.op == OP_NEG) {
        if (!d_is_num(x)) {
          d_fail(result, DERR_TYPE);
          break;
        }
        stack[sp++] = (x.tag == ValueTag::Float)
                          ? Value::from_float(vm_semantics::canonicalize_vm_float(-x.f))
                          : Value::from_int(vm_semantics::wrap_int_neg(x.i));
      } else {
        if (x.tag != ValueTag::Bool) {
          d_fail(result, DERR_TYPE);
          break;
        }
        stack[sp++] = Value::from_bool(!x.b);
      }
      continue;
    }

    if (ins.op == OP_ADD || ins.op == OP_SUB || ins.op == OP_MUL || ins.op == OP_DIV ||
        ins.op == OP_MOD) {
      if (sp < 2) {
        d_fail(result, DERR_VALUE);
        break;
      }
      Value b = stack[--sp];
      Value a = stack[--sp];
      double a_num = 0.0;
      double b_num = 0.0;
      bool any_float = false;
      if (!d_to_numeric_pair(a, b, a_num, b_num, any_float)) {
        d_fail(result, DERR_TYPE);
        break;
      }
      if ((ins.op == OP_DIV || ins.op == OP_MOD) && b_num == 0.0) {
        d_fail(result, DERR_ZERODIV);
        break;
      }
      if (ins.op == OP_ADD) {
        stack[sp++] = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(a_num + b_num))
                                : Value::from_int(vm_semantics::wrap_int_add(
                                      static_cast<long long>(a_num), static_cast<long long>(b_num)));
      } else if (ins.op == OP_SUB) {
        stack[sp++] = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(a_num - b_num))
                                : Value::from_int(vm_semantics::wrap_int_sub(
                                      static_cast<long long>(a_num), static_cast<long long>(b_num)));
      } else if (ins.op == OP_MUL) {
        stack[sp++] = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(a_num * b_num))
                                : Value::from_int(vm_semantics::wrap_int_mul(
                                      static_cast<long long>(a_num), static_cast<long long>(b_num)));
      } else if (ins.op == OP_DIV) {
        stack[sp++] = Value::from_float(vm_semantics::canonicalize_vm_float(a_num / b_num));
      } else {
        stack[sp++] = any_float
                          ? Value::from_float(vm_semantics::canonicalize_vm_float(d_float_mod(a_num, b_num)))
                          : Value::from_int(d_int_mod(static_cast<long long>(a_num),
                                                      static_cast<long long>(b_num)));
      }
      continue;
    }

    if (ins.op == OP_LT || ins.op == OP_LE || ins.op == OP_GT || ins.op == OP_GE ||
        ins.op == OP_EQ || ins.op == OP_NE) {
      if (sp < 2) {
        d_fail(result, DERR_VALUE);
        break;
      }
      Value b = stack[--sp];
      Value a = stack[--sp];
      bool cmp = false;
      DeviceErrCode derr = DERR_TYPE;
      if (!d_compare(ins.op, a, b, cmp, derr)) {
        d_fail(result, derr);
        break;
      }
      stack[sp++] = Value::from_bool(cmp);
      continue;
    }

    if (ins.op == OP_JMP) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a > meta.code_len) {
        d_fail(result, DERR_VALUE);
        break;
      }
      ip = ins.a;
      continue;
    }

    if (ins.op == OP_JMP_IF_FALSE || ins.op == OP_JMP_IF_TRUE) {
      if (sp < 1) {
        d_fail(result, DERR_VALUE);
        break;
      }
      if (!d_has_a(ins) || ins.a < 0 || ins.a > meta.code_len) {
        d_fail(result, DERR_VALUE);
        break;
      }
      Value c = stack[--sp];
      if (c.tag != ValueTag::Bool) {
        d_fail(result, DERR_TYPE);
        break;
      }
      if (ins.op == OP_JMP_IF_FALSE && !c.b) ip = ins.a;
      if (ins.op == OP_JMP_IF_TRUE && c.b) ip = ins.a;
      continue;
    }

    if (ins.op == OP_CALL_BUILTIN) {
      const int bid = d_has_a(ins) ? ins.a : -1;
      const int argc = d_has_b(ins) ? ins.b : -1;
      if (argc < 0 || sp < argc) {
        d_fail(result, (argc < 0) ? DERR_TYPE : DERR_VALUE);
        break;
      }
      Value args_buf[4];
      if (argc > 4) {
        d_fail(result, DERR_TYPE);
        break;
      }
      for (int i = 0; i < argc; ++i) {
        args_buf[i] = stack[sp - argc + i];
      }
      sp -= argc;
      DeviceErrCode derr = DERR_TYPE;
      Value ret = Value::none();
      if (!d_builtin_call(bid, args_buf, argc, payload_tables, payload_state, ret, derr)) {
        d_fail(result, derr);
        break;
      }
      if (sp >= MAX_STACK) {
        d_fail(result, DERR_VALUE);
        break;
      }
      stack[sp++] = ret;
      continue;
    }

    if (ins.op == OP_RETURN) {
      if (sp < 1) {
        d_fail(result, DERR_VALUE);
        break;
      }
      result.is_error = 0;
      result.value = stack[sp - 1];
      returned = true;
      break;
    }

    d_fail(result, DERR_TYPE);
    break;
  }

  if (!returned && !result.is_error) {
    d_fail(result, DERR_VALUE);
  }
  return result;
}

__global__ void vm_multi_kernel_shared_cases(const Value* all_consts, const DInstr* all_code,
                                             const DProgramMeta* metas,
                                             const Value* shared_case_local_vals,
                                             const unsigned char* shared_case_local_set,
                                             const DStringPayloadEntry* string_payload_entries,
                                             int string_payload_entry_count,
                                             const char* string_payload_bytes,
                                             const DListPayloadEntry* list_payload_entries,
                                             int list_payload_entry_count,
                                             const Value* list_payload_values,
                                             int n_programs, int fuel, DResult* all_out) {
  const int prog_idx = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (prog_idx < 0 || prog_idx >= n_programs) return;

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

  for (int local_case = tid; local_case < meta.case_count; local_case += static_cast<int>(blockDim.x)) {
    const int out_idx = meta.case_offset + local_case;
    all_out[out_idx] = d_exec_one_case(
        meta, shared_code, all_consts, shared_case_local_vals, shared_case_local_set, payload_tables, local_case, fuel);
  }
}

__global__ void vm_multi_fitness_kernel_shared_cases(
    const Value* all_consts, const DInstr* all_code, const DProgramMeta* metas,
    const Value* shared_case_local_vals, const unsigned char* shared_case_local_set,
    const Value* shared_answer,
    const DStringPayloadEntry* string_payload_entries, int string_payload_entry_count,
    const char* string_payload_bytes,
    const DListPayloadEntry* list_payload_entries, int list_payload_entry_count,
    const Value* list_payload_values,
    int n_programs, int fuel, double penalty, double* fitness_out) {
  const int prog_idx = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (prog_idx < 0 || prog_idx >= n_programs) return;

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
    const DResult result = d_exec_one_case(
        meta, shared_code, all_consts, shared_case_local_vals, shared_case_local_set, payload_tables, local_case, fuel);
    if (result.is_error) {
      local_score = d_canonicalize_fitness_accumulator(local_score - fabs(penalty));
      continue;
    }

    double case_score = 0.0;
    if (d_fitness_score_for_values(result.value, shared_answer[local_case], penalty, case_score)) {
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
