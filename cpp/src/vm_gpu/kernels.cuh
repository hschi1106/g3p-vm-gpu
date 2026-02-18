#pragma once

#include <cmath>
#include <cstdint>

#include "device_builtins.cuh"
#include "device_exec.cuh"

namespace g3pvm::gpu_detail {

__device__ inline DResult d_exec_one_case(const DProgramMeta& meta,
                                          const DInstr* shared_code,
                                          const Value* all_consts,
                                          const Value* shared_case_local_vals,
                                          const unsigned char* shared_case_local_set,
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
        stack[sp++] = (x.tag == ValueTag::Float) ? Value::from_float(-x.f) : Value::from_int(-x.i);
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
        stack[sp++] = any_float ? Value::from_float(a_num + b_num)
                                : Value::from_int(static_cast<long long>(a_num) +
                                                  static_cast<long long>(b_num));
      } else if (ins.op == OP_SUB) {
        stack[sp++] = any_float ? Value::from_float(a_num - b_num)
                                : Value::from_int(static_cast<long long>(a_num) -
                                                  static_cast<long long>(b_num));
      } else if (ins.op == OP_MUL) {
        stack[sp++] = any_float ? Value::from_float(a_num * b_num)
                                : Value::from_int(static_cast<long long>(a_num) *
                                                  static_cast<long long>(b_num));
      } else if (ins.op == OP_DIV) {
        stack[sp++] = Value::from_float(a_num / b_num);
      } else {
        stack[sp++] = any_float
                          ? Value::from_float(d_float_mod(a_num, b_num))
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
      if (!d_builtin_call(bid, args_buf, argc, ret, derr)) {
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
                                             int n_programs, int fuel, DResult* all_out) {
  const int prog_idx = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (prog_idx < 0 || prog_idx >= n_programs) return;

  const DProgramMeta meta = metas[prog_idx];

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
        meta, shared_code, all_consts, shared_case_local_vals, shared_case_local_set, local_case, fuel);
  }
}

__global__ void vm_multi_fitness_kernel_shared_cases(
    const Value* all_consts, const DInstr* all_code, const DProgramMeta* metas,
    const Value* shared_case_local_vals, const unsigned char* shared_case_local_set,
    const Value* shared_answer, int n_programs, int fuel, int* fitness_out) {
  const int prog_idx = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (prog_idx < 0 || prog_idx >= n_programs) return;

  const DProgramMeta meta = metas[prog_idx];

  extern __shared__ DInstr shared_code[];
  __shared__ int block_exact_match_count;
  __shared__ int block_runtime_error_count;
  __shared__ double block_abs_error_sum;
  if (tid == 0) {
    block_exact_match_count = 0;
    block_runtime_error_count = 0;
    block_abs_error_sum = 0.0;
  }
  __syncthreads();

  if (meta.is_valid && meta.code_len > 0) {
    for (int i = tid; i < meta.code_len; i += static_cast<int>(blockDim.x)) {
      shared_code[i] = all_code[meta.code_offset + i];
    }
  }
  __syncthreads();

  int local_exact_match_count = 0;
  int local_runtime_error_count = 0;
  double local_abs_error_sum = 0.0;
  for (int local_case = tid; local_case < meta.case_count; local_case += static_cast<int>(blockDim.x)) {
    const DResult result = d_exec_one_case(
        meta, shared_code, all_consts, shared_case_local_vals, shared_case_local_set, local_case, fuel);
    if (result.is_error) {
      local_runtime_error_count += 1;
      continue;
    }

    if (d_value_equal_for_fitness(result.value, shared_answer[local_case])) {
      local_exact_match_count += 1;
    }

    double pred_num = 0.0;
    double expected_num = 0.0;
    bool any_float = false;
    if (vm_semantics::to_numeric_pair(result.value, shared_answer[local_case], pred_num, expected_num, any_float)) {
      (void)any_float;
      local_abs_error_sum += fabs(pred_num - expected_num);
    }
  }

  if (local_exact_match_count != 0) {
    atomicAdd(&block_exact_match_count, local_exact_match_count);
  }
  if (local_runtime_error_count != 0) {
    atomicAdd(&block_runtime_error_count, local_runtime_error_count);
  }
  if (local_abs_error_sum != 0.0) {
    atomicAdd(&block_abs_error_sum, local_abs_error_sum);
  }
  __syncthreads();
  if (tid == 0) {
    const double case_count = static_cast<double>(meta.case_count);
    const double mean_abs_error = (case_count > 0.0) ? (block_abs_error_sum / case_count) : 0.0;
    const int rounded_mean_abs_error = static_cast<int>(mean_abs_error + 0.5);
    fitness_out[prog_idx] =
        block_exact_match_count - rounded_mean_abs_error - block_runtime_error_count * 10;
  }
}

}  // namespace g3pvm::gpu_detail
