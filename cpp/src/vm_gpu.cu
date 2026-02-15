#include "g3pvm/vm_gpu.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace g3pvm {

namespace {

constexpr int MAX_STACK = 64;
constexpr int MAX_LOCALS = 64;
constexpr std::uint8_t DINSTR_HAS_A = 1;
constexpr std::uint8_t DINSTR_HAS_B = 2;

enum DeviceErrCode : int {
  DERR_NAME = 0,
  DERR_TYPE = 1,
  DERR_ZERODIV = 2,
  DERR_VALUE = 3,
  DERR_TIMEOUT = 4,
};

enum DeviceOp : int {
  OP_PUSH_CONST = 0,
  OP_LOAD = 1,
  OP_STORE = 2,
  OP_NEG = 3,
  OP_NOT = 4,
  OP_ADD = 5,
  OP_SUB = 6,
  OP_MUL = 7,
  OP_DIV = 8,
  OP_MOD = 9,
  OP_LT = 10,
  OP_LE = 11,
  OP_GT = 12,
  OP_GE = 13,
  OP_EQ = 14,
  OP_NE = 15,
  OP_JMP = 16,
  OP_JMP_IF_FALSE = 17,
  OP_JMP_IF_TRUE = 18,
  OP_CALL_BUILTIN = 19,
  OP_RETURN = 20,
};

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

__device__ bool d_is_num(const Value& v) {
  return v.tag == ValueTag::Int || v.tag == ValueTag::Float;
}

__device__ bool d_to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                                  bool& any_float) {
  if (!d_is_num(a) || !d_is_num(b)) return false;
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

__device__ double d_floor(double x) {
  long long i = static_cast<long long>(x);
  if (static_cast<double>(i) > x) return static_cast<double>(i - 1);
  return static_cast<double>(i);
}

__device__ double d_float_mod(double a, double b) {
  return a - d_floor(a / b) * b;
}

__device__ long long d_int_mod(long long a, long long b) {
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

__device__ void d_fail(DResult& out, DeviceErrCode code) {
  out.is_error = 1;
  out.err_code = static_cast<int>(code);
}

__device__ bool d_compare(const int op, const Value& a, const Value& b, bool& out_bool,
                          DeviceErrCode& err) {
  double a_num = 0.0;
  double b_num = 0.0;
  bool any_float = false;
  if (d_to_numeric_pair(a, b, a_num, b_num, any_float)) {
    if (op == OP_LT) out_bool = a_num < b_num;
    else if (op == OP_LE) out_bool = a_num <= b_num;
    else if (op == OP_GT) out_bool = a_num > b_num;
    else if (op == OP_GE) out_bool = a_num >= b_num;
    else if (op == OP_EQ) out_bool = a_num == b_num;
    else if (op == OP_NE) out_bool = a_num != b_num;
    else {
      err = DERR_TYPE;
      return false;
    }
    return true;
  }

  if (a.tag == ValueTag::Bool && b.tag == ValueTag::Bool) {
    if (op == OP_EQ) {
      out_bool = (a.b == b.b);
      return true;
    }
    if (op == OP_NE) {
      out_bool = (a.b != b.b);
      return true;
    }
    err = DERR_TYPE;
    return false;
  }

  if (a.tag == ValueTag::None || b.tag == ValueTag::None) {
    if (op == OP_EQ) {
      out_bool = (a.tag == ValueTag::None && b.tag == ValueTag::None);
      return true;
    }
    if (op == OP_NE) {
      out_bool = !(a.tag == ValueTag::None && b.tag == ValueTag::None);
      return true;
    }
    err = DERR_TYPE;
    return false;
  }

  err = DERR_TYPE;
  return false;
}

__device__ bool d_value_equal_for_fitness(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::None) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int) return a.i == b.i;
  if (a.tag == ValueTag::Float) return fabs(a.f - b.f) <= 1e-12;
  return false;
}

__device__ bool d_builtin_call(int bid, const Value* args, int argc, Value& out, DeviceErrCode& err) {
  if (bid == 0) {
    if (argc != 1) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    if (!d_is_num(x)) {
      err = DERR_TYPE;
      return false;
    }
    out = (x.tag == ValueTag::Float) ? Value::from_float(x.f < 0 ? -x.f : x.f)
                                      : Value::from_int(x.i < 0 ? -x.i : x.i);
    return true;
  }

  if (bid == 1 || bid == 2) {
    if (argc != 2) {
      err = DERR_TYPE;
      return false;
    }
    double a = 0.0;
    double b = 0.0;
    bool any_float = false;
    if (!d_to_numeric_pair(args[0], args[1], a, b, any_float)) {
      err = DERR_TYPE;
      return false;
    }
    const double pick = (bid == 1) ? ((a <= b) ? a : b) : ((a >= b) ? a : b);
    out = any_float ? Value::from_float(pick) : Value::from_int(static_cast<long long>(pick));
    return true;
  }

  if (bid == 3) {
    if (argc != 3) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!d_is_num(x) || !d_is_num(lo) || !d_is_num(hi)) {
      err = DERR_TYPE;
      return false;
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      if (lo2 > hi2) {
        err = DERR_VALUE;
        return false;
      }
      out = (x2 < lo2) ? Value::from_float(lo2)
                       : ((x2 > hi2) ? Value::from_float(hi2) : Value::from_float(x2));
      return true;
    }
    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    if (lo2 > hi2) {
      err = DERR_VALUE;
      return false;
    }
    out = (x2 < lo2) ? Value::from_int(lo2)
                     : ((x2 > hi2) ? Value::from_int(hi2) : Value::from_int(x2));
    return true;
  }

  err = DERR_NAME;
  return false;
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

  Value stack[MAX_STACK];
  Value locals[MAX_LOCALS];
  unsigned char local_set[MAX_LOCALS];

  for (int local_case = tid; local_case < meta.case_count; local_case += static_cast<int>(blockDim.x)) {
    const int out_idx = meta.case_offset + local_case;
    DResult result;
    result.is_error = 0;
    result.err_code = DERR_VALUE;
    result.value = Value::none();

    if (!meta.is_valid) {
      d_fail(result, static_cast<DeviceErrCode>(meta.err_code));
      all_out[out_idx] = result;
      continue;
    }

    const int base = local_case * MAX_LOCALS;
    for (int i = 0; i < meta.n_locals; ++i) {
      locals[i] = shared_case_local_vals[base + i];
      local_set[i] = shared_case_local_set[base + i];
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
        if (!local_set[ins.a]) {
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
        local_set[ins.a] = 1;
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
    all_out[out_idx] = result;
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
  __shared__ int block_score;
  if (tid == 0) {
    block_score = 0;
  }
  __syncthreads();

  if (meta.is_valid && meta.code_len > 0) {
    for (int i = tid; i < meta.code_len; i += static_cast<int>(blockDim.x)) {
      shared_code[i] = all_code[meta.code_offset + i];
    }
  }
  __syncthreads();

  int local_score = 0;
  Value stack[MAX_STACK];
  Value locals[MAX_LOCALS];
  unsigned char local_set[MAX_LOCALS];

  for (int local_case = tid; local_case < meta.case_count; local_case += static_cast<int>(blockDim.x)) {
    if (!meta.is_valid) {
      local_score -= 10;
      continue;
    }

    DResult result;
    result.is_error = 0;
    result.err_code = DERR_VALUE;
    result.value = Value::none();

    const int base = local_case * MAX_LOCALS;
    for (int i = 0; i < meta.n_locals; ++i) {
      locals[i] = shared_case_local_vals[base + i];
      local_set[i] = shared_case_local_set[base + i];
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
        if (!local_set[ins.a]) {
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
        local_set[ins.a] = 1;
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

    if (result.is_error) {
      local_score -= 10;
    } else if (d_value_equal_for_fitness(result.value, shared_answer[local_case])) {
      local_score += 1;
    }
  }

  if (local_score != 0) {
    atomicAdd(&block_score, local_score);
  }
  __syncthreads();
  if (tid == 0) {
    fitness_out[prog_idx] = block_score;
  }
}

int host_opcode(const std::string& op) {
  if (op == "PUSH_CONST") return OP_PUSH_CONST;
  if (op == "LOAD") return OP_LOAD;
  if (op == "STORE") return OP_STORE;
  if (op == "NEG") return OP_NEG;
  if (op == "NOT") return OP_NOT;
  if (op == "ADD") return OP_ADD;
  if (op == "SUB") return OP_SUB;
  if (op == "MUL") return OP_MUL;
  if (op == "DIV") return OP_DIV;
  if (op == "MOD") return OP_MOD;
  if (op == "LT") return OP_LT;
  if (op == "LE") return OP_LE;
  if (op == "GT") return OP_GT;
  if (op == "GE") return OP_GE;
  if (op == "EQ") return OP_EQ;
  if (op == "NE") return OP_NE;
  if (op == "JMP") return OP_JMP;
  if (op == "JMP_IF_FALSE") return OP_JMP_IF_FALSE;
  if (op == "JMP_IF_TRUE") return OP_JMP_IF_TRUE;
  if (op == "CALL_BUILTIN") return OP_CALL_BUILTIN;
  if (op == "RETURN") return OP_RETURN;
  return -1;
}

ErrCode from_device_err(const int code) {
  if (code == DERR_NAME) return ErrCode::Name;
  if (code == DERR_TYPE) return ErrCode::Type;
  if (code == DERR_ZERODIV) return ErrCode::ZeroDiv;
  if (code == DERR_TIMEOUT) return ErrCode::Timeout;
  return ErrCode::Value;
}

const char* device_err_message(const int code) {
  if (code == DERR_NAME) return "gpu vm name error";
  if (code == DERR_TYPE) return "gpu vm type error";
  if (code == DERR_ZERODIV) return "gpu vm zero division";
  if (code == DERR_TIMEOUT) return "gpu vm timeout";
  return "gpu vm value error";
}

template <typename T>
bool cuda_alloc_and_copy_in(const std::vector<T>& host, T** dev) {
  if (host.empty()) {
    *dev = nullptr;
    return true;
  }
  if (cudaMalloc(reinterpret_cast<void**>(dev), sizeof(T) * host.size()) != cudaSuccess) return false;
  if (cudaMemcpy(*dev, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*dev);
    *dev = nullptr;
    return false;
  }
  return true;
}

std::vector<std::vector<VMResult>> multi_single_error(ErrCode code, const std::string& message) {
  return std::vector<std::vector<VMResult>>{
      std::vector<VMResult>{VMResult{true, Value::none(), Err{code, message}}}};
}

bool select_least_used_device(cudaDeviceProp& props_out, std::string& message_out) {
  int device_count = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess || device_count <= 0) {
    message_out = "cuda device unavailable";
    if (count_err != cudaSuccess) {
      message_out += " err=";
      message_out += cudaGetErrorString(count_err);
    }
    return false;
  }

  int best_dev = -1;
  double best_used_ratio = 2.0;
  std::size_t best_used_bytes = std::numeric_limits<std::size_t>::max();

  for (int dev = 0; dev < device_count; ++dev) {
    if (cudaSetDevice(dev) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total_bytes == 0) {
      continue;
    }
    const std::size_t used_bytes = total_bytes - free_bytes;
    const double used_ratio = static_cast<double>(used_bytes) / static_cast<double>(total_bytes);

    if (best_dev < 0 || used_ratio < best_used_ratio ||
        (std::fabs(used_ratio - best_used_ratio) <= 1e-12 && used_bytes < best_used_bytes)) {
      best_dev = dev;
      best_used_ratio = used_ratio;
      best_used_bytes = used_bytes;
    }
  }

  if (best_dev < 0) {
    message_out = "cuda device unavailable err=no usable device found";
    return false;
  }

  const cudaError_t set_err = cudaSetDevice(best_dev);
  if (set_err != cudaSuccess) {
    message_out = "cuda device select failure err=";
    message_out += cudaGetErrorString(set_err);
    return false;
  }

  const cudaError_t prop_err = cudaGetDeviceProperties(&props_out, best_dev);
  if (prop_err != cudaSuccess) {
    message_out = "cuda device query failure err=";
    message_out += cudaGetErrorString(prop_err);
    return false;
  }

  return true;
}

void cuda_cleanup_multi(Value* d_consts, DInstr* d_code, DProgramMeta* d_metas,
                        Value* d_case_local_vals, unsigned char* d_case_local_set, DResult* d_out) {
  if (d_consts) cudaFree(d_consts);
  if (d_code) cudaFree(d_code);
  if (d_metas) cudaFree(d_metas);
  if (d_case_local_vals) cudaFree(d_case_local_vals);
  if (d_case_local_set) cudaFree(d_case_local_set);
  if (d_out) cudaFree(d_out);
}

}  // namespace

std::vector<std::vector<VMResult>> run_bytecode_gpu_multi_batch(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    int fuel,
    int blocksize) {
  if (programs.empty() || shared_cases.empty()) {
    return multi_single_error(ErrCode::Value, "programs must not be empty");
  }
  if (blocksize <= 0) {
    return multi_single_error(ErrCode::Value, "invalid gpu blocksize");
  }

  cudaDeviceProp props;
  std::string select_msg;
  if (!select_least_used_device(props, select_msg)) {
    return multi_single_error(ErrCode::Value, select_msg);
  }

  if (blocksize > props.maxThreadsPerBlock) {
    return multi_single_error(ErrCode::Value, "gpu blocksize exceeds maxThreadsPerBlock");
  }

  std::vector<std::vector<VMResult>> result_by_program;
  result_by_program.resize(programs.size());

  std::vector<DProgramMeta> metas(programs.size());
  std::vector<DInstr> all_code;
  std::vector<Value> all_consts;
  std::size_t total_cases = 0;
  std::size_t max_code_len = 0;
  const int shared_case_count = static_cast<int>(shared_cases.size());

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const BytecodeProgram& prog = programs[p];

    DProgramMeta meta;
    meta.code_offset = static_cast<int>(all_code.size());
    meta.const_offset = static_cast<int>(all_consts.size());
    meta.n_locals = prog.n_locals;
    meta.case_offset = static_cast<int>(total_cases);
    meta.case_count = shared_case_count;
    meta.case_local_offset = 0;
    meta.is_valid = 1;
    meta.err_code = DERR_VALUE;

    if (prog.n_locals > MAX_LOCALS) {
      meta.is_valid = 0;
      meta.err_code = DERR_VALUE;
    }

    all_consts.insert(all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      const int op = host_opcode(ins.op);
      if (op < 0) {
        meta.is_valid = 0;
        meta.err_code = DERR_TYPE;
        continue;
      }
      DInstr di;
      di.op = static_cast<std::uint8_t>(op);
      di.flags = static_cast<std::uint8_t>((ins.has_a ? DINSTR_HAS_A : 0) | (ins.has_b ? DINSTR_HAS_B : 0));
      di.a = static_cast<std::int32_t>(ins.a);
      di.b = static_cast<std::int32_t>(ins.b);
      all_code.push_back(di);
    }
    meta.code_len = static_cast<int>(all_code.size()) - meta.code_offset;
    if (static_cast<std::size_t>(meta.code_len) > max_code_len) {
      max_code_len = static_cast<std::size_t>(meta.code_len);
    }

    if (prog.n_locals < 0 || prog.n_locals > MAX_LOCALS) {
      meta.is_valid = 0;
      meta.err_code = DERR_VALUE;
    }
    total_cases += static_cast<std::size_t>(shared_case_count);
    metas[p] = meta;
  }

  if (total_cases == 0) {
    return multi_single_error(ErrCode::Value, "cases must not be empty");
  }

  const std::size_t shared_bytes = max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    return multi_single_error(ErrCode::Value, "shared memory requirement exceeded");
  }

  std::vector<Value> packed_case_local_vals(shared_cases.size() * MAX_LOCALS, Value::none());
  std::vector<unsigned char> packed_case_local_set(shared_cases.size() * MAX_LOCALS, 0);
  for (std::size_t case_idx = 0; case_idx < shared_cases.size(); ++case_idx) {
    const std::size_t base = case_idx * MAX_LOCALS;
    for (const LocalBinding& binding : shared_cases[case_idx]) {
      if (binding.idx >= 0 && binding.idx < MAX_LOCALS) {
        packed_case_local_vals[base + static_cast<std::size_t>(binding.idx)] = binding.value;
        packed_case_local_set[base + static_cast<std::size_t>(binding.idx)] = 1;
      }
    }
  }

  Value* d_consts = nullptr;
  DInstr* d_code = nullptr;
  DProgramMeta* d_metas = nullptr;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  DResult* d_out = nullptr;

  if (!cuda_alloc_and_copy_in(all_consts, &d_consts) || !cuda_alloc_and_copy_in(all_code, &d_code) ||
      !cuda_alloc_and_copy_in(metas, &d_metas) ||
      !cuda_alloc_and_copy_in(packed_case_local_vals, &d_shared_case_local_vals) ||
      !cuda_alloc_and_copy_in(packed_case_local_set, &d_shared_case_local_set)) {
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, "cuda allocation failure");
  }

  if (cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(DResult) * total_cases) != cudaSuccess) {
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, "cuda allocation failure");
  }

  vm_multi_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), blocksize, shared_bytes>>>(
      d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set,
      static_cast<int>(programs.size()), fuel, d_out);

  const cudaError_t launch_err = cudaGetLastError();
  const cudaError_t sync_err = cudaDeviceSynchronize();
  if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
    std::string msg = "cuda kernel execution failure";
    if (launch_err != cudaSuccess) {
      msg += " launch=";
      msg += cudaGetErrorString(launch_err);
    }
    if (sync_err != cudaSuccess) {
      msg += " sync=";
      msg += cudaGetErrorString(sync_err);
    }
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, msg);
  }

  std::vector<DResult> host_out(total_cases);
  if (cudaMemcpy(host_out.data(), d_out, sizeof(DResult) * total_cases, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);
    return multi_single_error(ErrCode::Value, "cuda copy-back failure");
  }

  cuda_cleanup_multi(d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_out);

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const DProgramMeta& meta = metas[p];
    std::vector<VMResult>& outp = result_by_program[p];
    outp.reserve(static_cast<std::size_t>(meta.case_count));
    for (int local_case = 0; local_case < meta.case_count; ++local_case) {
      const DResult& r = host_out[static_cast<std::size_t>(meta.case_offset + local_case)];
      if (r.is_error) {
        outp.push_back(
            VMResult{true, Value::none(), Err{from_device_err(r.err_code), device_err_message(r.err_code)}});
      } else {
        outp.push_back(VMResult{false, r.value, Err{ErrCode::Value, ""}});
      }
    }
  }

  return result_by_program;
}

std::vector<int> run_bytecode_gpu_multi_fitness_shared_cases(
    const std::vector<BytecodeProgram>& programs,
    const std::vector<InputCase>& shared_cases,
    const std::vector<Value>& shared_answer,
    int fuel,
    int blocksize) {
  if (programs.empty() || shared_cases.empty()) {
    return {};
  }
  if (shared_answer.size() != shared_cases.size()) {
    return {};
  }
  if (blocksize <= 0) {
    return {};
  }

  cudaDeviceProp props;
  std::string select_msg;
  if (!select_least_used_device(props, select_msg)) {
    return {};
  }
  if (blocksize > props.maxThreadsPerBlock) {
    return {};
  }

  const int shared_case_count = static_cast<int>(shared_cases.size());
  std::vector<DProgramMeta> metas(programs.size());
  std::vector<DInstr> all_code;
  std::vector<Value> all_consts;
  std::size_t total_cases = 0;
  std::size_t max_code_len = 0;

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const BytecodeProgram& prog = programs[p];

    DProgramMeta meta;
    meta.code_offset = static_cast<int>(all_code.size());
    meta.const_offset = static_cast<int>(all_consts.size());
    meta.n_locals = prog.n_locals;
    meta.case_offset = static_cast<int>(total_cases);
    meta.case_count = shared_case_count;
    meta.case_local_offset = 0;
    meta.is_valid = 1;
    meta.err_code = DERR_VALUE;

    if (prog.n_locals > MAX_LOCALS || prog.n_locals < 0) {
      meta.is_valid = 0;
      meta.err_code = DERR_VALUE;
    }

    all_consts.insert(all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      const int op = host_opcode(ins.op);
      if (op < 0) {
        meta.is_valid = 0;
        meta.err_code = DERR_TYPE;
        continue;
      }
      DInstr di;
      di.op = static_cast<std::uint8_t>(op);
      di.flags = static_cast<std::uint8_t>((ins.has_a ? DINSTR_HAS_A : 0) | (ins.has_b ? DINSTR_HAS_B : 0));
      di.a = static_cast<std::int32_t>(ins.a);
      di.b = static_cast<std::int32_t>(ins.b);
      all_code.push_back(di);
    }
    meta.code_len = static_cast<int>(all_code.size()) - meta.code_offset;
    if (static_cast<std::size_t>(meta.code_len) > max_code_len) {
      max_code_len = static_cast<std::size_t>(meta.code_len);
    }

    total_cases += static_cast<std::size_t>(shared_case_count);
    metas[p] = meta;
  }

  if (total_cases == 0) {
    return {};
  }

  const std::size_t shared_bytes = max_code_len * sizeof(DInstr);
  if (shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    return {};
  }

  std::vector<Value> packed_case_local_vals(shared_cases.size() * MAX_LOCALS, Value::none());
  std::vector<unsigned char> packed_case_local_set(shared_cases.size() * MAX_LOCALS, 0);
  for (std::size_t case_idx = 0; case_idx < shared_cases.size(); ++case_idx) {
    const std::size_t base = case_idx * MAX_LOCALS;
    for (const LocalBinding& binding : shared_cases[case_idx]) {
      if (binding.idx >= 0 && binding.idx < MAX_LOCALS) {
        packed_case_local_vals[base + static_cast<std::size_t>(binding.idx)] = binding.value;
        packed_case_local_set[base + static_cast<std::size_t>(binding.idx)] = 1;
      }
    }
  }

  Value* d_consts = nullptr;
  DInstr* d_code = nullptr;
  DProgramMeta* d_metas = nullptr;
  Value* d_shared_case_local_vals = nullptr;
  unsigned char* d_shared_case_local_set = nullptr;
  Value* d_expected = nullptr;
  int* d_fitness = nullptr;

  if (!cuda_alloc_and_copy_in(all_consts, &d_consts) || !cuda_alloc_and_copy_in(all_code, &d_code) ||
      !cuda_alloc_and_copy_in(metas, &d_metas) ||
      !cuda_alloc_and_copy_in(packed_case_local_vals, &d_shared_case_local_vals) ||
      !cuda_alloc_and_copy_in(packed_case_local_set, &d_shared_case_local_set) ||
      !cuda_alloc_and_copy_in(shared_answer, &d_expected)) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_metas) cudaFree(d_metas);
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
    if (d_fitness) cudaFree(d_fitness);
    return {};
  }

  if (cudaMalloc(reinterpret_cast<void**>(&d_fitness), sizeof(int) * programs.size()) != cudaSuccess) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_metas) cudaFree(d_metas);
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
    if (d_fitness) cudaFree(d_fitness);
    return {};
  }

  if (cudaMemset(d_fitness, 0, sizeof(int) * programs.size()) != cudaSuccess) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_metas) cudaFree(d_metas);
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
    if (d_fitness) cudaFree(d_fitness);
    return {};
  }

  vm_multi_fitness_kernel_shared_cases<<<static_cast<unsigned int>(programs.size()), blocksize, shared_bytes>>>(
      d_consts, d_code, d_metas, d_shared_case_local_vals, d_shared_case_local_set, d_expected,
      static_cast<int>(programs.size()), fuel, d_fitness);

  const cudaError_t launch_err = cudaGetLastError();
  const cudaError_t sync_err = cudaDeviceSynchronize();
  if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_metas) cudaFree(d_metas);
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
    if (d_fitness) cudaFree(d_fitness);
    return {};
  }

  std::vector<int> host_fitness(programs.size(), 0);
  if (cudaMemcpy(host_fitness.data(), d_fitness, sizeof(int) * programs.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_metas) cudaFree(d_metas);
    if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
    if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
    if (d_expected) cudaFree(d_expected);
    if (d_fitness) cudaFree(d_fitness);
    return {};
  }

  if (d_consts) cudaFree(d_consts);
  if (d_code) cudaFree(d_code);
  if (d_metas) cudaFree(d_metas);
  if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
  if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
  if (d_expected) cudaFree(d_expected);
  if (d_fitness) cudaFree(d_fitness);
  return host_fitness;
}

}  // namespace g3pvm
