#include "g3pvm/vm_gpu.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace g3pvm {

namespace {

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
  int op = -1;
  int a = 0;
  int b = 0;
  int has_a = 0;
  int has_b = 0;
};

struct DResult {
  int is_error = 0;
  int err_code = DERR_VALUE;
  Value value = Value::none();
};

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

__device__ bool d_builtin_call(int bid, const Value* args, int argc, Value& out, DeviceErrCode& err) {
  if (bid == 0) {  // abs
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

  if (bid == 1 || bid == 2) {  // min/max
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

  if (bid == 3) {  // clip
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

__global__ void vm_kernel(const Value* consts, int n_consts, const DInstr* code, int n_code,
                          int n_locals, const int* input_idxs, const Value* input_vals,
                          int n_inputs, int fuel, DResult* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  constexpr int MAX_STACK = 1024;
  constexpr int MAX_LOCALS = 512;
  Value stack[MAX_STACK];
  int sp = 0;

  Value locals[MAX_LOCALS];
  unsigned char local_set[MAX_LOCALS];
  if (n_locals > MAX_LOCALS) {
    d_fail(*out, DERR_VALUE);
    return;
  }
  for (int i = 0; i < n_locals; ++i) {
    locals[i] = Value::none();
    local_set[i] = 0;
  }
  for (int i = 0; i < n_inputs; ++i) {
    const int idx = input_idxs[i];
    if (idx >= 0 && idx < n_locals) {
      locals[idx] = input_vals[i];
      local_set[idx] = 1;
    }
  }

  int ip = 0;
  out->is_error = 0;
  out->err_code = DERR_VALUE;
  out->value = Value::none();

  while (ip < n_code) {
    if (fuel <= 0) {
      d_fail(*out, DERR_TIMEOUT);
      return;
    }
    fuel -= 1;

    const DInstr ins = code[ip];
    ip += 1;

    if (ins.op == OP_PUSH_CONST) {
      if (!ins.has_a || ins.a < 0 || ins.a >= n_consts || sp >= MAX_STACK) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      stack[sp++] = consts[ins.a];
      continue;
    }

    if (ins.op == OP_LOAD) {
      if (!ins.has_a || ins.a < 0 || ins.a >= n_locals || sp >= MAX_STACK) {
        d_fail(*out, DERR_NAME);
        return;
      }
      if (!local_set[ins.a]) {
        d_fail(*out, DERR_NAME);
        return;
      }
      stack[sp++] = locals[ins.a];
      continue;
    }

    if (ins.op == OP_STORE) {
      if (!ins.has_a || ins.a < 0 || ins.a >= n_locals) {
        d_fail(*out, DERR_NAME);
        return;
      }
      if (sp < 1) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      locals[ins.a] = stack[--sp];
      local_set[ins.a] = 1;
      continue;
    }

    if (ins.op == OP_NEG || ins.op == OP_NOT) {
      if (sp < 1) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      Value x = stack[--sp];
      if (ins.op == OP_NEG) {
        if (!d_is_num(x)) {
          d_fail(*out, DERR_TYPE);
          return;
        }
        stack[sp++] = (x.tag == ValueTag::Float) ? Value::from_float(-x.f) : Value::from_int(-x.i);
      } else {
        if (x.tag != ValueTag::Bool) {
          d_fail(*out, DERR_TYPE);
          return;
        }
        stack[sp++] = Value::from_bool(!x.b);
      }
      continue;
    }

    if (ins.op == OP_ADD || ins.op == OP_SUB || ins.op == OP_MUL || ins.op == OP_DIV ||
        ins.op == OP_MOD) {
      if (sp < 2) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      Value b = stack[--sp];
      Value a = stack[--sp];
      double a_num = 0.0;
      double b_num = 0.0;
      bool any_float = false;
      if (!d_to_numeric_pair(a, b, a_num, b_num, any_float)) {
        d_fail(*out, DERR_TYPE);
        return;
      }
      if ((ins.op == OP_DIV || ins.op == OP_MOD) && b_num == 0.0) {
        d_fail(*out, DERR_ZERODIV);
        return;
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
        d_fail(*out, DERR_VALUE);
        return;
      }
      Value b = stack[--sp];
      Value a = stack[--sp];
      bool cmp = false;
      DeviceErrCode derr = DERR_TYPE;
      if (!d_compare(ins.op, a, b, cmp, derr)) {
        d_fail(*out, derr);
        return;
      }
      stack[sp++] = Value::from_bool(cmp);
      continue;
    }

    if (ins.op == OP_JMP) {
      if (!ins.has_a || ins.a < 0 || ins.a > n_code) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      ip = ins.a;
      continue;
    }

    if (ins.op == OP_JMP_IF_FALSE || ins.op == OP_JMP_IF_TRUE) {
      if (sp < 1) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      if (!ins.has_a || ins.a < 0 || ins.a > n_code) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      Value c = stack[--sp];
      if (c.tag != ValueTag::Bool) {
        d_fail(*out, DERR_TYPE);
        return;
      }
      if (ins.op == OP_JMP_IF_FALSE && !c.b) ip = ins.a;
      if (ins.op == OP_JMP_IF_TRUE && c.b) ip = ins.a;
      continue;
    }

    if (ins.op == OP_CALL_BUILTIN) {
      const int bid = ins.has_a ? ins.a : -1;
      const int argc = ins.has_b ? ins.b : -1;
      if (argc < 0 || sp < argc) {
        d_fail(*out, (argc < 0) ? DERR_TYPE : DERR_VALUE);
        return;
      }
      Value args_buf[4];
      if (argc > 4) {
        d_fail(*out, DERR_TYPE);
        return;
      }
      for (int i = 0; i < argc; ++i) {
        args_buf[i] = stack[sp - argc + i];
      }
      sp -= argc;
      DeviceErrCode derr = DERR_TYPE;
      Value ret = Value::none();
      if (!d_builtin_call(bid, args_buf, argc, ret, derr)) {
        d_fail(*out, derr);
        return;
      }
      if (sp >= MAX_STACK) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      stack[sp++] = ret;
      continue;
    }

    if (ins.op == OP_RETURN) {
      if (sp < 1) {
        d_fail(*out, DERR_VALUE);
        return;
      }
      out->is_error = 0;
      out->value = stack[sp - 1];
      return;
    }

    d_fail(*out, DERR_TYPE);
    return;
  }

  d_fail(*out, DERR_VALUE);
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

}  // namespace

VMResult run_bytecode_gpu(const BytecodeProgram& program, const std::vector<std::pair<int, Value>>& inputs,
                          int fuel) {
  if (program.n_locals > 512) {
    return VMResult{true, Value::none(), Err{ErrCode::Value, "gpu locals capacity exceeded"}};
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return VMResult{true, Value::none(), Err{ErrCode::Value, "cuda device unavailable"}};
  }

  std::vector<DInstr> code;
  code.reserve(program.code.size());
  for (const Instr& ins : program.code) {
    const int op = host_opcode(ins.op);
    if (op < 0) {
      return VMResult{true, Value::none(), Err{ErrCode::Type, "unknown opcode"}};
    }
    DInstr di;
    di.op = op;
    di.a = ins.a;
    di.b = ins.b;
    di.has_a = ins.has_a ? 1 : 0;
    di.has_b = ins.has_b ? 1 : 0;
    code.push_back(di);
  }

  std::vector<int> input_idxs;
  std::vector<Value> input_vals;
  input_idxs.reserve(inputs.size());
  input_vals.reserve(inputs.size());
  for (const auto& iv : inputs) {
    input_idxs.push_back(iv.first);
    input_vals.push_back(iv.second);
  }

  Value* d_consts = nullptr;
  DInstr* d_code = nullptr;
  int* d_input_idxs = nullptr;
  Value* d_input_vals = nullptr;
  DResult* d_out = nullptr;

  if (!cuda_alloc_and_copy_in(program.consts, &d_consts) || !cuda_alloc_and_copy_in(code, &d_code) ||
      !cuda_alloc_and_copy_in(input_idxs, &d_input_idxs) ||
      !cuda_alloc_and_copy_in(input_vals, &d_input_vals)) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_input_idxs) cudaFree(d_input_idxs);
    if (d_input_vals) cudaFree(d_input_vals);
    return VMResult{true, Value::none(), Err{ErrCode::Value, "cuda allocation failure"}};
  }
  if (cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(DResult)) != cudaSuccess) {
    if (d_consts) cudaFree(d_consts);
    if (d_code) cudaFree(d_code);
    if (d_input_idxs) cudaFree(d_input_idxs);
    if (d_input_vals) cudaFree(d_input_vals);
    return VMResult{true, Value::none(), Err{ErrCode::Value, "cuda allocation failure"}};
  }

  vm_kernel<<<1, 1>>>(d_consts, static_cast<int>(program.consts.size()), d_code,
                      static_cast<int>(code.size()), program.n_locals, d_input_idxs, d_input_vals,
                      static_cast<int>(inputs.size()), fuel, d_out);

  cudaError_t launch_err = cudaGetLastError();
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
    cudaFree(d_consts);
    cudaFree(d_code);
    cudaFree(d_input_idxs);
    cudaFree(d_input_vals);
    cudaFree(d_out);
    std::string msg = "cuda kernel execution failure";
    if (launch_err != cudaSuccess) {
      msg += " launch=";
      msg += cudaGetErrorString(launch_err);
    }
    if (sync_err != cudaSuccess) {
      msg += " sync=";
      msg += cudaGetErrorString(sync_err);
    }
    return VMResult{true, Value::none(), Err{ErrCode::Value, msg}};
  }

  DResult host_out;
  if (cudaMemcpy(&host_out, d_out, sizeof(DResult), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_consts);
    cudaFree(d_code);
    cudaFree(d_input_idxs);
    cudaFree(d_input_vals);
    cudaFree(d_out);
    return VMResult{true, Value::none(), Err{ErrCode::Value, "cuda copy-back failure"}};
  }

  cudaFree(d_consts);
  cudaFree(d_code);
  cudaFree(d_input_idxs);
  cudaFree(d_input_vals);
  cudaFree(d_out);

  if (host_out.is_error) {
    return VMResult{true, Value::none(), Err{from_device_err(host_out.err_code), "gpu vm error"}};
  }
  return VMResult{false, host_out.value, Err{ErrCode::Value, ""}};
}

}  // namespace g3pvm
