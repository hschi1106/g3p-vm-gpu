#pragma once

#include <cstdint>

#include "builtins_device.cuh"
#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/runtime/gpu/device_types_gpu.hpp"

namespace g3pvm::gpu_detail {

__device__ inline void d_fail(DResult& out, DeviceErrCode code) {
  out.is_error = 1;
  out.err_code = static_cast<int>(code);
}

__device__ inline DResult execute_bytecode_device(const DProgramMeta& meta,
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

}  // namespace g3pvm::gpu_detail
