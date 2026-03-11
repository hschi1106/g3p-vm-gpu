#pragma once

#include <cstdint>

namespace g3pvm::gpu_detail {

constexpr int MAX_STACK = 64;
constexpr int MAX_LOCALS = 64;
constexpr int DMAX_THREAD_PAYLOAD_ENTRIES = 32;
constexpr int DMAX_THREAD_STRING_BYTES = 512;
constexpr int DMAX_THREAD_LIST_VALUES = 128;
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

enum DeviceBuiltinId : int {
  DBUILTIN_ABS = 0,
  DBUILTIN_MIN = 1,
  DBUILTIN_MAX = 2,
  DBUILTIN_CLAMP = 3,
  DBUILTIN_LEN = 4,
  DBUILTIN_CONCAT = 5,
  DBUILTIN_SLICE = 6,
  DBUILTIN_INDEX = 7,
};

}  // namespace g3pvm::gpu_detail
