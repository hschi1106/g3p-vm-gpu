#pragma once

#include <cstdint>

#include "g3pvm/core/opcode.hpp"

namespace g3pvm::gpu_detail {

constexpr int MAX_STACK = 64;
constexpr int MAX_LOCALS = 64;
constexpr int DMAX_THREAD_PAYLOAD_ENTRIES = 32;
constexpr int DMAX_THREAD_STRING_BYTES = 512;
constexpr int DMAX_THREAD_LIST_VALUES = 128;
constexpr std::uint8_t DINSTR_HAS_A = 1;
constexpr std::uint8_t DINSTR_HAS_B = 2;

constexpr int OP_PUSH_CONST = static_cast<int>(Opcode::PushConst);
constexpr int OP_LOAD = static_cast<int>(Opcode::Load);
constexpr int OP_STORE = static_cast<int>(Opcode::Store);
constexpr int OP_NEG = static_cast<int>(Opcode::Neg);
constexpr int OP_NOT = static_cast<int>(Opcode::Not);
constexpr int OP_ADD = static_cast<int>(Opcode::Add);
constexpr int OP_SUB = static_cast<int>(Opcode::Sub);
constexpr int OP_MUL = static_cast<int>(Opcode::Mul);
constexpr int OP_DIV = static_cast<int>(Opcode::Div);
constexpr int OP_MOD = static_cast<int>(Opcode::Mod);
constexpr int OP_LT = static_cast<int>(Opcode::Lt);
constexpr int OP_LE = static_cast<int>(Opcode::Le);
constexpr int OP_GT = static_cast<int>(Opcode::Gt);
constexpr int OP_GE = static_cast<int>(Opcode::Ge);
constexpr int OP_EQ = static_cast<int>(Opcode::Eq);
constexpr int OP_NE = static_cast<int>(Opcode::Ne);
constexpr int OP_JMP = static_cast<int>(Opcode::Jmp);
constexpr int OP_JMP_IF_FALSE = static_cast<int>(Opcode::JmpIfFalse);
constexpr int OP_JMP_IF_TRUE = static_cast<int>(Opcode::JmpIfTrue);
constexpr int OP_CALL_BUILTIN = static_cast<int>(Opcode::CallBuiltin);
constexpr int OP_RETURN = static_cast<int>(Opcode::Return);

}  // namespace g3pvm::gpu_detail
