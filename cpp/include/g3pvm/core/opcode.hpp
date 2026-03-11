#pragma once

#include <cstdint>
#include <string_view>

namespace g3pvm {

enum class Opcode : std::uint8_t {
  PushConst = 0,
  Load = 1,
  Store = 2,
  Neg = 3,
  Not = 4,
  Add = 5,
  Sub = 6,
  Mul = 7,
  Div = 8,
  Mod = 9,
  Lt = 10,
  Le = 11,
  Gt = 12,
  Ge = 13,
  Eq = 14,
  Ne = 15,
  Jmp = 16,
  JmpIfFalse = 17,
  JmpIfTrue = 18,
  CallBuiltin = 19,
  Return = 20,
};

inline const char* opcode_name(Opcode op) {
  switch (op) {
    case Opcode::PushConst:
      return "PUSH_CONST";
    case Opcode::Load:
      return "LOAD";
    case Opcode::Store:
      return "STORE";
    case Opcode::Neg:
      return "NEG";
    case Opcode::Not:
      return "NOT";
    case Opcode::Add:
      return "ADD";
    case Opcode::Sub:
      return "SUB";
    case Opcode::Mul:
      return "MUL";
    case Opcode::Div:
      return "DIV";
    case Opcode::Mod:
      return "MOD";
    case Opcode::Lt:
      return "LT";
    case Opcode::Le:
      return "LE";
    case Opcode::Gt:
      return "GT";
    case Opcode::Ge:
      return "GE";
    case Opcode::Eq:
      return "EQ";
    case Opcode::Ne:
      return "NE";
    case Opcode::Jmp:
      return "JMP";
    case Opcode::JmpIfFalse:
      return "JMP_IF_FALSE";
    case Opcode::JmpIfTrue:
      return "JMP_IF_TRUE";
    case Opcode::CallBuiltin:
      return "CALL_BUILTIN";
    case Opcode::Return:
      return "RETURN";
  }
  return "PUSH_CONST";
}

inline bool opcode_from_name(std::string_view name, Opcode& out) {
  if (name == "PUSH_CONST") out = Opcode::PushConst;
  else if (name == "LOAD") out = Opcode::Load;
  else if (name == "STORE") out = Opcode::Store;
  else if (name == "NEG") out = Opcode::Neg;
  else if (name == "NOT") out = Opcode::Not;
  else if (name == "ADD") out = Opcode::Add;
  else if (name == "SUB") out = Opcode::Sub;
  else if (name == "MUL") out = Opcode::Mul;
  else if (name == "DIV") out = Opcode::Div;
  else if (name == "MOD") out = Opcode::Mod;
  else if (name == "LT") out = Opcode::Lt;
  else if (name == "LE") out = Opcode::Le;
  else if (name == "GT") out = Opcode::Gt;
  else if (name == "GE") out = Opcode::Ge;
  else if (name == "EQ") out = Opcode::Eq;
  else if (name == "NE") out = Opcode::Ne;
  else if (name == "JMP") out = Opcode::Jmp;
  else if (name == "JMP_IF_FALSE") out = Opcode::JmpIfFalse;
  else if (name == "JMP_IF_TRUE") out = Opcode::JmpIfTrue;
  else if (name == "CALL_BUILTIN") out = Opcode::CallBuiltin;
  else if (name == "RETURN") out = Opcode::Return;
  else return false;
  return true;
}

}  // namespace g3pvm
