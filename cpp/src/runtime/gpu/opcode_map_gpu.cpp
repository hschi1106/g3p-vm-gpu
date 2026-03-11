#include "opcode_map_gpu.hpp"

#include "g3pvm/runtime/gpu/constants_gpu.hpp"

namespace g3pvm::gpu_detail {

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

}  // namespace g3pvm::gpu_detail
