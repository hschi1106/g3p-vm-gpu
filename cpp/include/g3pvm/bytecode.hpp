#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/value.hpp"

namespace g3pvm {

struct Instr {
  std::string op;
  int a = 0;
  int b = 0;
  bool has_a = false;
  bool has_b = false;
};

struct BytecodeProgram {
  std::vector<Value> consts;
  std::vector<Instr> code;
  int n_locals = 0;
  std::unordered_map<std::string, int> var2idx;
};

}  // namespace g3pvm
