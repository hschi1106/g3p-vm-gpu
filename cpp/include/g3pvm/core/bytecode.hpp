#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "g3pvm/core/opcode.hpp"
#include "g3pvm/core/value.hpp"

namespace g3pvm {

struct Instr {
  Opcode op = Opcode::PushConst;
  int a = 0;
  int b = 0;
  bool has_a = false;
  bool has_b = false;
};

struct PhaseProgram {
  std::vector<Value> consts;
  std::vector<Instr> code;
  int n_locals = 0;
  std::unordered_map<std::string, int> var2idx;
  std::unordered_map<int, int> binder_locals;
};

struct AsgpDcSegment {
  int solve_xs_name = 0;
  int solve_n_name = 0;
  int solve_lo_name = 0;
  int divide_n_name = 0;
  int combine_left_name = 0;
  int combine_right_name = 0;
  PhaseProgram solve;
  PhaseProgram divide;
  PhaseProgram combine;
};

struct AsgpDp1dSegment {
  int lo = 0;
  int hi = 0;
  int base_state = 0;
  Value boundary_value = Value::invalid();
  int dep_kind = 0;
  std::vector<int> dep_offsets;
  int solve_state_name = 0;
  int transition_state_name = 0;
  std::vector<int> transition_dep_names;
  PhaseProgram solve;
  PhaseProgram transition;
};

struct AsgpDp2dSegment {
  int i_lo = 0;
  int i_hi = 0;
  int j_lo = 0;
  int j_hi = 0;
  int base_i = 0;
  int base_j = 0;
  Value boundary_value = Value::invalid();
  int dep_kind = 0;
  int solve_i_name = 0;
  int solve_j_name = 0;
  int transition_i_name = 0;
  int transition_j_name = 0;
  std::vector<int> transition_dep_names;
  PhaseProgram solve;
  PhaseProgram transition;
};

struct BytecodeProgram {
  std::vector<Value> consts;
  std::vector<Instr> code;
  int n_locals = 0;
  std::unordered_map<std::string, int> var2idx;
  std::vector<AsgpDcSegment> asgp_dc_segments;
  std::vector<AsgpDp1dSegment> asgp_dp1d_segments;
  std::vector<AsgpDp2dSegment> asgp_dp2d_segments;
};

}  // namespace g3pvm
