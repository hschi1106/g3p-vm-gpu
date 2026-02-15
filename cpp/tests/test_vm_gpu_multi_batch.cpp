#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::ErrCode;
using g3pvm::InputCase;
using g3pvm::LocalBinding;
using g3pvm::Value;
using g3pvm::ValueTag;
using g3pvm::VMResult;

g3pvm::Instr ins(const std::string& op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(const std::string& op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool should_skip_gpu(const std::vector<std::vector<VMResult>>& out) {
  if (out.size() != 1 || out[0].size() != 1) return false;
  if (!out[0][0].is_error) return false;
  if (out[0][0].err.message.find("cuda device unavailable") == std::string::npos) return false;
  std::cout << "g3pvm_test_vm_gpu_multi_batch: SKIP (" << out[0][0].err.message << ")\n";
  return true;
}

BytecodeProgram make_add_one_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(1)};
  p.code = {
      ins_a("LOAD", 0),
      ins_a("PUSH_CONST", 0),
      ins("ADD"),
      ins("RETURN"),
  };
  return p;
}

BytecodeProgram make_return_const_program(int v) {
  BytecodeProgram p;
  p.consts = {Value::from_int(v)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins("RETURN"),
  };
  return p;
}

BytecodeProgram make_type_error_program() {
  BytecodeProgram p;
  p.consts = {Value::from_bool(true)};
  p.code = {
      ins_a("PUSH_CONST", 0),
      ins("NEG"),
  };
  return p;
}

BytecodeProgram make_timeout_program() {
  BytecodeProgram p;
  p.code = {
      ins_a("JMP", 0),
  };
  return p;
}

bool test_multi_stride() {
  std::vector<BytecodeProgram> programs;
  programs.push_back(make_add_one_program());
  programs.push_back(make_return_const_program(7));

  std::vector<std::vector<InputCase>> cases_by_program(2);
  cases_by_program[0].reserve(600);
  for (int i = 0; i < 600; ++i) {
    InputCase one;
    one.push_back(LocalBinding{0, Value::from_int(i)});
    cases_by_program[0].push_back(one);
  }
  cases_by_program[1].assign(300, InputCase{});

  std::vector<std::vector<VMResult>> out =
      g3pvm::run_bytecode_gpu_multi_batch(programs, cases_by_program, 1000, 256);

  if (!check(out.size() == 2, "multi stride program count mismatch")) return false;
  if (!check(out[0].size() == 600, "multi stride program 0 case count mismatch")) return false;
  if (!check(out[1].size() == 300, "multi stride program 1 case count mismatch")) return false;

  for (int i = 0; i < 600; ++i) {
    if (!check(!out[0][i].is_error, "multi stride p0 case should succeed")) return false;
    if (!check(out[0][i].value.tag == ValueTag::Int, "multi stride p0 should return int")) return false;
    if (!check(out[0][i].value.i == i + 1, "multi stride p0 value mismatch")) return false;
  }

  for (int i = 0; i < 300; ++i) {
    if (!check(!out[1][i].is_error, "multi stride p1 case should succeed")) return false;
    if (!check(out[1][i].value.tag == ValueTag::Int, "multi stride p1 should return int")) return false;
    if (!check(out[1][i].value.i == 7, "multi stride p1 value mismatch")) return false;
  }

  return true;
}

bool test_multi_mixed_outcomes() {
  std::vector<BytecodeProgram> programs;
  programs.push_back(make_add_one_program());
  programs.push_back(make_type_error_program());
  programs.push_back(make_timeout_program());

  std::vector<std::vector<InputCase>> cases_by_program(3);
  cases_by_program[0] = {
      InputCase{LocalBinding{0, Value::from_int(10)}},
      InputCase{LocalBinding{0, Value::from_int(20)}},
  };
  cases_by_program[1] = {InputCase{}, InputCase{}};
  cases_by_program[2] = {InputCase{}, InputCase{}};

  std::vector<std::vector<VMResult>> out =
      g3pvm::run_bytecode_gpu_multi_batch(programs, cases_by_program, 5, 128);

  if (!check(out.size() == 3, "mixed outcomes program count mismatch")) return false;

  if (!check(!out[0][0].is_error && out[0][0].value.i == 11, "mixed p0 case0 mismatch")) return false;
  if (!check(!out[0][1].is_error && out[0][1].value.i == 21, "mixed p0 case1 mismatch")) return false;

  if (!check(out[1][0].is_error && out[1][0].err.code == ErrCode::Type, "mixed p1 case0 should TypeError")) {
    return false;
  }
  if (!check(out[1][1].is_error && out[1][1].err.code == ErrCode::Type, "mixed p1 case1 should TypeError")) {
    return false;
  }

  if (!check(out[2][0].is_error && out[2][0].err.code == ErrCode::Timeout,
             "mixed p2 case0 should Timeout")) {
    return false;
  }
  if (!check(out[2][1].is_error && out[2][1].err.code == ErrCode::Timeout,
             "mixed p2 case1 should Timeout")) {
    return false;
  }

  return true;
}

bool test_invalid_program_isolation() {
  std::vector<BytecodeProgram> programs;
  programs.push_back(make_add_one_program());

  BytecodeProgram bad = make_return_const_program(3);
  bad.n_locals = 1000;
  programs.push_back(bad);

  std::vector<std::vector<InputCase>> cases_by_program(2);
  cases_by_program[0] = {InputCase{LocalBinding{0, Value::from_int(41)}}};
  cases_by_program[1] = {InputCase{}};

  std::vector<std::vector<VMResult>> out =
      g3pvm::run_bytecode_gpu_multi_batch(programs, cases_by_program, 100, 64);

  if (!check(out.size() == 2, "invalid isolation program count mismatch")) return false;
  if (!check(out[0].size() == 1 && out[1].size() == 1, "invalid isolation case count mismatch")) return false;

  if (!check(!out[0][0].is_error, "valid program should still succeed")) return false;
  if (!check(out[0][0].value.tag == ValueTag::Int && out[0][0].value.i == 42,
             "valid program returned wrong value")) {
    return false;
  }

  if (!check(out[1][0].is_error, "invalid program should return error")) return false;
  if (!check(out[1][0].err.code == ErrCode::Value, "invalid program should be ValueError")) return false;

  return true;
}

}  // namespace

int main() {
  std::vector<BytecodeProgram> probe_programs = {make_return_const_program(1)};
  std::vector<std::vector<InputCase>> probe_cases = {{InputCase{}}};
  std::vector<std::vector<VMResult>> probe_out =
      g3pvm::run_bytecode_gpu_multi_batch(probe_programs, probe_cases, 10, 1);
  if (should_skip_gpu(probe_out)) {
    return 0;
  }

  if (!test_multi_stride()) return 1;
  if (!test_multi_mixed_outcomes()) return 1;
  if (!test_invalid_program_isolation()) return 1;

  std::cout << "g3pvm_test_vm_gpu_multi_batch: OK\n";
  return 0;
}
