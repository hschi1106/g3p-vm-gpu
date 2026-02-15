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

bool should_skip_gpu(const std::vector<VMResult>& out) {
  if (out.empty()) return false;
  if (!out[0].is_error) return false;
  std::cout << "g3pvm_test_vm_gpu_batch: SKIP (" << out[0].err.message << ")\n";
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

bool test_batch_stride_mapping() {
  BytecodeProgram p = make_add_one_program();

  std::vector<InputCase> cases;
  cases.reserve(600);
  for (int i = 0; i < 600; ++i) {
    InputCase one;
    one.push_back(LocalBinding{0, Value::from_int(i)});
    cases.push_back(one);
  }

  const std::vector<VMResult> out = g3pvm::run_bytecode_gpu_batch(p, cases, 1000, 256);
  if (!check(out.size() == cases.size(), "stride batch output size mismatch")) return false;

  for (int i = 0; i < static_cast<int>(out.size()); ++i) {
    if (!check(!out[i].is_error, "stride case should succeed")) return false;
    if (!check(out[i].value.tag == ValueTag::Int, "stride case result should be int")) return false;
    if (!check(out[i].value.i == i + 1, "stride case value mismatch")) return false;
  }
  return true;
}

bool test_batch_mixed_results() {
  BytecodeProgram p = make_add_one_program();

  std::vector<InputCase> cases(3);
  cases[0].push_back(LocalBinding{0, Value::from_int(2)});
  // cases[1] keeps local unset -> NameError
  cases[2].push_back(LocalBinding{0, Value::from_int(7)});

  const std::vector<VMResult> out = g3pvm::run_bytecode_gpu_batch(p, cases, 1000, 256);
  if (!check(out.size() == 3, "mixed batch output size mismatch")) return false;

  if (!check(!out[0].is_error, "case 0 should succeed")) return false;
  if (!check(out[0].value.tag == ValueTag::Int && out[0].value.i == 3, "case 0 value mismatch")) {
    return false;
  }

  if (!check(out[1].is_error, "case 1 should fail")) return false;
  if (!check(out[1].err.code == ErrCode::Name, "case 1 should be NameError")) return false;

  if (!check(!out[2].is_error, "case 2 should succeed")) return false;
  if (!check(out[2].value.tag == ValueTag::Int && out[2].value.i == 8, "case 2 value mismatch")) {
    return false;
  }
  return true;
}

bool test_invalid_blocksize() {
  BytecodeProgram p = make_add_one_program();
  std::vector<InputCase> cases(1);
  cases[0].push_back(LocalBinding{0, Value::from_int(1)});

  const std::vector<VMResult> out = g3pvm::run_bytecode_gpu_batch(p, cases, 1000, 0);
  if (!check(out.size() == 1, "invalid blocksize output size mismatch")) return false;
  if (!check(out[0].is_error, "invalid blocksize should fail")) return false;
  if (!check(out[0].err.code == ErrCode::Value, "invalid blocksize should return ValueError")) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  BytecodeProgram smoke = make_add_one_program();
  std::vector<InputCase> smoke_cases(1);
  smoke_cases[0].push_back(LocalBinding{0, Value::from_int(4)});
  std::vector<VMResult> smoke_out = g3pvm::run_bytecode_gpu_batch(smoke, smoke_cases, 100, 1);
  if (should_skip_gpu(smoke_out)) {
    return 0;
  }

  if (!test_batch_stride_mapping()) return 1;
  if (!test_batch_mixed_results()) return 1;
  if (!test_invalid_blocksize()) return 1;

  std::cout << "g3pvm_test_vm_gpu_batch: OK\n";
  return 0;
}
