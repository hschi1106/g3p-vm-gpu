#include <chrono>
#include <cstdlib>
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
using g3pvm::VMResult;

g3pvm::Instr ins(const std::string& op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(const std::string& op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }

BytecodeProgram make_pass_program() {
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

BytecodeProgram make_fail_program() {
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

int parse_int(const char* s, int fallback) {
  if (s == nullptr) return fallback;
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0') return fallback;
  return static_cast<int>(v);
}

}  // namespace

int main(int argc, char** argv) {
  const int program_count = parse_int((argc > 1) ? argv[1] : nullptr, 4096);
  const int cases_per_program = parse_int((argc > 2) ? argv[2] : nullptr, 1024);
  const int pass_programs = parse_int((argc > 3) ? argv[3] : nullptr, 2048);
  const int fail_programs = parse_int((argc > 4) ? argv[4] : nullptr, 1024);
  const int timeout_programs = parse_int((argc > 5) ? argv[5] : nullptr, 1024);
  const int fuel = parse_int((argc > 6) ? argv[6] : nullptr, 64);
  const int blocksize = parse_int((argc > 7) ? argv[7] : nullptr, 256);

  if (program_count <= 0 || cases_per_program <= 0 || pass_programs < 0 || fail_programs < 0 ||
      timeout_programs < 0 || fuel <= 0 || blocksize <= 0) {
    std::cerr << "invalid arguments\n";
    return 2;
  }
  if (pass_programs + fail_programs + timeout_programs != program_count) {
    std::cerr << "bucket counts must sum to program_count\n";
    return 2;
  }

  std::vector<BytecodeProgram> programs;
  programs.reserve(static_cast<std::size_t>(program_count));

  const BytecodeProgram pass_p = make_pass_program();
  const BytecodeProgram fail_p = make_fail_program();
  const BytecodeProgram timeout_p = make_timeout_program();

  for (int i = 0; i < pass_programs; ++i) programs.push_back(pass_p);
  for (int i = 0; i < fail_programs; ++i) programs.push_back(fail_p);
  for (int i = 0; i < timeout_programs; ++i) programs.push_back(timeout_p);

  std::vector<std::vector<InputCase>> cases_by_program;
  cases_by_program.resize(static_cast<std::size_t>(program_count));

  for (int pi = 0; pi < program_count; ++pi) {
    auto& v = cases_by_program[static_cast<std::size_t>(pi)];
    v.resize(static_cast<std::size_t>(cases_per_program));
    if (pi < pass_programs) {
      for (int ci = 0; ci < cases_per_program; ++ci) {
        v[static_cast<std::size_t>(ci)].push_back(LocalBinding{0, Value::from_int(ci)});
      }
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<VMResult>> out =
      g3pvm::run_bytecode_gpu_multi_batch(programs, cases_by_program, fuel, blocksize);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  if (out.size() == 1 && out[0].size() == 1 && out[0][0].is_error &&
      out[0][0].err.message.find("cuda device unavailable") != std::string::npos) {
    std::cout << "SKIP " << out[0][0].err.message << "\n";
    return 0;
  }

  long long ret_ok = 0;
  long long err_type = 0;
  long long err_timeout = 0;
  long long err_other = 0;
  long long total = 0;

  for (const auto& po : out) {
    for (const auto& r : po) {
      total += 1;
      if (!r.is_error) {
        ret_ok += 1;
      } else if (r.err.code == ErrCode::Type) {
        err_type += 1;
      } else if (r.err.code == ErrCode::Timeout) {
        err_timeout += 1;
      } else {
        err_other += 1;
      }
    }
  }

  const long long expected_ok = static_cast<long long>(pass_programs) * cases_per_program;
  const long long expected_type = static_cast<long long>(fail_programs) * cases_per_program;
  const long long expected_timeout = static_cast<long long>(timeout_programs) * cases_per_program;

  std::cout << "OK programs=" << program_count << " cases_per_program=" << cases_per_program
            << " total_cases=" << total << " fuel=" << fuel << " blocksize=" << blocksize
            << " elapsed_ms=" << ms << "\n";
  std::cout << "RESULT return=" << ret_ok << " type_err=" << err_type << " timeout_err=" << err_timeout
            << " other_err=" << err_other << "\n";
  std::cout << "EXPECT return=" << expected_ok << " type_err=" << expected_type
            << " timeout_err=" << expected_timeout << "\n";

  if (ret_ok != expected_ok || err_type != expected_type || err_timeout != expected_timeout || err_other != 0) {
    std::cerr << "mismatch against expected distribution\n";
    return 1;
  }

  return 0;
}
