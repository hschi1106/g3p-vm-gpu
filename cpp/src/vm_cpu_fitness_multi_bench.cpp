#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::InputCase;
using g3pvm::LocalBinding;
using g3pvm::Value;

g3pvm::Instr ins(const std::string& op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(const std::string& op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }

BytecodeProgram make_pass_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(1)};
  p.code = {ins_a("LOAD", 0), ins_a("PUSH_CONST", 0), ins("ADD"), ins("RETURN")};
  return p;
}

BytecodeProgram make_fail_program() {
  BytecodeProgram p;
  p.consts = {Value::from_bool(true)};
  p.code = {ins_a("PUSH_CONST", 0), ins("NEG")};
  return p;
}

BytecodeProgram make_timeout_program() {
  BytecodeProgram p;
  p.code = {ins_a("JMP", 0)};
  return p;
}

int parse_int(const char* s, int fallback) {
  if (!s) return fallback;
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

  if (program_count <= 0 || cases_per_program <= 0 || pass_programs < 0 || fail_programs < 0 ||
      timeout_programs < 0 || fuel <= 0) {
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

  std::vector<InputCase> shared_cases;
  shared_cases.reserve(static_cast<std::size_t>(cases_per_program));
  std::vector<Value> shared_answer;
  shared_answer.reserve(static_cast<std::size_t>(cases_per_program));
  for (int ci = 0; ci < cases_per_program; ++ci) {
    shared_cases.push_back(InputCase{LocalBinding{0, Value::from_int(ci)}});
    // Shared oracle targets pass-program behavior: LOAD(0)+1.
    shared_answer.push_back(Value::from_int(ci + 1));
  }

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<int> fitness =
      g3pvm::run_bytecode_cpu_multi_fitness_shared_cases(programs, shared_cases, shared_answer, fuel);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  if (fitness.size() != programs.size()) {
    std::cerr << "fitness size mismatch\n";
    return 1;
  }

  long long ok_programs = 0;
  long long neg_programs = 0;
  for (std::size_t i = 0; i < fitness.size(); ++i) {
    const int f = fitness[i];
    if (i < static_cast<std::size_t>(pass_programs)) {
      if (f != cases_per_program) {
        std::cerr << "unexpected pass fitness\n";
        return 1;
      }
      ok_programs += 1;
    } else {
      if (f != -10 * cases_per_program) {
        std::cerr << "unexpected error fitness\n";
        return 1;
      }
      neg_programs += 1;
    }
  }

  std::cout << "OK programs=" << program_count << " cases_per_program=" << cases_per_program
            << " fuel=" << fuel << " elapsed_ms=" << ms << " pass_fit=" << ok_programs
            << " neg_fit=" << neg_programs << "\n";
  return 0;
}
