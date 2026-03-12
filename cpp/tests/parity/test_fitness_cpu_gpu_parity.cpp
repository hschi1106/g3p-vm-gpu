#include <climits>
#include <cmath>
#include <iostream>
#include <vector>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::CaseBindings;
using g3pvm::InputBinding;
using g3pvm::Opcode;
using g3pvm::Value;

g3pvm::Instr ins(Opcode op) { return g3pvm::Instr{op, 0, 0, false, false}; }
g3pvm::Instr ins_a(Opcode op, int a) { return g3pvm::Instr{op, a, 0, true, false}; }
g3pvm::Instr ins_ab(Opcode op, int a, int b) { return g3pvm::Instr{op, a, b, true, true}; }

BytecodeProgram make_add_one_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(1)};
  p.code = {ins_a(Opcode::Load, 0), ins_a(Opcode::PushConst, 0), ins(Opcode::Add), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_type_error_program() {
  BytecodeProgram p;
  p.consts = {Value::from_bool(true)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Neg)};
  return p;
}

BytecodeProgram make_timeout_program() {
  BytecodeProgram p;
  p.code = {ins_a(Opcode::Jmp, 0)};
  return p;
}

BytecodeProgram make_return_const_program(int v) {
  BytecodeProgram p;
  p.consts = {Value::from_int(v)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_return_string_program() {
  BytecodeProgram p;
  p.consts = {Value::from_string_hash_len(0x1234ULL, 3)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_wrap_add_program() {
  BytecodeProgram p;
  p.consts = {Value::from_int(LLONG_MAX), Value::from_int(1)};
  p.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1), ins(Opcode::Add), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_float_mod_div_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_float(2.207), Value::from_int(3)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Mod),
      ins(Opcode::Div),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_nested_exact_string_payload_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {
      g3pvm::payload::make_string_value("wepw"),
      Value::from_int(-4),
      Value::from_int(-2),
      Value::from_int(1),
      Value::from_int(-4),
      g3pvm::payload::make_string_value("vww"),
      g3pvm::payload::make_string_value("cnwl"),
  };
  p.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Gt),
      ins_a(Opcode::JmpIfFalse, 6),
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::Jmp, 7),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Neg),
      ins(Opcode::Neg),
      ins(Opcode::Neg),
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::PushConst, 2),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Slice), 3),
      ins_a(Opcode::PushConst, 3),
      ins_a(Opcode::PushConst, 4),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Slice), 3),
      ins_a(Opcode::PushConst, 5),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Concat), 2),
      ins_a(Opcode::PushConst, 6),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Concat), 2),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Len), 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(g3pvm::BuiltinId::Min), 2),
      ins(Opcode::Return),
  };
  return p;
}

bool approx(double a, double b) {
  return std::fabs(a - b) <= 1e-9;
}

bool exact(double a, double b) {
  return a == b;
}

g3pvm::FitnessEvalResult eval_gpu_via_session(const std::vector<BytecodeProgram>& programs,
                                              const std::vector<CaseBindings>& shared_cases,
                                              const std::vector<Value>& shared_answer,
                                              int fuel,
                                              int blocksize,
                                              double penalty) {
  g3pvm::FitnessSessionGpu session;
  g3pvm::FitnessEvalResult init = session.init(shared_cases, shared_answer, fuel, blocksize, penalty);
  if (!init.ok) {
    return init;
  }
  return session.eval_programs(programs);
}

}  // namespace

int main() {
  const double penalty = 1.0;
  constexpr int kParityBlocksize = 128;

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_add_one_program());
    programs.push_back(make_return_const_program(7));
    programs.push_back(make_return_string_program());
    programs.push_back(make_type_error_program());
    programs.push_back(make_timeout_program());

    std::vector<CaseBindings> shared_cases;
    std::vector<Value> shared_answer;
    for (int i = 0; i < 64; ++i) {
      shared_cases.push_back(CaseBindings{InputBinding{0, Value::from_int(i)}});
      shared_answer.push_back(Value::from_int(i + 1));
    }

    const std::vector<double> cpu_fit =
        g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const g3pvm::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on numeric cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on numeric cases at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: exact numeric program should score 0 MAE\n";
      return 1;
    }
    if (!approx(cpu_fit[1], -1674.0)) {
      std::cerr << "FAIL: constant numeric program should accumulate negative MAE\n";
      return 1;
    }
    if (!approx(cpu_fit[2], -64.0 * penalty)) {
      std::cerr << "FAIL: non-numeric actual against numeric expected should accumulate penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[3], -64.0 * penalty) || !approx(cpu_fit[4], -64.0 * penalty)) {
      std::cerr << "FAIL: runtime errors on numeric cases should accumulate penalty\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_return_string_program());
    programs.push_back(make_return_const_program(7));
    programs.push_back(make_type_error_program());

    std::vector<CaseBindings> shared_cases(16);
    std::vector<Value> shared_answer(16, Value::from_string_hash_len(0x9999ULL, 3));

    const std::vector<double> cpu_fit =
        g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const g3pvm::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on binary cases: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on binary cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on binary cases at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: same-tag binary mismatch should score 0\n";
      return 1;
    }
    if (!approx(cpu_fit[1], -16.0 * penalty)) {
      std::cerr << "FAIL: binary type mismatch should accumulate penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[2], -16.0 * penalty)) {
      std::cerr << "FAIL: runtime errors on binary cases should accumulate penalty\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_wrap_add_program());

    std::vector<CaseBindings> shared_cases(4);
    std::vector<Value> shared_answer(4, Value::from_int(LLONG_MIN));

    const std::vector<double> cpu_fit =
        g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const g3pvm::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on wrap cases: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on wrap cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on wrap cases at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: wrap add should match expected wrapped result\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_float_mod_div_program());

    std::vector<CaseBindings> shared_cases{
        CaseBindings{InputBinding{0, Value::from_float(-0.008797653959)}},
    };
    std::vector<Value> shared_answer{Value::from_float(0.99124093216)};

    const std::vector<double> cpu_fit =
        g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const g3pvm::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on float mod/div case: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on float mod/div case\n";
      return 1;
    }
    if (cpu_fit[0] != gpu_fit.fitness[0]) {
      std::cerr << "FAIL: cpu/gpu fitness mismatch on float mod/div case\n";
      return 1;
    }
  }

  {
    g3pvm::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_nested_exact_string_payload_program());
    programs.push_back(make_return_const_program(7));

    std::vector<CaseBindings> shared_cases;
    std::vector<Value> shared_answer;
    shared_cases.reserve(1024);
    shared_answer.reserve(1024);
    for (int i = 0; i < 1024; ++i) {
      const long long x = static_cast<long long>(i) - 512LL;
      shared_cases.push_back(CaseBindings{InputBinding{0, Value::from_int(x)}});
      shared_answer.push_back(Value::from_int(x + 1));
    }

    const std::vector<double> cpu_fit =
        g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, 256, penalty, 256);
    const g3pvm::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 256, 256, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on nested exact payload case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on nested exact payload case\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on nested exact payload case at " << i << "\n";
        return 1;
      }
    }
  }

  std::cout << "g3pvm_test_fitness_cpu_gpu_parity: OK\n";
  return 0;
}
