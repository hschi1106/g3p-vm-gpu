#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/runtime/cpu/builtins_cpu.hpp"
#include "g3pvm/core/builtin.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"

// Keep this file directly buildable without adding new library targets.
#include "../../src/cli/json.cpp"
#include "../../src/cli/codec.cpp"

namespace {

using g3pvm::Value;
using g3pvm::cli_detail::JsonParser;
using g3pvm::cli_detail::JsonValue;
using g3pvm::evo::EvolutionConfig;
using g3pvm::evo::EvalCase;
using g3pvm::evo::ProgramGenome;
using g3pvm::evo::ScoredGenome;

std::string value_debug_string(const Value& v) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  if (v.tag == g3pvm::ValueTag::Int) {
    oss << "int(" << v.i << ")";
  } else if (v.tag == g3pvm::ValueTag::Float) {
    oss << "float(" << v.f << ")";
  } else if (v.tag == g3pvm::ValueTag::Bool) {
    oss << "bool(" << (v.b ? "true" : "false") << ")";
  } else if (v.tag == g3pvm::ValueTag::None) {
    oss << "none";
  } else if (v.tag == g3pvm::ValueTag::String) {
    oss << "string(hash=" << Value::container_hash48(v) << ",len=" << Value::container_len(v) << ")";
  } else if (v.tag == g3pvm::ValueTag::List) {
    oss << "list(hash=" << Value::container_hash48(v) << ",len=" << Value::container_len(v) << ")";
  }
  return oss.str();
}

void dump_consts(const g3pvm::BytecodeProgram& bc) {
  std::cout << "consts:\n";
  for (std::size_t i = 0; i < bc.consts.size(); ++i) {
    std::cout << "const[" << i << "]=" << value_debug_string(bc.consts[i]) << "\n";
  }
}

void trace_cpu_case(const g3pvm::BytecodeProgram& program,
                    const std::vector<std::pair<int, Value>>& inputs,
                    int fuel) {
  struct LocalSlot {
    bool is_set = false;
    Value value = Value::none();
  };

  std::vector<Value> stack;
  std::vector<LocalSlot> locals(static_cast<std::size_t>(program.n_locals));
  for (const auto& input : inputs) {
    locals[static_cast<std::size_t>(input.first)].is_set = true;
    locals[static_cast<std::size_t>(input.first)].value = input.second;
  }

  int ip = 0;
  int step = 0;
  while (ip < static_cast<int>(program.code.size())) {
    if (fuel <= 0) {
      std::cout << "trace timeout\n";
      return;
    }
    fuel -= 1;
    const g3pvm::Instr& ins = program.code[static_cast<std::size_t>(ip)];
    std::cout << "trace step=" << step++ << " ip=" << ip << " op=" << g3pvm::opcode_name(ins.op);
    if (ins.has_a) {
      std::cout << " a=" << ins.a;
    }
    if (ins.has_b) {
      std::cout << " b=" << ins.b;
    }
    std::cout << "\n";
    ip += 1;

    auto print_stack = [&]() {
      std::cout << "  stack:";
      for (const Value& value : stack) {
        std::cout << " " << value_debug_string(value);
      }
      std::cout << "\n";
    };

    if (ins.op == g3pvm::Opcode::PushConst) {
      stack.push_back(program.consts[static_cast<std::size_t>(ins.a)]);
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Load) {
      stack.push_back(locals[static_cast<std::size_t>(ins.a)].value);
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Store) {
      locals[static_cast<std::size_t>(ins.a)].is_set = true;
      locals[static_cast<std::size_t>(ins.a)].value = stack.back();
      stack.pop_back();
      std::cout << "  store local[" << ins.a << "]="
                << value_debug_string(locals[static_cast<std::size_t>(ins.a)].value) << "\n";
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Neg) {
      const Value x = stack.back();
      stack.pop_back();
      if (x.tag == g3pvm::ValueTag::Float) {
        stack.push_back(Value::from_float(g3pvm::vm_semantics::canonicalize_vm_float(-x.f)));
      } else {
        stack.push_back(Value::from_int(g3pvm::vm_semantics::wrap_int_neg(x.i)));
      }
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Not) {
      const Value x = stack.back();
      stack.pop_back();
      stack.push_back(Value::from_bool(!x.b));
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Add || ins.op == g3pvm::Opcode::Sub || ins.op == g3pvm::Opcode::Mul ||
        ins.op == g3pvm::Opcode::Div || ins.op == g3pvm::Opcode::Mod) {
      const Value b = stack.back();
      stack.pop_back();
      const Value a = stack.back();
      stack.pop_back();
      double a_num = 0.0;
      double b_num = 0.0;
      bool any_float = false;
      g3pvm::vm_semantics::to_numeric_pair(a, b, a_num, b_num, any_float);
      std::cout << "  arith lhs=" << value_debug_string(a) << " rhs=" << value_debug_string(b) << "\n";
      if (ins.op == g3pvm::Opcode::Add) {
        stack.push_back(any_float ? Value::from_float(g3pvm::vm_semantics::canonicalize_vm_float(a_num + b_num))
                                  : Value::from_int(g3pvm::vm_semantics::wrap_int_add(
                                        static_cast<long long>(a_num), static_cast<long long>(b_num))));
      } else if (ins.op == g3pvm::Opcode::Sub) {
        stack.push_back(any_float ? Value::from_float(g3pvm::vm_semantics::canonicalize_vm_float(a_num - b_num))
                                  : Value::from_int(g3pvm::vm_semantics::wrap_int_sub(
                                        static_cast<long long>(a_num), static_cast<long long>(b_num))));
      } else if (ins.op == g3pvm::Opcode::Mul) {
        stack.push_back(any_float ? Value::from_float(g3pvm::vm_semantics::canonicalize_vm_float(a_num * b_num))
                                  : Value::from_int(g3pvm::vm_semantics::wrap_int_mul(
                                        static_cast<long long>(a_num), static_cast<long long>(b_num))));
      } else if (ins.op == g3pvm::Opcode::Div) {
        stack.push_back(Value::from_float(g3pvm::vm_semantics::canonicalize_vm_float(a_num / b_num)));
      } else if (any_float) {
        const double mod_value = g3pvm::vm_semantics::py_float_mod(a_num, b_num);
        std::cout << "  mod_raw=" << std::setprecision(17) << mod_value << "\n";
        stack.push_back(Value::from_float(g3pvm::vm_semantics::canonicalize_vm_float(mod_value)));
      } else {
        stack.push_back(Value::from_int(g3pvm::vm_semantics::py_int_mod(
            static_cast<long long>(a_num), static_cast<long long>(b_num))));
      }
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Lt || ins.op == g3pvm::Opcode::Le || ins.op == g3pvm::Opcode::Gt ||
        ins.op == g3pvm::Opcode::Ge || ins.op == g3pvm::Opcode::Eq || ins.op == g3pvm::Opcode::Ne) {
      const Value b = stack.back();
      stack.pop_back();
      const Value a = stack.back();
      stack.pop_back();
      bool out_bool = false;
      g3pvm::vm_semantics::CmpOp cmp_op = g3pvm::vm_semantics::CmpOp::EQ;
      if (ins.op == g3pvm::Opcode::Lt) cmp_op = g3pvm::vm_semantics::CmpOp::LT;
      else if (ins.op == g3pvm::Opcode::Le) cmp_op = g3pvm::vm_semantics::CmpOp::LE;
      else if (ins.op == g3pvm::Opcode::Gt) cmp_op = g3pvm::vm_semantics::CmpOp::GT;
      else if (ins.op == g3pvm::Opcode::Ge) cmp_op = g3pvm::vm_semantics::CmpOp::GE;
      else if (ins.op == g3pvm::Opcode::Ne) cmp_op = g3pvm::vm_semantics::CmpOp::NE;
      g3pvm::vm_semantics::compare_values(cmp_op, a, b, out_bool);
      stack.push_back(Value::from_bool(out_bool));
      std::cout << "  cmp lhs=" << value_debug_string(a) << " rhs=" << value_debug_string(b)
                << " -> " << value_debug_string(stack.back()) << "\n";
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Jmp) {
      ip = ins.a;
      continue;
    }
    if (ins.op == g3pvm::Opcode::JmpIfFalse || ins.op == g3pvm::Opcode::JmpIfTrue) {
      const Value c = stack.back();
      stack.pop_back();
      std::cout << "  branch cond=" << value_debug_string(c) << "\n";
      if (ins.op == g3pvm::Opcode::JmpIfFalse && !c.b) ip = ins.a;
      if (ins.op == g3pvm::Opcode::JmpIfTrue && c.b) ip = ins.a;
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::CallBuiltin) {
      const int argc = ins.b;
      std::vector<Value> args;
      const std::size_t start = stack.size() - static_cast<std::size_t>(argc);
      for (std::size_t i = start; i < stack.size(); ++i) {
        args.push_back(stack[i]);
      }
      stack.resize(start);
      g3pvm::BuiltinId builtin_id = g3pvm::BuiltinId::Abs;
      if (!g3pvm::builtin_id_from_int(ins.a, builtin_id)) {
        throw std::runtime_error("unknown builtin id in probe");
      }
      const g3pvm::BuiltinResult out = g3pvm::builtin_call(builtin_id, args);
      stack.push_back(out.value);
      std::cout << "  builtin " << g3pvm::builtin_name(builtin_id);
      for (const Value& arg : args) {
        std::cout << " " << value_debug_string(arg);
      }
      std::cout << " -> " << value_debug_string(out.value) << "\n";
      print_stack();
      continue;
    }
    if (ins.op == g3pvm::Opcode::Return) {
      std::cout << "  return " << value_debug_string(stack.back()) << "\n";
      return;
    }
    std::cout << "  unhandled trace opcode\n";
    return;
  }
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path);
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

bool is_integer_number(double x) {
  const long long i = static_cast<long long>(x);
  return static_cast<double>(i) == x;
}

Value decode_typed_or_raw_value(const JsonValue& v) {
  if (v.kind == JsonValue::Kind::Object) {
    auto it = v.object_v.find("type");
    if (it != v.object_v.end()) {
      return g3pvm::cli_detail::decode_typed_value(v);
    }
  }
  if (v.kind == JsonValue::Kind::Null) {
    return Value::none();
  }
  if (v.kind == JsonValue::Kind::Bool) {
    return Value::from_bool(v.bool_v);
  }
  if (v.kind == JsonValue::Kind::Number) {
    if (is_integer_number(v.number_v)) {
      return Value::from_int(static_cast<long long>(v.number_v));
    }
    return Value::from_float(v.number_v);
  }
  throw std::runtime_error("unsupported raw value type");
}

g3pvm::evo::NamedInputs decode_inputs(const JsonValue& raw) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("case.inputs must be an object");
  }
  g3pvm::evo::NamedInputs out;
  for (const auto& kv : raw.object_v) {
    out[kv.first] = decode_typed_or_raw_value(kv.second);
  }
  return out;
}

std::vector<EvalCase> load_cases_v1(const std::string& path) {
  const JsonValue payload = JsonParser(read_text_file(path)).parse();
  const auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "fitness-cases-v1") {
    throw std::runtime_error("input JSON must include format_version=fitness-cases-v1");
  }
  const auto cases_it = payload.object_v.find("cases");
  if (cases_it == payload.object_v.end() || cases_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input JSON must include list field: cases");
  }

  std::vector<EvalCase> out;
  out.reserve(cases_it->second.array_v.size());
  for (const JsonValue& row : cases_it->second.array_v) {
    const auto inputs_it = row.object_v.find("inputs");
    const auto expected_it = row.object_v.find("expected");
    if (inputs_it == row.object_v.end() || expected_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] must include inputs/expected");
    }
    out.push_back(EvalCase{decode_inputs(inputs_it->second), decode_typed_or_raw_value(expected_it->second)});
  }
  return out;
}

std::vector<ProgramGenome> init_population(const EvolutionConfig& cfg) {
  std::vector<ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(cfg.population_size));
  for (int i = 0; i < cfg.population_size; ++i) {
    out.push_back(g3pvm::evo::generate_random_genome(cfg.seed + static_cast<std::uint64_t>(i), cfg.limits));
  }
  return out;
}

std::vector<ProgramGenome> next_population_from_scored(const std::vector<ScoredGenome>& scored,
                                                       const EvolutionConfig& cfg,
                                                       std::mt19937_64* rng) {
  std::vector<ProgramGenome> next_population;
  next_population.reserve(static_cast<std::size_t>(cfg.population_size));
  const int offspring_count = cfg.population_size;
  std::vector<ProgramGenome> selected_parents = g3pvm::evo::tournament_selection_without_replacement(
      scored, *rng, cfg.selection_pressure, offspring_count);

  std::vector<ProgramGenome> offspring = selected_parents;
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);

  if (selected_parents.size() > 1) {
    std::shuffle(offspring.begin(), offspring.end(), *rng);
    for (std::size_t i = 0; i + 1 < offspring.size(); i += 2) {
      if (prob_dist(*rng) >= cfg.crossover_rate) {
        continue;
      }
      auto children = g3pvm::evo::crossover(offspring[i], offspring[i + 1], seed_dist(*rng), cfg.limits);
      offspring[i] = std::move(children.first);
      offspring[i + 1] = std::move(children.second);
    }
  }

  for (ProgramGenome& child : offspring) {
    if (prob_dist(*rng) < cfg.mutation_rate) {
      child = g3pvm::evo::mutate(child, seed_dist(*rng), cfg.limits, cfg.mutation_subtree_prob);
    }
  }
  next_population.insert(next_population.end(),
                         std::make_move_iterator(offspring.begin()),
                         std::make_move_iterator(offspring.end()));
  return next_population;
}

g3pvm::CaseBindings to_case_inputs(const EvalCase& one_case,
                                 const std::vector<std::string>& input_names) {
  g3pvm::CaseBindings out;
  out.reserve(input_names.size());
  for (std::size_t i = 0; i < input_names.size(); ++i) {
    const auto it = one_case.inputs.find(input_names[i]);
    if (it != one_case.inputs.end()) {
      out.push_back(g3pvm::InputBinding{static_cast<int>(i), it->second});
    }
  }
  return out;
}

}  // namespace

int main() {
  const std::vector<EvalCase> cases = load_cases_v1("data/fixtures/simple_exp_1024.json");

  EvolutionConfig cpu_cfg;
  cpu_cfg.population_size = 2048;
  cpu_cfg.generations = 40;
  cpu_cfg.mutation_rate = 0.5;
  cpu_cfg.mutation_subtree_prob = 0.8;
  cpu_cfg.crossover_rate = 0.9;
  cpu_cfg.selection_pressure = 3;
  cpu_cfg.seed = 0;
  cpu_cfg.fuel = 20000;
  cpu_cfg.gpu_blocksize = 256;
  cpu_cfg.eval_engine = g3pvm::evo::EvalEngine::CPU;

  if (const char* generations_env = std::getenv("G3PVM_PROBE_GENERATIONS")) {
    cpu_cfg.generations = std::atoi(generations_env);
  }

  EvolutionConfig gpu_cfg = cpu_cfg;
  gpu_cfg.eval_engine = g3pvm::evo::EvalEngine::GPU;
  const std::vector<std::string> input_names = {"x"};

  std::mt19937_64 rng(cpu_cfg.seed);
  std::vector<ProgramGenome> population = init_population(cpu_cfg);

  for (int gen = 0; gen < cpu_cfg.generations; ++gen) {
    const std::vector<g3pvm::evo::ScoredGenome> cpu_scored =
        g3pvm::evo::evaluate_population(population, cases, cpu_cfg);
    std::vector<g3pvm::evo::ScoredGenome> gpu_scored;
    try {
      gpu_scored = g3pvm::evo::evaluate_population(population, cases, gpu_cfg);
    } catch (const std::runtime_error& err) {
      if (std::string(err.what()).find("cuda device unavailable") != std::string::npos) {
        std::cout << "simple_exp_population_probe: SKIP (" << err.what() << ")\n";
        return 0;
      }
      throw;
    }

    for (std::size_t i = 0; i < cpu_scored.size(); ++i) {
      if (cpu_scored[i].genome.meta.program_key != gpu_scored[i].genome.meta.program_key ||
          cpu_scored[i].fitness != gpu_scored[i].fitness) {
        std::cout << "generation=" << gen << " index=" << i << "\n";
        std::cout << "cpu_program_key=" << cpu_scored[i].genome.meta.program_key << "\n";
        std::cout << "gpu_program_key=" << gpu_scored[i].genome.meta.program_key << "\n";
        std::cout << "cpu_fitness=" << cpu_scored[i].fitness << "\n";
        std::cout << "gpu_fitness=" << gpu_scored[i].fitness << "\n";
        const ProgramGenome& mismatched = cpu_scored[i].genome;
        std::cout << "program_key=" << mismatched.meta.program_key << "\n";
        const g3pvm::BytecodeProgram bc =
            g3pvm::evo::compile_for_eval(mismatched, input_names);
        std::cout << std::setprecision(17);
        std::cout << "bytecode_consts=" << bc.consts.size() << " bytecode_code=" << bc.code.size() << "\n";
        dump_consts(bc);
        for (std::size_t op_idx = 0; op_idx < bc.code.size(); ++op_idx) {
          const auto& ins = bc.code[op_idx];
          std::cout << "op[" << op_idx << "]=" << g3pvm::opcode_name(ins.op);
          if (ins.has_a) {
            std::cout << " a=" << ins.a;
          }
          if (ins.has_b) {
            std::cout << " b=" << ins.b;
          }
          std::cout << "\n";
        }
        for (std::size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
          const std::vector<g3pvm::CaseBindings> shared_cases{to_case_inputs(cases[case_idx], input_names)};
          const std::vector<Value> shared_answer{cases[case_idx].expected};
          const double cpu_case =
              g3pvm::eval_fitness_cpu({bc}, shared_cases, shared_answer, cpu_cfg.fuel, cpu_cfg.penalty,
                                      cpu_cfg.gpu_blocksize)[0];
          g3pvm::FitnessSessionGpu gpu_case_session;
          const g3pvm::FitnessSessionInitResult init =
              gpu_case_session.init(shared_cases, shared_answer, gpu_cfg.fuel, gpu_cfg.gpu_blocksize,
                                    gpu_cfg.penalty);
          if (!init.ok) {
            throw std::runtime_error("case-level gpu fitness init failed: " + init.err.message);
          }
          const g3pvm::FitnessEvalResult gpu_case = gpu_case_session.eval_programs({bc});
          if (!gpu_case.ok) {
            throw std::runtime_error("case-level gpu fitness failed: " + gpu_case.err.message);
          }
          if (cpu_case != gpu_case.fitness[0]) {
            std::cout << "case_index=" << case_idx << "\n";
            std::cout << "x=" << cases[case_idx].inputs.at("x").f << "\n";
            std::cout << "expected=" << cases[case_idx].expected.f << "\n";
            std::cout << "cpu_case_fitness=" << cpu_case << "\n";
            std::cout << "gpu_case_fitness=" << gpu_case.fitness[0] << "\n";
            trace_cpu_case(bc, {{0, cases[case_idx].inputs.at("x")}}, cpu_cfg.fuel);
            break;
          }
        }
        return 1;
      }
    }

    const std::vector<ScoredGenome> scored = g3pvm::evo::evaluate_population(population, cases, cpu_cfg);
    population = next_population_from_scored(scored, cpu_cfg, &rng);
  }

  std::cout << "simple_exp_population_probe: OK\n";
  return 0;
}
