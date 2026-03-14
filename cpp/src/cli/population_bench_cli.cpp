#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#endif

// Keep directly buildable.
#include "json.cpp"
#include "codec.cpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::CaseBindings;
using g3pvm::ExecResult;
using g3pvm::InputBinding;
using g3pvm::Value;
using g3pvm::cli_detail::JsonValue;
using g3pvm::evo::EvalCase;
using g3pvm::evo::EvolutionConfig;
using g3pvm::evo::NamedInputs;
using g3pvm::evo::ProgramGenome;
using g3pvm::evo::ScoredGenome;

struct CliOptions {
  std::string cases_path;
  std::string out_population_json;
  std::string engine = "cpu";
  std::string repro_backend = "cpu";
  bool repro_overlap = false;
  int blocksize = 1024;
  int fuel = 20000;
  double mutation_rate = 0.5;
  double mutation_subtree_prob = 0.8;
  double crossover_rate = 0.9;
  double penalty = 1.0;
  int selection_pressure = 3;
  int population_size = 1024;
  std::uint64_t seed_start = 0;
  int probe_cases = 32;
  double min_success_rate = 0.10;
  int max_attempts = 200000;
  int max_expr_depth = 5;
  int max_stmts_per_block = 6;
  int max_total_nodes = 80;
  int max_for_k = 16;
  int max_call_args = 3;
};

struct AcceptedSeed {
  std::uint64_t seed = 0;
  int probe_successes = 0;
  int node_count = 0;
  std::string program_key;
};

struct GeneratedPopulation {
  g3pvm::evo::Limits limits;
  std::vector<ProgramGenome> population;
  std::vector<std::uint64_t> seeds;
  int attempts = 0;
  int probe_case_count = 0;
  int min_successes = 0;
};

struct CompiledPopulation {
  std::vector<BytecodeProgram> programs;
  double compile_ms = 0.0;
};

struct EvalRun {
  std::vector<double> fitness;
  double eval_ms = 0.0;
  double session_init_ms = 0.0;
  double pack_upload_ms = 0.0;
  double kernel_ms = 0.0;
  double copyback_ms = 0.0;
};

using ReproductionRun = g3pvm::evo::repro::ReproductionResult;

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("missing file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
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
  if (v.kind == JsonValue::Kind::Null) return Value::none();
  if (v.kind == JsonValue::Kind::Bool) return Value::from_bool(v.bool_v);
  if (v.kind == JsonValue::Kind::Number) {
    if (is_integer_number(v.number_v)) return Value::from_int(static_cast<long long>(v.number_v));
    return Value::from_float(v.number_v);
  }
  throw std::runtime_error("unsupported raw value type");
}

NamedInputs decode_inputs(const JsonValue& raw) {
  if (raw.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("case.inputs must be an object");
  }
  NamedInputs out;
  for (const auto& kv : raw.object_v) {
    out[kv.first] = decode_typed_or_raw_value(kv.second);
  }
  return out;
}

std::vector<EvalCase> parse_cases_v1(const JsonValue& payload) {
  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "fitness-cases-v1") {
    throw std::runtime_error("input JSON must include format_version=fitness-cases-v1");
  }
  auto cases_it = payload.object_v.find("cases");
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
  if (out.empty()) {
    throw std::runtime_error("cases must not be empty");
  }
  return out;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;
  auto parse_on_off = [](const std::string& raw, const char* flag) -> bool {
    if (raw == "on") return true;
    if (raw == "off") return false;
    throw std::runtime_error(std::string(flag) + " must be on or off");
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* key) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + key);
      return argv[++i];
    };
    if (arg == "--cases") opts.cases_path = need_value("--cases");
    else if (arg == "--out-population-json") opts.out_population_json = need_value("--out-population-json");
    else if (arg == "--engine") opts.engine = need_value("--engine");
    else if (arg == "--repro-backend") opts.repro_backend = need_value("--repro-backend");
    else if (arg == "--repro-overlap") opts.repro_overlap = parse_on_off(need_value("--repro-overlap"), "--repro-overlap");
    else if (arg == "--blocksize") opts.blocksize = std::stoi(need_value("--blocksize"));
    else if (arg == "--fuel") opts.fuel = std::stoi(need_value("--fuel"));
    else if (arg == "--mutation-rate") opts.mutation_rate = std::stod(need_value("--mutation-rate"));
    else if (arg == "--mutation-subtree-prob") opts.mutation_subtree_prob = std::stod(need_value("--mutation-subtree-prob"));
    else if (arg == "--crossover-rate") opts.crossover_rate = std::stod(need_value("--crossover-rate"));
    else if (arg == "--penalty") opts.penalty = std::stod(need_value("--penalty"));
    else if (arg == "--selection-pressure") opts.selection_pressure = std::stoi(need_value("--selection-pressure"));
    else if (arg == "--population-size") opts.population_size = std::stoi(need_value("--population-size"));
    else if (arg == "--seed-start") opts.seed_start = static_cast<std::uint64_t>(std::stoull(need_value("--seed-start")));
    else if (arg == "--probe-cases") opts.probe_cases = std::stoi(need_value("--probe-cases"));
    else if (arg == "--min-success-rate") opts.min_success_rate = std::stod(need_value("--min-success-rate"));
    else if (arg == "--max-attempts") opts.max_attempts = std::stoi(need_value("--max-attempts"));
    else if (arg == "--max-expr-depth") opts.max_expr_depth = std::stoi(need_value("--max-expr-depth"));
    else if (arg == "--max-stmts-per-block") opts.max_stmts_per_block = std::stoi(need_value("--max-stmts-per-block"));
    else if (arg == "--max-total-nodes") opts.max_total_nodes = std::stoi(need_value("--max-total-nodes"));
    else if (arg == "--max-for-k") opts.max_for_k = std::stoi(need_value("--max-for-k"));
    else if (arg == "--max-call-args") opts.max_call_args = std::stoi(need_value("--max-call-args"));
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (opts.cases_path.empty()) throw std::runtime_error("--cases is required");
  if (opts.engine != "cpu" && opts.engine != "gpu") {
    throw std::runtime_error("--engine must be cpu or gpu");
  }
  if (opts.repro_backend != "cpu" && opts.repro_backend != "gpu") {
    throw std::runtime_error("--repro-backend must be cpu or gpu");
  }
  if (opts.population_size <= 0) throw std::runtime_error("--population-size must be > 0");
  if (opts.probe_cases <= 0) throw std::runtime_error("--probe-cases must be > 0");
  if (opts.min_success_rate < 0.0 || opts.min_success_rate > 1.0) {
    throw std::runtime_error("--min-success-rate must be in [0, 1]");
  }
  if (opts.max_attempts <= 0) throw std::runtime_error("--max-attempts must be > 0");
  return opts;
}

std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    if (c == '"') oss << "\\\"";
    else if (c == '\\') oss << "\\\\";
    else if (c == '\n') oss << "\\n";
    else if (c == '\r') oss << "\\r";
    else if (c == '\t') oss << "\\t";
    else oss << c;
  }
  return oss.str();
}

std::vector<std::string> build_canonical_input_names(const std::vector<EvalCase>& cases) {
  std::set<std::string> names;
  for (const EvalCase& one_case : cases) {
    for (const auto& kv : one_case.inputs) {
      names.insert(kv.first);
    }
  }
  return std::vector<std::string>(names.begin(), names.end());
}

std::vector<CaseBindings> build_shared_case_bindings(const std::vector<EvalCase>& cases,
                                                     const std::vector<std::string>& input_names) {
  std::vector<CaseBindings> out;
  out.reserve(cases.size());
  for (const EvalCase& one_case : cases) {
    CaseBindings bindings;
    bindings.reserve(input_names.size());
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      auto it = one_case.inputs.find(input_names[i]);
      if (it != one_case.inputs.end()) {
        bindings.push_back(InputBinding{static_cast<int>(i), it->second});
      }
    }
    out.push_back(std::move(bindings));
  }
  return out;
}

std::vector<Value> build_expected_values(const std::vector<EvalCase>& cases) {
  std::vector<Value> out;
  out.reserve(cases.size());
  for (const EvalCase& one_case : cases) {
    out.push_back(one_case.expected);
  }
  return out;
}

std::vector<std::pair<int, Value>> case_inputs_to_locals(const EvalCase& one_case,
                                                         const std::vector<std::string>& input_names) {
  std::vector<std::pair<int, Value>> out;
  out.reserve(input_names.size());
  for (std::size_t i = 0; i < input_names.size(); ++i) {
    auto it = one_case.inputs.find(input_names[i]);
    if (it != one_case.inputs.end()) {
      out.push_back({static_cast<int>(i), it->second});
    }
  }
  return out;
}

void write_population_seed_set(const std::string& path,
                               const g3pvm::evo::Limits& limits,
                               const std::vector<std::uint64_t>& seeds,
                               const std::vector<AcceptedSeed>* accepted,
                               const std::string& cases_path,
                               int probe_case_count,
                               double min_success_rate,
                               int fuel,
                               int attempts) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open out-population-json path");
  }
  out << "{\n";
  out << "  \"format_version\": \"population-seeds-v1\",\n";
  out << "  \"cases_path\": \"" << json_escape(cases_path) << "\",\n";
  out << "  \"population_size\": " << seeds.size() << ",\n";
  out << "  \"probe_cases\": " << probe_case_count << ",\n";
  out << "  \"min_success_rate\": " << std::setprecision(17) << min_success_rate << ",\n";
  out << "  \"fuel\": " << fuel << ",\n";
  out << "  \"attempts\": " << attempts << ",\n";
  out << "  \"limits\": {\n";
  out << "    \"max_expr_depth\": " << limits.max_expr_depth << ",\n";
  out << "    \"max_stmts_per_block\": " << limits.max_stmts_per_block << ",\n";
  out << "    \"max_total_nodes\": " << limits.max_total_nodes << ",\n";
  out << "    \"max_for_k\": " << limits.max_for_k << ",\n";
  out << "    \"max_call_args\": " << limits.max_call_args << "\n";
  out << "  },\n";
  out << "  \"seeds\": [\n";
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    out << "    {\"seed\": " << seeds[i];
    if (accepted != nullptr) {
      const AcceptedSeed& row = accepted->at(i);
      out << ", \"probe_successes\": " << row.probe_successes
          << ", \"node_count\": " << row.node_count
          << ", \"program_key\": \"" << json_escape(row.program_key) << "\"";
    }
    out << "}";
    if (i + 1 < seeds.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

GeneratedPopulation generate_population(const CliOptions& args,
                                        const std::vector<EvalCase>& cases,
                                        const std::vector<std::string>& input_names) {
  const g3pvm::evo::Limits limits{
      args.max_expr_depth,
      args.max_stmts_per_block,
      args.max_total_nodes,
      args.max_for_k,
      args.max_call_args,
  };
  const int probe_case_count = std::min<int>(args.probe_cases, static_cast<int>(cases.size()));
  const int min_successes =
      std::max(1, static_cast<int>(std::ceil(args.min_success_rate * static_cast<double>(probe_case_count))));

  std::vector<AcceptedSeed> accepted;
  accepted.reserve(static_cast<std::size_t>(args.population_size));
  std::vector<ProgramGenome> population;
  population.reserve(static_cast<std::size_t>(args.population_size));

  std::uint64_t candidate_seed = args.seed_start;
  int attempts = 0;
  while (static_cast<int>(accepted.size()) < args.population_size && attempts < args.max_attempts) {
    ++attempts;
    const ProgramGenome genome = g3pvm::evo::generate_random_genome(candidate_seed, limits);
    const BytecodeProgram program = g3pvm::evo::compile_for_eval(genome, input_names);

    int successes = 0;
    for (int i = 0; i < probe_case_count; ++i) {
      const ExecResult out = g3pvm::execute_bytecode_cpu(
          program, case_inputs_to_locals(cases[static_cast<std::size_t>(i)], input_names), args.fuel);
      if (!out.is_error) {
        ++successes;
      }
    }

    if (successes >= min_successes) {
      accepted.push_back(AcceptedSeed{
          candidate_seed,
          successes,
          genome.meta.node_count,
          genome.meta.program_key,
      });
      population.push_back(genome);
    }
    ++candidate_seed;
  }

  if (static_cast<int>(accepted.size()) != args.population_size) {
    throw std::runtime_error("failed to collect enough accepted genomes");
  }

  GeneratedPopulation out;
  out.limits = limits;
  out.seeds.reserve(accepted.size());
  for (const AcceptedSeed& one : accepted) {
    out.seeds.push_back(one.seed);
  }
  out.population = std::move(population);
  out.attempts = attempts;
  out.probe_case_count = probe_case_count;
  out.min_successes = min_successes;

  if (!args.out_population_json.empty()) {
    write_population_seed_set(args.out_population_json,
                              out.limits,
                              out.seeds,
                              &accepted,
                              args.cases_path,
                              probe_case_count,
                              args.min_success_rate,
                              args.fuel,
                              attempts);
  }
  return out;
}

CompiledPopulation compile_population(const std::vector<ProgramGenome>& population,
                                      const std::vector<std::string>& input_names) {
  CompiledPopulation out;
  out.programs.reserve(population.size());
  for (const ProgramGenome& genome : population) {
    const auto t0 = std::chrono::steady_clock::now();
    out.programs.push_back(g3pvm::evo::compile_for_eval(genome, input_names));
    const auto t1 = std::chrono::steady_clock::now();
    out.compile_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return out;
}

void canonicalize_fitness_vector(std::vector<double>* fitness) {
  if (fitness == nullptr) {
    return;
  }
  for (double& value : *fitness) {
    value = g3pvm::evo::canonicalize_fitness_for_ranking(value);
  }
}

EvalRun evaluate_compiled_population(const std::vector<BytecodeProgram>& programs,
                                     const std::vector<CaseBindings>& shared_case_bindings,
                                     const std::vector<Value>& expected_values,
                                     const CliOptions& args) {
  EvalRun out;
  if (args.engine == "cpu") {
    const auto t0 = std::chrono::steady_clock::now();
    out.fitness = g3pvm::eval_fitness_cpu(
        programs, shared_case_bindings, expected_values, args.fuel, args.penalty, args.blocksize);
    const auto t1 = std::chrono::steady_clock::now();
    out.eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    canonicalize_fitness_vector(&out.fitness);
    return out;
  }
#ifdef G3PVM_HAS_CUDA
  g3pvm::FitnessSessionGpu session;
  const auto t0 = std::chrono::steady_clock::now();
  const auto init_result =
      session.init(shared_case_bindings, expected_values, args.fuel, args.blocksize, args.penalty);
  const auto t1 = std::chrono::steady_clock::now();
  if (!init_result.ok) {
    throw std::runtime_error("gpu session init failed: " + init_result.err.message);
  }
  const auto fit = session.eval_programs(programs);
  const auto t2 = std::chrono::steady_clock::now();
  if (!fit.ok) {
    throw std::runtime_error("gpu eval failed: " + fit.err.message);
  }
  out.fitness = fit.fitness;
  canonicalize_fitness_vector(&out.fitness);
  out.session_init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  out.eval_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
  out.pack_upload_ms = fit.pack_programs_ms + fit.upload_programs_ms;
  out.kernel_ms = fit.kernel_exec_ms;
  out.copyback_ms = fit.copyback_ms;
  return out;
#else
  throw std::runtime_error("gpu benchmark requested but CUDA support is not built");
#endif
}

ReproductionRun reproduce_population(const std::vector<ScoredGenome>& scored,
                                     const EvolutionConfig& cfg,
                                     std::mt19937_64* rng) {
  return g3pvm::evo::repro::run_reproduction_backend(scored, cfg, *rng);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions args = parse_cli(argc, argv);
    const JsonValue cases_payload = g3pvm::cli_detail::JsonParser(read_text_file(args.cases_path)).parse();
    const std::vector<EvalCase> cases = parse_cases_v1(cases_payload);
    const std::vector<std::string> input_names = build_canonical_input_names(cases);
    const GeneratedPopulation generated = generate_population(args, cases, input_names);
    const std::vector<ProgramGenome>& population = generated.population;

    const std::vector<CaseBindings> shared_case_bindings = build_shared_case_bindings(cases, input_names);
    const std::vector<Value> expected_values = build_expected_values(cases);

    EvolutionConfig cfg;
    cfg.population_size = static_cast<int>(population.size());
    cfg.generations = 1;
    cfg.mutation_rate = args.mutation_rate;
    cfg.mutation_subtree_prob = args.mutation_subtree_prob;
    cfg.crossover_rate = args.crossover_rate;
    cfg.penalty = args.penalty;
    cfg.eval_engine = (args.engine == "gpu") ? g3pvm::evo::EvalEngine::GPU : g3pvm::evo::EvalEngine::CPU;
    cfg.reproduction_backend = g3pvm::evo::repro::parse_reproduction_backend_name(args.repro_backend);
    cfg.repro_overlap = args.repro_overlap;
    cfg.gpu_blocksize = args.blocksize;
    cfg.selection_pressure = args.selection_pressure;
    cfg.seed = 0;
    cfg.fuel = args.fuel;
    cfg.limits = generated.limits;

    const bool overlap_gpu =
        args.engine == "gpu" &&
        cfg.reproduction_backend == g3pvm::evo::repro::ReproductionBackend::Gpu &&
        args.repro_overlap;
    struct OverlapPrepared {
      g3pvm::evo::repro::GpuReproPreparedData prepared;
      g3pvm::evo::repro::ReproductionStats stats;
    };
    std::future<OverlapPrepared> overlap_future;

    const auto all_t0 = std::chrono::steady_clock::now();
    const auto compile = compile_population(population, input_names);
    if (overlap_gpu) {
      std::mt19937_64 overlap_rng(cfg.seed);
      const std::uint64_t repro_seed = overlap_rng();
      overlap_future = std::async(std::launch::async, [population, cfg, repro_seed]() {
        OverlapPrepared out;
        out.prepared =
            g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, repro_seed, &out.stats);
        return out;
      });
    }
    const EvalRun eval = evaluate_compiled_population(compile.programs, shared_case_bindings, expected_values, args);

    std::vector<ScoredGenome> scored_unsorted;
    scored_unsorted.reserve(population.size());
    long double fitness_sum = 0.0L;
    for (std::size_t i = 0; i < population.size(); ++i) {
      scored_unsorted.push_back(ScoredGenome{population[i], eval.fitness[i]});
      fitness_sum += static_cast<long double>(eval.fitness[i]);
    }
    std::vector<ScoredGenome> scored = scored_unsorted;
    std::sort(scored.begin(), scored.end(), g3pvm::evo::scored_genome_sorts_before);

    std::mt19937_64 rng(cfg.seed);
    const auto repro_t0 = std::chrono::steady_clock::now();
    ReproductionRun reproduction;
    if (overlap_gpu) {
      OverlapPrepared overlap = overlap_future.get();
      reproduction = g3pvm::evo::repro::run_gpu_repro_backend_prepared(
          scored_unsorted, cfg, overlap.prepared, &overlap.stats);
    } else {
      reproduction = reproduce_population(scored, cfg, &rng);
    }
    const auto repro_t1 = std::chrono::steady_clock::now();
    const auto all_t1 = std::chrono::steady_clock::now();

    const double repro_ms = std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count();
    const double total_ms = std::chrono::duration<double, std::milli>(all_t1 - all_t0).count();

    std::cout << "BENCH engine=" << args.engine
              << " repro_backend=" << g3pvm::evo::repro::reproduction_backend_name(cfg.reproduction_backend)
              << " repro_overlap=" << (cfg.repro_overlap ? "on" : "off")
              << " population_size=" << population.size()
              << " compile_ms=" << std::fixed << std::setprecision(3) << compile.compile_ms
              << " eval_ms=" << eval.eval_ms
              << " repro_ms=" << repro_ms
              << " selection_ms=" << reproduction.stats.selection_ms
              << " crossover_ms=" << reproduction.stats.crossover_ms
              << " mutation_ms=" << reproduction.stats.mutation_ms
              << " repro_prepare_inputs_ms=" << reproduction.stats.prepare_inputs_ms
              << " repro_setup_ms=" << reproduction.stats.setup_ms
              << " repro_preprocess_ms=" << reproduction.stats.preprocess_ms
              << " repro_pack_ms=" << reproduction.stats.pack_ms
              << " repro_upload_ms=" << reproduction.stats.upload_ms
              << " repro_kernel_ms=" << reproduction.stats.kernel_ms
              << " repro_copyback_ms=" << reproduction.stats.copyback_ms
              << " repro_decode_ms=" << reproduction.stats.decode_ms
              << " repro_teardown_ms=" << reproduction.stats.teardown_ms
              << " repro_selection_kernel_ms=" << reproduction.stats.selection_kernel_ms
              << " repro_variation_kernel_ms=" << reproduction.stats.variation_kernel_ms
              << " total_ms=" << total_ms
              << " pack_upload_ms=" << eval.pack_upload_ms
              << " kernel_ms=" << eval.kernel_ms
              << " copyback_ms=" << eval.copyback_ms
              << " mean_fitness=" << std::setprecision(17)
              << static_cast<double>(fitness_sum / static_cast<long double>(eval.fitness.size()))
              << " best_fitness=" << scored.front().fitness
              << " best_program_key=" << scored.front().genome.meta.program_key
              << " population_source=generated"
              << " generation_attempts=" << generated.attempts
              << " probe_cases=" << generated.probe_case_count
              << " min_successes=" << generated.min_successes;
    if (!args.out_population_json.empty()) {
      std::cout << " out_population_json=" << args.out_population_json;
    }
    std::cout << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
