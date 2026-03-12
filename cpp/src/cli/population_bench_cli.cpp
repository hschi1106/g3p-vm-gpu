#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
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
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/runtime/cpu/fitness_cpu.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/runtime/gpu/fitness_gpu.hpp"
#endif

// Keep directly buildable.
#include "json.cpp"
#include "codec.cpp"

namespace {

using g3pvm::Value;
using g3pvm::cli_detail::JsonValue;
using g3pvm::evo::EvolutionConfig;
using g3pvm::evo::EvalCase;
using g3pvm::evo::NamedInputs;
using g3pvm::evo::ProgramGenome;
using g3pvm::evo::ScoredGenome;
using g3pvm::InputBinding;
using g3pvm::CaseBindings;

struct CliOptions {
  std::string cases_path;
  std::string population_json;
  std::string mode = "one-gen-e2e";
  std::string engine = "cpu";
  int blocksize = 256;
  int fuel = 20000;
  double mutation_rate = 0.5;
  double mutation_subtree_prob = 0.8;
  double crossover_rate = 0.9;
  double penalty = 1.0;
  int selection_pressure = 3;
};

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
  return out;
}

double canonicalize_fitness_for_ranking(double fitness) {
  if (!std::isfinite(fitness) || fitness == 0.0) {
    return fitness == 0.0 ? 0.0 : fitness;
  }
  int exponent = 0;
  const double mantissa = std::frexp(fitness, &exponent);
  constexpr int kMantissaBits = 40;
  const long long quantized_mantissa = std::llround(std::ldexp(mantissa, kMantissaBits));
  return std::ldexp(static_cast<double>(quantized_mantissa), exponent - kMantissaBits);
}

bool scored_genome_sorts_before(const ScoredGenome& a, const ScoredGenome& b) {
  const double fitness_a = canonicalize_fitness_for_ranking(a.fitness);
  const double fitness_b = canonicalize_fitness_for_ranking(b.fitness);
  if (fitness_a != fitness_b) {
    return fitness_a > fitness_b;
  }
  return a.genome.meta.program_key < b.genome.meta.program_key;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* key) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + key);
      return argv[++i];
    };
    if (arg == "--cases") opts.cases_path = need_value("--cases");
    else if (arg == "--population-json") opts.population_json = need_value("--population-json");
    else if (arg == "--mode") opts.mode = need_value("--mode");
    else if (arg == "--engine") opts.engine = need_value("--engine");
    else if (arg == "--blocksize") opts.blocksize = std::stoi(need_value("--blocksize"));
    else if (arg == "--fuel") opts.fuel = std::stoi(need_value("--fuel"));
    else if (arg == "--mutation-rate") opts.mutation_rate = std::stod(need_value("--mutation-rate"));
    else if (arg == "--mutation-subtree-prob") opts.mutation_subtree_prob = std::stod(need_value("--mutation-subtree-prob"));
    else if (arg == "--crossover-rate") opts.crossover_rate = std::stod(need_value("--crossover-rate"));
    else if (arg == "--penalty") opts.penalty = std::stod(need_value("--penalty"));
    else if (arg == "--selection-pressure") opts.selection_pressure = std::stoi(need_value("--selection-pressure"));
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (opts.cases_path.empty()) throw std::runtime_error("--cases is required");
  if (opts.population_json.empty()) throw std::runtime_error("--population-json is required");
  if (opts.mode != "eval-only" && opts.mode != "one-gen-e2e") {
    throw std::runtime_error("--mode must be eval-only or one-gen-e2e");
  }
  if (opts.engine != "cpu" && opts.engine != "gpu") {
    throw std::runtime_error("--engine must be cpu or gpu");
  }
  return opts;
}

struct PopulationSeedSet {
  g3pvm::evo::Limits limits;
  std::vector<std::uint64_t> seeds;
};

PopulationSeedSet parse_population_seed_set(const JsonValue& root) {
  if (root.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("population-json must be object");
  }
  const auto fv_it = root.object_v.find("format_version");
  if (fv_it == root.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "population-seeds-v1") {
    throw std::runtime_error("population-json must include format_version=population-seeds-v1");
  }
  const auto limits_it = root.object_v.find("limits");
  const auto seeds_it = root.object_v.find("seeds");
  if (limits_it == root.object_v.end() || limits_it->second.kind != JsonValue::Kind::Object ||
      seeds_it == root.object_v.end() || seeds_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("population-json missing limits/seeds");
  }
  const auto& obj = limits_it->second.object_v;
  PopulationSeedSet out;
  out.limits.max_expr_depth = static_cast<int>(obj.at("max_expr_depth").number_v);
  out.limits.max_stmts_per_block = static_cast<int>(obj.at("max_stmts_per_block").number_v);
  out.limits.max_total_nodes = static_cast<int>(obj.at("max_total_nodes").number_v);
  out.limits.max_for_k = static_cast<int>(obj.at("max_for_k").number_v);
  out.limits.max_call_args = static_cast<int>(obj.at("max_call_args").number_v);
  out.seeds.reserve(seeds_it->second.array_v.size());
  for (const JsonValue& row : seeds_it->second.array_v) {
    out.seeds.push_back(static_cast<std::uint64_t>(row.object_v.at("seed").number_v));
  }
  return out;
}

std::vector<ProgramGenome> make_population_from_seeds(const PopulationSeedSet& seed_set) {
  std::vector<ProgramGenome> out;
  out.reserve(seed_set.seeds.size());
  for (std::uint64_t seed : seed_set.seeds) {
    out.push_back(g3pvm::evo::make_random_genome(seed, seed_set.limits));
  }
  return out;
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

struct CompiledPopulation {
  std::vector<g3pvm::BytecodeProgram> programs;
  double compile_ms = 0.0;
};

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

struct EvalRun {
  std::vector<double> fitness;
  double eval_ms = 0.0;
  double session_init_ms = 0.0;
  double pack_upload_ms = 0.0;
  double kernel_ms = 0.0;
  double copyback_ms = 0.0;
};

void canonicalize_fitness_vector(std::vector<double>* fitness) {
  if (fitness == nullptr) {
    return;
  }
  for (double& value : *fitness) {
    value = canonicalize_fitness_for_ranking(value);
  }
}

EvalRun evaluate_compiled_population(const std::vector<g3pvm::BytecodeProgram>& programs,
                                     const std::vector<CaseBindings>& shared_cases,
                                     const std::vector<Value>& shared_answer,
                                     const CliOptions& args) {
  EvalRun out;
  if (args.engine == "cpu") {
    const auto t0 = std::chrono::steady_clock::now();
    out.fitness = g3pvm::eval_fitness_cpu(programs, shared_cases, shared_answer, args.fuel, args.penalty);
    const auto t1 = std::chrono::steady_clock::now();
    out.eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    canonicalize_fitness_vector(&out.fitness);
    return out;
  }
#ifdef G3PVM_HAS_CUDA
  g3pvm::FitnessSessionGpu session;
  const auto t0 = std::chrono::steady_clock::now();
  const auto init_result = session.init(shared_cases, shared_answer, args.fuel, args.blocksize, args.penalty);
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

struct ReproductionRun {
  std::vector<ProgramGenome> next_population;
  double selection_ms = 0.0;
  double crossover_ms = 0.0;
  double mutation_ms = 0.0;
};

ReproductionRun reproduce_population(const std::vector<ScoredGenome>& scored,
                                     const EvolutionConfig& cfg,
                                     std::mt19937_64* rng) {
  ReproductionRun out;
  out.next_population.reserve(static_cast<std::size_t>(cfg.population_size));
  const int offspring_count = static_cast<int>(cfg.population_size);
  std::vector<ProgramGenome> selected_parents;
  selected_parents.reserve(static_cast<std::size_t>(offspring_count));

  const auto selection_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < offspring_count; ++i) {
    selected_parents.push_back(g3pvm::evo::tournament_selection(scored, *rng, cfg.selection_pressure));
  }
  const auto selection_t1 = std::chrono::steady_clock::now();
  out.selection_ms = std::chrono::duration<double, std::milli>(selection_t1 - selection_t0).count();

  std::vector<ProgramGenome> offspring = selected_parents;
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  std::uniform_int_distribution<std::uint64_t> seed_dist(0, 2000000000ULL);

  const auto crossover_t0 = std::chrono::steady_clock::now();
  if (selected_parents.size() > 1) {
    std::uniform_int_distribution<std::size_t> mate_pick(0, selected_parents.size() - 1);
    for (std::size_t i = 0; i < offspring.size(); ++i) {
      if (prob_dist(*rng) >= cfg.crossover_rate) {
        continue;
      }
      std::size_t mate_idx = mate_pick(*rng);
      if (mate_idx == i) {
        mate_idx = (mate_idx + 1) % selected_parents.size();
      }
      offspring[i] =
          g3pvm::evo::crossover(selected_parents[i], selected_parents[mate_idx], seed_dist(*rng), cfg.limits);
    }
  }
  const auto crossover_t1 = std::chrono::steady_clock::now();
  out.crossover_ms = std::chrono::duration<double, std::milli>(crossover_t1 - crossover_t0).count();

  const auto mutation_t0 = std::chrono::steady_clock::now();
  for (ProgramGenome& child : offspring) {
    if (prob_dist(*rng) < cfg.mutation_rate) {
      child = g3pvm::evo::mutate(child, seed_dist(*rng), cfg.limits, cfg.mutation_subtree_prob);
    }
  }
  const auto mutation_t1 = std::chrono::steady_clock::now();
  out.mutation_ms = std::chrono::duration<double, std::milli>(mutation_t1 - mutation_t0).count();

  out.next_population.insert(out.next_population.end(),
                             std::make_move_iterator(offspring.begin()),
                             std::make_move_iterator(offspring.end()));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions args = parse_cli(argc, argv);
    const JsonValue cases_payload = g3pvm::cli_detail::JsonParser(read_text_file(args.cases_path)).parse();
    const JsonValue population_payload = g3pvm::cli_detail::JsonParser(read_text_file(args.population_json)).parse();
    const std::vector<EvalCase> cases = parse_cases_v1(cases_payload);
    const PopulationSeedSet seed_set = parse_population_seed_set(population_payload);
    std::vector<ProgramGenome> population = make_population_from_seeds(seed_set);
    const std::vector<std::string> input_names = build_canonical_input_names(cases);
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
    cfg.gpu_blocksize = args.blocksize;
    cfg.selection_pressure = args.selection_pressure;
    cfg.seed = 0;
    cfg.fuel = args.fuel;
    cfg.limits = seed_set.limits;

    if (args.mode == "eval-only") {
      const auto compile = compile_population(population, input_names);
      const EvalRun eval = evaluate_compiled_population(compile.programs, shared_case_bindings, expected_values, args);
      long double checksum = 0.0L;
      for (double one : eval.fitness) checksum += one;
      std::cout << "BENCH mode=eval-only engine=" << args.engine
                << " population_size=" << population.size()
                << " compile_ms=" << std::fixed << std::setprecision(3) << compile.compile_ms
                << " eval_ms=" << eval.eval_ms
                << " total_ms="
                << (compile.compile_ms + eval.eval_ms)
                << " pack_upload_ms=" << eval.pack_upload_ms
                << " kernel_ms=" << eval.kernel_ms
                << " copyback_ms=" << eval.copyback_ms
                << " checksum=" << std::setprecision(17) << static_cast<double>(checksum) << "\n";
      return 0;
    }

    const auto all_t0 = std::chrono::steady_clock::now();
    const auto compile = compile_population(population, input_names);
    const EvalRun eval = evaluate_compiled_population(compile.programs, shared_case_bindings, expected_values, args);

    std::vector<ScoredGenome> scored;
    scored.reserve(population.size());
    long double fitness_sum = 0.0L;
    for (std::size_t i = 0; i < population.size(); ++i) {
      scored.push_back(ScoredGenome{population[i], eval.fitness[i]});
      fitness_sum += static_cast<long double>(eval.fitness[i]);
    }
    std::sort(scored.begin(), scored.end(), scored_genome_sorts_before);

    std::mt19937_64 rng(cfg.seed);
    const auto repro_t0 = std::chrono::steady_clock::now();
    const ReproductionRun reproduction = reproduce_population(scored, cfg, &rng);
    const auto repro_t1 = std::chrono::steady_clock::now();
    const auto all_t1 = std::chrono::steady_clock::now();

    const double repro_ms = std::chrono::duration<double, std::milli>(repro_t1 - repro_t0).count();
    const double total_ms = std::chrono::duration<double, std::milli>(all_t1 - all_t0).count();

    std::cout << "BENCH mode=one-gen-e2e engine=" << args.engine
              << " population_size=" << population.size()
              << " compile_ms=" << std::fixed << std::setprecision(3) << compile.compile_ms
              << " eval_ms=" << eval.eval_ms
              << " repro_ms=" << repro_ms
              << " selection_ms=" << reproduction.selection_ms
              << " crossover_ms=" << reproduction.crossover_ms
              << " mutation_ms=" << reproduction.mutation_ms
              << " total_ms=" << total_ms
              << " pack_upload_ms=" << eval.pack_upload_ms
              << " kernel_ms=" << eval.kernel_ms
              << " copyback_ms=" << eval.copyback_ms
              << " mean_fitness=" << std::setprecision(17)
              << static_cast<double>(fitness_sum / static_cast<long double>(eval.fitness.size()))
              << " best_fitness=" << scored.front().fitness
              << " best_program_key=" << scored.front().genome.meta.program_key << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
