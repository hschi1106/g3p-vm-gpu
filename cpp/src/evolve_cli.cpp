#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/evo_ast.hpp"
#include "g3pvm/evolve.hpp"
#include "vm_cpu_cli/codec.hpp"
#include "vm_cpu_cli/json.hpp"

// Keep evolve_cli.cpp directly buildable in ad-hoc environments.
#include "vm_cpu_cli/json.cpp"
#include "vm_cpu_cli/codec.cpp"

namespace {

using g3pvm::Value;
using g3pvm::ValueTag;
using g3pvm::cli_detail::JsonValue;

struct CliOptions {
  std::string cases_path;
  std::string engine = "cpu";
  int blocksize = 256;
  int population_size = 64;
  int generations = 40;
  int elitism = 2;
  double mutation_rate = 0.5;
  double crossover_rate = 0.9;
  std::string crossover_method = "hybrid";
  std::string selection = "tournament";
  int tournament_k = 3;
  double truncation_ratio = 0.5;
  std::uint64_t seed = 0;
  int fuel = 20000;
  int max_expr_depth = 5;
  int max_stmts_per_block = 6;
  int max_total_nodes = 80;
  int max_for_k = 16;
  int max_call_args = 3;
  std::string show_program = "none";
  std::string timing = "summary";
  std::string out_json;
};

std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    if (c == '"') {
      oss << "\\\"";
    } else if (c == '\\') {
      oss << "\\\\";
    } else if (c == '\n') {
      oss << "\\n";
    } else if (c == '\r') {
      oss << "\\r";
    } else if (c == '\t') {
      oss << "\\t";
    } else {
      oss << c;
    }
  }
  return oss.str();
}

void write_value_json(std::ostream& out, const Value& v) {
  if (v.tag == ValueTag::None) {
    out << "null";
    return;
  }
  if (v.tag == ValueTag::Bool) {
    out << (v.b ? "true" : "false");
    return;
  }
  if (v.tag == ValueTag::Int) {
    out << v.i;
    return;
  }
  if (std::isfinite(v.f)) {
    out << std::setprecision(17) << v.f;
    return;
  }
  out << "null";
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

std::vector<g3pvm::evo::FitnessCase> parse_cases_v1(const JsonValue& payload) {
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("input JSON must be object");
  }

  auto fv_it = payload.object_v.find("format_version");
  if (fv_it == payload.object_v.end() || fv_it->second.kind != JsonValue::Kind::String ||
      fv_it->second.string_v != "fitness-cases-v1") {
    throw std::runtime_error("input JSON must include format_version=fitness-cases-v1");
  }

  auto cases_it = payload.object_v.find("cases");
  if (cases_it == payload.object_v.end() || cases_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input JSON must include list field: cases");
  }

  std::vector<g3pvm::evo::FitnessCase> out;
  out.reserve(cases_it->second.array_v.size());
  for (const JsonValue& row : cases_it->second.array_v) {
    if (row.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("cases[i] must be object");
    }
    auto inputs_it = row.object_v.find("inputs");
    auto expected_it = row.object_v.find("expected");
    if (inputs_it == row.object_v.end() || expected_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] must include inputs/expected");
    }
    out.push_back(g3pvm::evo::FitnessCase{decode_inputs(inputs_it->second),
                                          decode_typed_or_raw_value(expected_it->second)});
  }
  if (out.empty()) {
    throw std::runtime_error("cases must not be empty");
  }
  return out;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* key) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + key);
      }
      return argv[++i];
    };

    if (arg == "--cases") {
      opts.cases_path = need_value("--cases");
    } else if (arg == "--engine") {
      opts.engine = need_value("--engine");
    } else if (arg == "--blocksize") {
      opts.blocksize = std::stoi(need_value("--blocksize"));
    } else if (arg == "--population-size") {
      opts.population_size = std::stoi(need_value("--population-size"));
    } else if (arg == "--generations") {
      opts.generations = std::stoi(need_value("--generations"));
    } else if (arg == "--elitism") {
      opts.elitism = std::stoi(need_value("--elitism"));
    } else if (arg == "--mutation-rate") {
      opts.mutation_rate = std::stod(need_value("--mutation-rate"));
    } else if (arg == "--crossover-rate") {
      opts.crossover_rate = std::stod(need_value("--crossover-rate"));
    } else if (arg == "--crossover-method") {
      opts.crossover_method = need_value("--crossover-method");
    } else if (arg == "--selection") {
      opts.selection = need_value("--selection");
    } else if (arg == "--tournament-k") {
      opts.tournament_k = std::stoi(need_value("--tournament-k"));
    } else if (arg == "--truncation-ratio") {
      opts.truncation_ratio = std::stod(need_value("--truncation-ratio"));
    } else if (arg == "--seed") {
      opts.seed = static_cast<std::uint64_t>(std::stoull(need_value("--seed")));
    } else if (arg == "--fuel") {
      opts.fuel = std::stoi(need_value("--fuel"));
    } else if (arg == "--max-expr-depth") {
      opts.max_expr_depth = std::stoi(need_value("--max-expr-depth"));
    } else if (arg == "--max-stmts-per-block") {
      opts.max_stmts_per_block = std::stoi(need_value("--max-stmts-per-block"));
    } else if (arg == "--max-total-nodes") {
      opts.max_total_nodes = std::stoi(need_value("--max-total-nodes"));
    } else if (arg == "--max-for-k") {
      opts.max_for_k = std::stoi(need_value("--max-for-k"));
    } else if (arg == "--max-call-args") {
      opts.max_call_args = std::stoi(need_value("--max-call-args"));
    } else if (arg == "--show-program") {
      opts.show_program = need_value("--show-program");
    } else if (arg == "--timing") {
      opts.timing = need_value("--timing");
    } else if (arg == "--out-json") {
      opts.out_json = need_value("--out-json");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opts.cases_path.empty()) {
    throw std::runtime_error("--cases is required");
  }
  if (opts.engine != "cpu" && opts.engine != "gpu") {
    throw std::runtime_error("--engine must be cpu or gpu");
  }
  if (opts.blocksize <= 0) {
    throw std::runtime_error("--blocksize must be > 0");
  }
  if (opts.timing != "none" && opts.timing != "summary" && opts.timing != "per_gen" && opts.timing != "all") {
    throw std::runtime_error("--timing must be one of: none|summary|per_gen|all");
  }

  return opts;
}

g3pvm::evo::SelectionMethod parse_selection(const std::string& s) {
  if (s == "tournament") return g3pvm::evo::SelectionMethod::Tournament;
  if (s == "roulette") return g3pvm::evo::SelectionMethod::Roulette;
  if (s == "rank") return g3pvm::evo::SelectionMethod::Rank;
  if (s == "truncation") return g3pvm::evo::SelectionMethod::Truncation;
  if (s == "random") return g3pvm::evo::SelectionMethod::Random;
  throw std::runtime_error("unknown selection method");
}

g3pvm::evo::CrossoverMethod parse_crossover_method(const std::string& s) {
  if (s == "top_level_splice") return g3pvm::evo::CrossoverMethod::TopLevelSplice;
  if (s == "typed_subtree") return g3pvm::evo::CrossoverMethod::TypedSubtree;
  if (s == "hybrid") return g3pvm::evo::CrossoverMethod::Hybrid;
  throw std::runtime_error("unknown crossover method");
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("missing cases file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions args = parse_cli(argc, argv);
    const std::string text = read_text_file(args.cases_path);

    g3pvm::cli_detail::JsonParser parser(text);
    const JsonValue payload = parser.parse();
    const std::vector<g3pvm::evo::FitnessCase> cases = parse_cases_v1(payload);

    g3pvm::evo::EvolutionConfig cfg;
    cfg.population_size = args.population_size;
    cfg.generations = args.generations;
    cfg.elitism = args.elitism;
    cfg.mutation_rate = args.mutation_rate;
    cfg.crossover_rate = args.crossover_rate;
    cfg.crossover_method = parse_crossover_method(args.crossover_method);
    cfg.selection_method = parse_selection(args.selection);
    cfg.eval_engine = (args.engine == "gpu") ? g3pvm::evo::EvalEngine::GPU : g3pvm::evo::EvalEngine::CPU;
    cfg.gpu_blocksize = args.blocksize;
    cfg.tournament_k = args.tournament_k;
    cfg.truncation_ratio = args.truncation_ratio;
    cfg.seed = args.seed;
    cfg.fuel = args.fuel;
    cfg.limits = g3pvm::evo::Limits{args.max_expr_depth,
                                    args.max_stmts_per_block,
                                    args.max_total_nodes,
                                    args.max_for_k,
                                    args.max_call_args};

    const g3pvm::evo::EvolutionRun run = g3pvm::evo::evolve_population_profiled(cases, cfg);
    const g3pvm::evo::EvolutionResult& result = run.result;

    struct HistoryRow {
      int generation = 0;
      double best_fitness = 0.0;
      double mean_fitness = 0.0;
      std::string hash_key;
    };
    std::vector<HistoryRow> history_rows;
    history_rows.reserve(result.history_best.size());

    for (int i = 0; i < static_cast<int>(result.history_best.size()); ++i) {
      const auto& best = result.history_best[static_cast<std::size_t>(i)];
      const double best_fit = result.history_best_fitness[static_cast<std::size_t>(i)];
      const double mean_fit = result.history_mean_fitness[static_cast<std::size_t>(i)];
      history_rows.push_back(HistoryRow{i, best_fit, mean_fit, best.genome.meta.hash_key});

      std::cout << "GEN " << std::setfill('0') << std::setw(3) << i << std::setfill(' ') << " best="
                << std::fixed << std::setprecision(6) << best_fit << " mean=" << std::fixed
                << std::setprecision(6) << mean_fit << " hash=" << best.genome.meta.hash_key << "\n";

      if (args.show_program == "ast" || args.show_program == "both") {
        std::cout << "AST " << std::setfill('0') << std::setw(3) << i << std::setfill(' ') << ": "
                  << g3pvm::evo::ast_to_string(best.genome.ast) << "\n";
      }
      if (args.show_program == "bytecode" || args.show_program == "both") {
        const g3pvm::BytecodeProgram bc = g3pvm::evo::compile_for_eval(best.genome);
        std::cout << "BYTECODE " << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                  << ": n_locals=" << bc.n_locals << " consts=" << bc.consts.size() << " code="
                  << bc.code.size() << "\n";

        std::cout << "BYTECODE_HEAD";
        const std::size_t cap = std::min<std::size_t>(12, bc.code.size());
        for (std::size_t j = 0; j < cap; ++j) {
          std::cout << " " << j << ":" << bc.code[j].op;
        }
        std::cout << "\n";
      }
    }

    std::cout << "FINAL best=" << std::fixed << std::setprecision(6) << result.best.fitness
              << " hash=" << result.best.genome.meta.hash_key
              << " selection=" << g3pvm::evo::selection_method_name(cfg.selection_method)
              << " crossover=" << g3pvm::evo::crossover_method_name(cfg.crossover_method) << "\n";

    if (args.timing == "summary" || args.timing == "all") {
      double gen_eval_sum = 0.0;
      double gen_repro_sum = 0.0;
      for (double v : run.timing.generation_eval_ms) gen_eval_sum += v;
      for (double v : run.timing.generation_repro_ms) gen_repro_sum += v;
      std::cout << "TIMING phase=init_population ms=" << std::fixed << std::setprecision(3)
                << run.timing.init_population_ms << "\n";
      std::cout << "TIMING phase=generations_eval_total ms=" << std::fixed << std::setprecision(3)
                << gen_eval_sum << "\n";
      std::cout << "TIMING phase=generations_repro_total ms=" << std::fixed << std::setprecision(3)
                << gen_repro_sum << "\n";
      std::cout << "TIMING phase=generations_selection_total ms=" << std::fixed << std::setprecision(3)
                << run.timing.generations_selection_ms_total << "\n";
      std::cout << "TIMING phase=generations_crossover_total ms=" << std::fixed << std::setprecision(3)
                << run.timing.generations_crossover_ms_total << "\n";
      std::cout << "TIMING phase=generations_mutation_total ms=" << std::fixed << std::setprecision(3)
                << run.timing.generations_mutation_ms_total << "\n";
      std::cout << "TIMING phase=generations_elite_copy_total ms=" << std::fixed << std::setprecision(3)
                << run.timing.generations_elite_copy_ms_total << "\n";
      std::cout << "TIMING phase=cpu_generations_program_compile_total ms=" << std::fixed << std::setprecision(3)
                << run.timing.cpu_generations_program_compile_ms_total << "\n";
      std::cout << "TIMING phase=final_eval ms=" << std::fixed << std::setprecision(3)
                << run.timing.final_eval_ms << "\n";
      if (cfg.eval_engine == g3pvm::evo::EvalEngine::GPU) {
        std::cout << "TIMING phase=gpu_session_init ms=" << std::fixed << std::setprecision(3)
                  << run.timing.gpu_session_init_ms << "\n";
        std::cout << "TIMING phase=gpu_generations_program_compile_total ms=" << std::fixed
                  << std::setprecision(3) << run.timing.gpu_generations_program_compile_ms_total << "\n";
        std::cout << "TIMING phase=gpu_generations_pack_upload_total ms=" << std::fixed << std::setprecision(3)
                  << run.timing.gpu_generations_pack_upload_ms_total << "\n";
        std::cout << "TIMING phase=gpu_generations_kernel_total ms=" << std::fixed << std::setprecision(3)
                  << run.timing.gpu_generations_kernel_ms_total << "\n";
        std::cout << "TIMING phase=gpu_generations_copyback_total ms=" << std::fixed << std::setprecision(3)
                  << run.timing.gpu_generations_copyback_ms_total << "\n";
      }
      std::cout << "TIMING phase=total ms=" << std::fixed << std::setprecision(3)
                << run.timing.total_ms << "\n";
    }

    if (args.timing == "per_gen" || args.timing == "all") {
      for (std::size_t i = 0; i < run.timing.generation_total_ms.size(); ++i) {
        std::cout << "TIMING gen=" << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                  << " eval_ms=" << std::fixed << std::setprecision(3) << run.timing.generation_eval_ms[i]
                  << " repro_ms=" << run.timing.generation_repro_ms[i]
                  << " total_ms=" << run.timing.generation_total_ms[i]
                  << " selection_ms=" << run.timing.generation_selection_ms[i]
                  << " crossover_ms=" << run.timing.generation_crossover_ms[i]
                  << " mutation_ms=" << run.timing.generation_mutation_ms[i]
                  << " elite_ms=" << run.timing.generation_elite_copy_ms[i]
                  << " cpu_compile_ms=" << run.timing.generation_cpu_program_compile_ms[i] << "\n";
        if (cfg.eval_engine == g3pvm::evo::EvalEngine::GPU) {
          std::cout << "TIMING gpu_gen=" << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                    << " compile_ms=" << std::fixed << std::setprecision(3)
                    << run.timing.generation_gpu_program_compile_ms[i]
                    << " pack_upload_ms=" << run.timing.generation_gpu_pack_upload_ms[i]
                    << " kernel_ms=" << run.timing.generation_gpu_kernel_ms[i]
                    << " copyback_ms=" << run.timing.generation_gpu_copyback_ms[i] << "\n";
        }
      }
    }

    if (!args.out_json.empty()) {
      std::ofstream out(args.out_json);
      if (!out) {
        throw std::runtime_error("failed to open out-json path");
      }

      out << "{\n";
      out << "  \"meta\": {\n";
      out << "    \"cases_path\": \"" << json_escape(args.cases_path) << "\",\n";
      out << "    \"population_size\": " << cfg.population_size << ",\n";
      out << "    \"generations\": " << cfg.generations << ",\n";
      out << "    \"selection\": \"" << g3pvm::evo::selection_method_name(cfg.selection_method) << "\",\n";
      out << "    \"crossover_method\": \"" << g3pvm::evo::crossover_method_name(cfg.crossover_method)
          << "\",\n";
      out << "    \"eval_engine\": \"" << g3pvm::evo::eval_engine_name(cfg.eval_engine) << "\",\n";
      out << "    \"gpu_blocksize\": " << cfg.gpu_blocksize << ",\n";
      out << "    \"seed\": " << cfg.seed << ",\n";
      out << "    \"timing\": {\n";
      out << "      \"init_population_ms\": " << std::setprecision(17) << run.timing.init_population_ms << ",\n";
      out << "      \"gpu_session_init_ms\": " << run.timing.gpu_session_init_ms << ",\n";
      out << "      \"final_eval_ms\": " << run.timing.final_eval_ms << ",\n";
      out << "      \"cpu_generations_program_compile_ms_total\": "
          << run.timing.cpu_generations_program_compile_ms_total << ",\n";
      out << "      \"gpu_generations_program_compile_ms_total\": "
          << run.timing.gpu_generations_program_compile_ms_total << ",\n";
      out << "      \"gpu_generations_pack_upload_ms_total\": " << run.timing.gpu_generations_pack_upload_ms_total
          << ",\n";
      out << "      \"gpu_generations_kernel_ms_total\": " << run.timing.gpu_generations_kernel_ms_total << ",\n";
      out << "      \"gpu_generations_copyback_ms_total\": " << run.timing.gpu_generations_copyback_ms_total
          << ",\n";
      out << "      \"generations_selection_ms_total\": " << run.timing.generations_selection_ms_total << ",\n";
      out << "      \"generations_crossover_ms_total\": " << run.timing.generations_crossover_ms_total << ",\n";
      out << "      \"generations_mutation_ms_total\": " << run.timing.generations_mutation_ms_total << ",\n";
      out << "      \"generations_elite_copy_ms_total\": " << run.timing.generations_elite_copy_ms_total << ",\n";
      out << "      \"total_ms\": " << run.timing.total_ms << "\n";
      out << "    }\n";
      out << "  },\n";

      out << "  \"history\": [\n";
      for (std::size_t i = 0; i < history_rows.size(); ++i) {
        const auto& row = history_rows[i];
        out << "    {\"generation\": " << row.generation << ", \"best_fitness\": "
            << std::setprecision(17) << row.best_fitness << ", \"mean_fitness\": " << row.mean_fitness
            << ", \"hash_key\": \"" << json_escape(row.hash_key) << "\"}";
        if (i + 1 < history_rows.size()) {
          out << ",";
        }
        out << "\n";
      }
      out << "  ],\n";

      auto dump_vec = [&](const char* name, const std::vector<double>& values, bool last) {
        out << "    \"" << name << "\": [";
        for (std::size_t i = 0; i < values.size(); ++i) {
          if (i > 0) out << ", ";
          out << std::setprecision(17) << values[i];
        }
        out << "]";
        if (!last) out << ",";
        out << "\n";
      };

      out << "  \"timing\": {\n";
      dump_vec("generation_eval_ms", run.timing.generation_eval_ms, false);
      dump_vec("generation_repro_ms", run.timing.generation_repro_ms, false);
      dump_vec("generation_cpu_program_compile_ms", run.timing.generation_cpu_program_compile_ms, false);
      dump_vec("generation_gpu_program_compile_ms", run.timing.generation_gpu_program_compile_ms, false);
      dump_vec("generation_gpu_pack_upload_ms", run.timing.generation_gpu_pack_upload_ms, false);
      dump_vec("generation_gpu_kernel_ms", run.timing.generation_gpu_kernel_ms, false);
      dump_vec("generation_gpu_copyback_ms", run.timing.generation_gpu_copyback_ms, false);
      dump_vec("generation_selection_ms", run.timing.generation_selection_ms, false);
      dump_vec("generation_crossover_ms", run.timing.generation_crossover_ms, false);
      dump_vec("generation_mutation_ms", run.timing.generation_mutation_ms, false);
      dump_vec("generation_elite_copy_ms", run.timing.generation_elite_copy_ms, false);
      dump_vec("generation_total_ms", run.timing.generation_total_ms, true);
      out << "  },\n";

      out << "  \"final\": {\n";
      out << "    \"best_fitness\": " << std::setprecision(17) << result.best.fitness << ",\n";
      out << "    \"hash_key\": \"" << json_escape(result.best.genome.meta.hash_key) << "\",\n";
      out << "    \"ast_repr\": \"" << json_escape(g3pvm::evo::ast_to_string(result.best.genome.ast)) << "\",\n";
      out << "    \"ast_names\": [";
      for (std::size_t i = 0; i < result.best.genome.ast.names.size(); ++i) {
        if (i > 0) out << ", ";
        out << "\"" << json_escape(result.best.genome.ast.names[i]) << "\"";
      }
      out << "],\n";
      out << "    \"ast_consts\": [";
      for (std::size_t i = 0; i < result.best.genome.ast.consts.size(); ++i) {
        if (i > 0) out << ", ";
        write_value_json(out, result.best.genome.ast.consts[i]);
      }
      out << "]\n";
      out << "  }\n";
      out << "}\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "invalid cases payload: " << e.what() << "\n";
    return 2;
  }
}
