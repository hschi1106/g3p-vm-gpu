#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
  std::string cases_format = "auto";
  std::string input_indices = "auto";
  std::string input_names = "x";
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

std::vector<g3pvm::evo::FitnessCase> parse_simple_cases(const JsonValue& payload) {
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("input JSON must be object");
  }
  auto cases_it = payload.object_v.find("cases");
  if (cases_it == payload.object_v.end() || cases_it->second.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("input JSON must include list field: cases");
  }

  std::vector<g3pvm::evo::FitnessCase> out;
  for (const JsonValue& row : cases_it->second.array_v) {
    if (row.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("cases[i] must be object");
    }

    auto inputs_it = row.object_v.find("inputs");
    if (inputs_it == row.object_v.end()) {
      inputs_it = row.object_v.find("input");
    }
    if (inputs_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] missing inputs/input");
    }

    auto expected_it = row.object_v.find("expected");
    if (expected_it == row.object_v.end()) {
      expected_it = row.object_v.find("output");
    }
    if (expected_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] missing expected/output");
    }

    out.push_back(g3pvm::evo::FitnessCase{decode_inputs(inputs_it->second), decode_typed_or_raw_value(expected_it->second)});
  }

  return out;
}

std::map<int, Value> parse_idx_case_entries(const JsonValue& row) {
  if (row.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("shared_cases[i] must be a list");
  }
  std::map<int, Value> out;
  for (const JsonValue& item : row.array_v) {
    if (item.kind != JsonValue::Kind::Object) {
      throw std::runtime_error("shared_cases[i][j] must be object");
    }
    const JsonValue& idx_node = g3pvm::cli_detail::require_object_field(item, "idx");
    const JsonValue& value_node = g3pvm::cli_detail::require_object_field(item, "value");
    const int idx = g3pvm::cli_detail::require_int(idx_node, "idx");
    out[idx] = decode_typed_or_raw_value(value_node);
  }
  return out;
}

std::vector<int> parse_indices(const std::string& raw) {
  std::vector<int> out;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    out.push_back(std::stoi(token));
  }
  if (out.empty()) {
    throw std::runtime_error("input indices must not be empty");
  }
  return out;
}

std::vector<std::string> parse_input_names(const std::string& raw, int n) {
  if (raw.empty()) {
    if (n == 1) return {"x"};
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      out.push_back("x" + std::to_string(i));
    }
    return out;
  }

  std::vector<std::string> out;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  if (static_cast<int>(out.size()) != n) {
    throw std::runtime_error("--input-names count mismatch");
  }
  return out;
}

bool value_equal_exact(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::None) return true;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  if (a.tag == ValueTag::Int) return a.i == b.i;
  return a.f == b.f;
}

std::vector<int> auto_pick_indices(const std::vector<std::map<int, Value>>& case_maps,
                                   const std::vector<Value>& expected_vals) {
  std::set<int> idx_set;
  for (const auto& one : case_maps) {
    for (const auto& kv : one) {
      idx_set.insert(kv.first);
    }
  }
  if (idx_set.empty()) {
    throw std::runtime_error("shared_cases does not contain any idx values");
  }

  std::vector<int> idxs(idx_set.begin(), idx_set.end());
  if (std::find(idxs.begin(), idxs.end(), 0) != idxs.end() && idxs.size() > 1) {
    bool equal_all = true;
    for (std::size_t i = 0; i < case_maps.size(); ++i) {
      auto it = case_maps[i].find(0);
      if (it == case_maps[i].end() || !value_equal_exact(it->second, expected_vals[i])) {
        equal_all = false;
        break;
      }
    }
    if (equal_all) {
      std::vector<int> out;
      for (int idx : idxs) {
        if (idx != 0) {
          out.push_back(idx);
        }
      }
      return out;
    }
  }
  return idxs;
}

std::vector<g3pvm::evo::FitnessCase> parse_psb2_fixture_cases(const JsonValue& payload,
                                                               const std::string& input_indices_raw,
                                                               const std::string& input_names_raw) {
  const JsonValue* root = &payload;
  if (payload.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("fixture root must be object");
  }
  auto bpi_it = payload.object_v.find("bytecode_program_inputs");
  if (bpi_it != payload.object_v.end()) {
    root = &bpi_it->second;
  }
  if (root->kind != JsonValue::Kind::Object) {
    throw std::runtime_error("fixture root must be object");
  }

  const JsonValue& shared_cases_node = g3pvm::cli_detail::require_object_field(*root, "shared_cases");
  const JsonValue& shared_answer_node = g3pvm::cli_detail::require_object_field(*root, "shared_answer");
  if (shared_cases_node.kind != JsonValue::Kind::Array || shared_answer_node.kind != JsonValue::Kind::Array) {
    throw std::runtime_error("fixture must include shared_cases(list) and shared_answer(list)");
  }
  if (shared_cases_node.array_v.size() != shared_answer_node.array_v.size()) {
    throw std::runtime_error("shared_cases and shared_answer length mismatch");
  }

  std::vector<Value> expected_vals;
  expected_vals.reserve(shared_answer_node.array_v.size());
  for (const JsonValue& v : shared_answer_node.array_v) {
    expected_vals.push_back(decode_typed_or_raw_value(v));
  }

  std::vector<std::map<int, Value>> case_maps;
  case_maps.reserve(shared_cases_node.array_v.size());
  for (const JsonValue& row : shared_cases_node.array_v) {
    case_maps.push_back(parse_idx_case_entries(row));
  }

  std::vector<int> input_indices;
  if (input_indices_raw == "auto") {
    input_indices = auto_pick_indices(case_maps, expected_vals);
  } else {
    input_indices = parse_indices(input_indices_raw);
  }
  const std::vector<std::string> input_names = parse_input_names(input_names_raw, static_cast<int>(input_indices.size()));

  std::vector<g3pvm::evo::FitnessCase> out;
  out.reserve(case_maps.size());
  for (std::size_t i = 0; i < case_maps.size(); ++i) {
    g3pvm::evo::NamedInputs inputs;
    for (std::size_t j = 0; j < input_indices.size(); ++j) {
      const int idx = input_indices[j];
      const std::string& name = input_names[j];
      auto it = case_maps[i].find(idx);
      if (it == case_maps[i].end()) {
        throw std::runtime_error("shared_cases[i] missing requested idx");
      }
      inputs[name] = it->second;
    }
    out.push_back(g3pvm::evo::FitnessCase{inputs, expected_vals[i]});
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
    } else if (arg == "--cases-format") {
      opts.cases_format = need_value("--cases-format");
    } else if (arg == "--input-indices") {
      opts.input_indices = need_value("--input-indices");
    } else if (arg == "--input-names") {
      opts.input_names = need_value("--input-names");
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

    std::vector<g3pvm::evo::FitnessCase> cases;
    if (args.cases_format == "simple") {
      cases = parse_simple_cases(payload);
    } else if (args.cases_format == "psb2_fixture") {
      cases = parse_psb2_fixture_cases(payload, args.input_indices, args.input_names);
    } else if (args.cases_format == "auto") {
      if (payload.kind == JsonValue::Kind::Object && payload.object_v.find("cases") != payload.object_v.end()) {
        cases = parse_simple_cases(payload);
      } else {
        cases = parse_psb2_fixture_cases(payload, args.input_indices, args.input_names);
      }
    } else {
      throw std::runtime_error("invalid --cases-format");
    }

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
      std::cout << "TIMING phase=final_eval ms=" << std::fixed << std::setprecision(3)
                << run.timing.final_eval_ms << "\n";
      if (cfg.eval_engine == g3pvm::evo::EvalEngine::GPU) {
        std::cout << "TIMING phase=gpu_session_init ms=" << std::fixed << std::setprecision(3)
                  << run.timing.gpu_session_init_ms << "\n";
        std::cout << "TIMING phase=gpu_generations_program_compile_total ms=" << std::fixed
                  << std::setprecision(3) << run.timing.gpu_generations_program_compile_ms_total << "\n";
        std::cout << "TIMING phase=gpu_generations_pack_upload_total ms=" << std::fixed
                  << std::setprecision(3) << run.timing.gpu_generations_pack_upload_ms_total << "\n";
        std::cout << "TIMING phase=gpu_generations_kernel_total ms=" << std::fixed << std::setprecision(3)
                  << run.timing.gpu_generations_kernel_ms_total << "\n";
        std::cout << "TIMING phase=gpu_generations_copyback_total ms=" << std::fixed
                  << std::setprecision(3) << run.timing.gpu_generations_copyback_ms_total << "\n";
      }
      std::cout << "TIMING phase=total ms=" << std::fixed << std::setprecision(3)
                << run.timing.total_ms << "\n";
    }
    if (args.timing == "per_gen" || args.timing == "all") {
      for (std::size_t i = 0; i < run.timing.generation_total_ms.size(); ++i) {
        std::cout << "TIMING gen=" << std::setfill('0') << std::setw(3) << i << std::setfill(' ')
                  << " eval_ms=" << std::fixed << std::setprecision(3) << run.timing.generation_eval_ms[i]
                  << " repro_ms=" << run.timing.generation_repro_ms[i]
                  << " total_ms=" << run.timing.generation_total_ms[i] << "\n";
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
      out << "      \"gpu_generations_program_compile_ms_total\": "
          << run.timing.gpu_generations_program_compile_ms_total << ",\n";
      out << "      \"gpu_generations_pack_upload_ms_total\": " << run.timing.gpu_generations_pack_upload_ms_total
          << ",\n";
      out << "      \"gpu_generations_kernel_ms_total\": " << run.timing.gpu_generations_kernel_ms_total << ",\n";
      out << "      \"gpu_generations_copyback_ms_total\": " << run.timing.gpu_generations_copyback_ms_total
          << ",\n";
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

      out << "  \"timing\": {\n";
      out << "    \"generation_eval_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_eval_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_eval_ms[i];
      }
      out << "],\n";
      out << "    \"generation_repro_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_repro_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_repro_ms[i];
      }
      out << "],\n";
      out << "    \"generation_gpu_program_compile_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_gpu_program_compile_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_gpu_program_compile_ms[i];
      }
      out << "],\n";
      out << "    \"generation_gpu_pack_upload_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_gpu_pack_upload_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_gpu_pack_upload_ms[i];
      }
      out << "],\n";
      out << "    \"generation_gpu_kernel_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_gpu_kernel_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_gpu_kernel_ms[i];
      }
      out << "],\n";
      out << "    \"generation_gpu_copyback_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_gpu_copyback_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_gpu_copyback_ms[i];
      }
      out << "],\n";
      out << "    \"generation_total_ms\": [";
      for (std::size_t i = 0; i < run.timing.generation_total_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(17) << run.timing.generation_total_ms[i];
      }
      out << "]\n";
      out << "  },\n";

      out << "  \"final\": {\n";
      out << "    \"best_fitness\": " << std::setprecision(17) << result.best.fitness << ",\n";
      out << "    \"hash_key\": \"" << json_escape(result.best.genome.meta.hash_key) << "\",\n";
      out << "    \"ast_repr\": \"" << json_escape(g3pvm::evo::ast_to_string(result.best.genome.ast)) << "\"\n";
      out << "  }\n";
      out << "}\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "invalid cases payload: " << e.what() << "\n";
    return 2;
  }
}
