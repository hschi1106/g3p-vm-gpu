#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"

// Keep directly buildable.
#include "json.cpp"
#include "codec.cpp"

namespace {

using g3pvm::Value;
using g3pvm::cli_detail::JsonValue;
using g3pvm::evo::FitnessCase;
using g3pvm::evo::NamedInputs;

struct CliOptions {
  std::string cases_path;
  std::string out_json;
  int population_size = 1024;
  std::uint64_t seed_start = 0;
  int probe_cases = 32;
  double min_success_rate = 0.10;
  int max_attempts = 200000;
  int fuel = 20000;
  int max_expr_depth = 5;
  int max_stmts_per_block = 6;
  int max_total_nodes = 80;
  int max_for_k = 16;
  int max_call_args = 3;
};

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
    if (is_integer_number(v.number_v)) {
      return Value::from_int(static_cast<long long>(v.number_v));
    }
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

std::vector<FitnessCase> parse_cases_v1(const JsonValue& payload) {
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

  std::vector<FitnessCase> out;
  out.reserve(cases_it->second.array_v.size());
  for (const JsonValue& row : cases_it->second.array_v) {
    const auto inputs_it = row.object_v.find("inputs");
    const auto expected_it = row.object_v.find("expected");
    if (inputs_it == row.object_v.end() || expected_it == row.object_v.end()) {
      throw std::runtime_error("cases[i] must include inputs/expected");
    }
    out.push_back(FitnessCase{decode_inputs(inputs_it->second), decode_typed_or_raw_value(expected_it->second)});
  }
  if (out.empty()) {
    throw std::runtime_error("cases must not be empty");
  }
  return out;
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("missing file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::vector<std::string> collect_case_input_names(const std::vector<FitnessCase>& cases) {
  std::set<std::string> names;
  for (const auto& one_case : cases) {
    for (const auto& kv : one_case.inputs) {
      names.insert(kv.first);
    }
  }
  return std::vector<std::string>(names.begin(), names.end());
}

std::vector<std::pair<int, Value>> case_inputs_to_locals(const FitnessCase& one_case,
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
    } else if (arg == "--out-json") {
      opts.out_json = need_value("--out-json");
    } else if (arg == "--population-size") {
      opts.population_size = std::stoi(need_value("--population-size"));
    } else if (arg == "--seed-start") {
      opts.seed_start = static_cast<std::uint64_t>(std::stoull(need_value("--seed-start")));
    } else if (arg == "--probe-cases") {
      opts.probe_cases = std::stoi(need_value("--probe-cases"));
    } else if (arg == "--min-success-rate") {
      opts.min_success_rate = std::stod(need_value("--min-success-rate"));
    } else if (arg == "--max-attempts") {
      opts.max_attempts = std::stoi(need_value("--max-attempts"));
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
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opts.cases_path.empty()) throw std::runtime_error("--cases is required");
  if (opts.out_json.empty()) throw std::runtime_error("--out-json is required");
  if (opts.population_size <= 0) throw std::runtime_error("--population-size must be > 0");
  if (opts.probe_cases <= 0) throw std::runtime_error("--probe-cases must be > 0");
  if (opts.min_success_rate < 0.0 || opts.min_success_rate > 1.0) {
    throw std::runtime_error("--min-success-rate must be in [0, 1]");
  }
  if (opts.max_attempts <= 0) throw std::runtime_error("--max-attempts must be > 0");
  return opts;
}

struct AcceptedSeed {
  std::uint64_t seed = 0;
  int probe_successes = 0;
  int node_count = 0;
  std::string program_key;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions args = parse_cli(argc, argv);
    const JsonValue payload = g3pvm::cli_detail::JsonParser(read_text_file(args.cases_path)).parse();
    const std::vector<FitnessCase> cases = parse_cases_v1(payload);
    const std::vector<std::string> input_names = collect_case_input_names(cases);

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

    std::uint64_t candidate_seed = args.seed_start;
    int attempts = 0;
    while (static_cast<int>(accepted.size()) < args.population_size && attempts < args.max_attempts) {
      ++attempts;
      const g3pvm::evo::ProgramGenome genome = g3pvm::evo::make_random_genome(candidate_seed, limits);
      const g3pvm::BytecodeProgram program = g3pvm::evo::compile_for_eval_with_preset_locals(genome, input_names);

      int successes = 0;
      for (int i = 0; i < probe_case_count; ++i) {
        const g3pvm::ExecResult out =
            g3pvm::execute_bytecode_cpu(
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
      }
      ++candidate_seed;
    }

    if (static_cast<int>(accepted.size()) != args.population_size) {
      std::cerr << "failed to collect enough accepted genomes: accepted=" << accepted.size()
                << " attempts=" << attempts << "\n";
      return 1;
    }

    std::ofstream out(args.out_json);
    if (!out) {
      throw std::runtime_error("failed to open out-json path");
    }

    out << "{\n";
    out << "  \"format_version\": \"population-seeds-v1\",\n";
    out << "  \"cases_path\": \"" << args.cases_path << "\",\n";
    out << "  \"population_size\": " << args.population_size << ",\n";
    out << "  \"seed_start\": " << args.seed_start << ",\n";
    out << "  \"probe_cases\": " << probe_case_count << ",\n";
    out << "  \"min_success_rate\": " << std::setprecision(17) << args.min_success_rate << ",\n";
    out << "  \"fuel\": " << args.fuel << ",\n";
    out << "  \"limits\": {\n";
    out << "    \"max_expr_depth\": " << limits.max_expr_depth << ",\n";
    out << "    \"max_stmts_per_block\": " << limits.max_stmts_per_block << ",\n";
    out << "    \"max_total_nodes\": " << limits.max_total_nodes << ",\n";
    out << "    \"max_for_k\": " << limits.max_for_k << ",\n";
    out << "    \"max_call_args\": " << limits.max_call_args << "\n";
    out << "  },\n";
    out << "  \"seeds\": [\n";
    for (std::size_t i = 0; i < accepted.size(); ++i) {
      const auto& row = accepted[i];
      out << "    {\"seed\": " << row.seed
          << ", \"probe_successes\": " << row.probe_successes
          << ", \"node_count\": " << row.node_count
          << ", \"program_key\": \"" << json_escape(row.program_key) << "\"}";
      if (i + 1 < accepted.size()) {
        out << ",";
      }
      out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    std::cout << "OK accepted=" << accepted.size()
              << " attempts=" << attempts
              << " probe_cases=" << probe_case_count
              << " min_successes=" << min_successes
              << " out_json=" << args.out_json << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
