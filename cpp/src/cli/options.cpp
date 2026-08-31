#include "g3pvm/cli/options.hpp"

#include <stdexcept>

namespace g3pvm::cli_detail {

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions opts;
  auto parse_on_off = [](const std::string& raw, const char* flag) -> bool {
    if (raw == "on") return true;
    if (raw == "off") return false;
    throw std::runtime_error(std::string(flag) + " must be on or off");
  };

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
    } else if (arg == "--population-json") {
      opts.population_json = need_value("--population-json");
    } else if (arg == "--grammar-config") {
      opts.grammar_config_path = need_value("--grammar-config");
    } else if (arg == "--eval-ast-json") {
      opts.eval_ast_json = need_value("--eval-ast-json");
    } else if (arg == "--engine") {
      opts.engine = need_value("--engine");
    } else if (arg == "--repro-backend") {
      opts.repro_backend = need_value("--repro-backend");
    } else if (arg == "--cpu-repro-ablation") {
      opts.cpu_repro_ablation = need_value("--cpu-repro-ablation");
    } else if (arg == "--repro-overlap") {
      opts.repro_overlap = parse_on_off(need_value("--repro-overlap"), "--repro-overlap");
    } else if (arg == "--skip-final-eval") {
      opts.skip_final_eval = parse_on_off(need_value("--skip-final-eval"), "--skip-final-eval");
    } else if (arg == "--retain-final-population") {
      opts.retain_final_population =
          parse_on_off(need_value("--retain-final-population"), "--retain-final-population");
    } else if (arg == "--blocksize") {
      opts.blocksize = std::stoi(need_value("--blocksize"));
    } else if (arg == "--population-size") {
      opts.population_size = std::stoi(need_value("--population-size"));
    } else if (arg == "--generations") {
      opts.generations = std::stoi(need_value("--generations"));
    } else if (arg == "--mutation-rate") {
      opts.mutation_rate = std::stod(need_value("--mutation-rate"));
    } else if (arg == "--mutation-subtree-prob") {
      opts.mutation_subtree_prob = std::stod(need_value("--mutation-subtree-prob"));
    } else if (arg == "--penalty") {
      opts.penalty = std::stod(need_value("--penalty"));
    } else if (arg == "--selection-pressure") {
      opts.selection_pressure = std::stoi(need_value("--selection-pressure"));
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
  if (opts.repro_backend != "cpu" && opts.repro_backend != "gpu") {
    throw std::runtime_error("--repro-backend must be cpu or gpu");
  }
  if (opts.cpu_repro_ablation != "none" && opts.cpu_repro_ablation != "gpu_selection" &&
      opts.cpu_repro_ablation != "gpu_candidates" &&
      opts.cpu_repro_ablation != "gpu_coupled_donor") {
    throw std::runtime_error(
        "--cpu-repro-ablation must be one of: none|gpu_selection|gpu_candidates|gpu_coupled_donor");
  }
  if (opts.repro_backend != "cpu" && opts.cpu_repro_ablation != "none") {
    throw std::runtime_error("--cpu-repro-ablation requires --repro-backend cpu");
  }
  if (opts.blocksize <= 0) {
    throw std::runtime_error("--blocksize must be > 0");
  }
  if (opts.selection_pressure <= 0) {
    throw std::runtime_error("--selection-pressure must be > 0");
  }
  if (opts.mutation_subtree_prob < 0.0 || opts.mutation_subtree_prob > 1.0) {
    throw std::runtime_error("--mutation-subtree-prob must be in [0, 1]");
  }
  if (opts.penalty < 0.0) {
    throw std::runtime_error("--penalty must be >= 0");
  }
  if (opts.timing != "none" && opts.timing != "summary" &&
      opts.timing != "per_gen" && opts.timing != "all") {
    throw std::runtime_error("--timing must be one of: none|summary|per_gen|all");
  }
  return opts;
}

}  // namespace g3pvm::cli_detail
