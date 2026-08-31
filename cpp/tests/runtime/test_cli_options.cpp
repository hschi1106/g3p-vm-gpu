#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/cli/options.hpp"

namespace {

g3pvm::cli_detail::CliOptions parse(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) argv.push_back(arg.data());
  return g3pvm::cli_detail::parse_cli_options(static_cast<int>(argv.size()), argv.data());
}

void expect_error(const std::vector<std::string>& args, const std::string& message) {
  try {
    (void)parse(args);
  } catch (const std::runtime_error& error) {
    assert(error.what() == message);
    return;
  }
  assert(false && "expected parse failure");
}

}  // namespace

int main() {
  {
    const auto opts = parse({"g3pvm_evolve_cli", "--cases", "cases.json"});
    assert(opts.cases_path == "cases.json");
    assert(opts.engine == "cpu");
    assert(opts.repro_backend == "cpu");
    assert(opts.cpu_repro_ablation == "none");
    assert(!opts.repro_overlap);
    assert(!opts.skip_final_eval);
    assert(!opts.retain_final_population);
    assert(opts.blocksize == 1024);
    assert(opts.population_size == 64);
    assert(opts.generations == 40);
    assert(opts.mutation_rate == 0.5);
    assert(opts.mutation_subtree_prob == 0.8);
    assert(opts.penalty == 1.0);
    assert(opts.selection_pressure == 2);
    assert(opts.seed == 0);
    assert(opts.fuel == 20000);
    assert(opts.max_expr_depth == 7);
    assert(opts.max_stmts_per_block == 6);
    assert(opts.max_total_nodes == 80);
    assert(opts.max_for_k == 16);
    assert(opts.max_call_args == 3);
    assert(opts.show_program == "none");
    assert(opts.timing == "summary");
  }
  {
    const auto opts = parse({
        "g3pvm_evolve_cli", "--cases", "cases.json",
        "--population-json", "population.json", "--grammar-config", "grammar.json",
        "--eval-ast-json", "ast.json", "--engine", "gpu", "--repro-backend", "cpu",
        "--cpu-repro-ablation", "gpu_candidates", "--repro-overlap", "on",
        "--skip-final-eval", "on", "--retain-final-population", "on",
        "--blocksize", "256", "--population-size", "32", "--generations", "9",
        "--mutation-rate", "0.25", "--mutation-subtree-prob", "0.4",
        "--penalty", "2.5", "--selection-pressure", "4", "--seed", "42",
        "--fuel", "500", "--max-expr-depth", "8", "--max-stmts-per-block", "7",
        "--max-total-nodes", "100", "--max-for-k", "12", "--max-call-args", "5",
        "--show-program", "best", "--timing", "all", "--out-json", "run.json"});
    assert(opts.population_json == "population.json");
    assert(opts.grammar_config_path == "grammar.json");
    assert(opts.eval_ast_json == "ast.json");
    assert(opts.engine == "gpu");
    assert(opts.cpu_repro_ablation == "gpu_candidates");
    assert(opts.repro_overlap && opts.skip_final_eval && opts.retain_final_population);
    assert(opts.blocksize == 256 && opts.population_size == 32 && opts.generations == 9);
    assert(opts.mutation_rate == 0.25 && opts.mutation_subtree_prob == 0.4);
    assert(opts.penalty == 2.5 && opts.selection_pressure == 4 && opts.seed == 42);
    assert(opts.fuel == 500 && opts.max_expr_depth == 8 && opts.max_stmts_per_block == 7);
    assert(opts.max_total_nodes == 100 && opts.max_for_k == 12 && opts.max_call_args == 5);
    assert(opts.show_program == "best" && opts.timing == "all" && opts.out_json == "run.json");
  }

  expect_error({"cli"}, "--cases is required");
  expect_error({"cli", "--cases"}, "missing value for --cases");
  expect_error({"cli", "--cases", "x", "--unknown"}, "unknown argument: --unknown");
  expect_error({"cli", "--cases", "x", "--engine", "other"}, "--engine must be cpu or gpu");
  expect_error({"cli", "--cases", "x", "--repro-backend", "other"},
               "--repro-backend must be cpu or gpu");
  expect_error({"cli", "--cases", "x", "--cpu-repro-ablation", "other"},
               "--cpu-repro-ablation must be one of: none|gpu_selection|gpu_candidates|gpu_coupled_donor");
  expect_error({"cli", "--cases", "x", "--repro-backend", "gpu", "--cpu-repro-ablation", "gpu_selection"},
               "--cpu-repro-ablation requires --repro-backend cpu");
  expect_error({"cli", "--cases", "x", "--repro-overlap", "maybe"},
               "--repro-overlap must be on or off");
  expect_error({"cli", "--cases", "x", "--blocksize", "0"}, "--blocksize must be > 0");
  expect_error({"cli", "--cases", "x", "--selection-pressure", "0"},
               "--selection-pressure must be > 0");
  expect_error({"cli", "--cases", "x", "--mutation-subtree-prob", "1.1"},
               "--mutation-subtree-prob must be in [0, 1]");
  expect_error({"cli", "--cases", "x", "--penalty", "-1"}, "--penalty must be >= 0");
  expect_error({"cli", "--cases", "x", "--timing", "verbose"},
               "--timing must be one of: none|summary|per_gen|all");
}
