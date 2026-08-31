#include <exception>
#include <iostream>

#include "g3pvm/cli/commands.hpp"
#include "g3pvm/cli/options.hpp"

int main(int argc, char** argv) {
  try {
    const g3pvm::cli_detail::CliOptions options =
        g3pvm::cli_detail::parse_cli_options(argc, argv);
    if (!options.eval_ast_json.empty()) {
      return g3pvm::cli_detail::run_eval_ast_command(options);
    }
    return g3pvm::cli_detail::run_evolve_command(options);
  } catch (const std::exception& error) {
    std::cerr << "g3pvm_evolve_cli error: " << error.what() << "\n";
    return 2;
  }
}
