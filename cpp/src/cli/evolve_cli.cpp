#include <exception>
#include <iostream>

#include "gagp/cli/commands.hpp"
#include "gagp/cli/options.hpp"

int main(int argc, char** argv) {
  try {
    const gagp::cli_detail::CliOptions options =
        gagp::cli_detail::parse_cli_options(argc, argv);
    if (!options.eval_ast_json.empty()) {
      return gagp::cli_detail::run_eval_ast_command(options);
    }
    return gagp::cli_detail::run_evolve_command(options);
  } catch (const std::exception& error) {
    std::cerr << "gagp_evolve_cli error: " << error.what() << "\n";
    return 2;
  }
}
