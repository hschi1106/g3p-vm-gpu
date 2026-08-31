#pragma once

#include "g3pvm/cli/options.hpp"

namespace g3pvm::cli_detail {

int run_eval_ast_command(const CliOptions& options);
int run_evolve_command(const CliOptions& options);

}  // namespace g3pvm::cli_detail
