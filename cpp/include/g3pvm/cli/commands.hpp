#pragma once

#include "g3pvm/cli/json.hpp"
#include "g3pvm/evolution/ast_program.hpp"
#include "g3pvm/cli/options.hpp"

namespace g3pvm::cli_detail {

evo::AstProgram decode_ast_json(const JsonValue& raw);
int run_eval_ast_command(const CliOptions& options);
int run_evolve_command(const CliOptions& options);

}  // namespace g3pvm::cli_detail
