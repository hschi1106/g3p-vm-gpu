#pragma once

#include <string>

#include "gagp/cli/json.hpp"
#include "gagp/cli/options.hpp"
#include "gagp/evolution/ast_program.hpp"
#include "gagp/evolution/grammar_config.hpp"

namespace gagp::cli_detail {

evo::AstProgram decode_ast_json(const JsonValue& raw);
std::string encode_ast_json(const evo::AstProgram& ast);
evo::GrammarConfig decode_grammar_config_json(const JsonValue& raw);
int run_eval_ast_command(const CliOptions& options);
int run_evolve_command(const CliOptions& options);

}  // namespace gagp::cli_detail
