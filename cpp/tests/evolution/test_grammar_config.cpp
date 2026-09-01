#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "g3pvm/cli/commands.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/evolution/ast_program.hpp"

namespace {

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool throws_with(const g3pvm::cli_detail::JsonValue& raw, const std::string& expected) {
  try {
    (void)g3pvm::cli_detail::decode_grammar_config_json(raw);
  } catch (const std::exception& ex) {
    return check(std::string(ex.what()).find(expected) != std::string::npos,
                 "unexpected grammar diagnostic: " + std::string(ex.what()));
  }
  return check(false, "invalid grammar config was accepted");
}

}  // namespace

int main(int argc, char** argv) {
  if (!check(argc == 2, "expected canonical grammar-config path")) return 1;
  std::ifstream input(argv[1]);
  if (!check(static_cast<bool>(input), "could not open canonical grammar config")) return 1;
  std::ostringstream text;
  text << input.rdbuf();

  const auto raw = g3pvm::cli_detail::JsonParser(text.str()).parse();
  const auto config = g3pvm::cli_detail::decode_grammar_config_json(raw);
  if (!check(config.statement_return && config.expression_const,
             "canonical config must retain required grammar forms")) return 1;
  if (!check(config.allows_type(g3pvm::evo::RType::Int) &&
                 !config.allows_type(g3pvm::evo::RType::Char),
             "canonical config value domain mismatch")) return 1;
  if (!check(config.allows_node_kind(g3pvm::evo::NodeKind::CALL_ABS) &&
                 !config.allows_node_kind(g3pvm::evo::NodeKind::CALL_CHR),
             "canonical config builtin domain mismatch")) return 1;

  auto unknown = raw;
  unknown.object_v["values"].object_v["integer"] = {};
  if (!throws_with(unknown, "unknown field: values.integer")) return 1;

  auto no_return = raw;
  no_return.object_v["statements"].object_v["return"].bool_v = false;
  if (!throws_with(no_return, "must enable statements.return")) return 1;

  auto no_numeric = raw;
  no_numeric.object_v["values"].object_v["int"].bool_v = false;
  no_numeric.object_v["values"].object_v["float"].bool_v = false;
  if (!throws_with(no_numeric, "must enable values.int or values.float")) return 1;

  auto no_calls = raw;
  no_calls.object_v["expressions"].object_v["call"].bool_v = false;
  const auto calls_disabled = g3pvm::cli_detail::decode_grammar_config_json(no_calls);
  if (!check(!calls_disabled.allows_node_kind(g3pvm::evo::NodeKind::CALL_ABS),
             "expressions.call=false must disable builtin nodes")) return 1;

  std::cout << "g3pvm_test_grammar_config: OK\n";
  return 0;
}
