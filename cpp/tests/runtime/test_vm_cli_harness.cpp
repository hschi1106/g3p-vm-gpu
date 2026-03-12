#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/core/errors.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"

// Keep this harness directly buildable by the Python parity test.
#include "../../src/cli/json.cpp"
#include "../../src/cli/codec.cpp"

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  try {
    std::stringstream buf;
    buf << std::cin.rdbuf();
    const std::string text = buf.str();
    if (text.empty()) {
      return 2;
    }

    g3pvm::cli_detail::JsonParser parser(text);
    const g3pvm::cli_detail::JsonValue root = parser.parse();
    if (root.kind != g3pvm::cli_detail::JsonValue::Kind::Object) {
      throw std::runtime_error("top-level JSON must be object");
    }

    auto format_it = root.object_v.find("format_version");
    if (format_it != root.object_v.end()) {
      const std::string format = g3pvm::cli_detail::require_string(format_it->second, "format_version");
      if (format != "bytecode-json-v0.1") {
        throw std::runtime_error("unsupported format_version");
      }
    }

    const int fuel = g3pvm::cli_detail::require_int(
        g3pvm::cli_detail::require_object_field(root, "fuel"), "fuel");
    const std::vector<g3pvm::BytecodeProgram> programs =
        g3pvm::cli_detail::decode_programs(g3pvm::cli_detail::require_object_field(root, "programs"));
    const std::vector<g3pvm::CaseBindings> shared_cases =
        g3pvm::cli_detail::decode_cases(g3pvm::cli_detail::require_object_field(root, "shared_cases"));

    if (root.object_v.find("shared_answer") != root.object_v.end()) {
      std::cout << "ERR ValueError\n";
      std::cout << "MSG test harness only supports raw cpu execution\n";
      return 0;
    }

    std::vector<std::vector<g3pvm::ExecResult>> out;
    out.resize(programs.size());
    for (std::size_t p = 0; p < programs.size(); ++p) {
      auto& per_prog = out[p];
      per_prog.reserve(shared_cases.size());
      for (const auto& one_case : shared_cases) {
        std::vector<std::pair<int, g3pvm::Value>> inputs;
        inputs.reserve(one_case.size());
        for (const auto& binding : one_case) {
          inputs.push_back({binding.idx, binding.value});
        }
        per_prog.push_back(g3pvm::execute_bytecode_cpu(programs[p], inputs, fuel));
      }
    }

    if (out.size() == 1 && out[0].size() == 1) {
      const g3pvm::ExecResult& result = out[0][0];
      if (result.is_error) {
        std::cout << "ERR " << g3pvm::err_code_name(result.err.code) << "\n";
        if (!result.err.message.empty()) {
          std::cout << "MSG " << result.err.message << "\n";
        }
        return 0;
      }
      std::cout << "OK ";
      g3pvm::cli_detail::print_value(result.value);
      return 0;
    }

    int total = 0;
    int ok = 0;
    int err = 0;
    for (const auto& per_prog : out) {
      total += static_cast<int>(per_prog.size());
      for (const auto& result : per_prog) {
        if (result.is_error) {
          err += 1;
        } else {
          ok += 1;
        }
      }
    }
    std::cout << "OK programs " << out.size() << " cases " << total << " return " << ok << " error " << err
              << "\n";
    return 0;
  } catch (const std::exception&) {
    return 2;
  }
}
