#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"
#include "g3pvm/core/errors.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::CaseBindings;
using g3pvm::ExecResult;
using g3pvm::Value;
using g3pvm::ValueTag;
using g3pvm::cli_detail::JsonValue;

std::vector<std::pair<int, Value>> to_vm_inputs(const CaseBindings& one_case) {
  std::vector<std::pair<int, Value>> inputs;
  inputs.reserve(one_case.size());
  for (const auto& binding : one_case) inputs.push_back({binding.idx, binding.value});
  return inputs;
}

bool exact_value_equal(const Value& lhs, const Value& rhs) {
  if (lhs.tag != rhs.tag) return false;
  switch (lhs.tag) {
    case ValueTag::Int:
    case ValueTag::Char: return lhs.i == rhs.i;
    case ValueTag::Float: return lhs.f == rhs.f;
    case ValueTag::Bool: return lhs.b == rhs.b;
    case ValueTag::String: {
      std::string lhs_value;
      std::string rhs_value;
      if (g3pvm::payload::lookup_string(lhs, &lhs_value) &&
          g3pvm::payload::lookup_string(rhs, &rhs_value)) {
        return lhs_value == rhs_value;
      }
      return lhs.i == rhs.i;
    }
    case ValueTag::IntList:
    case ValueTag::FloatList:
    case ValueTag::StringList: {
      std::vector<Value> lhs_values;
      std::vector<Value> rhs_values;
      if (g3pvm::payload::lookup_list(lhs, &lhs_values) &&
          g3pvm::payload::lookup_list(rhs, &rhs_values)) {
        if (lhs_values.size() != rhs_values.size()) return false;
        for (std::size_t i = 0; i < lhs_values.size(); ++i) {
          if (!exact_value_equal(lhs_values[i], rhs_values[i])) return false;
        }
        return true;
      }
      return lhs.i == rhs.i;
    }
    case ValueTag::FallbackToken: return lhs.i == rhs.i;
    case ValueTag::Invalid: return true;
  }
  return false;
}

const JsonValue* optional_field(const JsonValue& object, const char* key) {
  if (object.kind != JsonValue::Kind::Object) throw std::runtime_error("expected object");
  const auto it = object.object_v.find(key);
  return it == object.object_v.end() ? nullptr : &it->second;
}

int optional_fuel(const JsonValue& object, int fallback) {
  const JsonValue* raw = optional_field(object, "fuel");
  return raw == nullptr ? fallback : g3pvm::cli_detail::require_int(*raw, "fuel");
}

std::string read_input(int argc, char** argv) {
  if (argc > 2) throw std::runtime_error("usage: g3pvm_test_vm_cli_harness [fixture.json]");
  std::stringstream buffer;
  if (argc == 2) {
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error(std::string("cannot open fixture: ") + argv[1]);
    buffer << input.rdbuf();
  } else {
    buffer << std::cin.rdbuf();
  }
  if (buffer.str().empty()) throw std::runtime_error("empty input");
  return buffer.str();
}

struct FixtureCounts {
  int scenarios = 0;
  int cases = 0;
  int passed = 0;
  int failed = 0;
};

void run_fixture_cases(const BytecodeProgram& program, const JsonValue& cases,
                       int fuel, const std::string& scenario_intent,
                       FixtureCounts* counts) {
  if (scenario_intent.empty()) throw std::runtime_error("fixture scenario intent must not be empty");
  if (fuel <= 0) throw std::runtime_error("fixture fuel must be positive");
  if (cases.kind != JsonValue::Kind::Array || cases.array_v.empty()) {
    throw std::runtime_error("fixture cases must be a non-empty array");
  }
  counts->scenarios += 1;
  for (const JsonValue& one_case : cases.array_v) {
    const CaseBindings bindings = g3pvm::cli_detail::decode_input_case(
        g3pvm::cli_detail::require_object_field(one_case, "inputs"));
    const JsonValue* expected_value = optional_field(one_case, "expected");
    const JsonValue* expected_error = optional_field(one_case, "expected_error");
    if ((expected_value == nullptr) == (expected_error == nullptr)) {
      throw std::runtime_error("fixture case requires exactly one of expected or expected_error");
    }

    const ExecResult result = g3pvm::execute_bytecode_cpu(program, to_vm_inputs(bindings), fuel);
    bool passed = false;
    if (expected_error != nullptr) {
      const std::string name =
          g3pvm::cli_detail::require_string(*expected_error, "expected_error");
      passed = result.is_error && name == g3pvm::err_code_name(result.err.code);
    } else {
      const Value expected = g3pvm::cli_detail::decode_typed_value(*expected_value);
      passed = !result.is_error && exact_value_equal(result.value, expected);
    }

    counts->cases += 1;
    if (passed) {
      counts->passed += 1;
    } else {
      counts->failed += 1;
      std::cerr << "FAIL fixture scenario: " << scenario_intent << " case "
                << counts->cases << "\n";
      if (result.is_error) {
        std::cerr << "  actual error: " << g3pvm::err_code_name(result.err.code)
                  << " (" << result.err.message << ")\n";
      } else {
        std::cerr << "  actual value tag: " << static_cast<int>(result.value.tag) << "\n";
      }
    }
  }
}

int run_fixture(const JsonValue& root, int default_fuel) {
  FixtureCounts counts;
  const JsonValue* scenarios = optional_field(root, "scenarios");
  if (scenarios != nullptr) {
    if (scenarios->kind != JsonValue::Kind::Array || scenarios->array_v.empty()) {
      throw std::runtime_error("fixture scenarios must be a non-empty array");
    }
    for (const JsonValue& scenario : scenarios->array_v) {
      const std::string intent = g3pvm::cli_detail::require_string(
          g3pvm::cli_detail::require_object_field(scenario, "intent"), "intent");
      const BytecodeProgram program = g3pvm::cli_detail::decode_program(
          g3pvm::cli_detail::require_object_field(scenario, "program"));
      run_fixture_cases(program, g3pvm::cli_detail::require_object_field(scenario, "cases"),
                        optional_fuel(scenario, default_fuel), intent, &counts);
    }
  } else {
    const BytecodeProgram program = g3pvm::cli_detail::decode_program(
        g3pvm::cli_detail::require_object_field(root, "program"));
    run_fixture_cases(program, g3pvm::cli_detail::require_object_field(root, "cases"),
                      default_fuel, "legacy single-program fixture", &counts);
  }

  std::cout << "OK fixture scenarios " << counts.scenarios << " cases " << counts.cases
            << " passed " << counts.passed << " failed " << counts.failed << "\n";
  return counts.failed == 0 ? 0 : 1;
}

int run_bytecode_request(const JsonValue& root, int fuel) {
  const std::vector<BytecodeProgram> programs = g3pvm::cli_detail::decode_programs(
      g3pvm::cli_detail::require_object_field(root, "programs"));
  const std::vector<CaseBindings> shared_cases = g3pvm::cli_detail::decode_cases(
      g3pvm::cli_detail::require_object_field(root, "shared_cases"));
  if (root.object_v.find("shared_answer") != root.object_v.end()) {
    std::cout << "ERR ValueError\nMSG test harness only supports raw cpu execution\n";
    return 0;
  }

  std::vector<std::vector<ExecResult>> output(programs.size());
  for (std::size_t p = 0; p < programs.size(); ++p) {
    for (const CaseBindings& one_case : shared_cases) {
      output[p].push_back(g3pvm::execute_bytecode_cpu(programs[p], to_vm_inputs(one_case), fuel));
    }
  }
  if (output.size() == 1 && output[0].size() == 1) {
    const ExecResult& result = output[0][0];
    if (result.is_error) {
      std::cout << "ERR " << g3pvm::err_code_name(result.err.code) << "\n";
      if (!result.err.message.empty()) std::cout << "MSG " << result.err.message << "\n";
    } else {
      std::cout << "OK ";
      g3pvm::cli_detail::print_value(result.value);
    }
    return 0;
  }

  int returned = 0;
  int errors = 0;
  for (const auto& per_program : output) {
    for (const ExecResult& result : per_program) result.is_error ? ++errors : ++returned;
  }
  std::cout << "OK programs " << output.size() << " cases "
            << returned + errors << " return " << returned << " error " << errors << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const JsonValue root = g3pvm::cli_detail::JsonParser(read_input(argc, argv)).parse();
    if (root.kind != JsonValue::Kind::Object) throw std::runtime_error("top-level JSON must be object");
    const std::string format = g3pvm::cli_detail::require_string(
        g3pvm::cli_detail::require_object_field(root, "format_version"), "format_version");
    if (format != "bytecode-json" && format != "bytecode-fixture") {
      throw std::runtime_error("unsupported format_version");
    }
    const int fuel = g3pvm::cli_detail::require_int(
        g3pvm::cli_detail::require_object_field(root, "fuel"), "fuel");
    return format == "bytecode-fixture" ? run_fixture(root, fuel)
                                         : run_bytecode_request(root, fuel);
  } catch (const std::exception& error) {
    std::cerr << "g3pvm_test_vm_cli_harness error: " << error.what() << "\n";
    return 2;
  }
}
