#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"
#include "vm_cpu_cli/codec.hpp"
#include "vm_cpu_cli/json.hpp"
#include "vm_cpu_cli/options.hpp"
#ifdef G3PVM_HAS_CUDA
#include "g3pvm/vm_gpu.hpp"
#endif

// Keep vm_cpu_cli.cpp directly buildable by tests that compile only this file.
#include "vm_cpu_cli/json.cpp"
#include "vm_cpu_cli/codec.cpp"
#include "vm_cpu_cli/options.cpp"

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  try {
    const g3pvm::cli_detail::CliOptions cli_opts = g3pvm::cli_detail::parse_cli_options(argc, argv);

    std::stringstream buf;
    buf << std::cin.rdbuf();
    const std::string text = buf.str();
    if (text.empty()) {
      return 2;
    }

    g3pvm::cli_detail::JsonParser parser(text);
    g3pvm::cli_detail::JsonValue root = parser.parse();

    const g3pvm::cli_detail::JsonValue* req = &root;
    if (root.kind == g3pvm::cli_detail::JsonValue::Kind::Object) {
      auto it = root.object_v.find("bytecode_program_inputs");
      if (it != root.object_v.end()) {
        req = &it->second;
      }
    }

    if (req->kind != g3pvm::cli_detail::JsonValue::Kind::Object) {
      throw std::runtime_error("top-level JSON must be object");
    }

    auto fv_it = req->object_v.find("format_version");
    if (fv_it != req->object_v.end()) {
      const std::string format = g3pvm::cli_detail::require_string(fv_it->second, "format_version");
      if (format != "bytecode-json-v0.1") {
        throw std::runtime_error("unsupported format_version");
      }
    }

    const int fuel = g3pvm::cli_detail::require_int(
        g3pvm::cli_detail::require_object_field(*req, "fuel"), "fuel");
    std::string engine = cli_opts.engine;
    if (engine.empty()) {
#ifdef G3PVM_HAS_CUDA
      engine = "gpu";
#else
      engine = "cpu";
#endif
    }

    const int blocksize = cli_opts.blocksize;

    std::vector<g3pvm::BytecodeProgram> programs =
        g3pvm::cli_detail::decode_programs(g3pvm::cli_detail::require_object_field(*req, "programs"));
    std::vector<g3pvm::InputCase> shared_cases =
        g3pvm::cli_detail::decode_cases(g3pvm::cli_detail::require_object_field(*req, "shared_cases"));

    auto shared_answer_it = req->object_v.find("shared_answer");
    if (shared_answer_it != req->object_v.end()) {
      std::vector<g3pvm::Value> shared_answer = g3pvm::cli_detail::decode_shared_answer(shared_answer_it->second);

      std::vector<int> fitness;
      if (engine == "cpu") {
        fitness = g3pvm::run_bytecode_cpu_multi_fitness_shared_cases(programs, shared_cases, shared_answer, fuel);
      } else if (engine == "gpu") {
#ifdef G3PVM_HAS_CUDA
        const g3pvm::GPUFitnessEvalResult gpu_fit = g3pvm::run_bytecode_gpu_multi_fitness_shared_cases_debug(
            programs, shared_cases, shared_answer, fuel, blocksize);
        if (!gpu_fit.ok) {
          std::cout << "ERR " << g3pvm::err_code_name(gpu_fit.err.code) << "\n";
          if (!gpu_fit.err.message.empty()) {
            std::cout << "MSG " << gpu_fit.err.message << "\n";
          } else {
            std::cout << "MSG fitness evaluation failure\n";
          }
          return 0;
        }
        fitness = gpu_fit.fitness;
#else
        throw std::runtime_error("gpu unsupported");
#endif
      } else {
        throw std::runtime_error("unknown engine");
      }

      if (fitness.empty()) {
        std::cout << "ERR ValueError\n";
        std::cout << "MSG fitness evaluation failure\n";
        return 0;
      }
      std::cout << "OK fitness_count " << fitness.size() << "\n";
      for (std::size_t i = 0; i < fitness.size(); ++i) {
        std::cout << "FIT " << i << " " << fitness[i] << "\n";
      }
      return 0;
    }

    std::vector<std::vector<g3pvm::VMResult>> out;
    if (engine == "cpu") {
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
          per_prog.push_back(g3pvm::run_bytecode(programs[p], inputs, fuel));
        }
      }
    } else if (engine == "gpu") {
#ifdef G3PVM_HAS_CUDA
      out = g3pvm::run_bytecode_gpu_multi_batch(programs, shared_cases, fuel, blocksize);
#else
      throw std::runtime_error("gpu unsupported");
#endif
    } else {
      throw std::runtime_error("unknown engine");
    }

    if (out.size() == 1 && out[0].size() == 1) {
      const g3pvm::VMResult& result = out[0][0];
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
      for (const auto& r : per_prog) {
        if (r.is_error) {
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
