#include "options.hpp"

#include <stdexcept>

namespace g3pvm::cli_detail {

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--engine") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --engine");
      }
      opts.engine = argv[++i];
      continue;
    }
    if (arg == "--blocksize") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --blocksize");
      }
      int value = 0;
      try {
        value = std::stoi(argv[++i]);
      } catch (...) {
        throw std::runtime_error("invalid --blocksize");
      }
      if (value <= 0) {
        throw std::runtime_error("invalid --blocksize");
      }
      opts.blocksize = value;
      continue;
    }
    throw std::runtime_error("unknown argument");
  }
  return opts;
}

}  // namespace g3pvm::cli_detail
