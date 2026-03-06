#pragma once

#include <string>

namespace g3pvm::cli_detail {

struct CliOptions {
  std::string engine;
  int blocksize = 256;
};

CliOptions parse_cli_options(int argc, char** argv);

}  // namespace g3pvm::cli_detail
