#pragma once

#include <string>
#include "gagp/evolution/ast_program.hpp"

namespace gagp::evo {

struct GenomeMeta {
  int node_count = 0;
  int max_depth = 0;
  bool uses_builtins = false;
  std::string program_key;
};

struct ProgramGenome {
  AstProgram ast;
  GenomeMeta meta;
};

GenomeMeta build_genome_meta(const AstProgram& ast);

}  // namespace gagp::evo
