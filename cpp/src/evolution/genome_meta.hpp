#pragma once

#include <string>

#include "g3pvm/evolution/genome.hpp"

namespace g3pvm::evo::genome_meta {

GenomeMeta build_meta_fast(const AstProgram& ast);

}  // namespace g3pvm::evo::genome_meta
