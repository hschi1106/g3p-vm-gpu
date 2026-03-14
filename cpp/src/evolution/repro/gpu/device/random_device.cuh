#pragma once

#include <cstdint>

namespace g3pvm::evo::repro {

__host__ __device__ inline std::uint64_t hash64(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

}  // namespace g3pvm::evo::repro
