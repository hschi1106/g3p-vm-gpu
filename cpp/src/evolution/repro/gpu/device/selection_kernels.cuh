#pragma once

#include <cstdint>

#include "random_device.cuh"

namespace g3pvm::evo::repro {

__global__ void tournament_select_kernel(const double* fitness,
                                         int population_size,
                                         int pair_count,
                                         int candidates_per_program,
                                         int donor_pool_size,
                                         int tournament_k,
                                         double mutation_ratio,
                                         std::uint64_t seed,
                                         int* parent_a,
                                         int* parent_b,
                                         int* cand_a,
                                         int* cand_b,
                                         int* donor_idx,
                                         unsigned char* is_mutation) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= pair_count) {
    return;
  }
  const std::uint64_t base = hash64(seed + static_cast<std::uint64_t>(idx) * 0x9e3779b97f4a7c15ULL);
  auto run_tournament = [&](std::uint64_t local_seed) {
    int best = static_cast<int>(local_seed % static_cast<std::uint64_t>(population_size));
    double best_fit = fitness[best];
    for (int i = 1; i < tournament_k; ++i) {
      local_seed = hash64(local_seed + static_cast<std::uint64_t>(i));
      const int cand = static_cast<int>(local_seed % static_cast<std::uint64_t>(population_size));
      const double cand_fit = fitness[cand];
      if (cand_fit > best_fit) {
        best = cand;
        best_fit = cand_fit;
      }
    }
    return best;
  };

  const int pa = run_tournament(base);
  const int pb = run_tournament(hash64(base ^ 0x123456789abcdefULL));
  parent_a[idx] = pa;
  parent_b[idx] = pb;

  const std::uint64_t cseed = hash64(base ^ 0xfeedbeefULL);
  cand_a[idx] = static_cast<int>(cseed % static_cast<std::uint64_t>(candidates_per_program));
  cand_b[idx] = static_cast<int>((cseed >> 7) % static_cast<std::uint64_t>(candidates_per_program));
  donor_idx[idx] = static_cast<int>((cseed >> 17) % static_cast<std::uint64_t>(donor_pool_size));

  const double pick = static_cast<double>(cseed & 0xffffULL) / 65535.0;
  is_mutation[idx] = (pick < mutation_ratio) ? 1 : 0;
}

}  // namespace g3pvm::evo::repro
