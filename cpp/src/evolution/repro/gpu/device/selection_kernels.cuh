#pragma once

#include <cstdint>

#include "random_device.cuh"

namespace g3pvm::evo::repro {

__device__ inline int d_clamp_tournament_size(int population_size, int tournament_k) {
  if (population_size <= 0) {
    return 0;
  }
  if (tournament_k < 1) {
    return 1;
  }
  return tournament_k > population_size ? population_size : tournament_k;
}

__device__ inline int d_round_winner_count(int population_size, int tournament_k) {
  return (population_size + tournament_k - 1) / tournament_k;
}

__device__ inline int d_gcd_int(int a, int b) {
  while (b != 0) {
    const int next = a % b;
    a = b;
    b = next;
  }
  return a < 0 ? -a : a;
}

__device__ inline int d_choose_stride(int population_size, std::uint64_t round_seed) {
  if (population_size <= 1) {
    return 1;
  }
  int stride = static_cast<int>(hash64(round_seed ^ 0x94d049bb133111ebULL) %
                                static_cast<std::uint64_t>(population_size));
  if (stride <= 0) {
    stride = 1;
  }
  while (d_gcd_int(stride, population_size) != 1) {
    ++stride;
    if (stride >= population_size) {
      stride = 1;
    }
  }
  return stride;
}

__device__ inline int d_permuted_index_for_round(int population_size,
                                                 int logical_index,
                                                 std::uint64_t round_seed) {
  if (population_size <= 1) {
    return 0;
  }
  const int offset =
      static_cast<int>(hash64(round_seed ^ 0xbf58476d1ce4e5b9ULL) % static_cast<std::uint64_t>(population_size));
  const int stride = d_choose_stride(population_size, round_seed);
  return (offset + logical_index * stride) % population_size;
}

__device__ inline int d_best_index_for_parent_slot(const double* fitness,
                                                   int population_size,
                                                   int tournament_k,
                                                   int slot_index,
                                                   std::uint64_t selection_seed) {
  const int clamped_tournament_k = d_clamp_tournament_size(population_size, tournament_k);
  const int winners_per_round = d_round_winner_count(population_size, clamped_tournament_k);
  const int round_index = slot_index / winners_per_round;
  const int slot_in_round = slot_index % winners_per_round;
  const int logical_begin = slot_in_round * clamped_tournament_k;
  const int logical_end =
      logical_begin + clamped_tournament_k < population_size ? logical_begin + clamped_tournament_k : population_size;
  const std::uint64_t round_seed =
      hash64(selection_seed + static_cast<std::uint64_t>(round_index + 1) * 0x9e3779b97f4a7c15ULL);

  int best = d_permuted_index_for_round(population_size, logical_begin, round_seed);
  double best_fit = fitness[best];
  for (int logical = logical_begin + 1; logical < logical_end; ++logical) {
    const int cand = d_permuted_index_for_round(population_size, logical, round_seed);
    const double cand_fit = fitness[cand];
    if (cand_fit > best_fit) {
      best = cand;
      best_fit = cand_fit;
    }
  }
  return best;
}

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

  const int slot_a = idx * 2;
  const int slot_b = slot_a + 1;
  parent_a[idx] = d_best_index_for_parent_slot(fitness, population_size, tournament_k, slot_a, seed);
  parent_b[idx] = d_best_index_for_parent_slot(fitness, population_size, tournament_k, slot_b, seed);

  const std::uint64_t cseed = hash64(seed + static_cast<std::uint64_t>(idx + 1) * 0x9e3779b97f4a7c15ULL);
  cand_a[idx] = static_cast<int>(cseed % static_cast<std::uint64_t>(candidates_per_program));
  cand_b[idx] = static_cast<int>((cseed >> 7) % static_cast<std::uint64_t>(candidates_per_program));
  donor_idx[idx] = static_cast<int>((cseed >> 17) % static_cast<std::uint64_t>(donor_pool_size));

  const double pick = static_cast<double>(cseed & 0xffffULL) / 65535.0;
  is_mutation[idx] = (pick < mutation_ratio) ? 1 : 0;
}

}  // namespace g3pvm::evo::repro
