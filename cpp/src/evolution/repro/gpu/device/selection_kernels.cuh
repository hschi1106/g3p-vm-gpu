#pragma once

#include <cstdint>

#include "pack_types.cuh"
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

__device__ inline unsigned int d_type_bit(RType type) {
  switch (type) {
    case RType::Int:
      return 1u << 0;
    case RType::Float:
      return 1u << 1;
    case RType::Bool:
      return 1u << 2;
    case RType::Char:
      return 1u << 3;
    case RType::String:
      return 1u << 4;
    case RType::IntList:
      return 1u << 5;
    case RType::FloatList:
      return 1u << 6;
    case RType::StringList:
      return 1u << 7;
    case RType::Any:
      return 1u << 8;
    default:
      return 0u;
  }
}

__device__ inline RType d_type_from_bit(unsigned int bit) {
  switch (bit) {
    case 0:
      return RType::Int;
    case 1:
      return RType::Float;
    case 2:
      return RType::Bool;
    case 3:
      return RType::Char;
    case 4:
      return RType::String;
    case 5:
      return RType::IntList;
    case 6:
      return RType::FloatList;
    case 7:
      return RType::StringList;
    case 8:
      return RType::Any;
    default:
      return RType::Invalid;
  }
}

__device__ inline bool d_candidate_is_valid(const DCandidateRange& candidate) {
  return candidate.start >= 0 && candidate.stop > candidate.start && d_type_bit(static_cast<RType>(candidate.aux)) != 0u;
}

__device__ inline bool d_candidate_keys_compatible(const DCandidateRange& a, const DCandidateRange& b) {
  return d_candidate_is_valid(a) &&
         d_candidate_is_valid(b) &&
         a.aux == b.aux &&
         a.scope_signature == b.scope_signature &&
         a.binder_signature == b.binder_signature &&
         a.scheme_kind == b.scheme_kind &&
         a.phase_name == b.phase_name &&
         a.visible_env_signature == b.visible_env_signature &&
         a.dp_dependency_arity == b.dp_dependency_arity;
}

__device__ inline int d_pick_candidate_index_for_type(const DCandidateRange* candidates,
                                                      int candidates_per_program,
                                                      RType target_type,
                                                      std::uint64_t seed) {
  int match_count = 0;
  for (int i = 0; i < candidates_per_program; ++i) {
    const DCandidateRange candidate = candidates[i];
    if (d_candidate_is_valid(candidate) && static_cast<RType>(candidate.aux) == target_type) {
      ++match_count;
    }
  }
  if (match_count <= 0) {
    return 0;
  }

  const int target_rank = static_cast<int>(seed % static_cast<std::uint64_t>(match_count));
  int seen = 0;
  for (int i = 0; i < candidates_per_program; ++i) {
    const DCandidateRange candidate = candidates[i];
    if (!d_candidate_is_valid(candidate) || static_cast<RType>(candidate.aux) != target_type) {
      continue;
    }
    if (seen == target_rank) {
      return i;
    }
    ++seen;
  }
  return 0;
}

__device__ inline void d_choose_typed_candidate_pair(const DCandidateRange* candidates,
                                                     int candidates_per_program,
                                                     int parent_a_index,
                                                     int parent_b_index,
                                                     std::uint64_t seed,
                                                     int* cand_a_out,
                                                     int* cand_b_out) {
  const DCandidateRange* candidates_a = candidates + parent_a_index * candidates_per_program;
  const DCandidateRange* candidates_b = candidates + parent_b_index * candidates_per_program;

  int compatible_count = 0;
  for (int i = 0; i < candidates_per_program; ++i) {
    for (int j = 0; j < candidates_per_program; ++j) {
      if (d_candidate_keys_compatible(candidates_a[i], candidates_b[j])) {
        compatible_count += 1;
      }
    }
  }

  if (compatible_count <= 0) {
    *cand_a_out = 0;
    *cand_b_out = 0;
    return;
  }

  const int chosen_rank = static_cast<int>(hash64(seed ^ 0x517cc1b727220a95ULL) %
                                           static_cast<std::uint64_t>(compatible_count));
  int seen = 0;
  for (int i = 0; i < candidates_per_program; ++i) {
    for (int j = 0; j < candidates_per_program; ++j) {
      if (!d_candidate_keys_compatible(candidates_a[i], candidates_b[j])) {
        continue;
      }
      if (seen == chosen_rank) {
        *cand_a_out = i;
        *cand_b_out = j;
        return;
      }
      seen += 1;
    }
  }

  *cand_a_out = 0;
  *cand_b_out = 0;
}

__global__ void tournament_select_kernel(const double* fitness,
                                         const DCandidateRange* candidates,
                                         int population_size,
                                         int pair_count,
                                         int candidates_per_program,
                                         int tournament_k,
                                         std::uint64_t seed,
                                         int* parent_a,
                                         int* parent_b,
                                         int* cand_a,
                                         int* cand_b) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= pair_count) {
    return;
  }

  const int slot_a = idx * 2;
  const int slot_b = slot_a + 1;
  const int pa = d_best_index_for_parent_slot(fitness, population_size, tournament_k, slot_a, seed);
  const int pb = d_best_index_for_parent_slot(fitness, population_size, tournament_k, slot_b, seed);
  parent_a[idx] = pa;
  parent_b[idx] = pb;

  const std::uint64_t cseed = hash64(seed + static_cast<std::uint64_t>(idx + 1) * 0x9e3779b97f4a7c15ULL);
  d_choose_typed_candidate_pair(candidates, candidates_per_program, pa, pb, cseed, &cand_a[idx], &cand_b[idx]);
}

}  // namespace g3pvm::evo::repro
