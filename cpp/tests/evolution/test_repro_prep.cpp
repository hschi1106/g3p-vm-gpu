#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "g3pvm/evolution/repro/backend.hpp"
#include "g3pvm/evolution/repro/gpu.hpp"
#include "g3pvm/evolution/repro/pack.hpp"
#include "g3pvm/evolution/repro/prep.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

#include "../../src/evolution/subtree_utils.hpp"

#ifdef G3PVM_HAS_CUDA
#include "../../src/evolution/repro/gpu/internal.hpp"
#endif

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

std::vector<g3pvm::evo::ProgramGenome> make_population(
    int count = 4,
    const g3pvm::evo::GrammarConfig& grammar = g3pvm::evo::GrammarConfig{}) {
  g3pvm::evo::Limits limits;
  limits.max_expr_depth = 5;
  limits.max_stmts_per_block = 6;
  limits.max_total_nodes = 80;
  limits.max_for_k = 16;
  limits.max_call_args = 3;

  std::vector<g3pvm::evo::ProgramGenome> out;
  out.reserve(static_cast<std::size_t>(count));
  for (int seed = 1; seed <= count; ++seed) {
    out.push_back(g3pvm::evo::generate_random_genome(seed, limits, grammar));
  }
  return out;
}

bool scalar_config_allows_program(const g3pvm::evo::ProgramGenome& genome) {
  const g3pvm::evo::GrammarConfig grammar = g3pvm::evo::GrammarConfig::scalar();
  for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
    if (!grammar.allows_node_kind(node.kind)) {
      return false;
    }
  }
  for (const g3pvm::Value& value : genome.ast.consts) {
    if (value.tag == g3pvm::ValueTag::String ||
        value.tag == g3pvm::ValueTag::IntList ||
        value.tag == g3pvm::ValueTag::FloatList ||
        value.tag == g3pvm::ValueTag::StringList) {
      return false;
    }
  }
  return true;
}

g3pvm::evo::ProgramGenome make_linear_rec_genome(int suffix) {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::LinearRecBinders;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  ProgramGenome genome;
  genome.ast.names = {
      "xs" + std::to_string(suffix),
      "__lr_u" + std::to_string(suffix),
      "__lr_v" + std::to_string(suffix),
      "__lr_i" + std::to_string(suffix),
  };
  genome.ast.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)}),
      Value::from_int(0),
      Value::from_int(suffix),
  };
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  genome.ast.linear_rec_binders = {LinearRecBinders{3, 1, 2, 3}};
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
  return genome;
}

bool program_contains_linear_rec(const g3pvm::evo::ProgramGenome& genome) {
  for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
    if (node.kind == g3pvm::evo::NodeKind::LINEAR_REC) {
      return true;
    }
  }
  return false;
}

g3pvm::evo::GrammarConfig grammar_with_only_asgp_form(g3pvm::evo::NodeKind form) {
  g3pvm::evo::GrammarConfig grammar;
  grammar.expression_asgp_dc = form == g3pvm::evo::NodeKind::ASGP_DC;
  grammar.expression_asgp_dp1d = form == g3pvm::evo::NodeKind::ASGP_DP1D;
  grammar.expression_asgp_dp2d = form == g3pvm::evo::NodeKind::ASGP_DP2D;
  return grammar;
}

g3pvm::evo::ProgramGenome make_asgp_dc_genome() {
  using g3pvm::Value;
  using g3pvm::evo::AsgpDcBinders;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  ProgramGenome genome;
  genome.ast.names = {"xs", "n", "lo", "r1", "r2"};
  genome.ast.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
      Value::from_int(0),
      Value::from_int(999),
  };
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CALL_INDEX, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 3, 0},
      AstNode{NodeKind::BOUND_VAR, 4, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  genome.ast.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 1, 3, 4}};
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
  return genome;
}

bool candidate_contains_asgp(const g3pvm::evo::ProgramGenome& genome,
                             const g3pvm::evo::repro::CandidateRange& candidate) {
  if (candidate.start < 0 || candidate.stop <= candidate.start ||
      static_cast<std::size_t>(candidate.stop) > genome.ast.nodes.size()) {
    return false;
  }
  for (int i = candidate.start; i < candidate.stop; ++i) {
    const g3pvm::evo::NodeKind kind = genome.ast.nodes[static_cast<std::size_t>(i)].kind;
    if (kind == g3pvm::evo::NodeKind::ASGP_DC ||
        kind == g3pvm::evo::NodeKind::ASGP_DP1D ||
        kind == g3pvm::evo::NodeKind::ASGP_DP2D) {
      return true;
    }
  }
  return false;
}

bool program_contains_asgp(const g3pvm::evo::ProgramGenome& genome) {
  for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
    if (node.kind == g3pvm::evo::NodeKind::ASGP_DC ||
        node.kind == g3pvm::evo::NodeKind::ASGP_DP1D ||
        node.kind == g3pvm::evo::NodeKind::ASGP_DP2D) {
      return true;
    }
  }
  return false;
}

bool test_preprocess_and_pack() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = make_population();
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);

  if (!check(prep.subtree_ends.size() == population.size(), "subtree_ends size mismatch")) return false;
  if (!check(prep.candidates.size() == population.size(), "candidates size mismatch")) return false;
  if (!check(static_cast<int>(prep.donor_pool.size()) ==
                 repro_cfg.donor_pool_size_per_type * g3pvm::evo::repro::kGpuReproDonorTypeCount,
             "donor_pool size mismatch")) {
    return false;
  }

  for (std::size_t i = 0; i < population.size(); ++i) {
    if (!check(prep.subtree_ends[i].size() == population[i].ast.nodes.size(),
               "subtree_ends node count mismatch")) {
      return false;
    }
    if (!check(!prep.candidates[i].empty(), "candidate set should not be empty")) {
      return false;
    }
  }

  const g3pvm::evo::repro::PackedHostData packed =
      g3pvm::evo::repro::pack_population(population, prep, repro_cfg);
  if (!check(static_cast<int>(packed.metas.size()) == repro_cfg.population_size, "meta size mismatch")) return false;
  if (!check(static_cast<int>(packed.donor_lens.size()) ==
                 repro_cfg.donor_pool_size_per_type * g3pvm::evo::repro::kGpuReproDonorTypeCount,
             "donor_lens size mismatch")) {
    return false;
  }
  if (!check(static_cast<int>(packed.candidates.size()) ==
                 repro_cfg.population_size * repro_cfg.candidates_per_program,
             "packed candidates size mismatch")) {
    return false;
  }
  if (!check(!packed.name_lookup.empty(), "name lookup should not be empty")) {
    return false;
  }
  return true;
}

bool test_gpu_repro_prep_includes_asgp_candidates() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 2;
  cfg.selection_pressure = 2;
  cfg.seed = 42;
  cfg.limits.max_expr_depth = 8;
  cfg.limits.max_stmts_per_block = 4;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = {
      make_asgp_dc_genome(),
      make_asgp_dc_genome(),
  };
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);

  bool saw_asgp_candidate = false;
  for (std::size_t p = 0; p < prep.candidates.size(); ++p) {
    for (const g3pvm::evo::repro::CandidateRange& candidate : prep.candidates[p]) {
      saw_asgp_candidate = saw_asgp_candidate || candidate_contains_asgp(population[p], candidate);
    }
  }
  if (!check(saw_asgp_candidate,
             "GPU reproduction candidates should expose ASGP metadata-bearing subtrees")) {
    return false;
  }
  return true;
}

bool test_gpu_repro_backend_preserves_asgp_metadata() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 2;
  cfg.selection_pressure = 2;
  cfg.seed = 43;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.mutation_rate = 0.0;
  cfg.limits.max_expr_depth = 8;
  cfg.limits.max_stmts_per_block = 4;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = {
      make_asgp_dc_genome(),
      make_asgp_dc_genome(),
  };
  std::vector<g3pvm::evo::ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(g3pvm::evo::ScoredGenome{population[i], static_cast<double>(i + 1)});
  }

  g3pvm::evo::repro::ReproductionStats prep_stats;
  try {
    auto prepared =
        g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, 123, &prep_stats);
    const g3pvm::evo::repro::CandidateRange asgp_root{
        3, 12, static_cast<int>(g3pvm::evo::repro::CandidateTag::Expr), static_cast<int>(g3pvm::evo::RType::Int)};
    for (int p = 0; p < prepared.config.population_size; ++p) {
      const int base = p * prepared.config.candidates_per_program;
      for (int i = 0; i < prepared.config.candidates_per_program; ++i) {
        prepared.packed.candidates[static_cast<std::size_t>(base + i)] = asgp_root;
      }
    }
    g3pvm::evo::repro::GpuReproArena arena;
    g3pvm::evo::repro::GpuReproHostStaging staging;
    struct Cleanup {
      g3pvm::evo::repro::GpuReproArena* arena = nullptr;
      g3pvm::evo::repro::GpuReproHostStaging* staging = nullptr;
      ~Cleanup() {
        if (staging != nullptr) g3pvm::evo::repro::destroy_gpu_repro_host_staging(staging);
        if (arena != nullptr) g3pvm::evo::repro::destroy_gpu_repro_arena(arena);
      }
    } cleanup{&arena, &staging};

    std::string message;
    if (!g3pvm::evo::repro::ensure_gpu_repro_arena_capacity(&arena, prepared.config, &message) ||
        !g3pvm::evo::repro::ensure_gpu_repro_host_staging_capacity(&staging, prepared.config, &message)) {
      if (message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_repro_prep: SKIP ASGP metadata gpu prepared (" << message << ")\n";
        return true;
      }
      std::cerr << "FAIL: ASGP metadata gpu setup failed: " << message << "\n";
      return false;
    }

    std::vector<double> ranked_fitness;
    ranked_fitness.reserve(scored.size());
    for (const auto& one : scored) {
      ranked_fitness.push_back(g3pvm::evo::canonicalize_fitness_for_ranking(one.fitness));
    }

    g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
    if (!g3pvm::evo::repro::upload_gpu_repro_inputs(prepared.packed, &arena, &run_stats, &message) ||
        !g3pvm::evo::repro::launch_gpu_repro_kernels(&arena, prepared.config, ranked_fitness, &run_stats, &message)) {
      std::cerr << "FAIL: ASGP metadata gpu kernels failed: " << message << "\n";
      return false;
    }

    g3pvm::evo::repro::GpuReproChildView copyback;
    if (!g3pvm::evo::repro::copyback_gpu_repro_children(
            arena, prepared.config, &staging, &copyback, &run_stats, &message)) {
      std::cerr << "FAIL: ASGP metadata gpu copyback failed: " << message << "\n";
      return false;
    }

    bool saw_valid_asgp_child = false;
    const int child_count = prepared.config.pair_count * 2;
    for (int child = 0; child < child_count; ++child) {
      if (copyback.child_meta[child].valid == 0) {
        continue;
      }
      const int node_base = copyback.child_node_offsets[child];
      for (int i = 0; i < copyback.child_used_len[child]; ++i) {
        const g3pvm::evo::NodeKind kind =
            static_cast<g3pvm::evo::NodeKind>(copyback.child_nodes[node_base + i].kind);
        if (kind == g3pvm::evo::NodeKind::ASGP_DC ||
            kind == g3pvm::evo::NodeKind::ASGP_DP1D ||
            kind == g3pvm::evo::NodeKind::ASGP_DP2D) {
          saw_valid_asgp_child = true;
        }
      }
    }
    if (!check(saw_valid_asgp_child, "gpu reproduction should mark ASGP children valid")) {
      return false;
    }

    const auto decoded =
        g3pvm::evo::repro::decode_gpu_repro_children(prepared.packed, copyback, scored, cfg);
    if (!check(static_cast<int>(decoded.size()) == cfg.population_size,
               "ASGP metadata decoded child count mismatch")) {
      return false;
    }
    bool saw_decoded_asgp_child = false;
    for (const g3pvm::evo::ProgramGenome& child : decoded) {
      if (program_contains_asgp(child)) {
        saw_decoded_asgp_child = true;
        if (!check(!child.ast.asgp_dc_binders.empty(),
                   "ASGP metadata child should preserve ASGP-DC binders")) {
          return false;
        }
      }
      const auto bc = g3pvm::evo::compile_for_eval(child);
      if (!check(!bc.code.empty(), "ASGP metadata child should compile")) {
        return false;
      }
    }
    if (!check(saw_decoded_asgp_child, "gpu reproduction should decode at least one ASGP child")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP ASGP metadata gpu prepared (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: ASGP metadata gpu reproduction failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

bool test_gpu_decode_preserves_asgp_metadata() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 2;
  cfg.selection_pressure = 2;
  cfg.seed = 44;
  cfg.mutation_rate = 0.0;
  cfg.mutation_subtree_prob = 0.0;
  cfg.limits.max_expr_depth = 8;
  cfg.limits.max_stmts_per_block = 4;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = {
      make_asgp_dc_genome(),
      make_asgp_dc_genome(),
  };
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);
  g3pvm::evo::repro::PackedHostData packed =
      g3pvm::evo::repro::pack_population(population, prep, repro_cfg);
  packed.candidates[0] = g3pvm::evo::repro::CandidateRange{
      4, 5, static_cast<int>(g3pvm::evo::repro::CandidateTag::Expr), static_cast<int>(g3pvm::evo::RType::IntList)};
  packed.candidates[static_cast<std::size_t>(repro_cfg.candidates_per_program)] =
      g3pvm::evo::repro::CandidateRange{
          4, 5, static_cast<int>(g3pvm::evo::repro::CandidateTag::Expr), static_cast<int>(g3pvm::evo::RType::IntList)};

  std::vector<g3pvm::evo::ScoredGenome> scored = {
      g3pvm::evo::ScoredGenome{population[0], 2.0},
      g3pvm::evo::ScoredGenome{population[1], 1.0},
  };

  std::vector<int> parent_a = {0};
  std::vector<int> parent_b = {1};
  std::vector<int> cand = {0};
  std::vector<int> child_node_offsets = {0, static_cast<int>(population[0].ast.nodes.size())};
  std::vector<int> child_name_offsets = {0, static_cast<int>(population[0].ast.names.size())};
  std::vector<int> child_const_offsets = {0, static_cast<int>(population[0].ast.consts.size())};
  std::vector<int> child_used_len = {
      static_cast<int>(population[0].ast.nodes.size()),
      static_cast<int>(population[1].ast.nodes.size()),
  };
  std::vector<int> child_name_counts = {
      static_cast<int>(population[0].ast.names.size()),
      static_cast<int>(population[1].ast.names.size()),
  };
  std::vector<int> child_const_counts = {
      static_cast<int>(population[0].ast.consts.size()),
      static_cast<int>(population[1].ast.consts.size()),
  };
  std::vector<g3pvm::evo::repro::PlainNode> child_nodes;
  for (const g3pvm::evo::ProgramGenome& genome : population) {
    for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
      child_nodes.push_back(g3pvm::evo::repro::PlainNode{static_cast<int>(node.kind), node.i0, node.i1});
    }
  }
  std::vector<std::uint64_t> child_name_ids;
  for (std::size_t p = 0; p < population.size(); ++p) {
    const std::size_t base = p * static_cast<std::size_t>(repro_cfg.max_names);
    for (int i = 0; i < child_name_counts[p]; ++i) {
      child_name_ids.push_back(packed.program_name_ids[base + static_cast<std::size_t>(i)]);
    }
  }
  std::vector<g3pvm::Value> child_consts;
  for (std::size_t p = 0; p < population.size(); ++p) {
    const std::size_t base = p * static_cast<std::size_t>(repro_cfg.max_consts);
    for (int i = 0; i < child_const_counts[p]; ++i) {
      child_consts.push_back(packed.program_consts[base + static_cast<std::size_t>(i)]);
    }
  }
  std::vector<g3pvm::evo::repro::PackedChildMeta> child_meta = {
      g3pvm::evo::repro::PackedChildMeta{
          population[0].meta.node_count,
          population[0].meta.max_depth,
          static_cast<unsigned char>(population[0].meta.uses_builtins ? 1 : 0),
          1,
      },
      g3pvm::evo::repro::PackedChildMeta{
          population[1].meta.node_count,
          population[1].meta.max_depth,
          static_cast<unsigned char>(population[1].meta.uses_builtins ? 1 : 0),
          1,
      },
  };

  g3pvm::evo::repro::GpuReproChildView view;
  view.config = repro_cfg;
  view.parent_a = parent_a.data();
  view.parent_b = parent_b.data();
  view.cand_a = cand.data();
  view.cand_b = cand.data();
  view.child_nodes = child_nodes.data();
  view.child_node_offsets = child_node_offsets.data();
  view.child_name_ids = child_name_ids.data();
  view.child_name_offsets = child_name_offsets.data();
  view.child_consts = child_consts.data();
  view.child_const_offsets = child_const_offsets.data();
  view.child_used_len = child_used_len.data();
  view.child_name_counts = child_name_counts.data();
  view.child_const_counts = child_const_counts.data();
  view.child_meta = child_meta.data();

  const std::vector<g3pvm::evo::ProgramGenome> decoded =
      g3pvm::evo::repro::decode_gpu_repro_children(packed, view, scored, cfg);
  if (!check(decoded.size() == 2, "ASGP metadata decode child count mismatch")) return false;
  if (!check(!decoded[0].ast.asgp_dc_binders.empty(),
             "ASGP metadata should be rebuilt from packed program metadata")) {
    return false;
  }
  const auto bc = g3pvm::evo::compile_for_eval(decoded[0]);
  return check(!bc.code.empty(), "decoded ASGP metadata child should compile");
}

bool test_gpu_donor_pool_emits_asgp_dc_metadata() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;

  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 32;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.limits.max_expr_depth = 8;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population =
      make_population(cfg.population_size, cfg.grammar);
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);
  const g3pvm::evo::repro::PackedHostData packed =
      g3pvm::evo::repro::pack_population(population, prep, repro_cfg);

  bool saw_asgp_dc_donor = false;
  bool saw_string_asgp_dc_donor = false;
  for (std::size_t i = 0; i < prep.donor_pool.size(); ++i) {
    const g3pvm::evo::repro::DonorProgram& donor = prep.donor_pool[i];
    if (donor.ast.asgp_dc_binders.empty()) {
      continue;
    }
    saw_asgp_dc_donor = true;
    saw_string_asgp_dc_donor = saw_string_asgp_dc_donor || donor.type == g3pvm::evo::RType::String;
    if (!check(donor.type == g3pvm::evo::RType::Int || donor.type == g3pvm::evo::RType::Float ||
                   donor.type == g3pvm::evo::RType::String,
               "GPU ASGP-DC donor should only appear in supported Int/Float/String buckets")) {
      return false;
    }
    if (!check(packed.donor_asgp_dc_counts[i] > 0,
               "packed GPU donor pool should carry ASGP-DC binder metadata")) {
      return false;
    }

    g3pvm::evo::AstProgram base;
    base.version = g3pvm::evo::k_ast_prefix_version_current;
    if (donor.type == g3pvm::evo::RType::String) {
      base.consts = {g3pvm::payload::make_string_value("")};
    } else {
      base.consts = {donor.type == g3pvm::evo::RType::Float ? Value::from_float(0.0)
                                                            : Value::from_int(0)};
    }
    base.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    g3pvm::evo::ProgramGenome wrapped;
    wrapped.ast = g3pvm::evo::subtree::replace_subtree(
        base, 3, 4, donor.ast, 0, donor.ast.nodes.size());
    wrapped.meta = g3pvm::evo::build_genome_meta(wrapped.ast);
    const auto bc = g3pvm::evo::compile_for_eval(wrapped);
    if (!check(!bc.code.empty(), "GPU ASGP-DC donor should compile after subtree replacement")) {
      return false;
    }
  }

  if (!check(saw_asgp_dc_donor, "GPU donor pool should emit ASGP-DC donors when enabled")) {
    return false;
  }
  return check(saw_string_asgp_dc_donor, "GPU donor pool should emit String ASGP-DC donors when enabled");
}

bool test_gpu_donor_pool_emits_asgp_dp_metadata() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;

  auto check_form = [](NodeKind form, std::uint64_t seed) {
    g3pvm::evo::EvolutionConfig cfg;
    cfg.population_size = 8;
    cfg.selection_pressure = 3;
    cfg.seed = seed;
    cfg.grammar = grammar_with_only_asgp_form(form);
    cfg.limits.max_expr_depth = 8;
    cfg.limits.max_stmts_per_block = 6;
    cfg.limits.max_total_nodes = 80;
    cfg.limits.max_for_k = 16;
    cfg.limits.max_call_args = 3;

    const std::vector<g3pvm::evo::ProgramGenome> population =
        make_population(cfg.population_size, cfg.grammar);
    const g3pvm::evo::repro::GpuReproConfig repro_cfg =
        g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
    const g3pvm::evo::repro::PreprocessOutput prep =
        g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);
    const g3pvm::evo::repro::PackedHostData packed =
        g3pvm::evo::repro::pack_population(population, prep, repro_cfg);

    bool saw_asgp_dp_donor = false;
    bool saw_string_asgp_dp_donor = false;
    for (std::size_t i = 0; i < prep.donor_pool.size(); ++i) {
      const g3pvm::evo::repro::DonorProgram& donor = prep.donor_pool[i];
      const bool has_dp1d = !donor.ast.asgp_dp1d_specs.empty();
      const bool has_dp2d = !donor.ast.asgp_dp2d_specs.empty();
      if ((form == NodeKind::ASGP_DP1D && !has_dp1d) ||
          (form == NodeKind::ASGP_DP2D && !has_dp2d)) {
        continue;
      }
      saw_asgp_dp_donor = true;
      saw_string_asgp_dp_donor = saw_string_asgp_dp_donor || donor.type == g3pvm::evo::RType::String;
      if (!check(donor.type == g3pvm::evo::RType::Int || donor.type == g3pvm::evo::RType::Float ||
                     donor.type == g3pvm::evo::RType::String,
                 "GPU ASGP-DP donor should only appear in supported Int/Float/String buckets")) {
        return false;
      }
      if (form == NodeKind::ASGP_DP1D &&
          !check(packed.donor_asgp_dp1d_counts[i] > 0,
                 "packed GPU donor pool should carry ASGP-DP1D metadata")) {
        return false;
      }
      if (form == NodeKind::ASGP_DP2D &&
          !check(packed.donor_asgp_dp2d_counts[i] > 0,
                 "packed GPU donor pool should carry ASGP-DP2D metadata")) {
        return false;
      }

      g3pvm::evo::AstProgram base;
      base.version = g3pvm::evo::k_ast_prefix_version_current;
      if (donor.type == g3pvm::evo::RType::String) {
        base.consts = {g3pvm::payload::make_string_value("")};
      } else {
        base.consts = {donor.type == g3pvm::evo::RType::Float ? Value::from_float(0.0)
                                                              : Value::from_int(0)};
      }
      base.nodes = {
          AstNode{NodeKind::PROGRAM, 0, 0},
          AstNode{NodeKind::BLOCK_CONS, 0, 0},
          AstNode{NodeKind::RETURN, 0, 0},
          AstNode{NodeKind::CONST, 0, 0},
          AstNode{NodeKind::BLOCK_NIL, 0, 0},
      };
      g3pvm::evo::ProgramGenome wrapped;
      wrapped.ast = g3pvm::evo::subtree::replace_subtree(
          base, 3, 4, donor.ast, 0, donor.ast.nodes.size());
      wrapped.meta = g3pvm::evo::build_genome_meta(wrapped.ast);
      const auto bc = g3pvm::evo::compile_for_eval(wrapped);
      if (!check(!bc.code.empty(), "GPU ASGP-DP donor should compile after subtree replacement")) {
        return false;
      }
    }
    if (!check(saw_asgp_dp_donor, form == NodeKind::ASGP_DP1D
                                      ? "GPU donor pool should emit ASGP-DP1D donors when enabled"
                                      : "GPU donor pool should emit ASGP-DP2D donors when enabled")) {
      return false;
    }
    return check(saw_string_asgp_dp_donor,
                 form == NodeKind::ASGP_DP1D
                     ? "GPU donor pool should emit String ASGP-DP1D donors when enabled"
                     : "GPU donor pool should emit String ASGP-DP2D donors when enabled");
  };

  if (!check_form(NodeKind::ASGP_DP1D, 45)) return false;
  return check_form(NodeKind::ASGP_DP2D, 46);
}

g3pvm::ValueTag value_tag_for_rtype(g3pvm::evo::RType type) {
  switch (type) {
    case g3pvm::evo::RType::Int:
      return g3pvm::ValueTag::Int;
    case g3pvm::evo::RType::Float:
      return g3pvm::ValueTag::Float;
    case g3pvm::evo::RType::Bool:
      return g3pvm::ValueTag::Bool;
    case g3pvm::evo::RType::Char:
      return g3pvm::ValueTag::Char;
    case g3pvm::evo::RType::String:
      return g3pvm::ValueTag::String;
    case g3pvm::evo::RType::IntList:
      return g3pvm::ValueTag::IntList;
    case g3pvm::evo::RType::FloatList:
      return g3pvm::ValueTag::FloatList;
    case g3pvm::evo::RType::StringList:
      return g3pvm::ValueTag::StringList;
    default:
      return g3pvm::ValueTag::Invalid;
  }
}

g3pvm::evo::ProgramGenome wrap_donor_as_return(const g3pvm::evo::repro::DonorProgram& donor) {
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  ProgramGenome genome;
  genome.ast.version = g3pvm::evo::k_ast_prefix_version_current;
  genome.ast.names = donor.ast.names;
  genome.ast.consts = donor.ast.consts;
  genome.ast.linear_rec_binders = donor.ast.linear_rec_binders;
  genome.ast.asgp_dc_binders = donor.ast.asgp_dc_binders;
  genome.ast.asgp_dp1d_specs = donor.ast.asgp_dp1d_specs;
  genome.ast.asgp_dp2d_specs = donor.ast.asgp_dp2d_specs;
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
  };
  genome.ast.nodes.insert(genome.ast.nodes.end(), donor.ast.nodes.begin(), donor.ast.nodes.end());
  genome.ast.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  for (g3pvm::evo::LinearRecBinders& binders : genome.ast.linear_rec_binders) {
    binders.node_index += 3;
  }
  for (g3pvm::evo::AsgpDcBinders& binders : genome.ast.asgp_dc_binders) {
    binders.node_index += 3;
  }
  for (g3pvm::evo::AsgpDp1dSpec& spec : genome.ast.asgp_dp1d_specs) {
    spec.node_index += 3;
  }
  for (g3pvm::evo::AsgpDp2dSpec& spec : genome.ast.asgp_dp2d_specs) {
    spec.node_index += 3;
  }
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
  return genome;
}

bool test_gpu_donor_pool_preserves_target_runtime_type() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.selection_pressure = 3;
  cfg.seed = 1042;
  cfg.limits.max_expr_depth = 8;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population =
      make_population(cfg.population_size, cfg.grammar);
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);

  std::vector<bool> saw_type(static_cast<std::size_t>(g3pvm::evo::RType::Invalid), false);
  for (const g3pvm::evo::repro::DonorProgram& donor : prep.donor_pool) {
    if (donor.type == g3pvm::evo::RType::Any) {
      continue;
    }
    const g3pvm::ValueTag expected = value_tag_for_rtype(donor.type);
    if (expected == g3pvm::ValueTag::Invalid) {
      continue;
    }
    saw_type[static_cast<std::size_t>(donor.type)] = true;
    const g3pvm::evo::ProgramGenome genome = wrap_donor_as_return(donor);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
    if (!check(!out.is_error, "GPU donor-pool donor should execute as a standalone return expression")) {
      return false;
    }
    if (!check(out.value.tag == expected,
               "GPU donor-pool donor runtime tag should match its donor bucket type")) {
      return false;
    }
  }

  for (g3pvm::evo::RType type : {
           g3pvm::evo::RType::Int,
           g3pvm::evo::RType::Float,
           g3pvm::evo::RType::Bool,
           g3pvm::evo::RType::Char,
           g3pvm::evo::RType::String,
           g3pvm::evo::RType::IntList,
           g3pvm::evo::RType::FloatList,
           g3pvm::evo::RType::StringList,
       }) {
    if (!check(saw_type[static_cast<std::size_t>(type)],
               "GPU donor pool should include every concrete public current donor type")) {
      return false;
    }
  }
  return true;
}

bool test_preprocess_respects_scalar_grammar_config() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.grammar = g3pvm::evo::GrammarConfig::scalar();
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population =
      make_population(cfg.population_size, cfg.grammar);
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);

  for (const auto& candidates : prep.candidates) {
    for (const g3pvm::evo::repro::CandidateRange& candidate : candidates) {
      const auto type = static_cast<g3pvm::evo::RType>(candidate.aux);
      if (type != g3pvm::evo::RType::Invalid &&
          !check(cfg.grammar.allows_type(type), "scalar gpu candidate type should be grammar-allowed")) {
        return false;
      }
    }
  }
  for (const g3pvm::evo::repro::DonorProgram& donor : prep.donor_pool) {
    if (!check(cfg.grammar.allows_type(donor.type), "scalar gpu donor type should be grammar-allowed")) {
      return false;
    }
    g3pvm::evo::ProgramGenome donor_genome;
    donor_genome.ast = donor.ast;
    if (!check(scalar_config_allows_program(donor_genome),
               "scalar gpu donor pool should not contain disabled sequence features")) {
      return false;
    }
  }
  return true;
}

bool test_gpu_preprocess_packs_typed_candidate_keys() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;

  g3pvm::evo::ProgramGenome genome;
  genome.ast.version = g3pvm::evo::k_ast_prefix_version_current;
  genome.ast.names = {"x"};
  genome.ast.consts = {Value::from_int(1), Value::from_int(2)};
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);

  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 2;
  cfg.selection_pressure = 2;
  cfg.seed = 42;
  const std::vector<g3pvm::evo::ProgramGenome> population = {genome, genome};
  const g3pvm::evo::repro::GpuReproConfig repro_cfg =
      g3pvm::evo::repro::make_gpu_repro_config(population, cfg);
  const g3pvm::evo::repro::PreprocessOutput prep =
      g3pvm::evo::repro::preprocess_population(population, repro_cfg, cfg.grammar);
  const g3pvm::evo::repro::PackedHostData packed =
      g3pvm::evo::repro::pack_population(population, prep, repro_cfg);

  const g3pvm::evo::repro::CandidateRange* assign_const = nullptr;
  const g3pvm::evo::repro::CandidateRange* scoped_var = nullptr;
  const g3pvm::evo::repro::CandidateRange* scoped_const = nullptr;
  for (const g3pvm::evo::repro::CandidateRange& candidate : prep.candidates[0]) {
    if (candidate.start == 3) assign_const = &candidate;
    if (candidate.start == 7) scoped_var = &candidate;
    if (candidate.start == 8) scoped_const = &candidate;
  }
  if (!check(assign_const != nullptr && scoped_var != nullptr && scoped_const != nullptr,
             "GPU preprocessing should expose typed candidates needed for key checks")) {
    return false;
  }
  if (!check(assign_const->scope_signature != scoped_var->scope_signature,
             "GPU candidate key should distinguish different visible environments")) {
    return false;
  }
  if (!check(scoped_var->scope_signature == scoped_const->scope_signature &&
                 scoped_var->visible_env_signature == scoped_const->visible_env_signature,
             "GPU candidate key should preserve common visible environment for compatible ordinary roots")) {
    return false;
  }
  const g3pvm::evo::repro::CandidateRange& packed_first = packed.candidates[0];
  if (!check(packed_first.scope_signature == prep.candidates[0][0].scope_signature &&
                 packed_first.binder_signature == prep.candidates[0][0].binder_signature &&
                 packed_first.visible_env_signature == prep.candidates[0][0].visible_env_signature,
             "packed GPU candidates should preserve typed key fields")) {
    return false;
  }
  return true;
}

g3pvm::evo::ProgramGenome make_bloated_const_table_genome() {
  g3pvm::evo::AstProgram ast;
  ast.version = g3pvm::evo::k_ast_prefix_version_current;
  ast.names = {"unused_name"};
  ast.consts.reserve(static_cast<std::size_t>(160));
  ast.consts.push_back(g3pvm::Value::from_int(7));
  for (int i = 1; i < 160; ++i) {
    ast.consts.push_back(g3pvm::Value::from_int(1000 + i));
  }
  ast.nodes = {
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::PROGRAM, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::BLOCK_CONS, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::RETURN, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::CONST, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::BLOCK_NIL, 0, 0},
  };
  g3pvm::evo::ProgramGenome genome;
  genome.ast = std::move(ast);
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
  return genome;
}

g3pvm::evo::ProgramGenome make_const_return_genome(int value) {
  g3pvm::evo::AstProgram ast;
  ast.version = g3pvm::evo::k_ast_prefix_version_current;
  ast.consts = {g3pvm::Value::from_int(value)};
  ast.nodes = {
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::PROGRAM, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::BLOCK_CONS, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::RETURN, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::CONST, 0, 0},
      g3pvm::evo::AstNode{g3pvm::evo::NodeKind::BLOCK_NIL, 0, 0},
  };
  g3pvm::evo::ProgramGenome genome;
  genome.ast = std::move(ast);
  genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
  return genome;
}

std::uint64_t test_hash_name(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

bool program_contains_bound_var(const g3pvm::evo::ProgramGenome& genome) {
  for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
    if (node.kind == g3pvm::evo::NodeKind::BOUND_VAR) {
      return true;
    }
  }
  return false;
}

bool test_gpu_repro_compacts_dead_tables_before_pack() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 2;
  cfg.selection_pressure = 2;
  cfg.seed = 42;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  std::vector<g3pvm::evo::ProgramGenome> population = {
      make_bloated_const_table_genome(),
      make_bloated_const_table_genome(),
  };
  if (!check(population.front().ast.consts.size() > g3pvm::evo::repro::kGpuReproMaxConsts,
             "test setup should exceed gpu repro const scratch")) {
    return false;
  }

  const g3pvm::evo::ProgramGenome compacted =
      g3pvm::evo::repro::compact_genome_tables(population.front());
  if (!check(compacted.ast.names.empty(), "compaction should remove unused names")) return false;
  if (!check(compacted.ast.consts.size() == 1, "compaction should remove unused consts")) return false;
  if (!check(compacted.ast.nodes[3].i0 == 0, "compaction should remap const node")) return false;

  g3pvm::evo::repro::ReproductionStats stats;
  const g3pvm::evo::repro::GpuReproPreparedData prepared =
      g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, 123, &stats);
  if (!check(prepared.config.max_consts <= g3pvm::evo::repro::kGpuReproMaxConsts,
             "prepare should compact dead consts before scratch-capacity checks")) {
    return false;
  }
  if (!check(prepared.config.max_names <= g3pvm::evo::repro::kGpuReproMaxNames,
             "prepare should compact dead names before scratch-capacity checks")) {
    return false;
  }
  return true;
}

bool test_decode_falls_back_on_escaped_bound_var_child() {
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 2;

  std::vector<g3pvm::evo::ScoredGenome> scored = {
      g3pvm::evo::ScoredGenome{make_const_return_genome(11), 2.0},
      g3pvm::evo::ScoredGenome{make_const_return_genome(22), 1.0},
  };

  g3pvm::evo::repro::PackedHostData packed;
  packed.config.population_size = 2;
  packed.config.pair_count = 1;
  packed.config.candidates_per_program = 1;
  packed.config.max_nodes = 5;
  packed.config.max_names = 1;
  const std::uint64_t map_name_id = test_hash_name("__map_u0");
  packed.name_lookup[map_name_id] = "__map_u0";
  packed.metas = {
      g3pvm::evo::repro::PackedProgramMeta{5, 0, 1, 0},
      g3pvm::evo::repro::PackedProgramMeta{5, 0, 1, 0},
  };
  packed.candidates = {
      g3pvm::evo::repro::CandidateRange{3, 4, static_cast<int>(g3pvm::evo::repro::CandidateTag::Expr),
                                        static_cast<int>(g3pvm::evo::RType::Int)},
      g3pvm::evo::repro::CandidateRange{3, 4, static_cast<int>(g3pvm::evo::repro::CandidateTag::Expr),
                                        static_cast<int>(g3pvm::evo::RType::Int)},
  };
  packed.program_linear_rec_binders.resize(2);

  std::vector<int> parent_a = {0};
  std::vector<int> parent_b = {1};
  std::vector<int> cand = {0};
  std::vector<int> child_node_offsets = {0, 5};
  std::vector<int> child_name_offsets = {0, 1};
  std::vector<int> child_const_offsets = {0, 0};
  std::vector<int> child_used_len = {5, 5};
  std::vector<int> child_name_counts = {1, 1};
  std::vector<int> child_const_counts = {0, 0};
  std::vector<std::uint64_t> child_name_ids = {map_name_id, map_name_id};
  std::vector<g3pvm::evo::repro::PlainNode> child_nodes = {
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::PROGRAM), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::BLOCK_CONS), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::RETURN), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::BOUND_VAR), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::BLOCK_NIL), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::PROGRAM), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::BLOCK_CONS), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::RETURN), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::BOUND_VAR), 0, 0},
      g3pvm::evo::repro::PlainNode{static_cast<int>(g3pvm::evo::NodeKind::BLOCK_NIL), 0, 0},
  };
  std::vector<g3pvm::Value> child_consts;
  std::vector<g3pvm::evo::repro::PackedChildMeta> child_meta = {
      g3pvm::evo::repro::PackedChildMeta{5, 4, 0, 1},
      g3pvm::evo::repro::PackedChildMeta{5, 4, 0, 1},
  };

  g3pvm::evo::repro::GpuReproChildView copyback;
  copyback.config = packed.config;
  copyback.parent_a = parent_a.data();
  copyback.parent_b = parent_b.data();
  copyback.cand_a = cand.data();
  copyback.cand_b = cand.data();
  copyback.child_nodes = child_nodes.data();
  copyback.child_node_offsets = child_node_offsets.data();
  copyback.child_name_ids = child_name_ids.data();
  copyback.child_name_offsets = child_name_offsets.data();
  copyback.child_consts = child_consts.data();
  copyback.child_const_offsets = child_const_offsets.data();
  copyback.child_used_len = child_used_len.data();
  copyback.child_name_counts = child_name_counts.data();
  copyback.child_const_counts = child_const_counts.data();
  copyback.child_meta = child_meta.data();

  const auto decoded = g3pvm::evo::repro::decode_gpu_repro_children(packed, copyback, scored, cfg);
  if (!check(decoded.size() == 2, "escaped bound var fallback decoded size mismatch")) return false;
  for (const auto& child : decoded) {
    if (!check(!program_contains_bound_var(child),
               "escaped bound var child should fall back instead of entering population")) {
      return false;
    }
    const auto bc = g3pvm::evo::compile_for_eval(child);
    if (!check(!bc.code.empty(), "fallback child should compile after escaped bound var rejection")) {
      return false;
    }
  }
  return true;
}

#ifdef G3PVM_HAS_CUDA
bool copyback_gpu_selection_parents(const g3pvm::evo::EvolutionConfig& cfg,
                                    const std::vector<g3pvm::evo::ProgramGenome>& population,
                                    const std::vector<double>& ranked_fitness,
                                    std::uint64_t seed,
                                    std::vector<int>* parent_a_out,
                                    std::vector<int>* parent_b_out,
                                    std::string* message_out) {
  g3pvm::evo::repro::ReproductionStats prep_stats;
  const auto prepared =
      g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, seed, &prep_stats);

  g3pvm::evo::repro::GpuReproArena arena;
  g3pvm::evo::repro::GpuReproHostStaging staging;
  struct Cleanup {
    g3pvm::evo::repro::GpuReproArena* arena = nullptr;
    g3pvm::evo::repro::GpuReproHostStaging* staging = nullptr;
    ~Cleanup() {
      if (staging != nullptr) g3pvm::evo::repro::destroy_gpu_repro_host_staging(staging);
      if (arena != nullptr) g3pvm::evo::repro::destroy_gpu_repro_arena(arena);
    }
  } cleanup{&arena, &staging};

  if (!g3pvm::evo::repro::ensure_gpu_repro_arena_capacity(&arena, prepared.config, message_out) ||
      !g3pvm::evo::repro::ensure_gpu_repro_host_staging_capacity(&staging, prepared.config, message_out)) {
    return false;
  }

  g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
  if (!g3pvm::evo::repro::upload_gpu_repro_inputs(prepared.packed, &arena, &run_stats, message_out) ||
      !g3pvm::evo::repro::launch_gpu_repro_kernels(&arena, prepared.config, ranked_fitness, &run_stats, message_out)) {
    return false;
  }

  g3pvm::evo::repro::GpuReproChildView copyback;
  if (!g3pvm::evo::repro::copyback_gpu_repro_children(
          arena, prepared.config, &staging, &copyback, &run_stats, message_out)) {
    return false;
  }

  parent_a_out->assign(copyback.parent_a, copyback.parent_a + prepared.config.pair_count);
  parent_b_out->assign(copyback.parent_b, copyback.parent_b + prepared.config.pair_count);
  return true;
}
#endif

bool test_gpu_prepared_backend_smoke() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = make_population();
  std::vector<g3pvm::evo::ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(g3pvm::evo::ScoredGenome{
        population[i],
        static_cast<double>(population.size() - i),
    });
  }

  g3pvm::evo::repro::ReproductionStats prep_stats;
  try {
    const auto prepared =
        g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, 123, &prep_stats);
    g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
    const auto reproduction =
        g3pvm::evo::repro::run_gpu_repro_backend_prepared(scored, cfg, prepared, &run_stats);
    if (!check(static_cast<int>(reproduction.next_population.size()) == cfg.population_size,
               "prepared gpu reproduction child count mismatch")) {
      return false;
    }
    for (const auto& child : reproduction.next_population) {
      const auto bc = g3pvm::evo::compile_for_eval(child);
      if (!check(!bc.code.empty(), "prepared gpu reproduction child should compile")) {
        return false;
      }
    }
    if (!check(run_stats.copyback_ms >= 0.0, "prepared gpu reproduction copyback stats missing")) {
      return false;
    }
    if (!check(run_stats.decode_ms >= 0.0, "prepared gpu reproduction decode stats missing")) {
      return false;
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP gpu prepared (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu reproduction prepared backend failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

bool test_gpu_prepared_backend_preserves_linear_rec_metadata() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.selection_pressure = 1;
  cfg.seed = 77;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.mutation_rate = 0.0;
  cfg.mutation_subtree_prob = 1.0;
  cfg.limits.max_expr_depth = 8;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 120;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  std::vector<g3pvm::evo::ProgramGenome> population;
  for (int i = 0; i < cfg.population_size; ++i) {
    population.push_back(make_linear_rec_genome(i + 1));
  }

  std::vector<g3pvm::evo::ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(g3pvm::evo::ScoredGenome{
        population[i],
        static_cast<double>(population.size() - i),
    });
  }

  g3pvm::evo::repro::ReproductionStats prep_stats;
  try {
    const auto prepared =
        g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, 321, &prep_stats);
    g3pvm::evo::repro::GpuReproArena arena;
    g3pvm::evo::repro::GpuReproHostStaging staging;
    struct Cleanup {
      g3pvm::evo::repro::GpuReproArena* arena = nullptr;
      g3pvm::evo::repro::GpuReproHostStaging* staging = nullptr;
      ~Cleanup() {
        if (staging != nullptr) g3pvm::evo::repro::destroy_gpu_repro_host_staging(staging);
        if (arena != nullptr) g3pvm::evo::repro::destroy_gpu_repro_arena(arena);
      }
    } cleanup{&arena, &staging};

    std::string message;
    if (!g3pvm::evo::repro::ensure_gpu_repro_arena_capacity(&arena, prepared.config, &message) ||
        !g3pvm::evo::repro::ensure_gpu_repro_host_staging_capacity(&staging, prepared.config, &message)) {
      if (message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "g3pvm_test_repro_prep: SKIP linear metadata gpu prepared (" << message << ")\n";
        return true;
      }
      std::cerr << "FAIL: linear metadata gpu setup failed: " << message << "\n";
      return false;
    }

    std::vector<double> ranked_fitness;
    ranked_fitness.reserve(scored.size());
    for (const auto& one : scored) {
      ranked_fitness.push_back(g3pvm::evo::canonicalize_fitness_for_ranking(one.fitness));
    }

    g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
    if (!g3pvm::evo::repro::upload_gpu_repro_inputs(prepared.packed, &arena, &run_stats, &message) ||
        !g3pvm::evo::repro::launch_gpu_repro_kernels(&arena, prepared.config, ranked_fitness, &run_stats, &message)) {
      std::cerr << "FAIL: linear metadata gpu kernels failed: " << message << "\n";
      return false;
    }

    g3pvm::evo::repro::GpuReproChildView copyback;
    if (!g3pvm::evo::repro::copyback_gpu_repro_children(
            arena, prepared.config, &staging, &copyback, &run_stats, &message)) {
      std::cerr << "FAIL: linear metadata gpu copyback failed: " << message << "\n";
      return false;
    }

    bool saw_valid_linear_child = false;
    const int child_count = prepared.config.pair_count * 2;
    for (int child = 0; child < child_count; ++child) {
      if (copyback.child_meta[child].valid == 0) {
        continue;
      }
      const int node_base = copyback.child_node_offsets[child];
      for (int i = 0; i < copyback.child_used_len[child]; ++i) {
        if (static_cast<g3pvm::evo::NodeKind>(copyback.child_nodes[node_base + i].kind) ==
            g3pvm::evo::NodeKind::LINEAR_REC) {
          saw_valid_linear_child = true;
        }
      }
    }
    if (!check(saw_valid_linear_child, "gpu reproduction should mark LinearRec children valid")) {
      return false;
    }

    const auto decoded =
        g3pvm::evo::repro::decode_gpu_repro_children(prepared.packed, copyback, scored, cfg);
    if (!check(static_cast<int>(decoded.size()) == cfg.population_size,
               "linear metadata decoded child count mismatch")) {
      return false;
    }
    for (const auto& child : decoded) {
      if (!check(program_contains_linear_rec(child), "linear metadata child should contain LINEAR_REC")) {
        return false;
      }
      if (!check(!child.ast.linear_rec_binders.empty(),
                 "linear metadata child should preserve LinearRec binders")) {
        return false;
      }
      const auto bc = g3pvm::evo::compile_for_eval(child);
      if (!check(!bc.code.empty(), "linear metadata child should compile")) {
        return false;
      }
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP linear metadata gpu prepared (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: linear metadata gpu reproduction failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

bool test_gpu_prepared_backend_scalar_config_smoke() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 4;
  cfg.selection_pressure = 3;
  cfg.seed = 42;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.grammar = g3pvm::evo::GrammarConfig::scalar();
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population =
      make_population(cfg.population_size, cfg.grammar);
  std::vector<g3pvm::evo::ScoredGenome> scored;
  scored.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    scored.push_back(g3pvm::evo::ScoredGenome{
        population[i],
        static_cast<double>(population.size() - i),
    });
  }

  g3pvm::evo::repro::ReproductionStats prep_stats;
  try {
    const auto prepared =
        g3pvm::evo::repro::prepare_gpu_repro_backend_inputs(population, cfg, 123, &prep_stats);
    g3pvm::evo::repro::ReproductionStats run_stats = prep_stats;
    const auto reproduction =
        g3pvm::evo::repro::run_gpu_repro_backend_prepared(scored, cfg, prepared, &run_stats);
    if (!check(static_cast<int>(reproduction.next_population.size()) == cfg.population_size,
               "scalar config gpu reproduction child count mismatch")) {
      return false;
    }
    for (const auto& child : reproduction.next_population) {
      const auto bc = g3pvm::evo::compile_for_eval(child);
      if (!check(!bc.code.empty(), "scalar config gpu reproduction child should compile")) {
        return false;
      }
    }
  } catch (const std::runtime_error& err) {
    const std::string message = err.what();
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP scalar gpu prepared (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: scalar config gpu reproduction prepared backend failed: " << message << "\n";
    return false;
  }
#endif
  return true;
}

bool test_gpu_selection_preserves_round_based_tournament_invariants() {
#ifdef G3PVM_HAS_CUDA
  g3pvm::evo::EvolutionConfig cfg;
  cfg.population_size = 8;
  cfg.seed = 42;
  cfg.reproduction_backend = g3pvm::evo::repro::ReproductionBackend::Gpu;
  cfg.limits.max_expr_depth = 5;
  cfg.limits.max_stmts_per_block = 6;
  cfg.limits.max_total_nodes = 80;
  cfg.limits.max_for_k = 16;
  cfg.limits.max_call_args = 3;

  const std::vector<g3pvm::evo::ProgramGenome> population = make_population(cfg.population_size);
  std::vector<double> ranked_fitness;
  ranked_fitness.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    ranked_fitness.push_back(
        g3pvm::evo::canonicalize_fitness_for_ranking(static_cast<double>(i + 1)));
  }

  std::string message;
  std::vector<int> parent_a;
  std::vector<int> parent_b;

  cfg.selection_pressure = cfg.population_size;
  if (!copyback_gpu_selection_parents(cfg, population, ranked_fitness, 777, &parent_a, &parent_b, &message)) {
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP gpu selection invariants (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu selection invariant setup failed: " << message << "\n";
    return false;
  }
  const int best_index = static_cast<int>(population.size()) - 1;
  for (std::size_t i = 0; i < parent_a.size(); ++i) {
    if (!check(parent_a[i] == best_index && parent_b[i] == best_index,
               "full-pressure GPU selection should always pick the best genome")) {
      return false;
    }
  }

  cfg.selection_pressure = 1;
  message.clear();
  if (!copyback_gpu_selection_parents(cfg, population, ranked_fitness, 778, &parent_a, &parent_b, &message)) {
    if (message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "g3pvm_test_repro_prep: SKIP gpu selection invariants (" << message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu selection invariant rerun failed: " << message << "\n";
    return false;
  }

  std::vector<bool> seen(population.size(), false);
  for (std::size_t pair = 0; pair < parent_a.size(); ++pair) {
    for (int parent : {parent_a[pair], parent_b[pair]}) {
      if (!check(parent >= 0 && parent < static_cast<int>(population.size()),
                 "gpu selection parent index out of range")) {
        return false;
      }
      if (!check(!seen[static_cast<std::size_t>(parent)],
                 "k=1 GPU selection should not repeat winners within the first round")) {
        return false;
      }
      seen[static_cast<std::size_t>(parent)] = true;
    }
  }
  for (bool one_seen : seen) {
    if (!check(one_seen, "k=1 GPU selection should visit every genome exactly once")) {
      return false;
    }
  }
#endif
  return true;
}

}  // namespace

int main() {
  if (!test_preprocess_and_pack()) return 1;
  if (!test_gpu_repro_prep_includes_asgp_candidates()) return 1;
  if (!test_gpu_repro_backend_preserves_asgp_metadata()) return 1;
  if (!test_gpu_decode_preserves_asgp_metadata()) return 1;
  if (!test_gpu_donor_pool_emits_asgp_dc_metadata()) return 1;
  if (!test_gpu_donor_pool_emits_asgp_dp_metadata()) return 1;
  if (!test_gpu_donor_pool_preserves_target_runtime_type()) return 1;
  if (!test_preprocess_respects_scalar_grammar_config()) return 1;
  if (!test_gpu_preprocess_packs_typed_candidate_keys()) return 1;
  if (!test_gpu_repro_compacts_dead_tables_before_pack()) return 1;
  if (!test_decode_falls_back_on_escaped_bound_var_child()) return 1;
  if (!test_gpu_prepared_backend_smoke()) return 1;
  if (!test_gpu_prepared_backend_preserves_linear_rec_metadata()) return 1;
  if (!test_gpu_prepared_backend_scalar_config_smoke()) return 1;
  if (!test_gpu_selection_preserves_round_based_tournament_invariants()) return 1;
  std::cout << "g3pvm_test_repro_prep: OK\n";
  return 0;
}
