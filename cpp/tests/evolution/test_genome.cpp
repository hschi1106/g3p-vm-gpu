#include <algorithm>
#include <iostream>
#include <string>

#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"
#include "../../src/evolution/subtree_utils.hpp"
#include "../../src/evolution/typed_expr_analysis.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool test_random_genome_compile_rate() {
  g3pvm::evo::Limits limits;
  const int n = 200;
  int compiled = 0;
  for (int i = 0; i < n; ++i) {
    const g3pvm::evo::ProgramGenome g = g3pvm::evo::generate_random_genome(static_cast<std::uint64_t>(i), limits);
    (void)g3pvm::evo::compile_for_eval(g);
    compiled += 1;
  }
  return check(static_cast<double>(compiled) / static_cast<double>(n) >= 0.99, "compile rate < 99%");
}

bool test_mutation_and_crossover_invariants() {
  g3pvm::evo::Limits limits;
  const g3pvm::evo::ProgramGenome a = g3pvm::evo::generate_random_genome(1, limits);
  const g3pvm::evo::ProgramGenome b = g3pvm::evo::generate_random_genome(2, limits);

  for (int i = 0; i < 80; ++i) {
    const g3pvm::evo::ProgramGenome m =
        g3pvm::evo::mutate(a, static_cast<std::uint64_t>(1000 + i), limits, 0.8);
    (void)g3pvm::evo::compile_for_eval(m);

    const auto children = g3pvm::evo::crossover(a, b, static_cast<std::uint64_t>(3000 + i), limits);
    (void)g3pvm::evo::compile_for_eval(children.first);
    (void)g3pvm::evo::compile_for_eval(children.second);
  }

  return true;
}

bool test_for_k_constraints() {
  g3pvm::evo::Limits limits{5, 6, 80, 8, 3};
  const g3pvm::evo::ProgramGenome base = g3pvm::evo::generate_random_genome(88, limits);
  for (int i = 0; i < 60; ++i) {
    const g3pvm::evo::ProgramGenome child =
        g3pvm::evo::mutate(base, static_cast<std::uint64_t>(10000 + i), limits, 0.8);
    (void)g3pvm::evo::compile_for_eval(child);
  }
  return true;
}

bool test_ast_cache_key_distinguishes_program_payload() {
  using g3pvm::Value;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;

  AstProgram a;
  a.nodes = {AstNode{NodeKind::PROGRAM, 0, 0}, AstNode{NodeKind::RETURN, 0, 0}, AstNode{NodeKind::CONST, 0, 0}};
  a.names = {"x"};
  a.consts = {Value::from_int(1)};
  a.version = "ast-prefix-v1";

  AstProgram b = a;
  b.consts[0] = Value::from_int(2);

  AstProgram c = a;
  c.names = {"y"};

  AstProgram d = a;
  d.version = "ast-prefix-v2";

  const std::string key_a = g3pvm::evo::ast_cache_key(a);
  const std::string key_b = g3pvm::evo::ast_cache_key(b);
  const std::string key_c = g3pvm::evo::ast_cache_key(c);
  const std::string key_d = g3pvm::evo::ast_cache_key(d);

  if (!check(key_a != key_b, "ast cache key should distinguish const payload")) {
    return false;
  }
  if (!check(key_a != key_c, "ast cache key should distinguish names")) {
    return false;
  }
  if (!check(key_a != key_d, "ast cache key should distinguish version")) {
    return false;
  }
  if (!check(key_a == g3pvm::evo::ast_cache_key(a), "ast cache key should be stable")) {
    return false;
  }
  return true;
}

bool test_build_genome_meta_tracks_max_expr_depth() {
  using g3pvm::Value;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;

  AstProgram program;
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::NEG, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.consts = {Value::from_int(1), Value::from_int(2), Value::from_int(3)};

  const g3pvm::evo::GenomeMeta meta = g3pvm::evo::build_genome_meta(program);
  return check(meta.max_depth == 3, "genome meta should track max expression depth");
}

bool test_random_genome_uses_requested_input_specs() {
  g3pvm::evo::Limits limits;
  const std::vector<g3pvm::evo::InputSpec> inputs = {{"xs", g3pvm::evo::RType::NumList}};
  bool saw_xs_var = false;
  for (std::uint64_t seed = 0; seed < 256; ++seed) {
    const g3pvm::evo::ProgramGenome g =
        g3pvm::evo::generate_random_genome_for_return_type(seed, g3pvm::evo::RType::Num, limits, inputs);
    if (!check(std::find(g.ast.names.begin(), g.ast.names.end(), "xs") != g.ast.names.end(),
               "generated genome should preserve requested input name")) {
      return false;
    }
    for (const g3pvm::evo::AstNode& node : g.ast.nodes) {
      if (node.kind == g3pvm::evo::NodeKind::VAR &&
          node.i0 >= 0 &&
          static_cast<std::size_t>(node.i0) < g.ast.names.size() &&
          g.ast.names[static_cast<std::size_t>(node.i0)] == "xs") {
        saw_xs_var = true;
        break;
      }
    }
    if (saw_xs_var) {
      break;
    }
  }
  return check(saw_xs_var, "generator should sometimes emit the requested list input as a variable");
}

bool test_typed_expr_analysis_treats_list_index_as_num() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::RType;
  using g3pvm::evo::typed_expr::TypedExprRoot;

  AstProgram program;
  program.names = {"xs"};
  program.consts = {Value::from_int(0), Value::from_int(1)};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CALL_MAX, 0, 0},
      AstNode{NodeKind::CALL_INDEX, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CALL_INDEX, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };

  const std::vector<std::size_t> subtree_end = g3pvm::evo::subtree::build_subtree_end(program);
  const std::vector<TypedExprRoot> roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(program, subtree_end);

  bool saw_index_num = false;
  bool saw_max_num = false;
  for (const TypedExprRoot& root : roots) {
    if (root.type != RType::Num) {
      continue;
    }
    if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::CALL_INDEX) {
      saw_index_num = true;
    }
    if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::CALL_MAX) {
      saw_max_num = true;
    }
  }
  if (!check(saw_index_num, "index(xs, i) should be tracked as a numeric typed root")) {
    return false;
  }
  return check(saw_max_num, "max(index(xs,0), index(xs,1)) should be tracked as a numeric typed root");
}

bool test_subtree_donor_generation_uses_existing_list_input_names() {
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::RType;

  std::mt19937_64 rng(123);
  AstProgram target;
  target.names = {"xs"};

  bool saw_xs_var = false;
  bool saw_spurious_x = false;
  for (int i = 0; i < 256; ++i) {
    AstProgram donor;
    donor.names = target.names;
    donor.nodes = g3pvm::evo::subtree::make_random_expr_nodes_for_type(rng, donor, RType::Num, 3);
    for (const auto& name : donor.names) {
      if (name == "x") {
        saw_spurious_x = true;
      }
    }
    for (const auto& node : donor.nodes) {
      if (node.kind == g3pvm::evo::NodeKind::VAR &&
          node.i0 >= 0 &&
          static_cast<std::size_t>(node.i0) < donor.names.size() &&
          donor.names[static_cast<std::size_t>(node.i0)] == "xs") {
        saw_xs_var = true;
      }
    }
  }
  if (!check(!saw_spurious_x, "subtree donor generation should not invent x when target only has xs")) {
    return false;
  }
  return check(saw_xs_var, "subtree donor generation should sometimes reference xs directly");
}

bool test_for_range_expr_compiles_and_executes() {
  using g3pvm::Value;
  using g3pvm::ValueTag;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  AstProgram program;
  program.names = {"x", "i"};
  program.consts = {
      Value::from_int(0),
      g3pvm::payload::make_num_list_value({
          Value::from_int(10),
          Value::from_int(20),
          Value::from_int(30),
          Value::from_int(40),
      }),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::FOR_RANGE, 1, 0},
      AstNode{NodeKind::CALL_LEN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::VAR, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };

  ProgramGenome genome;
  genome.ast = program;
  genome.meta = g3pvm::evo::build_genome_meta(program);
  if (!check(genome.meta.max_depth == 2, "FOR_RANGE bound should contribute expression depth")) {
    return false;
  }

  const g3pvm::BytecodeProgram bytecode = g3pvm::evo::compile_for_eval(genome);
  const g3pvm::ExecResult out = g3pvm::execute_bytecode_cpu(bytecode, {}, 20000);
  if (!check(!out.is_error, "FOR_RANGE program should execute successfully")) {
    return false;
  }
  if (!check(out.value.tag == ValueTag::Int, "FOR_RANGE program should return int")) {
    return false;
  }
  return check(out.value.i == 6, "FOR_RANGE program should sum loop indices");
}

bool scalar_config_allows_program(const g3pvm::evo::ProgramGenome& genome) {
  const g3pvm::evo::GrammarConfig cfg = g3pvm::evo::GrammarConfig::scalar();
  for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
    if (!cfg.allows_node_kind(node.kind)) {
      return false;
    }
  }
  for (const g3pvm::Value& value : genome.ast.consts) {
    if (value.tag == g3pvm::ValueTag::String ||
        value.tag == g3pvm::ValueTag::NumList ||
        value.tag == g3pvm::ValueTag::StringList) {
      return false;
    }
  }
  return true;
}

bool test_scalar_grammar_config_restricts_generation_and_mutation() {
  const g3pvm::evo::GrammarConfig cfg = g3pvm::evo::GrammarConfig::scalar();
  g3pvm::evo::Limits limits;
  for (std::uint64_t seed = 0; seed < 160; ++seed) {
    const g3pvm::evo::ProgramGenome genome = g3pvm::evo::generate_random_genome(seed, limits, cfg);
    if (!check(scalar_config_allows_program(genome), "scalar grammar config should not generate sequence features")) {
      return false;
    }
    (void)g3pvm::evo::compile_for_eval(genome);
  }

  const g3pvm::evo::ProgramGenome base = g3pvm::evo::generate_random_genome(4242, limits, cfg);
  for (std::uint64_t seed = 0; seed < 80; ++seed) {
    const g3pvm::evo::ProgramGenome child =
        g3pvm::evo::mutate(base, 9000 + seed, limits, 0.8, cfg);
    if (!check(scalar_config_allows_program(child), "scalar grammar config should not mutate in sequence features")) {
      return false;
    }
    (void)g3pvm::evo::compile_for_eval(child);
  }
  return true;
}

}  // namespace

int main() {
  if (!test_random_genome_compile_rate()) return 1;
  if (!test_mutation_and_crossover_invariants()) return 1;
  if (!test_for_k_constraints()) return 1;
  if (!test_ast_cache_key_distinguishes_program_payload()) return 1;
  if (!test_build_genome_meta_tracks_max_expr_depth()) return 1;
  if (!test_random_genome_uses_requested_input_specs()) return 1;
  if (!test_typed_expr_analysis_treats_list_index_as_num()) return 1;
  if (!test_subtree_donor_generation_uses_existing_list_input_names()) return 1;
  if (!test_for_range_expr_compiles_and_executes()) return 1;
  if (!test_scalar_grammar_config_restricts_generation_and_mutation()) return 1;
  std::cout << "g3pvm_test_genome: OK\n";
  return 0;
}
