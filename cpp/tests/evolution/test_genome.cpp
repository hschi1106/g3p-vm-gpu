#include <algorithm>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

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

bool test_random_genome_emits_structured_expressions_when_enabled() {
  using g3pvm::evo::NodeKind;

  g3pvm::evo::Limits limits{7, 6, 160, 16, 3};
  bool saw_map = false;
  bool saw_filter = false;
  bool saw_linear = false;
  bool saw_asgp = false;
  for (int i = 0; i < 1200; ++i) {
    const g3pvm::evo::ProgramGenome g =
        g3pvm::evo::generate_random_genome(static_cast<std::uint64_t>(9000 + i), limits);
    (void)g3pvm::evo::compile_for_eval(g);
    for (const g3pvm::evo::AstNode& node : g.ast.nodes) {
      saw_map = saw_map || node.kind == NodeKind::MAP_LIST;
      saw_filter = saw_filter || node.kind == NodeKind::FILTER_LIST;
      saw_linear = saw_linear || node.kind == NodeKind::LINEAR_REC;
      saw_asgp = saw_asgp || node.kind == NodeKind::ASGP_DC ||
                  node.kind == NodeKind::ASGP_DP1D ||
                  node.kind == NodeKind::ASGP_DP2D;
    }
    if (saw_map && saw_filter && saw_linear && saw_asgp) {
      return true;
    }
  }
  if (!check(saw_map, "random generation should emit MAP_LIST when enabled")) return false;
  if (!check(saw_filter, "random generation should emit FILTER_LIST when enabled")) return false;
  if (!check(saw_asgp, "random generation should emit ASGP-DC when enabled")) return false;
  return check(saw_linear, "random generation should emit LINEAR_REC when enabled");
}

bool test_subtree_mutation_emits_structured_list_donors_when_enabled() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  ProgramGenome base;
  base.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  base.ast.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
  };
  base.meta = g3pvm::evo::build_genome_meta(base.ast);

  g3pvm::evo::Limits limits{7, 6, 160, 16, 3};
  bool saw_map_or_filter = false;
  for (int i = 0; i < 200; ++i) {
    const ProgramGenome child =
        g3pvm::evo::mutate(base, static_cast<std::uint64_t>(12000 + i), limits, 1.0);
    (void)g3pvm::evo::compile_for_eval(child);
    for (const AstNode& node : child.ast.nodes) {
      if (node.kind == NodeKind::MAP_LIST || node.kind == NodeKind::FILTER_LIST) {
        saw_map_or_filter = true;
      }
    }
    if (saw_map_or_filter) return true;
  }
  return check(false, "subtree mutation should emit structured list donors when enabled");
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
  a.version = g3pvm::evo::k_ast_prefix_version_current;

  AstProgram b = a;
  b.consts[0] = Value::from_int(2);

  AstProgram c = a;
  c.names = {"y"};

  AstProgram d = a;
  d.version = "ast-prefix-future";

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

bool test_ast_prefix_old_is_rejected() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;

  g3pvm::evo::ProgramGenome genome;
  genome.ast.version = "ast-prefix-old";
  genome.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  genome.ast.consts = {Value::from_int(1)};
  try {
    (void)g3pvm::evo::compile_for_eval(genome);
  } catch (const std::runtime_error& e) {
    return check(std::string(e.what()).find("unsupported ast prefix version") != std::string::npos,
                 "ast-prefix-old should fail with version error");
  }
  return check(false, "ast-prefix-old should be rejected");
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
  const std::vector<g3pvm::evo::InputSpec> inputs = {{"xs", g3pvm::evo::RType::IntList}};
  bool saw_xs_var = false;
  for (std::uint64_t seed = 0; seed < 256; ++seed) {
    const g3pvm::evo::ProgramGenome g =
        g3pvm::evo::generate_random_genome_for_return_type(seed, g3pvm::evo::RType::Int, limits, inputs);
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

bool test_typed_expr_analysis_tracks_exact_list_index_types() {
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

  bool saw_index_int = false;
  bool saw_max_int = false;
  for (const TypedExprRoot& root : roots) {
    if (root.type != RType::Int) {
      continue;
    }
    if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::CALL_INDEX) {
      saw_index_int = true;
    }
    if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::CALL_MAX) {
      saw_max_int = true;
    }
  }
  if (!check(saw_index_int, "index(xs, i) should be tracked as an Int typed root")) {
    return false;
  }
  if (!check(saw_max_int, "max(index(xs,0), index(xs,1)) should be tracked as an Int typed root")) {
    return false;
  }

  AstProgram float_program;
  float_program.names = {"float_list"};
  float_program.consts = {Value::from_int(0)};
  float_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CALL_INDEX, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };

  const std::vector<std::size_t> float_subtree_end = g3pvm::evo::subtree::build_subtree_end(float_program);
  const std::vector<TypedExprRoot> float_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(float_program, float_subtree_end);
  bool saw_index_float = false;
  for (const TypedExprRoot& root : float_roots) {
    if (root.type == RType::Float &&
        root.start < float_program.nodes.size() &&
        float_program.nodes[root.start].kind == NodeKind::CALL_INDEX) {
      saw_index_float = true;
    }
  }
  if (!check(saw_index_float, "index(float_list, i) should be tracked as a Float typed root")) {
    return false;
  }

  auto has_root = [](const AstProgram& p,
                     const std::vector<TypedExprRoot>& roots,
                     std::size_t start,
                     NodeKind kind,
                     RType type) {
    return std::any_of(roots.begin(), roots.end(), [&](const TypedExprRoot& root) {
      return root.start == start && root.type == type &&
             root.start < p.nodes.size() && p.nodes[root.start].kind == kind;
    });
  };

  AstProgram string_list_program;
  string_list_program.consts = {
      g3pvm::payload::make_string_value("a"),
      g3pvm::payload::make_string_value("b"),
      g3pvm::payload::make_string_value("c"),
  };
  string_list_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CALL_REVERSE, 0, 0},
      AstNode{NodeKind::CALL_PREPEND, 0, 0},
      AstNode{NodeKind::CALL_APPEND, 0, 0},
      AstNode{NodeKind::CALL_SINGLETON, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  const std::vector<std::size_t> string_list_end = g3pvm::evo::subtree::build_subtree_end(string_list_program);
  const std::vector<TypedExprRoot> string_list_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(string_list_program, string_list_end);
  if (!check(has_root(string_list_program, string_list_roots, 6, NodeKind::CALL_SINGLETON, RType::StringList),
             "singleton(String) should be tracked as a StringList typed root")) {
    return false;
  }
  if (!check(has_root(string_list_program, string_list_roots, 5, NodeKind::CALL_APPEND, RType::StringList),
             "append(StringList, String) should be tracked as a StringList typed root")) {
    return false;
  }
  if (!check(has_root(string_list_program, string_list_roots, 4, NodeKind::CALL_PREPEND, RType::StringList),
             "prepend(StringList, String) should be tracked as a StringList typed root")) {
    return false;
  }
  if (!check(has_root(string_list_program, string_list_roots, 3, NodeKind::CALL_REVERSE, RType::StringList),
             "reverse(StringList) should be tracked as a StringList typed root")) {
    return false;
  }

  AstProgram char_singleton_program;
  char_singleton_program.consts = {Value::from_char('q')};
  char_singleton_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CALL_SINGLETON, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  const std::vector<std::size_t> char_singleton_end =
      g3pvm::evo::subtree::build_subtree_end(char_singleton_program);
  const std::vector<TypedExprRoot> char_singleton_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(char_singleton_program, char_singleton_end);
  if (!check(has_root(char_singleton_program, char_singleton_roots, 3, NodeKind::CALL_SINGLETON, RType::String),
             "singleton(Char) should be tracked as a String typed root")) {
    return false;
  }

  AstProgram valid_float_append_program;
  valid_float_append_program.consts = {
      g3pvm::payload::make_float_list_value({Value::from_float(1.0)}),
      Value::from_float(2.5),
  };
  valid_float_append_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CALL_APPEND, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  const std::vector<std::size_t> valid_float_append_end =
      g3pvm::evo::subtree::build_subtree_end(valid_float_append_program);
  const std::vector<TypedExprRoot> valid_float_append_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(valid_float_append_program, valid_float_append_end);
  if (!check(has_root(valid_float_append_program, valid_float_append_roots, 3, NodeKind::CALL_APPEND, RType::FloatList),
             "append(FloatList, Float) should be tracked as a FloatList typed root")) {
    return false;
  }

  AstProgram invalid_float_append_program = valid_float_append_program;
  invalid_float_append_program.consts[1] = Value::from_int(2);
  const std::vector<std::size_t> invalid_float_append_end =
      g3pvm::evo::subtree::build_subtree_end(invalid_float_append_program);
  const std::vector<TypedExprRoot> invalid_float_append_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(invalid_float_append_program, invalid_float_append_end);
  return check(!has_root(invalid_float_append_program,
                         invalid_float_append_roots,
                         3,
                         NodeKind::CALL_APPEND,
                         RType::FloatList),
               "append(FloatList, Int) should not be exposed as a FloatList typed root");
}

bool test_current_if_type_scope_analysis() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::RType;
  using g3pvm::evo::typed_expr::TypedExprRoot;

  auto root_at = [](const std::vector<TypedExprRoot>& roots,
                    std::size_t start,
                    RType type) -> const TypedExprRoot* {
    for (const TypedExprRoot& root : roots) {
      if (root.start == start && root.type == type) return &root;
    }
    return nullptr;
  };

  AstProgram valid_if_expr;
  valid_if_expr.consts = {Value::from_bool(true), Value::from_int(7), Value::from_int(11)};
  valid_if_expr.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::IF_EXPR, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  const std::vector<std::size_t> valid_if_expr_end =
      g3pvm::evo::subtree::build_subtree_end(valid_if_expr);
  const std::vector<TypedExprRoot> valid_if_expr_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(valid_if_expr, valid_if_expr_end);
  if (!check(root_at(valid_if_expr_roots, 3, RType::Int) != nullptr,
             "IF_EXPR with Bool condition and matching branch types should expose an Int root")) {
    return false;
  }

  AstProgram mismatched_branch_if_expr = valid_if_expr;
  mismatched_branch_if_expr.consts[2] = Value::from_float(11.0);
  const std::vector<std::size_t> mismatched_branch_end =
      g3pvm::evo::subtree::build_subtree_end(mismatched_branch_if_expr);
  const std::vector<TypedExprRoot> mismatched_branch_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(mismatched_branch_if_expr, mismatched_branch_end);
  if (!check(root_at(mismatched_branch_roots, 3, RType::Int) == nullptr &&
             root_at(mismatched_branch_roots, 3, RType::Float) == nullptr,
             "IF_EXPR with mismatched branch types should not expose a typed root")) {
    return false;
  }

  AstProgram non_bool_if_expr = valid_if_expr;
  non_bool_if_expr.consts[0] = Value::from_int(1);
  const std::vector<std::size_t> non_bool_if_expr_end =
      g3pvm::evo::subtree::build_subtree_end(non_bool_if_expr);
  const std::vector<TypedExprRoot> non_bool_if_expr_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(non_bool_if_expr, non_bool_if_expr_end);
  if (!check(root_at(non_bool_if_expr_roots, 3, RType::Int) == nullptr,
             "IF_EXPR with non-Bool condition should not expose a typed root")) {
    return false;
  }

  AstProgram if_stmt_program;
  if_stmt_program.names = {"x", "y"};
  if_stmt_program.consts = {
      Value::from_bool(true),
      Value::from_int(0),
      Value::from_float(0.5),
      Value::from_int(1),
      Value::from_int(2),
  };
  if_stmt_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::IF_STMT, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 1, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 1, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::CONST, 4, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  const std::vector<std::size_t> if_stmt_end =
      g3pvm::evo::subtree::build_subtree_end(if_stmt_program);
  const std::vector<TypedExprRoot> if_stmt_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(if_stmt_program, if_stmt_end);
  const TypedExprRoot* then_const = root_at(if_stmt_roots, 11, RType::Int);
  const TypedExprRoot* else_const = root_at(if_stmt_roots, 19, RType::Int);
  if (!check(then_const != nullptr && else_const != nullptr,
             "IF_STMT branches should expose same-type typed roots inside branch-local environments")) {
    return false;
  }
  return check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*then_const, *else_const),
               "IF_STMT branch-local visible environments should prevent cross-branch typed subtree matches");
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
    donor.nodes = g3pvm::evo::subtree::make_random_expr_nodes_for_type(rng, donor, RType::Int, 3);
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

bool test_subtree_mutation_emits_asgp_dc_donors_when_enabled() {
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  ProgramGenome base;
  base.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  base.ast.consts = {g3pvm::Value::from_int(0)};
  base.meta = g3pvm::evo::build_genome_meta(base.ast);

  g3pvm::evo::Limits limits{7, 6, 160, 16, 3};
  for (int i = 0; i < 300; ++i) {
    const ProgramGenome child =
        g3pvm::evo::mutate(base, static_cast<std::uint64_t>(15000 + i), limits, 1.0);
    bool saw_asgp_dc = false;
    for (const AstNode& node : child.ast.nodes) {
      saw_asgp_dc = saw_asgp_dc || node.kind == NodeKind::ASGP_DC;
    }
    if (!saw_asgp_dc) {
      continue;
    }
    if (!check(!child.ast.asgp_dc_binders.empty(),
               "ASGP-DC subtree mutation donor should carry binder metadata")) {
      return false;
    }
    (void)g3pvm::evo::compile_for_eval(child);
    return true;
  }
  return check(false, "subtree mutation should emit ASGP-DC donors when enabled");
}

g3pvm::evo::GrammarConfig grammar_with_only_asgp_form(g3pvm::evo::NodeKind form) {
  g3pvm::evo::GrammarConfig grammar;
  grammar.expression_asgp_dc = form == g3pvm::evo::NodeKind::ASGP_DC;
  grammar.expression_asgp_dp1d = form == g3pvm::evo::NodeKind::ASGP_DP1D;
  grammar.expression_asgp_dp2d = form == g3pvm::evo::NodeKind::ASGP_DP2D;
  return grammar;
}

bool test_asgp_dc_generation_can_use_existing_list_source() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::InputSpec;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;
  using g3pvm::evo::RType;

  const g3pvm::evo::GrammarConfig grammar = grammar_with_only_asgp_form(NodeKind::ASGP_DC);
  g3pvm::evo::Limits limits{8, 6, 180, 16, 3};

  {
    std::mt19937_64 rng(31000);
    bool saw_var_source = false;
    for (int i = 0; i < 800; ++i) {
      AstProgram donor;
      donor.version = g3pvm::evo::k_ast_prefix_version_current;
      donor.names = {"xs"};
      donor.nodes = g3pvm::evo::subtree::make_random_expr_nodes_for_type(
          rng, donor, RType::Int, 5, grammar, true);
      if (donor.nodes.size() < 2 || donor.nodes[0].kind != NodeKind::ASGP_DC ||
          donor.nodes[1].kind != NodeKind::VAR || donor.names[static_cast<std::size_t>(donor.nodes[1].i0)] != "xs") {
        continue;
      }
      saw_var_source = true;

      AstProgram base;
      base.version = g3pvm::evo::k_ast_prefix_version_current;
      base.names = {"xs"};
      base.consts = {
          g3pvm::payload::make_int_list_value({Value::from_int(2), Value::from_int(3), Value::from_int(4)}),
          Value::from_int(0),
      };
      base.nodes = {
          AstNode{NodeKind::PROGRAM, 0, 0},
          AstNode{NodeKind::BLOCK_CONS, 0, 0},
          AstNode{NodeKind::ASSIGN, 0, 0},
          AstNode{NodeKind::CONST, 0, 0},
          AstNode{NodeKind::BLOCK_CONS, 0, 0},
          AstNode{NodeKind::RETURN, 0, 0},
          AstNode{NodeKind::CONST, 1, 0},
          AstNode{NodeKind::BLOCK_NIL, 0, 0},
      };
      ProgramGenome wrapped;
      wrapped.ast = g3pvm::evo::subtree::replace_subtree(base, 6, 7, donor, 0, donor.nodes.size());
      wrapped.meta = g3pvm::evo::build_genome_meta(wrapped.ast);
      const g3pvm::ExecResult out =
          g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(wrapped), {}, 20000);
      if (!check(!out.is_error, "ASGP-DC donor with existing list source should execute on CPU")) {
        return false;
      }
      break;
    }
    if (!check(saw_var_source, "ASGP-DC subtree donor should sometimes use existing IntList source")) {
      return false;
    }
  }

  for (int i = 0; i < 2500; ++i) {
    const ProgramGenome genome = g3pvm::evo::generate_random_genome_for_return_type(
        32000 + static_cast<std::uint64_t>(i),
        RType::Int,
        limits,
        std::vector<InputSpec>{{"xs", RType::IntList}},
        grammar);
    for (std::size_t n = 0; n + 1 < genome.ast.nodes.size(); ++n) {
      if (genome.ast.nodes[n].kind == NodeKind::ASGP_DC &&
          genome.ast.nodes[n + 1].kind == NodeKind::VAR &&
          genome.ast.names[static_cast<std::size_t>(genome.ast.nodes[n + 1].i0)] == "xs") {
        (void)g3pvm::evo::compile_for_eval(genome);
        return true;
      }
    }
  }
  return check(false, "ASGP-DC random generation should sometimes use existing IntList input source");
}

bool test_asgp_dc_generation_can_emit_string_roots() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::InputSpec;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;
  using g3pvm::evo::RType;

  const g3pvm::evo::GrammarConfig grammar = grammar_with_only_asgp_form(NodeKind::ASGP_DC);
  g3pvm::evo::Limits limits{8, 6, 180, 16, 3};

  {
    ProgramGenome base;
    base.ast.version = g3pvm::evo::k_ast_prefix_version_current;
    base.ast.consts = {g3pvm::payload::make_string_value("")};
    base.ast.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    base.meta = g3pvm::evo::build_genome_meta(base.ast);

    bool saw_string_asgp_dc = false;
    for (int i = 0; i < 1800; ++i) {
      const ProgramGenome child =
          g3pvm::evo::mutate(base, 54000 + static_cast<std::uint64_t>(i), limits, 1.0, grammar);
      bool saw_form = false;
      bool saw_concat_phase = false;
      for (const AstNode& node : child.ast.nodes) {
        saw_form = saw_form || node.kind == NodeKind::ASGP_DC;
        saw_concat_phase = saw_concat_phase || node.kind == NodeKind::CALL_CONCAT;
      }
      if (!saw_form || !saw_concat_phase) {
        continue;
      }
      saw_string_asgp_dc = true;
      if (!check(!child.ast.asgp_dc_binders.empty(),
                 "String ASGP-DC mutation donor should carry binder metadata")) {
        return false;
      }
      const g3pvm::ExecResult out =
          g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(child), {}, 20000);
      if (!check(!out.is_error, "String ASGP-DC subtree mutation donor should execute on CPU")) {
        return false;
      }
      if (!check(out.value.tag == g3pvm::ValueTag::String,
                 "String ASGP-DC subtree mutation donor should return String")) {
        return false;
      }
      break;
    }
    if (!check(saw_string_asgp_dc, "subtree mutation should emit String ASGP-DC donors when enabled")) {
      return false;
    }
  }

  for (int i = 0; i < 3500; ++i) {
    const ProgramGenome genome = g3pvm::evo::generate_random_genome_for_return_type(
        56000 + static_cast<std::uint64_t>(i),
        RType::String,
        limits,
        std::vector<InputSpec>{{"text", RType::String}},
        grammar);
    bool saw_form = false;
    bool saw_concat_phase = false;
    for (const AstNode& node : genome.ast.nodes) {
      saw_form = saw_form || node.kind == NodeKind::ASGP_DC;
      saw_concat_phase = saw_concat_phase || node.kind == NodeKind::CALL_CONCAT;
    }
    if (!saw_form || !saw_concat_phase) {
      continue;
    }
    if (!check(!genome.ast.asgp_dc_binders.empty(),
               "random String ASGP-DC generation should carry binder metadata")) {
      return false;
    }
    (void)g3pvm::evo::compile_for_eval(genome);
    return true;
  }

  return check(false, "random generation should emit String ASGP-DC when enabled");
}

bool test_subtree_mutation_emits_asgp_dp_donors_when_enabled() {
  using g3pvm::evo::AstNode;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  auto base_for_type = [](g3pvm::evo::RType type) {
    ProgramGenome base;
    base.ast.version = g3pvm::evo::k_ast_prefix_version_current;
    base.ast.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    if (type == g3pvm::evo::RType::String) {
      base.ast.consts = {g3pvm::payload::make_string_value("")};
    } else {
      base.ast.consts = {type == g3pvm::evo::RType::Float ? g3pvm::Value::from_float(0.0)
                                                          : g3pvm::Value::from_int(0)};
    }
    base.meta = g3pvm::evo::build_genome_meta(base.ast);
    return base;
  };

  auto seek_form = [&](NodeKind form, g3pvm::evo::RType type, std::uint64_t seed_base) {
    const g3pvm::evo::GrammarConfig grammar = grammar_with_only_asgp_form(form);
    const ProgramGenome base = base_for_type(type);
    g3pvm::evo::Limits limits{7, 6, 160, 16, 3};
    for (int i = 0; i < 1200; ++i) {
      const ProgramGenome child =
          g3pvm::evo::mutate(base, seed_base + static_cast<std::uint64_t>(i), limits, 1.0, grammar);
      bool saw_form = false;
      for (const AstNode& node : child.ast.nodes) {
        saw_form = saw_form || node.kind == form;
      }
      if (!saw_form) {
        continue;
      }
      if (form == NodeKind::ASGP_DP1D &&
          !check(!child.ast.asgp_dp1d_specs.empty(),
                 "ASGP-DP1D subtree mutation donor should carry spec metadata")) {
        return false;
      }
      if (form == NodeKind::ASGP_DP2D &&
          !check(!child.ast.asgp_dp2d_specs.empty(),
                 "ASGP-DP2D subtree mutation donor should carry spec metadata")) {
        return false;
      }
      const g3pvm::BytecodeProgram bytecode = g3pvm::evo::compile_for_eval(child);
      const g3pvm::ExecResult out = g3pvm::execute_bytecode_cpu(bytecode, {}, 20000);
      if (!check(!out.is_error, "ASGP-DP subtree mutation donor should compile and execute on CPU")) {
        return false;
      }
      if (type == g3pvm::evo::RType::String &&
          !check(out.value.tag == g3pvm::ValueTag::String,
                 "String ASGP-DP subtree mutation donor should return String")) {
        return false;
      }
      return true;
    }
    return check(false, form == NodeKind::ASGP_DP1D
                            ? "subtree mutation should emit ASGP-DP1D donors when enabled"
                            : "subtree mutation should emit ASGP-DP2D donors when enabled");
  };

  if (!seek_form(NodeKind::ASGP_DP1D, g3pvm::evo::RType::Int, 16000)) return false;
  if (!seek_form(NodeKind::ASGP_DP1D, g3pvm::evo::RType::String, 17000)) return false;
  if (!seek_form(NodeKind::ASGP_DP2D, g3pvm::evo::RType::Float, 18000)) return false;
  return seek_form(NodeKind::ASGP_DP2D, g3pvm::evo::RType::String, 19000);
}

bool test_random_generation_emits_asgp_dp_when_enabled() {
  using g3pvm::evo::NodeKind;

  auto seek_form = [](NodeKind form, g3pvm::evo::RType type, std::uint64_t seed_base) {
    const g3pvm::evo::GrammarConfig grammar = grammar_with_only_asgp_form(form);
    g3pvm::evo::Limits limits{8, 6, 180, 16, 3};
    for (int i = 0; i < 2500; ++i) {
      const g3pvm::evo::ProgramGenome genome =
          g3pvm::evo::generate_random_genome_for_return_type(
              seed_base + static_cast<std::uint64_t>(i), type, limits, grammar);
      bool saw_form = false;
      bool saw_string_transition = false;
      for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
        saw_form = saw_form || node.kind == form;
        saw_string_transition = saw_string_transition || node.kind == g3pvm::evo::NodeKind::CALL_CONCAT;
      }
      if (!saw_form) {
        continue;
      }
      (void)g3pvm::evo::compile_for_eval(genome);
      if (form == NodeKind::ASGP_DP1D) {
        if (!check(!genome.ast.asgp_dp1d_specs.empty(),
                   "random ASGP-DP1D generation should carry spec metadata")) {
          return false;
        }
      } else if (!check(!genome.ast.asgp_dp2d_specs.empty(),
                        "random ASGP-DP2D generation should carry spec metadata")) {
        return false;
      }
      if (type == g3pvm::evo::RType::String &&
          !check(saw_string_transition, "random String ASGP-DP generation should use concat transition")) {
        return false;
      }
      return true;
    }
    return check(false, form == NodeKind::ASGP_DP1D
                            ? "random generation should emit ASGP-DP1D when enabled"
                            : "random generation should emit ASGP-DP2D when enabled");
  };

  if (!seek_form(NodeKind::ASGP_DP1D, g3pvm::evo::RType::Float, 20000)) return false;
  if (!seek_form(NodeKind::ASGP_DP1D, g3pvm::evo::RType::String, 21000)) return false;
  if (!seek_form(NodeKind::ASGP_DP2D, g3pvm::evo::RType::Int, 23000)) return false;
  return seek_form(NodeKind::ASGP_DP2D, g3pvm::evo::RType::String, 24000);
}

bool test_for_range_expr_compiles_and_executes() {
  using g3pvm::Value;
  using g3pvm::ValueTag;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;
  using g3pvm::evo::RType;
  using g3pvm::evo::typed_expr::TypedExprRoot;

  AstProgram program;
  program.names = {"x", "i"};
  program.consts = {
      Value::from_int(0),
      g3pvm::payload::make_int_list_value({
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

  const std::vector<std::size_t> subtree_end = g3pvm::evo::subtree::build_subtree_end(program);
  const std::vector<TypedExprRoot> roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(program, subtree_end);
  const TypedExprRoot* bound_len_root = nullptr;
  const TypedExprRoot* body_add_root = nullptr;
  for (const TypedExprRoot& root : roots) {
    if (root.start == 6) bound_len_root = &root;
    if (root.start == 10) body_add_root = &root;
  }
  if (!check(bound_len_root != nullptr && bound_len_root->type == RType::Int,
             "FOR_RANGE bound expression should be exposed as an Int typed root")) {
    return false;
  }
  if (!check(body_add_root != nullptr && body_add_root->type == RType::Int,
             "FOR_RANGE body expression should be exposed as an Int typed root")) {
    return false;
  }
  if (!check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*bound_len_root, *body_add_root),
             "FOR_RANGE bound and body roots should have different visible loop-index scopes")) {
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
    if (value.tag == g3pvm::ValueTag::Char ||
        value.tag == g3pvm::ValueTag::String ||
        value.tag == g3pvm::ValueTag::IntList ||
        value.tag == g3pvm::ValueTag::FloatList ||
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

bool test_current_structured_node_metadata() {
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::subtree::node_arity;

  if (!check(node_arity(NodeKind::BOUND_VAR) == 0, "BOUND_VAR should be leaf")) return false;
  if (!check(node_arity(NodeKind::MAP_LIST) == 2, "MAP_LIST arity should be 2")) return false;
  if (!check(node_arity(NodeKind::FILTER_LIST) == 2, "FILTER_LIST arity should be 2")) return false;
  if (!check(node_arity(NodeKind::LINEAR_REC) == 5, "LINEAR_REC arity should be 5")) return false;
  if (!check(node_arity(NodeKind::CALL_IDIV0) == 2, "CALL_IDIV0 arity should be 2")) return false;
  if (!check(node_arity(NodeKind::CALL_IMOD0) == 2, "CALL_IMOD0 arity should be 2")) return false;
  if (!check(node_arity(NodeKind::CALL_PREPEND) == 2, "CALL_PREPEND arity should be 2")) return false;
  if (!check(node_arity(NodeKind::CALL_SINGLETON) == 1, "CALL_SINGLETON arity should be 1")) return false;
  if (!check(node_arity(NodeKind::CALL_CHAR_TO_STRING) == 1, "CALL_CHAR_TO_STRING arity should be 1")) return false;
  if (!check(node_arity(NodeKind::CALL_STRING_TO_CHAR) == 1, "CALL_STRING_TO_CHAR arity should be 1")) return false;
  if (!check(node_arity(NodeKind::CALL_ORD) == 1, "CALL_ORD arity should be 1")) return false;
  if (!check(node_arity(NodeKind::CALL_CHR) == 1, "CALL_CHR arity should be 1")) return false;
  if (!check(node_arity(NodeKind::CALL_IS_LETTER) == 1, "CALL_IS_LETTER arity should be 1")) return false;
  if (!check(node_arity(NodeKind::CALL_TO_STRING) == 1, "CALL_TO_STRING arity should be 1")) return false;
  if (!check(node_arity(NodeKind::ASGP_DC) == 4, "ASGP_DC arity should be 4")) return false;
  if (!check(node_arity(NodeKind::ASGP_DP1D) == 3, "ASGP_DP1D arity should be 3")) return false;
  if (!check(node_arity(NodeKind::ASGP_DP2D) == 4, "ASGP_DP2D arity should be 4")) return false;

  g3pvm::evo::GrammarConfig scalar = g3pvm::evo::GrammarConfig::scalar();
  if (!check(!scalar.allows_node_kind(NodeKind::MAP_LIST), "scalar config should disable MAP_LIST")) return false;
  if (!check(!scalar.allows_node_kind(NodeKind::FILTER_LIST), "scalar config should disable FILTER_LIST")) return false;
  if (!check(!scalar.allows_node_kind(NodeKind::LINEAR_REC), "scalar config should disable LINEAR_REC")) return false;
  if (!check(!scalar.allows_node_kind(NodeKind::ASGP_DC), "scalar config should disable ASGP_DC")) return false;
  if (!check(!scalar.allows_node_kind(NodeKind::ASGP_DP1D), "scalar config should disable ASGP_DP1D")) return false;
  if (!check(!scalar.allows_node_kind(NodeKind::ASGP_DP2D), "scalar config should disable ASGP_DP2D")) return false;
  return true;
}

bool test_asgp_nodes_are_declared_and_dc_executes_on_cpu() {
  using g3pvm::Value;
  using g3pvm::evo::AsgpDcBinders;
  using g3pvm::evo::AsgpDp1dSpec;
  using g3pvm::evo::AsgpDp2dSpec;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  auto run_ast = [](const AstProgram& program, int fuel = 20000) {
    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    return g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, fuel);
  };

  {
    AstProgram program;
    program.names = {"xs", "n", "lo", "r1", "r2"};
    program.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3), Value::from_int(4)}),
        Value::from_int(0),
        Value::from_int(999),
    };
    program.nodes = {
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
    program.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 1, 3, 4}};

    const std::vector<std::size_t> subtree_end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<g3pvm::evo::typed_expr::TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, subtree_end);
    bool saw_asgp_dc_int = false;
    for (const g3pvm::evo::typed_expr::TypedExprRoot& root : roots) {
      if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::ASGP_DC &&
          root.type == g3pvm::evo::RType::Int) {
        saw_asgp_dc_int = true;
      }
    }
    if (!check(saw_asgp_dc_int, "typed analysis should infer ASGP-DC Int root")) return false;

    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
    if (!check(!out.is_error, "native ASGP-DC should compile and execute on CPU")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 10,
               "native ASGP-DC should sum IntList through clamped divide")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        Value::from_int(7),
        g3pvm::payload::make_int_list_value({Value::from_int(1)}),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 5, 0},
        AstNode{NodeKind::BOUND_VAR, 6, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{6, 1, 2, 3, 4, 5, 6}};

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DC solve phase should not see ordinary locals")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        Value::from_int(7),
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 5, 0},
        AstNode{NodeKind::BOUND_VAR, 6, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{6, 1, 2, 3, 4, 5, 6}};

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DC divide phase should not see ordinary locals")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        Value::from_int(7),
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{6, 1, 2, 3, 4, 5, 6}};

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DC combine phase should not see ordinary locals")) {
      return false;
    }
  }

  auto compile_rejects = [](const AstProgram& program) {
    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    try {
      (void)g3pvm::evo::compile_for_eval(genome);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };

  {
    AstProgram program;
    program.names = {"s", "d1"};
    program.consts = {Value::from_int(2), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 3, 0, 2, NodeKind::DP1_BACKWARD2, {1}, 0, 0, {1}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP1D should reject dependency arity mismatch")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1"};
    program.consts = {Value::from_int(1), Value::from_int(1), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_CROSS_BACKWARD, 0, 1, 0, 1, {2}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP2D should reject dependency arity mismatch")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CALL_LEN, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 0, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 4, 0},
        AstNode{NodeKind::BOUND_VAR, 5, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 3, 4, 5}};
    if (!check(compile_rejects(program),
               "native ASGP-DC divide phase should reject source binder access")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CALL_LEN, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 3, 4, 5}};
    if (!check(compile_rejects(program),
               "native ASGP-DC combine phase should reject source binder access")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(0),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {
        AsgpDcBinders{3, 0, 1, 2, 3, 4, 5},
        AsgpDcBinders{5, 0, 1, 2, 3, 4, 5},
    };

    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    bool rejected = false;
    try {
      (void)g3pvm::evo::compile_for_eval(genome);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!check(rejected, "native ASGP-DC phase body should reject nested ASGP source forms")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"tmp", "xs", "n", "lo", "dn", "r1", "r2"};
    program.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(0),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{3, 1, 2, 3, 4, 5, 6}};
    if (!check(compile_rejects(program),
               "native ASGP-DC phase body should reject assignment nodes")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"state", "xs", "n", "lo", "dn", "r1", "r2", "dep"};
    program.consts = {
        Value::from_int(1),
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(0),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 3, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 3, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 0, {7}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP1D phase body should reject nested ASGP source forms")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "state", "dep"};
    program.consts = {Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 1, NodeKind::DP2_DIAGONAL_BACKWARD, 0, 1, 0, 1, {3}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP2D phase body should reject nested ASGP source forms")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"tmp", "state", "dep"};
    program.consts = {Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 3, 0, 1, NodeKind::DP1_BACKWARD1, {1}, 1, 1, {2}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP1D phase body should reject assignment nodes")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"tmp", "i", "j", "ti", "tj", "dep"};
    program.consts = {Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 1, NodeKind::DP2_DIAGONAL_BACKWARD, 1, 2, 3, 4, {5}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP2D phase body should reject assignment nodes")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"s", "d1"};
    program.consts = {Value::from_int(4), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 5, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 0, {1}},
    };

    const std::vector<std::size_t> subtree_end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<g3pvm::evo::typed_expr::TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, subtree_end);
    bool saw_asgp_dp1d_int = false;
    for (const g3pvm::evo::typed_expr::TypedExprRoot& root : roots) {
      if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::ASGP_DP1D &&
          root.type == g3pvm::evo::RType::Int) {
        saw_asgp_dp1d_int = true;
      }
    }
    if (!check(saw_asgp_dp1d_int, "typed analysis should infer ASGP-DP1D Int root")) return false;

    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
    if (!check(!out.is_error, "native ASGP-DP1D should compile and execute on CPU")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 11,
               "native ASGP-DP1D should evaluate backward recurrence")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"s", "d1"};
    program.consts = {Value::from_int(-1), Value::from_int(1), Value::from_int(99)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 5, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 0, {1}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(!out.is_error, "native ASGP-DP1D boundary case should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 99,
               "native ASGP-DP1D should return out-of-bounds boundary value")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"s", "d1", "d2"};
    program.consts = {Value::from_int(20), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 20, 0, 2, NodeKind::DP1_BACKWARD2, {1, 2}, 0, 0, {1, 2}},
    };

    const g3pvm::ExecResult out = run_ast(program, 500);
    if (!check(!out.is_error, "native ASGP-DP1D memoized recurrence should fit bounded fuel")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 10946,
               "native ASGP-DP1D should memoize overlapping dependencies")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "s", "d1"};
    program.consts = {Value::from_int(7), Value::from_int(0), Value::from_int(1)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{6, 0, 3, 0, 1, NodeKind::DP1_BACKWARD1, {1}, 1, 1, {2}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DP1D solve phase should not see ordinary locals")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "s", "d1"};
    program.consts = {Value::from_int(7), Value::from_int(2), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{6, 0, 3, 0, 3, NodeKind::DP1_BACKWARD1, {1}, 1, 1, {2}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DP1D transition phase should not see ordinary locals")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"solve_s", "transition_s", "d1"};
    program.consts = {Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 3, 0, 1, NodeKind::DP1_BACKWARD1, {1}, 0, 1, {2}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP1D solve phase should reject transition dependency binders")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"solve_s", "transition_s", "d1"};
    program.consts = {Value::from_int(2), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 3, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 1, {2}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP1D transition phase should reject solve-only binders")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"s", "d1", "d2"};
    program.consts = {Value::from_int(1), Value::from_bool(true), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 3, 0, 2, NodeKind::DP1_BACKWARD2, {1, 2}, 0, 0, {1, 2}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Type,
               "native ASGP-DP1D should reject dependency type mismatch")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1", "d2", "d3"};
    program.consts = {Value::from_int(2), Value::from_int(2), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 3, 0},
        AstNode{NodeKind::BOUND_VAR, 4, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_NEIGHBORHOOD_BACKWARD3, 0, 1, 0, 1, {2, 3, 4}},
    };

    const std::vector<std::size_t> subtree_end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<g3pvm::evo::typed_expr::TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, subtree_end);
    bool saw_asgp_dp2d_int = false;
    for (const g3pvm::evo::typed_expr::TypedExprRoot& root : roots) {
      if (root.start < program.nodes.size() && program.nodes[root.start].kind == NodeKind::ASGP_DP2D &&
          root.type == g3pvm::evo::RType::Int) {
        saw_asgp_dp2d_int = true;
      }
    }
    if (!check(saw_asgp_dp2d_int, "typed analysis should infer ASGP-DP2D Int root")) return false;

    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
    if (!check(!out.is_error, "native ASGP-DP2D should compile and execute on CPU")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 13,
               "native ASGP-DP2D should evaluate neighborhood recurrence")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1", "d2", "d3"};
    program.consts = {Value::from_int(5), Value::from_int(5), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 3, 0},
        AstNode{NodeKind::BOUND_VAR, 4, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 5, 0, 5, 0, 0, 3, NodeKind::DP2_NEIGHBORHOOD_BACKWARD3, 0, 1, 0, 1, {2, 3, 4}},
    };

    const g3pvm::ExecResult out = run_ast(program, 1200);
    if (!check(!out.is_error, "native ASGP-DP2D memoized recurrence should fit bounded fuel")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 1683,
               "native ASGP-DP2D should memoize overlapping dependencies")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1", "d2"};
    program.consts = {Value::from_int(-1), Value::from_int(0), Value::from_int(1), Value::from_int(99)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 3, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_CROSS_BACKWARD, 0, 1, 0, 1, {2, 3}},
    };

    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
    if (!check(!out.is_error, "native ASGP-DP2D boundary case should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 99,
               "native ASGP-DP2D should return out-of-bounds boundary value")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "i", "j", "d1"};
    program.consts = {Value::from_int(7), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 3, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{6, 0, 3, 0, 3, 0, 0, 1, NodeKind::DP2_DIAGONAL_BACKWARD, 1, 2, 1, 2, {3}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DP2D solve phase should not see ordinary locals")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"hidden", "i", "j", "ti", "tj", "d1"};
    program.consts = {Value::from_int(7), Value::from_int(1), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::VAR, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{6, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_DIAGONAL_BACKWARD, 1, 2, 3, 4, {5}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Name,
               "native ASGP-DP2D transition phase should not see ordinary locals")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"state_i", "state_j", "solve_i", "solve_j", "transition_i", "transition_j", "d1"};
    program.consts = {Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 6, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 1, NodeKind::DP2_DIAGONAL_BACKWARD, 2, 3, 4, 5, {6}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP2D solve phase should reject transition dependency binders")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"state_i", "state_j", "solve_i", "solve_j", "transition_i", "transition_j", "d1"};
    program.consts = {Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 1, NodeKind::DP2_DIAGONAL_BACKWARD, 2, 3, 4, 5, {6}},
    };
    if (!check(compile_rejects(program),
               "native ASGP-DP2D transition phase should reject solve-only binders")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1"};
    program.consts = {Value::from_int(1), Value::from_int(0), Value::from_int(1),
                      Value::from_int(0), Value::from_bool(true)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::CONST, 4, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_DIAGONAL_BACKWARD, 0, 1, 0, 1, {2}},
    };

    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Type,
               "native ASGP-DP2D should reject transition type mismatch")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1", "d2", "d3"};
    program.consts = {Value::from_int(5), Value::from_int(5), Value::from_int(1), Value::from_int(0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 3, 0},
        AstNode{NodeKind::BOUND_VAR, 4, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 5, 0, 5, 0, 0, 3, NodeKind::DP2_NEIGHBORHOOD_BACKWARD3, 0, 1, 0, 1, {2, 3, 4}},
    };

    const g3pvm::ExecResult out = run_ast(program, 10);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Timeout,
               "native ASGP-DP2D should report timeout on fuel exhaustion")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "d1", "d2"};
    program.consts = {Value::from_int(0), Value::from_int(1), Value::from_bool(true)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 0, NodeKind::DP2_CROSS_BACKWARD, 0, 1, 0, 1, {2, 3}},
    };

    const g3pvm::ExecResult out = run_ast(program);
    if (!check(out.is_error && out.err.code == g3pvm::ErrCode::Type,
               "native ASGP-DP2D should reject dependency type mismatch")) {
      return false;
    }
  }
  return true;
}

bool test_replace_subtree_preserves_asgp_metadata() {
  using g3pvm::Value;
  using g3pvm::evo::AsgpDcBinders;
  using g3pvm::evo::AsgpDp1dSpec;
  using g3pvm::evo::AsgpDp2dSpec;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  auto run_ast = [](const AstProgram& program, int fuel = 20000) {
    ProgramGenome genome;
    genome.ast = program;
    genome.meta = g3pvm::evo::build_genome_meta(genome.ast);
    return g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, fuel);
  };

  {
    AstProgram base;
    base.names = {"x", "xs", "n", "lo", "r1", "r2"};
    base.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(0),
        Value::from_int(999),
    };
    base.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::ASSIGN, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CALL_INDEX, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 4, 0},
        AstNode{NodeKind::BOUND_VAR, 5, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    base.asgp_dc_binders = {AsgpDcBinders{6, 1, 2, 3, 2, 4, 5}};

    AstProgram donor;
    donor.consts = {Value::from_int(10), Value::from_int(20)};
    donor.nodes = {
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
    };

    const std::vector<std::size_t> end = g3pvm::evo::subtree::build_subtree_end(base);
    const AstProgram replaced = g3pvm::evo::subtree::replace_subtree(base, 3, end[3], donor, 0, donor.nodes.size());
    if (!check(replaced.asgp_dc_binders.size() == 1, "ASGP-DC metadata should survive unrelated replacement")) {
      return false;
    }
    if (!check(replaced.asgp_dc_binders[0].node_index == 8,
               "ASGP-DC metadata node_index should shift after insertion")) {
      return false;
    }
    const g3pvm::ExecResult out = run_ast(replaced);
    if (!check(!out.is_error, "shifted ASGP-DC metadata program should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 3,
               "shifted ASGP-DC metadata should still bind recursive phases")) {
      return false;
    }
  }

  {
    AstProgram base;
    base.consts = {Value::from_int(0)};
    base.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };

    AstProgram donor;
    donor.names = {"s", "d1"};
    donor.consts = {Value::from_int(-1), Value::from_int(1), Value::from_int(99)};
    donor.nodes = {
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
    };
    donor.asgp_dp1d_specs = {
        AsgpDp1dSpec{0, 0, 5, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 0, {1}},
    };

    const std::vector<std::size_t> end = g3pvm::evo::subtree::build_subtree_end(base);
    const AstProgram replaced = g3pvm::evo::subtree::replace_subtree(base, 3, end[3], donor, 0, donor.nodes.size());
    if (!check(replaced.asgp_dp1d_specs.size() == 1,
               "donor ASGP-DP1D metadata should be inserted with the subtree")) {
      return false;
    }
    const AsgpDp1dSpec& spec = replaced.asgp_dp1d_specs[0];
    if (!check(spec.node_index == 3, "donor ASGP-DP1D node_index should map to replacement start")) return false;
    if (!check(spec.boundary_const >= 0 &&
                   static_cast<std::size_t>(spec.boundary_const) < replaced.consts.size() &&
                   replaced.consts[static_cast<std::size_t>(spec.boundary_const)].tag == g3pvm::ValueTag::Int &&
                   replaced.consts[static_cast<std::size_t>(spec.boundary_const)].i == 99,
               "donor ASGP-DP1D boundary const should be remapped into the child")) {
      return false;
    }
    const g3pvm::ExecResult out = run_ast(replaced);
    if (!check(!out.is_error, "inserted ASGP-DP1D metadata program should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 99,
               "inserted ASGP-DP1D metadata should preserve boundary behavior")) {
      return false;
    }
  }

  return true;
}

bool test_current_structured_typed_expr_analysis() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::LinearRecBinders;
  using g3pvm::evo::ListTypeTag;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::RType;
  using g3pvm::evo::typed_expr::TypedExprRoot;

  AstProgram map_program;
  map_program.names = {"xs", "u"};
  map_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 1, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  map_program.consts = {Value::from_int(1)};
  if (!check(g3pvm::evo::build_genome_meta(map_program).max_depth == 3,
             "genome meta should track MAP_LIST expression depth")) {
    return false;
  }
  const std::vector<std::size_t> map_end = g3pvm::evo::subtree::build_subtree_end(map_program);
  const std::vector<TypedExprRoot> map_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(map_program, map_end);
  bool saw_map_int_list = false;
  bool exposed_map_bound_fragment = false;
  for (const TypedExprRoot& root : map_roots) {
    if (root.start < map_program.nodes.size() &&
        map_program.nodes[root.start].kind == NodeKind::MAP_LIST &&
        root.type == RType::IntList) {
      saw_map_int_list = true;
    }
    if (root.start == 5 || root.start == 6) {
      exposed_map_bound_fragment = true;
    }
  }
  if (!check(saw_map_int_list, "typed analysis should infer MAP_LIST IntList")) return false;
  if (!check(!exposed_map_bound_fragment,
             "typed analysis should not expose MAP_LIST BoundVar-dependent body fragments")) {
    return false;
  }

  AstProgram float_map_program;
  float_map_program.names = {"floats", "u"};
  float_map_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 1, static_cast<int>(ListTypeTag::Float)},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  float_map_program.consts = {Value::from_float(2.0)};
  const std::vector<std::size_t> float_map_end =
      g3pvm::evo::subtree::build_subtree_end(float_map_program);
  const std::vector<TypedExprRoot> float_map_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(float_map_program, float_map_end);
  bool saw_float_map_list = false;
  bool exposed_float_map_bound_fragment = false;
  for (const TypedExprRoot& root : float_map_roots) {
    if (root.start < float_map_program.nodes.size() &&
        float_map_program.nodes[root.start].kind == NodeKind::MAP_LIST &&
        root.type == RType::FloatList) {
      saw_float_map_list = true;
    }
    if (root.start == 5 || root.start == 6) {
      exposed_float_map_bound_fragment = true;
    }
  }
  if (!check(saw_float_map_list, "typed analysis should infer MAP_LIST FloatList")) return false;
  if (!check(!exposed_float_map_bound_fragment,
             "typed analysis should not expose Float MAP_LIST BoundVar-dependent body fragments")) {
    return false;
  }

  AstProgram string_filter_program;
  string_filter_program.names = {"strings", "u"};
  string_filter_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::FILTER_LIST, 1, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::GT, 0, 0},
      AstNode{NodeKind::CALL_LEN, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  string_filter_program.consts = {Value::from_int(1)};
  const std::vector<std::size_t> string_filter_end =
      g3pvm::evo::subtree::build_subtree_end(string_filter_program);
  const std::vector<TypedExprRoot> string_filter_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(string_filter_program, string_filter_end);
  bool saw_string_filter_list = false;
  bool exposed_string_filter_bound_fragment = false;
  for (const TypedExprRoot& root : string_filter_roots) {
    if (root.start < string_filter_program.nodes.size() &&
        string_filter_program.nodes[root.start].kind == NodeKind::FILTER_LIST &&
        root.type == RType::StringList) {
      saw_string_filter_list = true;
    }
    if (root.start == 5 || root.start == 6 || root.start == 7) {
      exposed_string_filter_bound_fragment = true;
    }
  }
  if (!check(saw_string_filter_list, "typed analysis should infer FILTER_LIST StringList")) return false;
  if (!check(!exposed_string_filter_bound_fragment,
             "typed analysis should not expose String FILTER_LIST BoundVar-dependent predicate fragments")) {
    return false;
  }

  AstProgram linear_program;
  linear_program.names = {"xs", "u", "v", "i"};
  linear_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 3, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  linear_program.consts = {Value::from_int(0), Value::from_int(0)};
  linear_program.linear_rec_binders = {LinearRecBinders{3, 1, 2, 3}};
  const std::vector<std::size_t> linear_end = g3pvm::evo::subtree::build_subtree_end(linear_program);
  const std::vector<TypedExprRoot> linear_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(linear_program, linear_end);
  bool saw_linear_int = false;
  bool exposed_linear_bound_fragment = false;
  for (const TypedExprRoot& root : linear_roots) {
    if (root.start < linear_program.nodes.size() &&
        linear_program.nodes[root.start].kind == NodeKind::LINEAR_REC &&
        root.type == RType::Int) {
      saw_linear_int = true;
    }
    if (root.start == 7 || root.start == 8 || root.start == 9 ||
        root.start == 10 || root.start == 11 || root.start == 12) {
      exposed_linear_bound_fragment = true;
    }
  }
  if (!check(!exposed_linear_bound_fragment,
             "typed analysis should not expose LINEAR_REC BoundVar-dependent body fragments")) {
    return false;
  }
  return check(saw_linear_int, "typed analysis should infer LINEAR_REC Int");
}

bool test_current_typed_subtree_key_tracks_scope_and_dp_arity() {
  using g3pvm::Value;
  using g3pvm::evo::AsgpDcBinders;
  using g3pvm::evo::AsgpDp1dSpec;
  using g3pvm::evo::AsgpDp2dSpec;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::RType;
  using g3pvm::evo::typed_expr::TypedExprRoot;

  auto root_at = [](const std::vector<TypedExprRoot>& roots, std::size_t start) -> const TypedExprRoot* {
    for (const TypedExprRoot& root : roots) {
      if (root.start == start) return &root;
    }
    return nullptr;
  };

  {
    AstProgram program;
    program.names = {"x"};
    program.consts = {Value::from_int(1), Value::from_int(2)};
    program.nodes = {
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
    const std::vector<std::size_t> end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, end);
    const TypedExprRoot* assign_const = nullptr;
    const TypedExprRoot* scoped_var = nullptr;
    const TypedExprRoot* scoped_const = nullptr;
    for (const TypedExprRoot& root : roots) {
      if (root.start == 3) assign_const = &root;
      if (root.start == 7) scoped_var = &root;
      if (root.start == 8) scoped_const = &root;
    }
    if (!check(assign_const != nullptr && scoped_var != nullptr && scoped_const != nullptr,
               "typed analysis should expose roots needed for key checks")) {
      return false;
    }
    if (!check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*assign_const, *scoped_var),
               "typed subtree key should reject same-type roots with different visible environments")) {
      return false;
    }
    if (!check(g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*scoped_var, *scoped_const),
               "typed subtree key should allow different ordinary node kinds in the same visible environment")) {
      return false;
    }
  }

  AstProgram asgp_dc_program;
  asgp_dc_program.names = {"xs", "n", "lo", "dn", "left", "right"};
  asgp_dc_program.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
      Value::from_int(7),
      Value::from_int(1),
      Value::from_int(9),
  };
  asgp_dc_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  asgp_dc_program.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 3, 4, 5}};

  const std::vector<std::size_t> asgp_dc_end = g3pvm::evo::subtree::build_subtree_end(asgp_dc_program);
  const std::vector<TypedExprRoot> asgp_dc_roots =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(asgp_dc_program, asgp_dc_end);
  const TypedExprRoot* asgp_dc_solve = root_at(asgp_dc_roots, 5);
  const TypedExprRoot* asgp_dc_divide = root_at(asgp_dc_roots, 6);
  if (!check(asgp_dc_solve != nullptr && asgp_dc_divide != nullptr,
             "typed analysis should expose ASGP-DC phase roots for key checks")) {
    return false;
  }
  if (!check(asgp_dc_solve->phase_name == 1 && asgp_dc_divide->phase_name == 2,
             "ASGP-DC phase roots should carry distinct phase names")) {
    return false;
  }
  if (!check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*asgp_dc_solve, *asgp_dc_divide),
             "typed subtree key should reject same-type ASGP-DC roots from different phases")) {
    return false;
  }

  auto dp1d_program = [](NodeKind dep_kind, std::vector<int> dep_offsets, std::vector<int> dep_names) {
    AstProgram program;
    program.names = {"state", "dep0", "dep1"};
    program.consts = {Value::from_int(0), Value::from_int(9), Value::from_int(99)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, dep_names.front(), 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 4, 0, 2, dep_kind, std::move(dep_offsets), 0, 0, std::move(dep_names)},
    };
    return program;
  };

  AstProgram dp_one = dp1d_program(NodeKind::DP1_BACKWARD1, {1}, {1});
  AstProgram dp_two = dp1d_program(NodeKind::DP1_BACKWARD2, {1, 2}, {1, 2});
  const std::vector<std::size_t> end_one = g3pvm::evo::subtree::build_subtree_end(dp_one);
  const std::vector<std::size_t> end_two = g3pvm::evo::subtree::build_subtree_end(dp_two);
  const std::vector<TypedExprRoot> roots_one =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(dp_one, end_one);
  const std::vector<TypedExprRoot> roots_two =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(dp_two, end_two);
  const TypedExprRoot* dp_one_root = root_at(roots_one, 3);
  const TypedExprRoot* dp_two_root = root_at(roots_two, 3);
  const TypedExprRoot* dp_one_solve = root_at(roots_one, 5);
  if (!check(dp_one_root != nullptr && dp_two_root != nullptr,
             "typed analysis should expose ASGP-DP1D roots for key checks")) {
    return false;
  }
  if (!check(dp_one_solve != nullptr, "typed analysis should expose ASGP-DP1D solve phase root")) {
    return false;
  }
  if (!check(asgp_dc_solve->scheme_kind != dp_one_solve->scheme_kind,
             "ASGP phase roots should carry scheme kind in typed subtree keys")) {
    return false;
  }
  if (!check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*asgp_dc_solve, *dp_one_solve),
             "typed subtree key should reject same-type ASGP roots from different schemes")) {
    return false;
  }
  if (!check(dp_one_root->dp_dependency_arity == 1 && dp_two_root->dp_dependency_arity == 2,
             "ASGP-DP typed subtree key should track dependency arity")) {
    return false;
  }
  if (!check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*dp_one_root, *dp_two_root),
             "typed subtree key should reject ASGP-DP roots with different dependency arity")) {
    return false;
  }

  auto dp2d_program = [](NodeKind dep_kind, std::vector<int> dep_names) {
    AstProgram program;
    program.names = {"i", "j", "dep0", "dep1", "dep2"};
    program.consts = {Value::from_int(0), Value::from_int(9), Value::from_int(99)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, dep_names.front(), 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 2, dep_kind, 0, 1, 0, 1, std::move(dep_names)},
    };
    return program;
  };

  AstProgram dp2d_one = dp2d_program(NodeKind::DP2_CROSS_BACKWARD, {2});
  AstProgram dp2d_three = dp2d_program(NodeKind::DP2_NEIGHBORHOOD_BACKWARD3, {2, 3, 4});
  const std::vector<std::size_t> end2d_one = g3pvm::evo::subtree::build_subtree_end(dp2d_one);
  const std::vector<std::size_t> end2d_three = g3pvm::evo::subtree::build_subtree_end(dp2d_three);
  const std::vector<TypedExprRoot> roots2d_one =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(dp2d_one, end2d_one);
  const std::vector<TypedExprRoot> roots2d_three =
      g3pvm::evo::typed_expr::collect_typed_expr_roots(dp2d_three, end2d_three);
  const TypedExprRoot* dp2d_one_root = root_at(roots2d_one, 3);
  const TypedExprRoot* dp2d_three_root = root_at(roots2d_three, 3);
  if (!check(dp2d_one_root != nullptr && dp2d_three_root != nullptr,
             "typed analysis should expose ASGP-DP2D roots for key checks")) {
    return false;
  }
  if (!check(dp2d_one_root->dp_dependency_arity == 1 && dp2d_three_root->dp_dependency_arity == 3,
             "ASGP-DP2D typed subtree key should track dependency arity")) {
    return false;
  }
  return check(!g3pvm::evo::typed_expr::typed_subtree_keys_compatible(*dp2d_one_root, *dp2d_three_root),
               "typed subtree key should reject ASGP-DP2D roots with different dependency arity");
}

bool test_subtree_mutation_preserves_structured_scope_boundaries() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::ListTypeTag;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  ProgramGenome base;
  base.ast.names = {"xs", "u"};
  base.ast.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
      Value::from_int(1),
  };
  base.ast.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 1, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  base.meta = g3pvm::evo::build_genome_meta(base.ast);

  g3pvm::evo::Limits limits{7, 6, 160, 16, 3};
  for (int i = 0; i < 160; ++i) {
    const ProgramGenome child =
        g3pvm::evo::mutate(base, static_cast<std::uint64_t>(52000 + i), limits, 1.0);
    const g3pvm::ExecResult out =
        g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(child), {}, 20000);
    if (!check(!out.is_error, "subtree mutation should preserve structured scope and target type")) {
      return false;
    }
  }
  return true;
}

bool test_subtree_donor_generation_preserves_target_value_type() {
  using g3pvm::ValueTag;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;
  using g3pvm::evo::RType;

  auto expected_tag = [](RType type) {
    switch (type) {
      case RType::Int:
        return ValueTag::Int;
      case RType::Float:
        return ValueTag::Float;
      case RType::Bool:
        return ValueTag::Bool;
      case RType::Char:
        return ValueTag::Char;
      case RType::String:
        return ValueTag::String;
      case RType::IntList:
        return ValueTag::IntList;
      case RType::FloatList:
        return ValueTag::FloatList;
      case RType::StringList:
        return ValueTag::StringList;
      default:
        return ValueTag::Invalid;
    }
  };

  const std::vector<RType> types = {
      RType::Int,
      RType::Float,
      RType::Bool,
      RType::Char,
      RType::String,
      RType::IntList,
      RType::FloatList,
      RType::StringList,
  };
  for (RType type : types) {
    for (int i = 0; i < 32; ++i) {
      AstProgram program;
      program.version = g3pvm::evo::k_ast_prefix_version_current;
      std::mt19937_64 rng(static_cast<std::uint64_t>(61000 + static_cast<int>(type) * 100 + i));
      const std::vector<AstNode> expr =
          g3pvm::evo::subtree::make_random_expr_nodes_for_type(rng, program, type, 3, {}, false);
      program.nodes = {
          AstNode{NodeKind::PROGRAM, 0, 0},
          AstNode{NodeKind::BLOCK_CONS, 0, 0},
          AstNode{NodeKind::RETURN, 0, 0},
      };
      program.nodes.insert(program.nodes.end(), expr.begin(), expr.end());
      program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});

      ProgramGenome genome;
      genome.ast = program;
      const g3pvm::ExecResult out =
          g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
      if (!check(!out.is_error, "subtree donor should execute for requested target type")) {
        return false;
      }
      if (!check(out.value.tag == expected_tag(type),
                 "subtree donor runtime value tag should match requested target type")) {
        return false;
      }
    }
  }
  return true;
}

bool test_asgp_typed_expr_analysis_roots_and_phase_guards() {
  using g3pvm::Value;
  using g3pvm::evo::AsgpDcBinders;
  using g3pvm::evo::AsgpDp1dSpec;
  using g3pvm::evo::AsgpDp2dSpec;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::RType;
  using g3pvm::evo::typed_expr::TypedExprRoot;

  auto has_root = [](const std::vector<TypedExprRoot>& roots, std::size_t start, RType type) {
    return std::any_of(roots.begin(), roots.end(), [&](const TypedExprRoot& root) {
      return root.start == start && root.type == type;
    });
  };

  {
    AstProgram program;
    program.names = {"xs", "n", "lo", "dn", "left", "right"};
    program.consts = {
        g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
        Value::from_int(7),
        Value::from_int(1),
    };
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DC, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::ADD, 0, 0},
        AstNode{NodeKind::BOUND_VAR, 4, 0},
        AstNode{NodeKind::BOUND_VAR, 5, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 3, 4, 5}};

    const std::vector<std::size_t> end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, end);
    if (!check(has_root(roots, 3, RType::Int), "typed analysis should infer whole ASGP-DC root")) {
      return false;
    }
    bool saw_phase_root = false;
    bool exposed_bound_fragment = false;
    for (const TypedExprRoot& root : roots) {
      if (root.start == 5 || root.start == 6) {
        saw_phase_root =
            saw_phase_root || g3pvm::evo::typed_expr::is_asgp_phase_body_root(program, end, root);
      }
      if (root.start == 7 || root.start == 8 || root.start == 9) {
        exposed_bound_fragment = true;
      }
    }
    if (!check(saw_phase_root, "ASGP-DC solve/divide roots should be classified as phase roots")) {
      return false;
    }
    if (!check(!exposed_bound_fragment, "ASGP-DC BoundVar fragments should not be exposed as typed roots")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"state", "next"};
    program.consts = {Value::from_int(0), Value::from_int(9), Value::from_int(99)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP1D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::BOUND_VAR, 1, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp1d_specs = {
        AsgpDp1dSpec{3, 0, 4, 0, 2, NodeKind::DP1_BACKWARD1, {1}, 0, 0, {1}},
    };

    const std::vector<std::size_t> end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, end);
    if (!check(has_root(roots, 3, RType::Int), "typed analysis should infer whole ASGP-DP1D root")) {
      return false;
    }
    bool saw_phase_root = false;
    for (const TypedExprRoot& root : roots) {
      if (root.start == 5) {
        saw_phase_root =
            saw_phase_root || g3pvm::evo::typed_expr::is_asgp_phase_body_root(program, end, root);
      }
    }
    if (!check(saw_phase_root, "ASGP-DP1D solve root should be classified as a phase root")) {
      return false;
    }
  }

  {
    AstProgram program;
    program.names = {"i", "j", "dep"};
    program.consts = {Value::from_int(0), Value::from_int(0), Value::from_float(4.5),
                      Value::from_float(-1.0)};
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
        AstNode{NodeKind::ASGP_DP2D, 0, 0},
        AstNode{NodeKind::CONST, 0, 0},
        AstNode{NodeKind::CONST, 1, 0},
        AstNode{NodeKind::CONST, 2, 0},
        AstNode{NodeKind::BOUND_VAR, 2, 0},
        AstNode{NodeKind::BLOCK_NIL, 0, 0},
    };
    program.asgp_dp2d_specs = {
        AsgpDp2dSpec{3, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_CROSS_BACKWARD, 0, 1, 0, 1, {2}},
    };

    const std::vector<std::size_t> end = g3pvm::evo::subtree::build_subtree_end(program);
    const std::vector<TypedExprRoot> roots =
        g3pvm::evo::typed_expr::collect_typed_expr_roots(program, end);
    if (!check(has_root(roots, 3, RType::Float), "typed analysis should infer whole ASGP-DP2D root")) {
      return false;
    }
    bool saw_phase_root = false;
    for (const TypedExprRoot& root : roots) {
      if (root.start == 6) {
        saw_phase_root =
            saw_phase_root || g3pvm::evo::typed_expr::is_asgp_phase_body_root(program, end, root);
      }
    }
    if (!check(saw_phase_root, "ASGP-DP2D solve root should be classified as a phase root")) {
      return false;
    }
  }

  return true;
}

bool test_linear_rec_metadata_affects_cache_key() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::LinearRecBinders;
  using g3pvm::evo::NodeKind;

  AstProgram a;
  a.names = {"xs", "u", "v", "i"};
  a.consts = {Value::from_int(0), Value::from_int(0)};
  a.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  a.linear_rec_binders = {LinearRecBinders{3, 1, 2, 3}};
  AstProgram b = a;
  b.linear_rec_binders = {LinearRecBinders{3, 2, 1, 3}};
  return check(g3pvm::evo::ast_cache_key(a) != g3pvm::evo::ast_cache_key(b),
               "cache key should include LinearRec binder metadata");
}

bool value_is_int_list(const g3pvm::Value& value, const std::vector<long long>& expected) {
  if (value.tag != g3pvm::ValueTag::IntList) return false;
  std::vector<g3pvm::Value> elems;
  if (!g3pvm::payload::lookup_list(value, &elems)) return false;
  if (elems.size() != expected.size()) return false;
  for (std::size_t i = 0; i < elems.size(); ++i) {
    if (elems[i].tag != g3pvm::ValueTag::Int || elems[i].i != expected[i]) return false;
  }
  return true;
}

bool value_is_exact_string(const g3pvm::Value& value, const std::string& expected) {
  if (value.tag != g3pvm::ValueTag::String) return false;
  std::string exact;
  return g3pvm::payload::lookup_string(value, &exact) && exact == expected;
}

bool test_native_cpu_structured_expressions_execute() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::LinearRecBinders;
  using g3pvm::evo::ListTypeTag;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  g3pvm::payload::clear();

  AstProgram map_program;
  map_program.names = {"u"};
  map_program.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)}),
      Value::from_int(2),
  };
  map_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome map_genome;
  map_genome.ast = map_program;
  const g3pvm::ExecResult map_out =
      g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(map_genome), {}, 20000);
  if (!check(!map_out.is_error, "native MAP_LIST should execute")) return false;
  if (!check(value_is_int_list(map_out.value, {2, 4, 6}), "native MAP_LIST should return doubled IntList")) return false;

  AstProgram filter_program;
  filter_program.names = {"u"};
  filter_program.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(3), Value::from_int(1), Value::from_int(4)}),
      Value::from_int(2),
  };
  filter_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::FILTER_LIST, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::GT, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome filter_genome;
  filter_genome.ast = filter_program;
  const g3pvm::ExecResult filter_out =
      g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(filter_genome), {}, 20000);
  if (!check(!filter_out.is_error, "native FILTER_LIST should execute")) return false;
  if (!check(value_is_int_list(filter_out.value, {3, 4}), "native FILTER_LIST should preserve filtered order")) return false;

  AstProgram linear_program;
  linear_program.names = {"u", "v", "i"};
  linear_program.consts = {
      g3pvm::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)}),
      Value::from_int(4),
      Value::from_int(0),
      Value::from_int(10),
      Value::from_int(100),
  };
  linear_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 4, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  linear_program.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  ProgramGenome linear_genome;
  linear_genome.ast = linear_program;
  const g3pvm::ExecResult linear_out =
      g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(linear_genome), {}, 20000);
  if (!check(!linear_out.is_error, "native LINEAR_REC should execute")) return false;
  if (!check(linear_out.value.tag == g3pvm::ValueTag::Int && linear_out.value.i == 30621,
             "native LINEAR_REC should run right-to-left recurrence")) return false;

  AstProgram float_linear_program;
  float_linear_program.names = {"u", "v", "i"};
  float_linear_program.consts = {
      g3pvm::payload::make_float_list_value({Value::from_float(1.5), Value::from_float(2.5)}),
      Value::from_int(0),
      Value::from_float(0.0),
  };
  float_linear_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  float_linear_program.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  ProgramGenome float_linear_genome;
  float_linear_genome.ast = float_linear_program;
  const g3pvm::ExecResult float_linear_out =
      g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(float_linear_genome), {}, 20000);
  if (!check(!float_linear_out.is_error, "native LINEAR_REC FloatList should execute")) return false;
  if (!check(float_linear_out.value.tag == g3pvm::ValueTag::Float && float_linear_out.value.f == 4.0,
             "native LINEAR_REC FloatList should accumulate Float result")) return false;

  AstProgram string_linear_program;
  string_linear_program.names = {"u", "v", "i"};
  const Value str_a = g3pvm::payload::make_string_value("a");
  const Value str_b = g3pvm::payload::make_string_value("b");
  const Value str_c = g3pvm::payload::make_string_value("c");
  string_linear_program.consts = {
      g3pvm::payload::make_string_list_value({str_a, str_b, str_c}),
      Value::from_int(0),
      g3pvm::payload::make_string_value(""),
  };
  string_linear_program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::CALL_CONCAT, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  string_linear_program.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  ProgramGenome string_linear_genome;
  string_linear_genome.ast = string_linear_program;
  const g3pvm::ExecResult string_linear_out =
      g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(string_linear_genome), {}, 20000);
  if (!check(!string_linear_out.is_error, "native LINEAR_REC StringList should execute")) return false;
  if (!check(value_is_exact_string(string_linear_out.value, "abc"),
             "native LINEAR_REC StringList should concatenate String result")) return false;

  return true;
}

bool test_native_cpu_current_builtin_ast_nodes_execute() {
  using g3pvm::Value;
  using g3pvm::evo::AstNode;
  using g3pvm::evo::AstProgram;
  using g3pvm::evo::NodeKind;
  using g3pvm::evo::ProgramGenome;

  auto run_return_expr = [](std::vector<AstNode> expr_nodes, std::vector<Value> consts) {
    AstProgram program;
    program.consts = std::move(consts);
    program.nodes = {
        AstNode{NodeKind::PROGRAM, 0, 0},
        AstNode{NodeKind::BLOCK_CONS, 0, 0},
        AstNode{NodeKind::RETURN, 0, 0},
    };
    program.nodes.insert(program.nodes.end(), expr_nodes.begin(), expr_nodes.end());
    program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
    ProgramGenome genome;
    genome.ast = program;
    return g3pvm::execute_bytecode_cpu(g3pvm::evo::compile_for_eval(genome), {}, 20000);
  };

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_IDIV0, 0, 0}, AstNode{NodeKind::CONST, 0, 0}, AstNode{NodeKind::CONST, 1, 0}},
        {Value::from_int(-7), Value::from_int(2)});
    if (!check(!out.is_error, "native CALL_IDIV0 should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == -3,
               "native CALL_IDIV0 should truncate toward zero")) return false;
  }

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_IMOD0, 0, 0}, AstNode{NodeKind::CONST, 0, 0}, AstNode{NodeKind::CONST, 1, 0}},
        {Value::from_int(7), Value::from_int(0)});
    if (!check(!out.is_error, "native CALL_IMOD0 should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Int && out.value.i == 0,
               "native CALL_IMOD0 should protect zero divisor")) return false;
  }

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_PREPEND, 0, 0}, AstNode{NodeKind::CONST, 0, 0}, AstNode{NodeKind::CONST, 1, 0}},
        {g3pvm::payload::make_int_list_value({Value::from_int(2), Value::from_int(3)}), Value::from_int(1)});
    if (!check(!out.is_error, "native CALL_PREPEND should execute")) return false;
    if (!check(value_is_int_list(out.value, {1, 2, 3}), "native CALL_PREPEND should add element first")) return false;
  }

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_SINGLETON, 0, 0}, AstNode{NodeKind::CONST, 0, 0}},
        {Value::from_int(42)});
    if (!check(!out.is_error, "native CALL_SINGLETON(Int) should execute")) return false;
    if (!check(value_is_int_list(out.value, {42}), "native CALL_SINGLETON(Int) should return IntList")) return false;
  }

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_CHAR_TO_STRING, 0, 0}, AstNode{NodeKind::CALL_TO_UPPER, 0, 0},
         AstNode{NodeKind::CALL_CHR, 0, 0}, AstNode{NodeKind::CONST, 0, 0}},
        {Value::from_int(97)});
    if (!check(!out.is_error, "native char conversion builtin chain should execute")) return false;
    std::string exact;
    if (!check(out.value.tag == g3pvm::ValueTag::String &&
               g3pvm::payload::lookup_string(out.value, &exact) && exact == "A",
               "native char conversion builtin chain should return string A")) return false;
  }

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_IS_VOWEL, 0, 0}, AstNode{NodeKind::CALL_STRING_TO_CHAR, 0, 0},
         AstNode{NodeKind::CONST, 0, 0}},
        {g3pvm::payload::make_string_value("E")});
    if (!check(!out.is_error, "native char predicate builtin chain should execute")) return false;
    if (!check(out.value.tag == g3pvm::ValueTag::Bool && out.value.b,
               "native char predicate builtin chain should return true")) return false;
  }

  {
    const g3pvm::ExecResult out = run_return_expr(
        {AstNode{NodeKind::CALL_TO_STRING, 0, 0}, AstNode{NodeKind::CONST, 0, 0}},
        {Value::from_int(123)});
    if (!check(!out.is_error, "native CALL_TO_STRING should execute")) return false;
    std::string exact;
    if (!check(out.value.tag == g3pvm::ValueTag::String &&
               g3pvm::payload::lookup_string(out.value, &exact) && exact == "123",
               "native CALL_TO_STRING should return string payload")) return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!test_random_genome_compile_rate()) return 1;
  if (!test_random_genome_emits_structured_expressions_when_enabled()) return 1;
  if (!test_subtree_mutation_emits_structured_list_donors_when_enabled()) return 1;
  if (!test_subtree_mutation_emits_asgp_dc_donors_when_enabled()) return 1;
  if (!test_asgp_dc_generation_can_use_existing_list_source()) return 1;
  if (!test_asgp_dc_generation_can_emit_string_roots()) return 1;
  if (!test_subtree_mutation_emits_asgp_dp_donors_when_enabled()) return 1;
  if (!test_random_generation_emits_asgp_dp_when_enabled()) return 1;
  if (!test_mutation_and_crossover_invariants()) return 1;
  if (!test_for_k_constraints()) return 1;
  if (!test_ast_cache_key_distinguishes_program_payload()) return 1;
  if (!test_ast_prefix_old_is_rejected()) return 1;
  if (!test_build_genome_meta_tracks_max_expr_depth()) return 1;
  if (!test_random_genome_uses_requested_input_specs()) return 1;
  if (!test_typed_expr_analysis_tracks_exact_list_index_types()) return 1;
  if (!test_current_if_type_scope_analysis()) return 1;
  if (!test_subtree_donor_generation_uses_existing_list_input_names()) return 1;
  if (!test_for_range_expr_compiles_and_executes()) return 1;
  if (!test_scalar_grammar_config_restricts_generation_and_mutation()) return 1;
  if (!test_current_structured_node_metadata()) return 1;
  if (!test_asgp_nodes_are_declared_and_dc_executes_on_cpu()) return 1;
  if (!test_replace_subtree_preserves_asgp_metadata()) return 1;
  if (!test_current_structured_typed_expr_analysis()) return 1;
  if (!test_current_typed_subtree_key_tracks_scope_and_dp_arity()) return 1;
  if (!test_subtree_mutation_preserves_structured_scope_boundaries()) return 1;
  if (!test_subtree_donor_generation_preserves_target_value_type()) return 1;
  if (!test_asgp_typed_expr_analysis_roots_and_phase_guards()) return 1;
  if (!test_linear_rec_metadata_affects_cache_key()) return 1;
  if (!test_native_cpu_structured_expressions_execute()) return 1;
  if (!test_native_cpu_current_builtin_ast_nodes_execute()) return 1;
  std::cout << "g3pvm_test_genome: OK\n";
  return 0;
}
