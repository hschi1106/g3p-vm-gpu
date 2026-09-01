#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/core/bytecode_verify.hpp"
#include "g3pvm/evolution/ast_verify.hpp"
#include "g3pvm/evolution/compiler.hpp"
#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/evolve.hpp"
#include "g3pvm/evolution/genome_generation.hpp"
#include "g3pvm/evolution/grammar_config.hpp"
#include "g3pvm/evolution/input_spec.hpp"
#include "g3pvm/evolution/mutation.hpp"
#include "g3pvm/evolution/repro/backend.hpp"
#include "g3pvm/evolution/selection.hpp"
#include "g3pvm/runtime/cpu/execute_bytecode_cpu.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

const std::vector<g3pvm::evo::InputSpec>& property_inputs() {
  static const std::vector<g3pvm::evo::InputSpec> inputs;
  return inputs;
}

bool verify_compile(const g3pvm::evo::ProgramGenome& genome,
                    const std::string& context, bool execute) {
  const auto verified = g3pvm::evo::verify_ast(genome.ast, property_inputs());
  if (!check(verified.ok,
             context + " verify failure: " +
                 g3pvm::evo::verify_code_name(verified.diagnostic.code) + " " +
                 verified.diagnostic.message + " program=" +
                 g3pvm::evo::ast_to_string(genome.ast))) return false;
  const g3pvm::BytecodeProgram bytecode =
      g3pvm::evo::compile_for_eval(genome, verified.verified);
  const auto bytecode_verified = g3pvm::verify_bytecode(bytecode);
  if (!check(bytecode_verified.ok, context + " bytecode verification failure")) return false;
  if (execute) (void)g3pvm::execute_bytecode_cpu(bytecode, {}, 2000);
  return true;
}

bool test_generation_mutation_and_crossover_seed_ranges() {
  g3pvm::evo::Limits limits;
  limits.max_expr_depth = 6;
  limits.max_stmts_per_block = 5;
  limits.max_total_nodes = 100;
  std::vector<g3pvm::evo::ProgramGenome> population;
  for (std::uint64_t seed = 0; seed < 128; ++seed) {
    g3pvm::evo::ProgramGenome generated =
        g3pvm::evo::generate_random_genome(seed, limits, property_inputs());
    if (!verify_compile(generated, "generation seed " + std::to_string(seed), true)) {
      return false;
    }
    const g3pvm::evo::ProgramGenome replay =
        g3pvm::evo::generate_random_genome(seed, limits, property_inputs());
    if (!check(generated.meta.program_key == replay.meta.program_key,
               "generation replay mismatch at seed " + std::to_string(seed))) return false;

    const auto generated_verified =
        g3pvm::evo::verify_ast(generated.ast, property_inputs());
    if (!check(generated_verified.ok, "generated AST annotation unavailable")) return false;
    g3pvm::evo::ProgramGenome mutated = g3pvm::evo::mutate(
        generated, generated_verified.verified, 10000 + seed, limits, 0.8);
    if (!verify_compile(mutated, "mutation seed " + std::to_string(seed), false)) {
      return false;
    }
    population.push_back(std::move(generated));
  }

  for (std::uint64_t seed = 0; seed < 64; ++seed) {
    const auto& first = population[static_cast<std::size_t>(seed * 2)];
    const auto& second = population[static_cast<std::size_t>(seed * 2 + 1)];
    const auto first_verified = g3pvm::evo::verify_ast(first.ast, property_inputs());
    const auto second_verified = g3pvm::evo::verify_ast(second.ast, property_inputs());
    if (!check(first_verified.ok && second_verified.ok,
               "crossover parent annotation unavailable")) return false;
    const auto children = g3pvm::evo::crossover(
        first, first_verified.verified, second, second_verified.verified,
        20000 + seed, limits);
    if (!verify_compile(children.first, "crossover first seed " + std::to_string(seed),
                        false) ||
        !verify_compile(children.second, "crossover second seed " + std::to_string(seed),
                        false)) return false;
  }
  return true;
}

bool test_regression_generation_seed_22_has_consistent_returns() {
  g3pvm::evo::Limits limits;
  limits.max_expr_depth = 6;
  limits.max_stmts_per_block = 5;
  limits.max_total_nodes = 100;
  const g3pvm::evo::ProgramGenome genome =
      g3pvm::evo::generate_random_genome(22, limits, property_inputs());
  return verify_compile(genome, "regression generation seed 22", false);
}

bool test_cpu_reproduction_verifies_every_child() {
  g3pvm::evo::EvolutionConfig config;
  config.population_size = 16;
  config.mutation_rate = 0.75;
  config.mutation_subtree_prob = 0.8;
  config.seed = 777;
  std::vector<g3pvm::evo::ScoredGenome> scored;
  for (int seed = 0; seed < config.population_size; ++seed) {
    scored.push_back({g3pvm::evo::generate_random_genome(
                          static_cast<std::uint64_t>(30000 + seed), config.limits,
                          property_inputs()),
                      static_cast<double>(seed)});
  }
  std::mt19937_64 rng(config.seed);
  const auto result =
      g3pvm::evo::repro::run_reproduction_backend(scored, config, rng);
  if (!check(result.next_population.size() ==
                 static_cast<std::size_t>(config.population_size),
             "CPU reproduction population size mismatch")) return false;
  for (std::size_t i = 0; i < result.next_population.size(); ++i) {
    if (!verify_compile(result.next_population[i],
                        "CPU reproduction child " + std::to_string(i), false)) return false;
  }
  return true;
}

bool test_scalar_grammar_property() {
  const g3pvm::evo::GrammarConfig scalar = g3pvm::evo::GrammarConfig::scalar();
  for (std::uint64_t seed = 0; seed < 128; ++seed) {
    const g3pvm::evo::ProgramGenome genome =
        g3pvm::evo::generate_random_genome(seed + 40000, g3pvm::evo::Limits{},
                                           property_inputs(), scalar);
    for (const g3pvm::evo::AstNode& node : genome.ast.nodes) {
      if (!check(scalar.allows_node_kind(node.kind),
                 "scalar grammar emitted a disabled node at seed " +
                     std::to_string(seed))) return false;
    }
    for (const g3pvm::Value& value : genome.ast.consts) {
      if (!check(value.tag != g3pvm::ValueTag::String &&
                     value.tag != g3pvm::ValueTag::IntList &&
                     value.tag != g3pvm::ValueTag::FloatList &&
                     value.tag != g3pvm::ValueTag::StringList,
                 "scalar grammar emitted a container constant at seed " +
                     std::to_string(seed))) return false;
    }
    if (!verify_compile(genome, "scalar grammar seed " + std::to_string(seed), false)) {
      return false;
    }
  }
  return true;
}

bool test_payload_retain_property() {
  g3pvm::payload::clear();
  std::vector<g3pvm::Value> roots;
  std::vector<g3pvm::Value> dropped;
  for (int i = 0; i < 64; ++i) {
    const g3pvm::Value string =
        g3pvm::payload::make_string_value("payload-" + std::to_string(i));
    const g3pvm::Value list = g3pvm::payload::make_string_list_value({string});
    (i % 2 == 0 ? roots : dropped).push_back(list);
  }
  g3pvm::payload::retain_only(roots);
  std::vector<g3pvm::Value> values;
  for (const g3pvm::Value& root : roots) {
    if (!check(g3pvm::payload::lookup_list(root, &values) && values.size() == 1,
               "live payload root was swept")) return false;
    std::string exact;
    if (!check(g3pvm::payload::lookup_string(values[0], &exact),
               "live transitive payload was swept")) return false;
  }
  for (const g3pvm::Value& value : dropped) {
    if (!check(!g3pvm::payload::lookup_list(value, &values),
               "dead payload root survived sweep")) return false;
  }
  const auto stats = g3pvm::payload::stats();
  return check(stats.list_entries == 32 && stats.string_entries == 32,
               "payload retain/sweep counts mismatch");
}

}  // namespace

int main() {
  if (!test_regression_generation_seed_22_has_consistent_returns()) return 1;
  if (!test_generation_mutation_and_crossover_seed_ranges()) return 1;
  if (!test_cpu_reproduction_verifies_every_child()) return 1;
  if (!test_scalar_grammar_property()) return 1;
  if (!test_payload_retain_property()) return 1;
  std::cout << "g3pvm_test_genome_properties: OK\n";
  return 0;
}
