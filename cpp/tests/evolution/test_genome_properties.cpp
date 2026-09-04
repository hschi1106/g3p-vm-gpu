#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gagp/core/bytecode_verify.hpp"
#include "gagp/evolution/ast_verify.hpp"
#include "gagp/evolution/compiler.hpp"
#include "gagp/evolution/crossover.hpp"
#include "gagp/evolution/evolve.hpp"
#include "gagp/evolution/genome_generation.hpp"
#include "gagp/evolution/grammar_config.hpp"
#include "gagp/evolution/input_spec.hpp"
#include "gagp/evolution/mutation.hpp"
#include "gagp/evolution/repro/backend.hpp"
#include "gagp/evolution/selection.hpp"
#include "gagp/runtime/cpu/execute_bytecode_cpu.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace {

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

const std::vector<gagp::evo::InputSpec>& property_inputs() {
  static const std::vector<gagp::evo::InputSpec> inputs;
  return inputs;
}

bool verify_compile(const gagp::evo::ProgramGenome& genome,
                    const std::string& context, bool execute) {
  const auto verified = gagp::evo::verify_ast(genome.ast, property_inputs());
  if (!check(verified.ok,
             context + " verify failure: " +
                 gagp::evo::verify_code_name(verified.diagnostic.code) + " " +
                 verified.diagnostic.message + " program=" +
                 gagp::evo::ast_to_string(genome.ast))) return false;
  const gagp::BytecodeProgram bytecode =
      gagp::evo::compile_for_eval(genome, verified.verified);
  const auto bytecode_verified = gagp::verify_bytecode(bytecode);
  if (!check(bytecode_verified.ok, context + " bytecode verification failure")) return false;
  if (execute) (void)gagp::execute_bytecode_cpu(bytecode, {}, 2000);
  return true;
}

bool test_generation_mutation_and_crossover_seed_ranges() {
  gagp::evo::Limits limits;
  limits.max_expr_depth = 6;
  limits.max_stmts_per_block = 5;
  limits.max_total_nodes = 100;
  std::vector<gagp::evo::ProgramGenome> population;
  for (std::uint64_t seed = 0; seed < 128; ++seed) {
    gagp::evo::ProgramGenome generated =
        gagp::evo::generate_random_genome(seed, limits, property_inputs());
    if (!verify_compile(generated, "generation seed " + std::to_string(seed), true)) {
      return false;
    }
    const gagp::evo::ProgramGenome replay =
        gagp::evo::generate_random_genome(seed, limits, property_inputs());
    if (!check(generated.meta.program_key == replay.meta.program_key,
               "generation replay mismatch at seed " + std::to_string(seed))) return false;

    const auto generated_verified =
        gagp::evo::verify_ast(generated.ast, property_inputs());
    if (!check(generated_verified.ok, "generated AST annotation unavailable")) return false;
    gagp::evo::ProgramGenome mutated = gagp::evo::mutate(
        generated, generated_verified.verified, 10000 + seed, limits, 0.8);
    if (!verify_compile(mutated, "mutation seed " + std::to_string(seed), false)) {
      return false;
    }
    population.push_back(std::move(generated));
  }

  for (std::uint64_t seed = 0; seed < 64; ++seed) {
    const auto& first = population[static_cast<std::size_t>(seed * 2)];
    const auto& second = population[static_cast<std::size_t>(seed * 2 + 1)];
    const auto first_verified = gagp::evo::verify_ast(first.ast, property_inputs());
    const auto second_verified = gagp::evo::verify_ast(second.ast, property_inputs());
    if (!check(first_verified.ok && second_verified.ok,
               "crossover parent annotation unavailable")) return false;
    const auto children = gagp::evo::crossover(
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
  gagp::evo::Limits limits;
  limits.max_expr_depth = 6;
  limits.max_stmts_per_block = 5;
  limits.max_total_nodes = 100;
  const gagp::evo::ProgramGenome genome =
      gagp::evo::generate_random_genome(22, limits, property_inputs());
  return verify_compile(genome, "regression generation seed 22", false);
}

bool test_cpu_reproduction_verifies_every_child() {
  gagp::evo::EvolutionConfig config;
  config.population_size = 16;
  config.mutation_rate = 0.75;
  config.mutation_subtree_prob = 0.8;
  config.seed = 777;
  std::vector<gagp::evo::ScoredGenome> scored;
  for (int seed = 0; seed < config.population_size; ++seed) {
    scored.push_back({gagp::evo::generate_random_genome(
                          static_cast<std::uint64_t>(30000 + seed), config.limits,
                          property_inputs()),
                      static_cast<double>(seed)});
  }
  std::mt19937_64 rng(config.seed);
  const auto result =
      gagp::evo::repro::run_reproduction_backend(scored, config, rng);
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
  const gagp::evo::GrammarConfig scalar = gagp::evo::GrammarConfig::scalar();
  for (std::uint64_t seed = 0; seed < 128; ++seed) {
    const gagp::evo::ProgramGenome genome =
        gagp::evo::generate_random_genome(seed + 40000, gagp::evo::Limits{},
                                           property_inputs(), scalar);
    for (const gagp::evo::AstNode& node : genome.ast.nodes) {
      if (!check(scalar.allows_node_kind(node.kind),
                 "scalar grammar emitted a disabled node at seed " +
                     std::to_string(seed))) return false;
    }
    for (const gagp::Value& value : genome.ast.consts) {
      if (!check(value.tag != gagp::ValueTag::String &&
                     value.tag != gagp::ValueTag::IntList &&
                     value.tag != gagp::ValueTag::FloatList &&
                     value.tag != gagp::ValueTag::StringList,
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
  gagp::payload::clear();
  std::vector<gagp::Value> roots;
  std::vector<gagp::Value> dropped;
  for (int i = 0; i < 64; ++i) {
    const gagp::Value string =
        gagp::payload::make_string_value("payload-" + std::to_string(i));
    const gagp::Value list = gagp::payload::make_string_list_value({string});
    (i % 2 == 0 ? roots : dropped).push_back(list);
  }
  gagp::payload::retain_only(roots);
  std::vector<gagp::Value> values;
  for (const gagp::Value& root : roots) {
    if (!check(gagp::payload::lookup_list(root, &values) && values.size() == 1,
               "live payload root was swept")) return false;
    std::string exact;
    if (!check(gagp::payload::lookup_string(values[0], &exact),
               "live transitive payload was swept")) return false;
  }
  for (const gagp::Value& value : dropped) {
    if (!check(!gagp::payload::lookup_list(value, &values),
               "dead payload root survived sweep")) return false;
  }
  const auto stats = gagp::payload::stats();
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
  std::cout << "gagp_test_genome_properties: OK\n";
  return 0;
}
