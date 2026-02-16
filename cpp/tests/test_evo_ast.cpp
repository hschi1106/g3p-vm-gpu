#include <iostream>
#include <string>

#include "g3pvm/evo_ast.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool test_random_genome_compile_rate() {
  const g3pvm::evo::Limits limits;
  const int n = 200;
  int compiled = 0;
  for (int i = 0; i < n; ++i) {
    const g3pvm::evo::ProgramGenome g = g3pvm::evo::make_random_genome(static_cast<std::uint64_t>(i), limits);
    const g3pvm::evo::ValidationResult vr = g3pvm::evo::validate_genome(g, limits);
    if (!check(vr.is_valid, "random genome validation failed")) return false;
    (void)g3pvm::evo::compile_for_eval(g);
    compiled += 1;
  }
  return check(static_cast<double>(compiled) / static_cast<double>(n) >= 0.99, "compile rate < 99%");
}

bool test_mutation_and_crossover_invariants() {
  const g3pvm::evo::Limits limits;
  const g3pvm::evo::ProgramGenome a = g3pvm::evo::make_random_genome(1, limits);
  const g3pvm::evo::ProgramGenome b = g3pvm::evo::make_random_genome(2, limits);

  for (int i = 0; i < 80; ++i) {
    const g3pvm::evo::ProgramGenome m = g3pvm::evo::mutate(a, static_cast<std::uint64_t>(1000 + i), limits);
    const auto mvr = g3pvm::evo::validate_genome(m, limits);
    if (!check(mvr.is_valid, "mutated genome invalid")) return false;
    (void)g3pvm::evo::compile_for_eval(m);

    const g3pvm::evo::ProgramGenome c0 = g3pvm::evo::crossover_top_level(a, b, static_cast<std::uint64_t>(2000 + i), limits);
    const g3pvm::evo::ProgramGenome c1 = g3pvm::evo::crossover_typed_subtree(a, b, static_cast<std::uint64_t>(3000 + i), limits);
    const g3pvm::evo::ProgramGenome c2 =
        g3pvm::evo::crossover(a, b, static_cast<std::uint64_t>(4000 + i), g3pvm::evo::CrossoverMethod::Hybrid, limits);

    for (const g3pvm::evo::ProgramGenome* c : {&c0, &c1, &c2}) {
      const auto cvr = g3pvm::evo::validate_genome(*c, limits);
      if (!check(cvr.is_valid, "crossover genome invalid")) return false;
      (void)g3pvm::evo::compile_for_eval(*c);
    }
  }

  return true;
}

bool test_for_k_constraints() {
  const g3pvm::evo::Limits limits{5, 6, 80, 8, 3};
  const g3pvm::evo::ProgramGenome base = g3pvm::evo::make_random_genome(88, limits);
  for (int i = 0; i < 60; ++i) {
    const g3pvm::evo::ProgramGenome child =
        g3pvm::evo::mutate(base, static_cast<std::uint64_t>(10000 + i), limits);
    const g3pvm::evo::ValidationResult vr = g3pvm::evo::validate_genome(child, limits);
    if (!check(vr.is_valid, "for_k constraints broken")) return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!test_random_genome_compile_rate()) return 1;
  if (!test_mutation_and_crossover_invariants()) return 1;
  if (!test_for_k_constraints()) return 1;
  std::cout << "g3pvm_test_evo_ast: OK\n";
  return 0;
}
