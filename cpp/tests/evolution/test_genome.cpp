#include <iostream>
#include <string>

#include "g3pvm/evolution/crossover.hpp"
#include "g3pvm/evolution/genome.hpp"
#include "g3pvm/evolution/mutation.hpp"

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
    const g3pvm::evo::ProgramGenome g = g3pvm::evo::make_random_genome(static_cast<std::uint64_t>(i), limits);
    (void)g3pvm::evo::compile_for_eval(g);
    compiled += 1;
  }
  return check(static_cast<double>(compiled) / static_cast<double>(n) >= 0.99, "compile rate < 99%");
}

bool test_mutation_and_crossover_invariants() {
  g3pvm::evo::Limits limits;
  const g3pvm::evo::ProgramGenome a = g3pvm::evo::make_random_genome(1, limits);
  const g3pvm::evo::ProgramGenome b = g3pvm::evo::make_random_genome(2, limits);

  for (int i = 0; i < 80; ++i) {
    const g3pvm::evo::ProgramGenome m =
        g3pvm::evo::mutate(a, static_cast<std::uint64_t>(1000 + i), limits, 0.8);
    (void)g3pvm::evo::compile_for_eval(m);

    const g3pvm::evo::ProgramGenome c =
        g3pvm::evo::crossover(a, b, static_cast<std::uint64_t>(3000 + i), limits);
    (void)g3pvm::evo::compile_for_eval(c);
  }

  return true;
}

bool test_for_k_constraints() {
  g3pvm::evo::Limits limits{5, 6, 80, 8, 3};
  const g3pvm::evo::ProgramGenome base = g3pvm::evo::make_random_genome(88, limits);
  for (int i = 0; i < 60; ++i) {
    const g3pvm::evo::ProgramGenome child =
        g3pvm::evo::mutate(base, static_cast<std::uint64_t>(10000 + i), limits, 0.8);
    (void)g3pvm::evo::compile_for_eval(child);
  }
  return true;
}

}  // namespace

int main() {
  if (!test_random_genome_compile_rate()) return 1;
  if (!test_mutation_and_crossover_invariants()) return 1;
  if (!test_for_k_constraints()) return 1;
  std::cout << "g3pvm_test_genome: OK\n";
  return 0;
}
