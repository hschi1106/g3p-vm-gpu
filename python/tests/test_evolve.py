import unittest

from src.g3p_vm_gpu.evolve import (
    EvolutionConfig,
    FitnessCase,
    SelectionMethod,
    evolve_population,
)


class TestEvolve(unittest.TestCase):
    def _simple_cases(self):
        return [
            FitnessCase(inputs={"x": 0, "y": 0}, expected=0),
            FitnessCase(inputs={"x": 1, "y": 2}, expected=3),
            FitnessCase(inputs={"x": -1, "y": 4}, expected=3),
            FitnessCase(inputs={"x": 3, "y": -2}, expected=1),
        ]

    def _run_one(self, method: SelectionMethod):
        cfg = EvolutionConfig(
            population_size=24,
            generations=8,
            elitism=2,
            mutation_rate=0.7,
            crossover_rate=0.9,
            crossover_method="hybrid",
            selection_method=method,
            seed=42,
        )
        result = evolve_population(self._simple_cases(), cfg)
        self.assertEqual(len(result.history_best_fitness), cfg.generations)
        self.assertEqual(len(result.history_mean_fitness), cfg.generations)
        self.assertEqual(len(result.final_population), cfg.population_size)
        self.assertGreaterEqual(result.best.fitness, min(result.history_best_fitness))

    def test_evolve_tournament(self):
        self._run_one(SelectionMethod.TOURNAMENT)

    def test_evolve_roulette(self):
        self._run_one(SelectionMethod.ROULETTE)

    def test_evolve_rank(self):
        self._run_one(SelectionMethod.RANK)

    def test_evolve_truncation(self):
        self._run_one(SelectionMethod.TRUNCATION)

    def test_evolve_random(self):
        self._run_one(SelectionMethod.RANDOM)

    def test_invalid_cases_raises(self):
        cfg = EvolutionConfig(population_size=8, generations=2)
        with self.assertRaises(ValueError):
            evolve_population([], cfg)


if __name__ == "__main__":
    unittest.main()
