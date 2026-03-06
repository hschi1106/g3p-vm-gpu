import unittest

from src.g3p_vm_gpu.evolution.evolve import EvolutionConfig, FitnessCase, evolve_population, score_output


class TestEvolutionLoop(unittest.TestCase):
    def _simple_cases(self):
        return [
            FitnessCase(inputs={"x": 0, "y": 0}, expected=0),
            FitnessCase(inputs={"x": 1, "y": 2}, expected=3),
            FitnessCase(inputs={"x": -1, "y": 4}, expected=3),
            FitnessCase(inputs={"x": 3, "y": -2}, expected=1),
        ]

    def test_score_output_uses_mixed_semantics(self):
        self.assertEqual(score_output(3, 3), 0.0)
        self.assertEqual(score_output(1, 3), -2.0)
        self.assertEqual(score_output(True, True), 1.0)
        self.assertEqual(score_output("ab", "ab"), 1.0)
        self.assertEqual(score_output("ab", "ac"), 0.0)

    def test_evolve_population_runs_with_tournament_only_api(self):
        cfg = EvolutionConfig(
            population_size=24,
            generations=8,
            elitism=2,
            mutation_rate=0.7,
            mutation_subtree_prob=0.75,
            crossover_rate=0.9,
            selection_pressure=4,
            seed=42,
        )
        result = evolve_population(self._simple_cases(), cfg)
        self.assertEqual(len(result.history_best_fitness), cfg.generations)
        self.assertEqual(len(result.history_mean_fitness), cfg.generations)
        self.assertEqual(len(result.final_population), cfg.population_size)
        self.assertGreaterEqual(result.best.fitness, min(result.history_best_fitness))

    def test_invalid_cases_raises(self):
        cfg = EvolutionConfig(population_size=8, generations=2)
        with self.assertRaises(ValueError):
            evolve_population([], cfg)

    def test_invalid_selection_pressure_raises(self):
        cfg = EvolutionConfig(population_size=8, generations=2, selection_pressure=0)
        with self.assertRaises(ValueError):
            evolve_population(self._simple_cases(), cfg)


if __name__ == "__main__":
    unittest.main()
