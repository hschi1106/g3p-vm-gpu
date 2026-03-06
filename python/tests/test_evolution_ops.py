import unittest

from src.g3p_vm_gpu.core.errors import Failed, Returned
from src.g3p_vm_gpu.evolution.crossover import crossover
from src.g3p_vm_gpu.evolution.genome import Limits, compile_for_eval
from src.g3p_vm_gpu.evolution.mutation import mutate
from src.g3p_vm_gpu.evolution.random_genome import make_random_genome
from src.g3p_vm_gpu.runtime.compiler import compile_program
from src.g3p_vm_gpu.runtime.interp import run_program
from src.g3p_vm_gpu.runtime.vm import ExecError, ExecReturn, exec_bytecode


class TestEvolutionOps(unittest.TestCase):
    def test_random_genome_compile_rate(self):
        limits = Limits()
        compiled = 0
        total = 400
        for i in range(total):
            genome = make_random_genome(seed=i, limits=limits)
            compile_program(genome.ast)
            compiled += 1
        self.assertGreaterEqual(compiled / total, 0.99)

    def test_mutation_produces_compilable_genomes(self):
        limits = Limits()
        base = make_random_genome(seed=123, limits=limits)
        for i in range(120):
            child = mutate(base, seed=i, limits=limits, mutation_subtree_prob=0.8)
            self.assertLessEqual(child.meta.node_count, limits.max_total_nodes)
            compile_for_eval(child)

    def test_crossover_produces_compilable_genomes(self):
        limits = Limits()
        parent_a = make_random_genome(seed=1, limits=limits)
        parent_b = make_random_genome(seed=2, limits=limits)
        for i in range(120):
            child = crossover(parent_a, parent_b, seed=i, limits=limits)
            self.assertLessEqual(child.meta.node_count, limits.max_total_nodes)
            compile_for_eval(child)

    def test_for_range_k_constraints_after_mutation(self):
        limits = Limits(max_for_k=8)
        base = make_random_genome(seed=88, limits=limits)
        for i in range(100):
            child = mutate(base, seed=10_000 + i, limits=limits)
            for node in child.ast.nodes:
                if node.kind.value == "FOR_RANGE":
                    self.assertLessEqual(node.i1, limits.max_for_k)

    def test_eval_parity_sample(self):
        limits = Limits()
        for i in range(80):
            genome = make_random_genome(seed=2000 + i, limits=limits)
            interp_env, interp_out = run_program(genome.ast, {}, fuel=20_000)
            vm_out = exec_bytecode(compile_for_eval(genome), {}, fuel=20_000)

            if isinstance(interp_out, Returned):
                self.assertIsInstance(vm_out, ExecReturn)
                self.assertEqual(interp_out.value, vm_out.value)
            elif isinstance(interp_out, Failed):
                self.assertIsInstance(vm_out, ExecError)
                self.assertEqual(interp_out.err.code, vm_out.err.code)
            else:
                self.fail("unexpected top-level interpreter outcome")

            self.assertIsInstance(interp_env, dict)


if __name__ == "__main__":
    unittest.main()
