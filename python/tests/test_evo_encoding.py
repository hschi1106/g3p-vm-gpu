import unittest

from src.g3p_vm_gpu.compiler import compile_program
from src.g3p_vm_gpu.errors import Failed, Returned
from src.g3p_vm_gpu.evo_encoding import (
    Limits,
    compile_for_eval,
    crossover,
    crossover_top_level,
    crossover_typed_subtree,
    make_random_genome,
    mutate,
    validate_genome,
)
from src.g3p_vm_gpu.interp import run_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


class TestEvoEncoding(unittest.TestCase):
    def test_random_genome_compile_rate(self):
        limits = Limits()
        n = 400
        compiled = 0
        for i in range(n):
            g = make_random_genome(seed=i, limits=limits)
            vr = validate_genome(g, limits)
            self.assertTrue(vr.is_valid, f"invalid genome at seed={i}: {vr.errors}")
            compile_program(g.ast)
            compiled += 1
        self.assertGreaterEqual(compiled / n, 0.99)

    def test_mutation_invariants(self):
        limits = Limits()
        base = make_random_genome(seed=123, limits=limits)
        for i in range(120):
            child = mutate(base, seed=i, limits=limits)
            vr = validate_genome(child, limits)
            self.assertTrue(vr.is_valid, f"invalid mutation at seed={i}: {vr.errors}")
            compile_for_eval(child)

    def test_crossover_invariants(self):
        limits = Limits()
        a = make_random_genome(seed=1, limits=limits)
        b = make_random_genome(seed=2, limits=limits)
        for i in range(120):
            child = crossover(a, b, seed=i, limits=limits)
            vr = validate_genome(child, limits)
            self.assertTrue(vr.is_valid, f"invalid crossover at seed={i}: {vr.errors}")
            compile_for_eval(child)

    def test_crossover_methods_all_valid(self):
        limits = Limits()
        a = make_random_genome(seed=10, limits=limits)
        b = make_random_genome(seed=20, limits=limits)
        for i in range(80):
            c_top = crossover_top_level(a, b, seed=i, limits=limits)
            c_sub = crossover_typed_subtree(a, b, seed=i, limits=limits)
            c_hyb = crossover(a, b, seed=i, limits=limits, method="hybrid")
            for c in (c_top, c_sub, c_hyb):
                vr = validate_genome(c, limits)
                self.assertTrue(vr.is_valid, f"invalid crossover method output at seed={i}: {vr.errors}")
                compile_for_eval(c)

    def test_crossover_unknown_method_raises(self):
        limits = Limits()
        a = make_random_genome(seed=11, limits=limits)
        b = make_random_genome(seed=22, limits=limits)
        with self.assertRaises(ValueError):
            crossover(a, b, seed=0, limits=limits, method="not_a_method")

    def test_for_range_k_constraints_after_mutation(self):
        limits = Limits(max_for_k=8)
        base = make_random_genome(seed=88, limits=limits)
        for i in range(100):
            child = mutate(base, seed=10_000 + i, limits=limits)
            vr = validate_genome(child, limits)
            self.assertTrue(vr.is_valid, f"k constraint broken at seed={i}: {vr.errors}")

    def test_eval_parity_sample(self):
        limits = Limits()
        for i in range(80):
            g = make_random_genome(seed=2000 + i, limits=limits)
            interp_env, interp_out = run_program(g.ast, {}, fuel=20_000)
            vm_out = run_bytecode(compile_for_eval(g), {}, fuel=20_000)

            if isinstance(interp_out, Returned):
                self.assertIsInstance(vm_out, VMReturn)
                self.assertEqual(interp_out.value, vm_out.value)
            elif isinstance(interp_out, Failed):
                self.assertIsInstance(vm_out, VMError)
                self.assertEqual(interp_out.err.code, vm_out.err.code)
            else:
                self.fail("unexpected top-level interpreter outcome")

            self.assertIsInstance(interp_env, dict)


if __name__ == "__main__":
    unittest.main()
