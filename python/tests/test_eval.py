import unittest

from src.g3p_vm_gpu.ast import build_program
from src.g3p_vm_gpu.errors import ErrCode, Failed, Returned
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.interp import run_program


class TestEval(unittest.TestCase):
    def test_short_circuit_and(self):
        prog = build_program([("return", ("and", ("const", False), ("div", ("const", 1), ("const", 0))))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, False)

    def test_short_circuit_or(self):
        prog = build_program([("return", ("or", ("const", True), ("div", ("const", 1), ("const", 0))))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, True)

    def test_if_requires_bool(self):
        prog = build_program([("if", ("const", 1), [("return", ("const", 1))], [("return", ("const", 2))])])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.TYPE)

    def test_for_loop(self):
        prog = build_program(
            [
                ("assign", "x", ("const", 0)),
                ("for", "i", 5, [("assign", "x", ("add", ("var", "x"), ("const", 1)))]),
                ("return", ("var", "x")),
            ]
        )
        _, out = run_program(prog, {}, fuel=1000)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 5)

    def test_return_stops_block(self):
        prog = build_program([("return", ("const", 1)), ("assign", "x", ("const", 999))])
        env, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 1)
        self.assertNotIn("x", env)

    def test_builtins_clip(self):
        prog = build_program([("return", ("call", "clip", [("const", 5), ("const", 0), ("const", 2)]))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 2)

    def test_timeout(self):
        prog = build_program([("return", ("const", 1))])
        _, out = run_program(prog, {}, fuel=0)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.TIMEOUT)

    def test_program_without_return_is_value_error(self):
        prog = build_program([("assign", "x", ("const", 1))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.VALUE)

    def test_none_eq_ne_with_non_none(self):
        eq_prog = build_program([("return", ("eq", ("const", None), ("const", 1)))])
        _, eq_out = run_program(eq_prog, {}, fuel=100)
        self.assertIsInstance(eq_out, Returned)
        self.assertEqual(eq_out.value, False)

        ne_prog = build_program([("return", ("ne", ("const", None), ("const", 1)))])
        _, ne_out = run_program(ne_prog, {}, fuel=100)
        self.assertIsInstance(ne_out, Returned)
        self.assertEqual(ne_out.value, True)

    def test_fuzz_random_programs(self):
        passed = 0
        failed = 0
        timeout = 0

        for i in range(1000):
            prog = make_random_program(seed=i, depth=3)
            env, out = run_program(prog, {}, fuel=10000)
            self.assertIn(type(out).__name__, ["Returned", "Failed"])
            if isinstance(out, Returned):
                passed += 1
            elif isinstance(out, Failed):
                if out.err.code == ErrCode.TIMEOUT:
                    timeout += 1
                else:
                    failed += 1
            self.assertIsInstance(env, dict)

        print(f"\nFuzz test results: {passed} passed, {failed} failed, {timeout} timeout")
        self.assertEqual(passed + failed + timeout, 1000)


if __name__ == "__main__":
    unittest.main()
