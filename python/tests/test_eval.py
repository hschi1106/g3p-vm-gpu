import unittest

from src.g3p_vm_gpu.core.ast import build_program
from src.g3p_vm_gpu.core.errors import ErrCode, Failed, Returned
from src.g3p_vm_gpu.evolution.random_program import make_random_program
from src.g3p_vm_gpu.runtime.builtins import builtin_call
from src.g3p_vm_gpu.runtime.interp import run_program


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
                ("for", "i", ("const", 5), [("assign", "x", ("add", ("var", "x"), ("const", 1)))]),
                ("return", ("var", "x")),
            ]
        )
        _, out = run_program(prog, {}, fuel=1000)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 5)

    def test_for_loop_old_constant_bound_syntax_is_rejected(self):
        with self.assertRaises(ValueError):
            build_program([("for", "i", 5, [("assign", "x", ("const", 1))])])

    def test_for_loop_dynamic_bound(self):
        prog = build_program(
            [
                ("assign", "x", ("const", 0)),
                (
                    "for",
                    "i",
                    ("call", "len", [("const", [10, 20, 30, 40])]),
                    [("assign", "x", ("add", ("var", "x"), ("var", "i")))],
                ),
                ("return", ("var", "x")),
            ]
        )
        _, out = run_program(prog, {}, fuel=1000)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 6)

    def test_for_loop_requires_non_negative_int_bound(self):
        prog = build_program(
            [
                ("for", "i", ("sub", ("const", 1), ("const", 3)), [("assign", "x", ("const", 1))]),
                ("return", ("const", 0)),
            ]
        )
        _, out = run_program(prog, {}, fuel=1000)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.TYPE)

    def test_for_loop_rejects_float_bound(self):
        prog = build_program(
            [
                ("for", "i", ("const", 1.0), [("assign", "x", ("const", 1))]),
                ("return", ("const", 0)),
            ]
        )
        _, out = run_program(prog, {}, fuel=1000)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.TYPE)

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

    def test_builtin_len_string_and_list(self):
        self.assertEqual(builtin_call("len", ["abcd"]), 4)
        self.assertEqual(builtin_call("len", [[1, 2, 3]]), 3)

    def test_builtin_len_via_ast(self):
        prog = build_program([("return", ("call", "len", [("const", "abcd")]))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 4)

    def test_builtin_len_type_error(self):
        out = builtin_call("len", [7])
        self.assertEqual(out.code, ErrCode.TYPE)

    def test_builtin_concat_string_and_list(self):
        self.assertEqual(builtin_call("concat", ["ab", "cd"]), "abcd")
        self.assertEqual(builtin_call("concat", [[1, 2], [3]]), [1, 2, 3])

    def test_builtin_concat_type_error(self):
        out = builtin_call("concat", ["ab", [1]])
        self.assertEqual(out.code, ErrCode.TYPE)

    def test_builtin_concat_via_ast(self):
        prog = build_program([("return", ("call", "concat", [("const", "ab"), ("const", "cd")]))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, "abcd")

    def test_builtin_slice_string_and_list(self):
        self.assertEqual(builtin_call("slice", ["abcdef", 1, 4]), "bcd")
        self.assertEqual(builtin_call("slice", [[1, 2, 3, 4], 1, 3]), [2, 3])

    def test_builtin_slice_type_error(self):
        out = builtin_call("slice", ["abcdef", 1.5, 4])
        self.assertEqual(out.code, ErrCode.TYPE)

    def test_builtin_slice_via_ast(self):
        prog = build_program([("return", ("call", "slice", [("const", "abcdef"), ("const", 1), ("const", 4)]))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, "bcd")

    def test_builtin_index_string_and_list(self):
        self.assertEqual(builtin_call("index", ["abcdef", 1]), "b")
        self.assertEqual(builtin_call("index", [[1, 2, 3], -1]), 3)

    def test_builtin_index_type_and_bounds_error(self):
        out = builtin_call("index", ["abcdef", 1.5])
        self.assertEqual(out.code, ErrCode.TYPE)
        out2 = builtin_call("index", ["ab", 5])
        self.assertEqual(out2.code, ErrCode.VALUE)

    def test_builtin_index_via_ast(self):
        prog = build_program([("return", ("call", "index", [("const", "abcdef"), ("const", 2)]))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, "c")

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

    def test_string_list_eq_ne(self):
        eq_prog = build_program([("return", ("eq", ("const", "ab"), ("const", "ab")))])
        _, eq_out = run_program(eq_prog, {}, fuel=100)
        self.assertIsInstance(eq_out, Returned)
        self.assertEqual(eq_out.value, True)

        ne_prog = build_program([("return", ("ne", ("const", [1, 2]), ("const", [1, 3])))])
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
