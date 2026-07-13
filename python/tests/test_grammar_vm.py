import unittest

from src.g3p_vm_gpu.core.ast import (
    Char,
    FloatList,
    IntList,
    StringList,
    build_program,
    make_char,
    make_float_list,
    make_int_list,
    make_string_list,
)
from src.g3p_vm_gpu.core.errors import ErrCode, Failed, Returned
from src.g3p_vm_gpu.evolution.stmt_codec import top_level_statements
from src.g3p_vm_gpu.runtime.compiler import compile_program
from src.g3p_vm_gpu.runtime.interp import run_program
from src.g3p_vm_gpu.runtime.vm import ExecError, ExecReturn, exec_bytecode


class TestGrammarVm(unittest.TestCase):
    def assert_interp_and_vm_return(self, expr, expected):
        program = build_program([("return", expr)])
        _env, out = run_program(program, {}, fuel=1000)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, expected)

        bytecode = compile_program(program)
        vm_out = exec_bytecode(bytecode, {}, fuel=1000)
        self.assertIsInstance(vm_out, ExecReturn)
        self.assertEqual(vm_out.value, expected)

    def assert_interp_and_vm_error(self, expr, code):
        program = build_program([("return", expr)])
        _env, out = run_program(program, {}, fuel=1000)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, code)

        bytecode = compile_program(program)
        vm_out = exec_bytecode(bytecode, {}, fuel=1000)
        self.assertIsInstance(vm_out, ExecError)
        self.assertEqual(vm_out.err.code, code)

    def assert_interp_and_vm_error_with_fuel(self, expr, code, interp_fuel, vm_fuel):
        program = build_program([("return", expr)])
        _env, out = run_program(program, {}, fuel=interp_fuel)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, code)

        vm_out = exec_bytecode(compile_program(program), {}, fuel=vm_fuel)
        self.assertIsInstance(vm_out, ExecError)
        self.assertEqual(vm_out.err.code, code)

    def test_string_index_char_conversion_round_trip(self):
        self.assert_interp_and_vm_return(
            ("call", "char_to_string", [("call", "index", [("const", "abc"), ("const", 1)])]),
            "b",
        )

    def test_char_predicate_and_case_builtin(self):
        self.assert_interp_and_vm_return(("call", "to_upper", [("const", make_char("q"))]), Char("Q"))
        self.assert_interp_and_vm_return(("call", "is_vowel", [("const", make_char("E"))]), True)

    def test_direct_list_bytecode_builtins(self):
        self.assert_interp_and_vm_return(
            ("call", "append", [("const", make_int_list([1, 2])), ("const", 3)]),
            IntList((1, 2, 3)),
        )
        self.assert_interp_and_vm_return(
            ("call", "prepend", [("const", make_float_list([2.0, 3.0])), ("const", 1.0)]),
            FloatList((1.0, 2.0, 3.0)),
        )

    def test_singleton_result_types(self):
        self.assert_interp_and_vm_return(("call", "singleton", [("const", make_char("z"))]), "z")
        self.assert_interp_and_vm_return(("call", "singleton", [("const", 4)]), IntList((4,)))
        self.assert_interp_and_vm_return(("call", "singleton", [("const", 4.5)]), FloatList((4.5,)))

    def test_protected_integer_ops(self):
        self.assert_interp_and_vm_return(("call", "idiv0", [("const", 7), ("const", 0)]), 0)
        self.assert_interp_and_vm_return(("call", "imod0", [("const", 7), ("const", 0)]), 0)
        self.assert_interp_and_vm_return(("call", "idiv0", [("const", -7), ("const", 2)]), -3)

    def test_exact_equality_rejects_char_string(self):
        self.assert_interp_and_vm_error(("eq", ("const", make_char("a")), ("const", "a")), ErrCode.TYPE)

    def test_mixed_int_float_numeric_ops_are_rejected(self):
        self.assert_interp_and_vm_error(("add", ("const", 1), ("const", 1.5)), ErrCode.TYPE)
        self.assert_interp_and_vm_error(("lt", ("const", 1), ("const", 1.5)), ErrCode.TYPE)
        self.assert_interp_and_vm_error(("call", "min", [("const", 1), ("const", 1.5)]), ErrCode.TYPE)
        self.assert_interp_and_vm_error(("call", "clip", [("const", 1), ("const", 0), ("const", 2.0)]), ErrCode.TYPE)

    def test_map_list_visits_left_to_right(self):
        self.assert_interp_and_vm_return(
            (
                "map_list",
                "x",
                ("const", make_int_list([1, 2, 3])),
                ("mul", ("bound", "x"), ("const", 2)),
                "int",
            ),
            IntList((2, 4, 6)),
        )

    def test_map_list_evaluates_source_once(self):
        source = ("const", make_int_list([1]))
        for _ in range(30):
            source = ("call", "reverse", [source])
        source = ("call", "concat", [source, ("const", make_int_list([2]))])
        expr = ("map_list", "x", source, ("bound", "x"), "int")
        program = build_program([("return", expr)])

        _env, out = run_program(program, {}, fuel=80)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, IntList((1, 2)))

        vm_out = exec_bytecode(compile_program(program), {}, fuel=140)
        self.assertIsInstance(vm_out, ExecReturn)
        self.assertEqual(vm_out.value, IntList((1, 2)))

    def test_map_list_returns_typed_empty_output(self):
        self.assert_interp_and_vm_return(
            (
                "map_list",
                "s",
                ("const", make_string_list([])),
                ("call", "concat", [("bound", "s"), ("const", "!")]),
                "string",
            ),
            StringList(()),
        )

    def test_filter_list_preserves_order(self):
        self.assert_interp_and_vm_return(
            (
                "filter_list",
                "x",
                ("const", make_int_list([3, 1, 4, 1, 5])),
                ("gt", ("bound", "x"), ("const", 2)),
            ),
            IntList((3, 4, 5)),
        )

    def test_filter_list_returns_source_typed_empty_output(self):
        self.assert_interp_and_vm_return(
            (
                "filter_list",
                "x",
                ("const", make_float_list([1.0, 2.0])),
                ("gt", ("bound", "x"), ("const", 5.0)),
            ),
            FloatList(()),
        )

    def test_filter_list_evaluates_source_once(self):
        source = ("const", make_int_list([1, 2, 3]))
        for _ in range(30):
            source = ("call", "reverse", [source])
        source = ("call", "concat", [source, ("const", make_int_list([4]))])
        expr = ("filter_list", "x", source, ("gt", ("bound", "x"), ("const", 1)))
        program = build_program([("return", expr)])

        _env, out = run_program(program, {}, fuel=70)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, IntList((2, 3, 4)))

        vm_out = exec_bytecode(compile_program(program), {}, fuel=150)
        self.assertIsInstance(vm_out, ExecReturn)
        self.assertEqual(vm_out.value, IntList((2, 3, 4)))

    def test_structured_expression_fuel_exhaustion_is_timeout(self):
        self.assert_interp_and_vm_error_with_fuel(
            (
                "map_list",
                "x",
                ("const", make_int_list([1, 2, 3])),
                ("mul", ("bound", "x"), ("const", 2)),
                "int",
            ),
            ErrCode.TIMEOUT,
            interp_fuel=8,
            vm_fuel=8,
        )
        self.assert_interp_and_vm_error_with_fuel(
            (
                "filter_list",
                "x",
                ("const", make_int_list([1, 2, 3])),
                ("gt", ("bound", "x"), ("const", 1)),
            ),
            ErrCode.TIMEOUT,
            interp_fuel=8,
            vm_fuel=8,
        )
        self.assert_interp_and_vm_error_with_fuel(
            (
                "linear_rec",
                "u",
                "v",
                "i",
                ("const", make_int_list([1, 2, 3])),
                ("const", 0),
                ("const", 0),
                ("add", ("bound", "u"), ("bound", "v")),
                ("bound", "u"),
            ),
            ErrCode.TIMEOUT,
            interp_fuel=8,
            vm_fuel=8,
        )

    def test_linear_rec_empty_case_only(self):
        self.assert_interp_and_vm_return(
            (
                "linear_rec",
                "u",
                "v",
                "i",
                ("const", make_int_list([])),
                ("const", 5),
                ("const", 42),
                ("div", ("const", 1), ("const", 0)),
                ("div", ("const", 1), ("const", 0)),
            ),
            42,
        )

    def test_linear_rec_singleton_case_only(self):
        self.assert_interp_and_vm_return(
            (
                "linear_rec",
                "u",
                "v",
                "i",
                ("const", make_int_list([7])),
                ("const", 5),
                ("const", 0),
                ("div", ("const", 1), ("const", 0)),
                ("add", ("mul", ("bound", "u"), ("const", 10)), ("bound", "i")),
            ),
            75,
        )

    def test_linear_rec_right_to_left_step_order(self):
        self.assert_interp_and_vm_return(
            (
                "linear_rec",
                "u",
                "v",
                "i",
                ("const", make_int_list([1, 2, 3])),
                ("const", 4),
                ("const", 0),
                ("add", ("mul", ("bound", "v"), ("const", 10)), ("bound", "u")),
                ("add", ("mul", ("bound", "u"), ("const", 100)), ("bound", "i")),
            ),
            30621,
        )

    def test_linear_rec_errors(self):
        self.assert_interp_and_vm_error(
            (
                "linear_rec",
                "u",
                "v",
                "i",
                ("const", "abc"),
                ("const", 0),
                ("const", 0),
                ("bound", "v"),
                ("bound", "u"),
            ),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            (
                "linear_rec",
                "u",
                "v",
                "i",
                ("const", make_int_list([1])),
                ("const", 0.0),
                ("const", 0),
                ("bound", "v"),
                ("bound", "u"),
            ),
            ErrCode.TYPE,
        )

    def test_linear_rec_codec_round_trip_preserves_binders(self):
        statements = [
            (
                "return",
                (
                    "linear_rec",
                    "u",
                    "v",
                    "i",
                    ("const", make_int_list([1, 2])),
                    ("const", 0),
                    ("const", 0),
                    ("add", ("bound", "u"), ("bound", "v")),
                    ("add", ("bound", "u"), ("bound", "i")),
                ),
            )
        ]
        program = build_program(statements)
        self.assertEqual(top_level_statements(program), statements)

    def test_nested_binders_are_capture_safe(self):
        self.assert_interp_and_vm_return(
            (
                "map_list",
                "x",
                ("const", make_int_list([1, 2])),
                (
                    "add",
                    ("bound", "x"),
                    (
                        "call",
                        "index",
                        [
                            (
                                "map_list",
                                "x",
                                ("const", make_int_list([10])),
                                ("bound", "x"),
                                "int",
                            ),
                            ("const", 0),
                        ],
                    ),
                ),
                "int",
            ),
            IntList((11, 12)),
        )

    def test_bound_var_is_distinct_from_ordinary_local(self):
        program = build_program(
            [
                ("assign", "x", ("const", 100)),
                (
                    "return",
                    (
                        "map_list",
                        "x",
                        ("const", make_int_list([1])),
                        ("add", ("bound", "x"), ("var", "x")),
                        "int",
                    ),
                ),
            ]
        )
        _env, out = run_program(program, {}, fuel=1000)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, IntList((101,)))
        vm_out = exec_bytecode(compile_program(program), {}, fuel=1000)
        self.assertIsInstance(vm_out, ExecReturn)
        self.assertEqual(vm_out.value, IntList((101,)))

    def test_structured_list_errors(self):
        self.assert_interp_and_vm_error(
            ("map_list", "c", ("const", "abc"), ("bound", "c"), "string"),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            ("filter_list", "x", ("const", make_int_list([1])), ("bound", "x")),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            ("map_list", "x", ("const", make_int_list([0, 1])), ("div", ("const", 1), ("bound", "x")), "int"),
            ErrCode.ZERODIV,
        )

    def test_asgp_dc_vm_matches_reference_interpreter(self):
        self.assert_interp_and_vm_return(
            (
                "asgp_dc",
                ("const", make_int_list([1, 2, 3, 4])),
                ("xs", "n", "lo"),
                ("call", "index", [("bound", "xs"), ("const", 0)]),
                "n",
                ("const", 999),
                ("r1", "r2"),
                ("add", ("bound", "r1"), ("bound", "r2")),
            ),
            10,
        )
        self.assert_interp_and_vm_return(
            (
                "asgp_dc",
                ("const", "abc"),
                ("xs", "n", "lo"),
                ("call", "char_to_string", [("call", "index", [("bound", "xs"), ("const", 0)])]),
                "n",
                ("const", 1),
                ("left", "right"),
                ("call", "concat", [("bound", "left"), ("bound", "right")]),
            ),
            "abc",
        )

    def test_asgp_dc_vm_errors_match_reference_interpreter(self):
        self.assert_interp_and_vm_error(
            (
                "asgp_dc",
                ("const", 1),
                ("xs", "n", "lo"),
                ("const", 0),
                "n",
                ("const", 1),
                ("r1", "r2"),
                ("const", 0),
            ),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            (
                "asgp_dc",
                ("const", make_int_list([1, 2])),
                ("xs", "n", "lo"),
                ("call", "index", [("bound", "xs"), ("const", 0)]),
                "n",
                ("const", 1.5),
                ("r1", "r2"),
                ("add", ("bound", "r1"), ("bound", "r2")),
            ),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            (
                "asgp_dc",
                ("const", make_int_list([1, 2])),
                ("xs", "n", "lo"),
                ("if_expr", ("eq", ("bound", "lo"), ("const", 0)), ("const", 1), ("const", True)),
                "n",
                ("const", 1),
                ("r1", "r2"),
                ("bound", "r1"),
            ),
            ErrCode.TYPE,
        )

    def test_asgp_dc_vm_phase_bodies_do_not_see_ordinary_locals(self):
        program = build_program(
            [
                ("assign", "hidden", ("const", 7)),
                (
                    "return",
                    (
                        "asgp_dc",
                        ("const", make_int_list([1])),
                        ("xs", "n", "lo"),
                        ("var", "hidden"),
                        "n",
                        ("const", 1),
                        ("r1", "r2"),
                        ("add", ("bound", "r1"), ("bound", "r2")),
                    ),
                ),
            ]
        )
        _env, out = run_program(program, {}, fuel=1000)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.NAME)
        vm_out = exec_bytecode(compile_program(program), {}, fuel=1000)
        self.assertIsInstance(vm_out, ExecError)
        self.assertEqual(vm_out.err.code, ErrCode.NAME)

    def test_asgp_dp1d_vm_matches_reference_interpreter(self):
        self.assert_interp_and_vm_return(
            (
                "asgp_dp1d",
                ("const", 4),
                (0, 5, 0, 0),
                "s",
                ("const", 1),
                ("backward1", (1,)),
                ("s", "d1"),
                ("add", ("bound", "d1"), ("bound", "s")),
            ),
            11,
        )
        self.assert_interp_and_vm_return(
            (
                "asgp_dp1d",
                ("const", 20),
                (0, 20, 0, 0),
                "s",
                ("const", 1),
                ("backward2", (1, 2)),
                ("s", "d1", "d2"),
                ("add", ("bound", "d1"), ("bound", "d2")),
            ),
            10946,
        )

    def test_asgp_dp1d_vm_errors_match_reference_interpreter(self):
        self.assert_interp_and_vm_error(
            (
                "asgp_dp1d",
                ("const", 1.5),
                (0, 3, 0, 0),
                "s",
                ("const", 1),
                ("backward1", (1,)),
                ("s", "d1"),
                ("bound", "d1"),
            ),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            (
                "asgp_dp1d",
                ("const", 1),
                (0, 3, 0, 0),
                "s",
                ("const", 1),
                ("backward1", (1,)),
                ("s", "d1"),
                ("const", True),
            ),
            ErrCode.TYPE,
        )

    def test_asgp_dp2d_vm_matches_reference_interpreter(self):
        self.assert_interp_and_vm_return(
            (
                "asgp_dp2d",
                ("const", 2),
                ("const", 2),
                (0, 3, 0, 3, 0, 0, 0),
                ("i", "j"),
                ("const", 1),
                "neighborhood_backward3",
                ("i", "j", "d1", "d2", "d3"),
                ("add", ("add", ("bound", "d1"), ("bound", "d2")), ("bound", "d3")),
            ),
            13,
        )
        self.assert_interp_and_vm_return(
            (
                "asgp_dp2d",
                ("const", -1),
                ("const", 0),
                (0, 3, 0, 3, 0, 0, 99),
                ("i", "j"),
                ("const", 1),
                "cross_backward",
                ("i", "j", "d1", "d2"),
                ("add", ("bound", "d1"), ("bound", "d2")),
            ),
            99,
        )

    def test_asgp_dp2d_vm_errors_match_reference_interpreter(self):
        self.assert_interp_and_vm_error(
            (
                "asgp_dp2d",
                ("const", 1.5),
                ("const", 0),
                (0, 3, 0, 3, 0, 0, 0),
                ("i", "j"),
                ("const", 1),
                "diagonal_backward",
                ("i", "j", "d1"),
                ("bound", "d1"),
            ),
            ErrCode.TYPE,
        )
        self.assert_interp_and_vm_error(
            (
                "asgp_dp2d",
                ("const", 1),
                ("const", 0),
                (0, 3, 0, 3, 0, 0, 0),
                ("i", "j"),
                ("const", 1),
                "diagonal_backward",
                ("i", "j", "d1"),
                ("const", True),
            ),
            ErrCode.TYPE,
        )

    def test_asgp_dp2d_vm_phase_bodies_do_not_see_ordinary_locals(self):
        program = build_program(
            [
                ("assign", "hidden", ("const", 7)),
                (
                    "return",
                    (
                        "asgp_dp2d",
                        ("const", 0),
                        ("const", 0),
                        (0, 3, 0, 3, 0, 0, 0),
                        ("i", "j"),
                        ("var", "hidden"),
                        "diagonal_backward",
                        ("i", "j", "d1"),
                        ("bound", "d1"),
                    ),
                ),
            ]
        )
        _env, out = run_program(program, {}, fuel=1000)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.NAME)
        vm_out = exec_bytecode(compile_program(program), {}, fuel=1000)
        self.assertIsInstance(vm_out, ExecError)
        self.assertEqual(vm_out.err.code, ErrCode.NAME)


if __name__ == "__main__":
    unittest.main()
