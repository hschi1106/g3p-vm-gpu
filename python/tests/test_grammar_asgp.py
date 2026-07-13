import unittest

from src.g3p_vm_gpu.core.ast import build_program, make_int_list
from src.g3p_vm_gpu.core.errors import ErrCode, Failed, Returned
from src.g3p_vm_gpu.runtime.interp import run_program


class TestGrammarAsgp(unittest.TestCase):
    def assert_interp_return(self, expr, expected, fuel=1000):
        program = build_program([("return", expr)])
        _env, out = run_program(program, {}, fuel=fuel)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, expected)

    def assert_interp_error(self, expr, code, fuel=1000):
        program = build_program([("return", expr)])
        _env, out = run_program(program, {}, fuel=fuel)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, code)

    def test_asgp_dc_sums_int_list_with_clamped_split(self):
        self.assert_interp_return(
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

    def test_asgp_dc_can_traverse_string_as_char_sequence(self):
        self.assert_interp_return(
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

    def test_asgp_dc_phase_bodies_see_only_phase_binders(self):
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

    def test_asgp_dc_rejects_non_sequence_source_and_non_int_split(self):
        self.assert_interp_error(
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
        self.assert_interp_error(
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

    def test_asgp_dc_requires_consistent_recursive_result_types(self):
        self.assert_interp_error(
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

    def test_asgp_dc_fuel_and_nested_asgp_errors(self):
        self.assert_interp_error(
            (
                "asgp_dc",
                ("const", make_int_list([1, 2, 3, 4])),
                ("xs", "n", "lo"),
                ("call", "index", [("bound", "xs"), ("const", 0)]),
                "n",
                ("const", 1),
                ("r1", "r2"),
                ("add", ("bound", "r1"), ("bound", "r2")),
            ),
            ErrCode.TIMEOUT,
            fuel=6,
        )
        nested = (
            "asgp_dc",
            ("const", make_int_list([1])),
            ("xs", "n", "lo"),
            ("const", 1),
            "n",
            ("const", 1),
            ("r1", "r2"),
            ("bound", "r1"),
        )
        self.assert_interp_error(
            (
                "asgp_dc",
                ("const", make_int_list([1])),
                ("xs", "n", "lo"),
                nested,
                "n",
                ("const", 1),
                ("r1", "r2"),
                ("bound", "r1"),
            ),
            ErrCode.TYPE,
        )

    def test_asgp_dp1d_backward_recurrence_and_boundary(self):
        self.assert_interp_return(
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
        self.assert_interp_return(
            (
                "asgp_dp1d",
                ("const", -1),
                (0, 5, 0, 99),
                "s",
                ("const", 1),
                ("backward1", (1,)),
                ("s", "d1"),
                ("add", ("bound", "d1"), ("bound", "s")),
            ),
            99,
        )

    def test_asgp_dp1d_memoizes_overlapping_dependencies(self):
        self.assert_interp_return(
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
            fuel=500,
        )

    def test_asgp_dp1d_phase_visibility_and_type_errors(self):
        program = build_program(
            [
                ("assign", "hidden", ("const", 7)),
                (
                    "return",
                    (
                        "asgp_dp1d",
                        ("const", 0),
                        (0, 3, 0, 0),
                        "s",
                        ("var", "hidden"),
                        ("backward1", (1,)),
                        ("s", "d1"),
                        ("bound", "d1"),
                    ),
                ),
            ]
        )
        _env, out = run_program(program, {}, fuel=1000)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.NAME)

        self.assert_interp_error(
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
        self.assert_interp_error(
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

    def test_asgp_dp1d_rejects_nested_asgp_in_phase(self):
        nested = (
            "asgp_dc",
            ("const", make_int_list([1])),
            ("xs", "n", "lo"),
            ("const", 1),
            "n",
            ("const", 1),
            ("r1", "r2"),
            ("bound", "r1"),
        )
        self.assert_interp_error(
            (
                "asgp_dp1d",
                ("const", 0),
                (0, 3, 0, 0),
                "s",
                nested,
                ("backward1", (1,)),
                ("s", "d1"),
                ("bound", "d1"),
            ),
            ErrCode.TYPE,
        )

    def test_asgp_dp2d_neighborhood_recurrence_and_boundary(self):
        transition = ("add", ("add", ("bound", "d1"), ("bound", "d2")), ("bound", "d3"))
        self.assert_interp_return(
            (
                "asgp_dp2d",
                ("const", 2),
                ("const", 2),
                (0, 3, 0, 3, 0, 0, 0),
                ("i", "j"),
                ("const", 1),
                "neighborhood_backward3",
                ("i", "j", "d1", "d2", "d3"),
                transition,
            ),
            13,
        )
        self.assert_interp_return(
            (
                "asgp_dp2d",
                ("const", -1),
                ("const", 0),
                (0, 3, 0, 3, 0, 0, 99),
                ("i", "j"),
                ("const", 1),
                "neighborhood_backward3",
                ("i", "j", "d1", "d2", "d3"),
                transition,
            ),
            99,
        )

    def test_asgp_dp2d_memoizes_overlapping_dependencies(self):
        self.assert_interp_return(
            (
                "asgp_dp2d",
                ("const", 5),
                ("const", 5),
                (0, 5, 0, 5, 0, 0, 0),
                ("i", "j"),
                ("const", 1),
                "neighborhood_backward3",
                ("i", "j", "d1", "d2", "d3"),
                ("add", ("add", ("bound", "d1"), ("bound", "d2")), ("bound", "d3")),
            ),
            1683,
            fuel=500,
        )

    def test_asgp_dp2d_phase_visibility_and_type_errors(self):
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

        self.assert_interp_error(
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
        self.assert_interp_error(
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

    def test_asgp_dp2d_rejects_nested_asgp_in_phase(self):
        nested = (
            "asgp_dc",
            ("const", make_int_list([1])),
            ("xs", "n", "lo"),
            ("const", 1),
            "n",
            ("const", 1),
            ("r1", "r2"),
            ("bound", "r1"),
        )
        self.assert_interp_error(
            (
                "asgp_dp2d",
                ("const", 0),
                ("const", 0),
                (0, 3, 0, 3, 0, 0, 0),
                ("i", "j"),
                nested,
                "diagonal_backward",
                ("i", "j", "d1"),
                ("bound", "d1"),
            ),
            ErrCode.TYPE,
        )


if __name__ == "__main__":
    unittest.main()
