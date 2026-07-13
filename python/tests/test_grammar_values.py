import unittest

from src.g3p_vm_gpu.core import ast as ast_mod
from src.g3p_vm_gpu.core.ast import (
    AstNode,
    AstProgram,
    Char,
    FloatList,
    IntList,
    NodeKind,
    StringList,
    build_program,
    make_char,
    make_float_list,
    make_int_list,
    make_string_list,
    normalize_value,
    validate_prefix_program,
)
from src.g3p_vm_gpu.core.errors import ErrCode
from src.g3p_vm_gpu.core.value_semantics import compare_values
from src.g3p_vm_gpu.runtime.compiler import compile_program


class TestGrammarValues(unittest.TestCase):
    def test_public_num_list_is_removed(self):
        self.assertFalse(hasattr(ast_mod, "NumList"))
        self.assertFalse(hasattr(ast_mod, "make_num_list"))

    def test_char_is_distinct_from_string(self):
        self.assertEqual(make_char("a"), Char("a"))
        self.assertEqual(normalize_value(Char("b")), Char("b"))
        with self.assertRaises(ValueError):
            make_char("")
        with self.assertRaises(ValueError):
            make_char("ab")
        self.assertNotEqual(type(make_char("a")), type("a"))

    def test_direct_list_tags_are_exact(self):
        self.assertEqual(make_int_list([1, 2]), IntList((1, 2)))
        self.assertEqual(make_float_list([1.0, 2.5]), FloatList((1.0, 2.5)))
        self.assertEqual(make_string_list(["a", "bc"]), StringList(("a", "bc")))
        with self.assertRaises(TypeError):
            make_int_list([1, True])
        with self.assertRaises(TypeError):
            make_float_list([1])

    def test_normalize_raw_lists_is_strict(self):
        self.assertEqual(normalize_value([1, 2]), IntList((1, 2)))
        self.assertEqual(normalize_value([1.0, 2.0]), FloatList((1.0, 2.0)))
        self.assertEqual(normalize_value(["a", "bc"]), StringList(("a", "bc")))
        with self.assertRaises(ValueError):
            normalize_value([])
        with self.assertRaises(ValueError):
            normalize_value([1, 2.0])
        with self.assertRaises(ValueError):
            normalize_value([Char("a")])

    def test_none_is_not_public_value(self):
        with self.assertRaises(ValueError):
            normalize_value(None)
        with self.assertRaises(ValueError):
            build_program([("return", ("const", None))])

    def test_ast_prefix_current_is_required(self):
        program = build_program([("return", ("const", 1))])
        self.assertEqual(program.version, "ast-prefix")
        validate_prefix_program(program)

        old = AstProgram(
            nodes=(AstNode(NodeKind.PROGRAM), AstNode(NodeKind.BLOCK_NIL)),
            names=(),
            consts=(),
            version="ast-prefix-old",
        )
        with self.assertRaises(ValueError):
            validate_prefix_program(old)

    def test_current_node_kinds_are_declared(self):
        for kind in (
            NodeKind.BOUND_VAR,
            NodeKind.MAP_LIST,
            NodeKind.FILTER_LIST,
            NodeKind.LINEAR_REC,
            NodeKind.ASGP_DC,
            NodeKind.ASGP_DP1D,
            NodeKind.ASGP_DP2D,
        ):
            self.assertIn(kind, ast_mod.EXPR_KINDS)
            self.assertIn(kind, ast_mod.NODE_ARITY)

    def test_asgp_nodes_are_declared_and_compile_to_segments(self):
        dc_program = build_program(
            [
                (
                    "return",
                    (
                        "asgp_dc",
                        ("const", make_int_list([1])),
                        ("xs", "n", "lo"),
                        ("const", 1),
                        "n",
                        ("const", 1),
                        ("r1", "r2"),
                        ("add", ("bound", "r1"), ("bound", "r2")),
                    ),
                )
            ]
        )
        validate_prefix_program(dc_program)
        dc_bytecode = compile_program(dc_program)
        self.assertEqual(len(dc_bytecode.asgp_dc_segments), 1)

        dp1_program = build_program(
            [
                (
                    "return",
                    (
                        "asgp_dp1d",
                        ("const", 1),
                        (0, 3, 0, 0),
                        "s",
                        ("const", 1),
                        ("backward1", (1,)),
                        ("s", "d1"),
                        ("bound", "d1"),
                    ),
                )
            ]
        )
        validate_prefix_program(dp1_program)
        dp1_bytecode = compile_program(dp1_program)
        self.assertEqual(len(dp1_bytecode.asgp_dp1d_segments), 1)

        dp2_program = build_program(
            [
                (
                    "return",
                    (
                        "asgp_dp2d",
                        ("const", 1),
                        ("const", 1),
                        (0, 3, 0, 3, 0, 0, 0),
                        ("i", "j"),
                        ("const", 1),
                        "diagonal_backward",
                        ("i", "j", "d1"),
                        ("bound", "d1"),
                    ),
                )
            ]
        )
        validate_prefix_program(dp2_program)
        dp2_bytecode = compile_program(dp2_program)
        self.assertEqual(len(dp2_bytecode.asgp_dp2d_segments), 1)

        nodes = [
            AstNode(NodeKind.PROGRAM),
            AstNode(NodeKind.BLOCK_CONS),
            AstNode(NodeKind.RETURN),
            AstNode(NodeKind.ASGP_DP2D),
        ]
        for idx in range(4):
            nodes.append(AstNode(NodeKind.CONST, i0=idx))
        nodes.append(AstNode(NodeKind.BLOCK_NIL))
        program_without_metadata = AstProgram(
            nodes=tuple(nodes),
            names=(),
            consts=tuple(range(4)),
            version="ast-prefix",
        )
        with self.assertRaisesRegex(ValueError, "ASGP-DP2D metadata mismatch"):
            validate_prefix_program(program_without_metadata)

    def test_equality_requires_exact_runtime_type(self):
        out = compare_values("EQ", 1, 1.0)
        self.assertEqual(out.code, ErrCode.TYPE)
        out = compare_values("EQ", Char("a"), "a")
        self.assertEqual(out.code, ErrCode.TYPE)
        out = compare_values("EQ", IntList((1,)), FloatList((1.0,)))
        self.assertEqual(out.code, ErrCode.TYPE)
        self.assertTrue(compare_values("EQ", IntList((1,)), IntList((1,))))


if __name__ == "__main__":
    unittest.main()
