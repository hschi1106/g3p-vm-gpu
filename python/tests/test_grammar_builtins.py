import unittest

from src.g3p_vm_gpu.core.ast import Char, FloatList, IntList, StringList, make_char
from src.g3p_vm_gpu.core.errors import ErrCode
from src.g3p_vm_gpu.runtime.builtins import builtin_call


class TestGrammarBuiltins(unittest.TestCase):
    def test_clip_uses_nested_min_max_when_bounds_are_reversed(self):
        self.assertEqual(builtin_call("clip", [5, 10, 0]), 0)
        self.assertEqual(builtin_call("clip", [5.0, 10.0, 0.5]), 0.5)

    def test_idiv0_is_exact_toward_zero_integer_division(self):
        self.assertEqual(builtin_call("idiv0", [-7, 2]), -3)
        self.assertEqual(builtin_call("idiv0", [7, -2]), -3)
        self.assertEqual(builtin_call("idiv0", [10**30 + 1, 3]), (10**30 + 1) // 3)

    def test_index_string_returns_char(self):
        self.assertEqual(builtin_call("index", ["abc", 1]), Char("b"))
        self.assertEqual(builtin_call("index", ["abc", -1]), Char("c"))

    def test_direct_list_builtins_are_strict(self):
        self.assertEqual(builtin_call("concat", [IntList((1,)), IntList((2,))]), IntList((1, 2)))
        self.assertEqual(builtin_call("concat", [FloatList((1.0,)), FloatList((2.0,))]), FloatList((1.0, 2.0)))
        self.assertEqual(builtin_call("append", [IntList((1,)), 2]), IntList((1, 2)))
        self.assertEqual(builtin_call("append", [FloatList((1.0,)), 2.0]), FloatList((1.0, 2.0)))
        self.assertEqual(builtin_call("prepend", [StringList(("b",)), "a"]), StringList(("a", "b")))

        out = builtin_call("concat", [IntList((1,)), FloatList((2.0,))])
        self.assertEqual(out.code, ErrCode.TYPE)
        out = builtin_call("append", [FloatList((1.0,)), 2])
        self.assertEqual(out.code, ErrCode.TYPE)

    def test_char_conversion_builtins(self):
        self.assertEqual(builtin_call("char_to_string", [make_char("x")]), "x")
        self.assertEqual(builtin_call("string_to_char", ["x"]), make_char("x"))
        self.assertEqual(builtin_call("ord", [make_char("A")]), 65)
        self.assertEqual(builtin_call("chr", [65]), make_char("A"))

        out = builtin_call("string_to_char", ["xy"])
        self.assertEqual(out.code, ErrCode.VALUE)
        out = builtin_call("chr", [-1])
        self.assertEqual(out.code, ErrCode.VALUE)

    def test_char_predicates_and_case(self):
        self.assertEqual(builtin_call("is_letter", [make_char("a")]), True)
        self.assertEqual(builtin_call("is_digit", [make_char("7")]), True)
        self.assertEqual(builtin_call("is_space", [make_char(" ")]), True)
        self.assertEqual(builtin_call("is_vowel", [make_char("E")]), True)
        self.assertEqual(builtin_call("to_lower", [make_char("A")]), make_char("a"))
        self.assertEqual(builtin_call("to_upper", [make_char("a")]), make_char("A"))

    def test_to_string_uses_current_canonical_float_format(self):
        self.assertEqual(builtin_call("to_string", [123]), "123")
        self.assertEqual(builtin_call("to_string", [1.5]), "1.5")
        self.assertEqual(builtin_call("to_string", [2.0]), "2")
        self.assertEqual(builtin_call("to_string", [-0.0]), "0")
        self.assertEqual(builtin_call("to_string", [1.2345678]), "1.234568")

    def test_singleton_result_types(self):
        self.assertEqual(builtin_call("singleton", [make_char("z")]), "z")
        self.assertEqual(builtin_call("singleton", [3]), IntList((3,)))
        self.assertEqual(builtin_call("singleton", [3.5]), FloatList((3.5,)))
        self.assertEqual(builtin_call("singleton", ["ab"]), StringList(("ab",)))


if __name__ == "__main__":
    unittest.main()
