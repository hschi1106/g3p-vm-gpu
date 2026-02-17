import unittest

from src.g3p_vm_gpu.ast import build_program
from src.g3p_vm_gpu.compiler import compile_program
from src.g3p_vm_gpu.errors import Failed, Returned
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.interp import run_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


class TestVMEquiv(unittest.TestCase):
    def _assert_equiv(self, prog):
        interp_env, interp_out = run_program(prog, {}, fuel=20_000)
        bc = compile_program(prog)
        vm_out = run_bytecode(bc, {}, fuel=20_000)

        if isinstance(interp_out, Returned):
            self.assertIsInstance(vm_out, VMReturn)
            self.assertEqual(interp_out.value, vm_out.value)
        elif isinstance(interp_out, Failed):
            self.assertIsInstance(vm_out, VMError)
            self.assertEqual(interp_out.err.code, vm_out.err.code)
        else:
            self.fail("run_program should not return Normal for top-level programs")

        for name, idx in bc.var2idx.items():
            if name.startswith("\x00for_i_"):
                continue
            if name in interp_env:
                self.assertTrue(idx < bc.n_locals)

    def test_manual_program(self):
        prog = build_program(
            [
                ("assign", "x", ("const", 0)),
                (
                    "for",
                    "i",
                    5,
                    [
                        ("assign", "x", ("add", ("var", "x"), ("const", 2))),
                        ("assign", "x", ("call", "clip", [("var", "x"), ("const", 0), ("const", 7)])),
                    ],
                ),
                ("return", ("var", "x")),
            ]
        )
        self._assert_equiv(prog)

    def test_short_circuit(self):
        prog = build_program([("return", ("and", ("const", False), ("div", ("const", 1), ("const", 0))))])
        self._assert_equiv(prog)

    def test_invalid_for_range_k(self):
        prog = build_program([("for", "i", -1, []), ("return", ("const", 0))])
        self._assert_equiv(prog)

    def test_program_without_return(self):
        prog = build_program([("assign", "x", ("const", 1))])
        self._assert_equiv(prog)

    def test_none_eq_ne_with_non_none(self):
        self._assert_equiv(build_program([("return", ("eq", ("const", None), ("const", 1)))]))
        self._assert_equiv(build_program([("return", ("ne", ("const", None), ("const", 1)))]))

    def test_fuzz_equivalence(self):
        for i in range(250):
            prog = make_random_program(seed=i, depth=3)
            self._assert_equiv(prog)


if __name__ == "__main__":
    unittest.main()
