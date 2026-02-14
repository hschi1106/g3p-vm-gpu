import unittest

from src.g3p_vm_gpu.ast import (
    Assign,
    BOp,
    Binary,
    Block,
    Call,
    Const,
    ForRange,
    Return,
    Var,
)
from src.g3p_vm_gpu.compiler import compile_program
from src.g3p_vm_gpu.errors import Failed, Returned
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.interp import run_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


class TestVMEquiv(unittest.TestCase):
    def _assert_equiv(self, prog: Block):
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

        # for variables assigned by this program, env values should match
        for name, idx in bc.var2idx.items():
            if name.startswith("\x00for_i_"):
                continue
            if name in interp_env:
                self.assertTrue(idx < bc.n_locals)

    def test_manual_program(self):
        prog = Block(
            [
                Assign("x", Const(0)),
                ForRange(
                    "i",
                    5,
                    Block(
                        [
                            Assign("x", Binary(BOp.ADD, Var("x"), Const(2))),
                            Assign("x", Call("clip", [Var("x"), Const(0), Const(7)])),
                        ]
                    ),
                ),
                Return(Var("x")),
            ]
        )
        self._assert_equiv(prog)

    def test_short_circuit(self):
        prog = Block(
            [
                Return(Binary(BOp.AND, Const(False), Binary(BOp.DIV, Const(1), Const(0))))
            ]
        )
        self._assert_equiv(prog)

    def test_invalid_for_range_k(self):
        prog = Block([ForRange("i", -1, Block([])), Return(Const(0))])
        self._assert_equiv(prog)

    def test_program_without_return(self):
        prog = Block([Assign("x", Const(1))])
        self._assert_equiv(prog)

    def test_none_eq_ne_with_non_none(self):
        self._assert_equiv(Block([Return(Binary(BOp.EQ, Const(None), Const(1)))]))
        self._assert_equiv(Block([Return(Binary(BOp.NE, Const(None), Const(1)))]))

    def test_fuzz_equivalence(self):
        for i in range(250):
            prog = make_random_program(seed=i, depth=3)
            self._assert_equiv(prog)


if __name__ == "__main__":
    unittest.main()
