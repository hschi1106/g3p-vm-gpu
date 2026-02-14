import unittest

from src.g3p_vm_gpu.ast import (
    Block, Assign, IfStmt, ForRange, Return,
    Const, Var, Unary, Binary, IfExpr, Call,
    UOp, BOp
)
from src.g3p_vm_gpu.interp import run_program, eval_expr
from src.g3p_vm_gpu.errors import Returned, Failed, ErrCode
from src.g3p_vm_gpu.fuzz import make_random_program


class TestEval(unittest.TestCase):
    def test_short_circuit_and(self):
        # False and (1/0) should not evaluate RHS
        e = Binary(BOp.AND, Const(False), Binary(BOp.DIV, Const(1), Const(0)))
        r, _ = eval_expr(e, {}, fuel=100)
        self.assertEqual(r, False)

    def test_short_circuit_or(self):
        e = Binary(BOp.OR, Const(True), Binary(BOp.DIV, Const(1), Const(0)))
        r, _ = eval_expr(e, {}, fuel=100)
        self.assertEqual(r, True)

    def test_if_requires_bool(self):
        prog = Block([
            IfStmt(Const(1), Block([Return(Const(1))]), Block([Return(Const(2))]))
        ])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.TYPE)

    def test_for_loop(self):
        prog = Block([
            Assign("x", Const(0)),
            ForRange("i", 5, Block([Assign("x", Binary(BOp.ADD, Var("x"), Const(1)))])),
            Return(Var("x"))
        ])
        _, out = run_program(prog, {}, fuel=1000)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 5)

    def test_return_stops_block(self):
        prog = Block([
            Return(Const(1)),
            Assign("x", Const(999)),
        ])
        env, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 1)
        self.assertNotIn("x", env)

    def test_builtins_clip(self):
        prog = Block([Return(Call("clip", [Const(5), Const(0), Const(2)]))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Returned)
        self.assertEqual(out.value, 2)

    def test_timeout(self):
        # fuel too small to even evaluate return
        prog = Block([Return(Const(1))])
        _, out = run_program(prog, {}, fuel=0)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.TIMEOUT)

    def test_program_without_return_is_value_error(self):
        prog = Block([Assign("x", Const(1))])
        _, out = run_program(prog, {}, fuel=100)
        self.assertIsInstance(out, Failed)
        self.assertEqual(out.err.code, ErrCode.VALUE)

    def test_none_eq_ne_with_non_none(self):
        eq_prog = Block([Return(Binary(BOp.EQ, Const(None), Const(1)))])
        _, eq_out = run_program(eq_prog, {}, fuel=100)
        self.assertIsInstance(eq_out, Returned)
        self.assertEqual(eq_out.value, False)

        ne_prog = Block([Return(Binary(BOp.NE, Const(None), Const(1)))])
        _, ne_out = run_program(ne_prog, {}, fuel=100)
        self.assertIsInstance(ne_out, Returned)
        self.assertEqual(ne_out.value, True)

    def test_fuzz_random_programs(self):
        """Test that interp can evaluate 1000 random fuzz-generated ASTs without crashing."""
        passed = 0
        failed = 0
        timeout = 0
        
        for i in range(1000):
            prog = make_random_program(seed=i, depth=3)
            env, out = run_program(prog, {}, fuel=10000)
            
            # Program should complete without crashing.
            # Top-level run_program outcomes are Returned or Failed.
            self.assertIn(type(out).__name__, ['Returned', 'Failed'])
            
            if isinstance(out, Returned):
                passed += 1
            elif isinstance(out, Failed):
                if out.err.code == ErrCode.TIMEOUT:
                    timeout += 1
                else:
                    failed += 1
        
        # Print statistics for debugging
        print(f"\nFuzz test results: {passed} passed, {failed} failed, {timeout} timeout")
        
        # The test passes as long as all programs complete without Python exceptions
        self.assertEqual(passed + failed + timeout, 1000)


if __name__ == "__main__":
    unittest.main()
