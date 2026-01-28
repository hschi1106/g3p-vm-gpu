from __future__ import annotations

from g3p_vm_gpu.ast import (
    Block, Assign, IfStmt, ForRange, Return,
    Const, Var, Binary, UOp, BOp, Unary, Call
)
from g3p_vm_gpu.interp import run_program
from g3p_vm_gpu.errors import Returned, Failed


def main():
    # program:
    # x = 0
    # for i in range(3):
    #   x = x + 1
    # if (x == 3) and True:
    #   return clip(x, 0, 2)
    # else:
    #   return 999
    prog = Block([
        Assign("x", Const(0)),
        ForRange("i", 3, Block([
            Assign("x", Binary(BOp.ADD, Var("x"), Const(1))),
        ])),
        IfStmt(
            Binary(BOp.AND,
                   Binary(BOp.EQ, Var("x"), Const(3)),
                   Const(True)),
            Block([Return(Call("clip", [Var("x"), Const(0), Const(2)]))]),
            Block([Return(Const(999))]),
        )
    ])

    env, out = run_program(prog, inputs={}, fuel=10_000)
    if isinstance(out, Returned):
        print("Returned:", out.value)
    elif isinstance(out, Failed):
        print("Error:", out.err.code.value, out.err.message)
    else:
        print("Finished without return. Env:", env)


if __name__ == "__main__":
    main()
