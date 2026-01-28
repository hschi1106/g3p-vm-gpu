from __future__ import annotations

import random
from typing import List

from .ast import (
    Block, Assign, IfStmt, ForRange, Return,
    Const, Var, Unary, Binary, IfExpr, Call,
    UOp, BOp, Expr, Stmt,
)

BUILTINS = ["abs", "min", "max", "clip"]


def rand_const(rng: random.Random):
    t = rng.choice(["int", "float", "bool", "none"])
    if t == "int":
        return Const(rng.randint(-5, 5))
    if t == "float":
        return Const(rng.uniform(-5.0, 5.0))
    if t == "bool":
        return Const(rng.choice([True, False]))
    return Const(None)


def rand_var(rng: random.Random, vars: List[str]):
    return Var(rng.choice(vars)) if vars else rand_const(rng)


def rand_expr(rng: random.Random, vars: List[str], depth: int) -> Expr:
    if depth <= 0:
        return rng.choice([rand_const(rng), rand_var(rng, vars)])

    choice = rng.randint(0, 6)
    if choice == 0:
        return rand_const(rng)
    if choice == 1:
        return rand_var(rng, vars)

    if choice == 2:
        op = rng.choice([UOp.NEG, UOp.NOT])
        return Unary(op, rand_expr(rng, vars, depth - 1))

    if choice == 3:
        op = rng.choice([
            BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD,
            BOp.LT, BOp.LE, BOp.GT, BOp.GE, BOp.EQ, BOp.NE,
            BOp.AND, BOp.OR,
        ])
        return Binary(op, rand_expr(rng, vars, depth - 1), rand_expr(rng, vars, depth - 1))

    if choice == 4:
        return IfExpr(
            rand_expr(rng, vars, depth - 1),
            rand_expr(rng, vars, depth - 1),
            rand_expr(rng, vars, depth - 1),
        )

    if choice == 5:
        f = rng.choice(BUILTINS)
        if f == "abs":
            args = [rand_expr(rng, vars, depth - 1)]
        elif f in ("min", "max"):
            args = [rand_expr(rng, vars, depth - 1), rand_expr(rng, vars, depth - 1)]
        else:
            args = [rand_expr(rng, vars, depth - 1), rand_expr(rng, vars, depth - 1), rand_expr(rng, vars, depth - 1)]
        return Call(f, args)

    # fallback: small binary
    return Binary(BOp.ADD, rand_expr(rng, vars, depth - 1), rand_expr(rng, vars, depth - 1))


def rand_stmt(rng: random.Random, vars: List[str], depth: int) -> Stmt:
    if depth <= 0:
        # force progress
        name = rng.choice(vars) if vars else "x"
        return Assign(name, rand_expr(rng, vars, 0))

    choice = rng.randint(0, 3)
    if choice == 0:
        name = rng.choice(vars) if vars else "x"
        return Assign(name, rand_expr(rng, vars, depth - 1))

    if choice == 1:
        return IfStmt(
            rand_expr(rng, vars, depth - 1),
            rand_block(rng, vars, depth - 1, max_stmts=2),
            rand_block(rng, vars, depth - 1, max_stmts=2),
        )

    if choice == 2:
        loop_var = rng.choice(["i", "j", "k"])
        k = rng.randint(0, 4)
        return ForRange(loop_var, k, rand_block(rng, vars + [loop_var], depth - 1, max_stmts=2))

    return Return(rand_expr(rng, vars, depth - 1))


def rand_block(rng: random.Random, vars: List[str], depth: int, max_stmts: int = 4) -> Block:
    n = rng.randint(1, max_stmts)
    stmts: List[Stmt] = []
    for _ in range(n):
        stmts.append(rand_stmt(rng, vars, depth))
        if isinstance(stmts[-1], Return):
            break
    return Block(stmts)


def make_random_program(seed: int = 0, depth: int = 4) -> Block:
    rng = random.Random(seed)
    vars = ["x", "y", "z"]
    return rand_block(rng, vars, depth, max_stmts=6)
