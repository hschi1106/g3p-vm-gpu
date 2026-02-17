from __future__ import annotations

import random
from typing import List

from .ast import AstProgram, build_program


_BUILTINS = ("abs", "min", "max", "clip")
_BIN = ("add", "sub", "mul", "div", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or")


def _rand_const(rng: random.Random) -> tuple:
    t = rng.choice(["int", "float", "bool", "none"])
    if t == "int":
        return ("const", rng.randint(-5, 5))
    if t == "float":
        return ("const", round(rng.uniform(-5.0, 5.0), 3))
    if t == "bool":
        return ("const", rng.choice([True, False]))
    return ("const", None)


def _rand_var(rng: random.Random, vars_: List[str]) -> tuple:
    if vars_ and rng.random() < 0.5:
        return ("var", rng.choice(vars_))
    return _rand_const(rng)


def _rand_expr(rng: random.Random, vars_: List[str], depth: int) -> tuple:
    if depth <= 0:
        return _rand_var(rng, vars_)

    mode = rng.randint(0, 6)
    if mode == 0:
        return _rand_const(rng)
    if mode == 1:
        return _rand_var(rng, vars_)
    if mode == 2:
        return (rng.choice(["neg", "not"]), _rand_expr(rng, vars_, depth - 1))
    if mode == 3:
        op = rng.choice(_BIN)
        return (op, _rand_expr(rng, vars_, depth - 1), _rand_expr(rng, vars_, depth - 1))
    if mode == 4:
        return (
            "if_expr",
            _rand_expr(rng, vars_, depth - 1),
            _rand_expr(rng, vars_, depth - 1),
            _rand_expr(rng, vars_, depth - 1),
        )
    if mode == 5:
        b = rng.choice(_BUILTINS)
        if b == "abs":
            args = [_rand_expr(rng, vars_, depth - 1)]
        elif b in ("min", "max"):
            args = [_rand_expr(rng, vars_, depth - 1), _rand_expr(rng, vars_, depth - 1)]
        else:
            args = [_rand_expr(rng, vars_, depth - 1), _rand_expr(rng, vars_, depth - 1), _rand_expr(rng, vars_, depth - 1)]
        return ("call", b, args)
    return ("add", _rand_expr(rng, vars_, depth - 1), _rand_expr(rng, vars_, depth - 1))


def _rand_stmt(rng: random.Random, vars_: List[str], depth: int) -> tuple:
    if depth <= 0:
        return ("assign", rng.choice(vars_) if vars_ else "x", _rand_expr(rng, vars_, 0))

    mode = rng.randint(0, 3)
    if mode == 0:
        return ("assign", rng.choice(vars_) if vars_ else "x", _rand_expr(rng, vars_, depth - 1))
    if mode == 1:
        return (
            "if",
            _rand_expr(rng, vars_, depth - 1),
            _rand_block(rng, vars_, depth - 1, max_stmts=2),
            _rand_block(rng, vars_, depth - 1, max_stmts=2),
        )
    if mode == 2:
        loop_var = rng.choice(["i", "j", "k"])
        return ("for", loop_var, rng.randint(0, 4), _rand_block(rng, vars_ + [loop_var], depth - 1, max_stmts=2))
    return ("return", _rand_expr(rng, vars_, depth - 1))


def _rand_block(rng: random.Random, vars_: List[str], depth: int, max_stmts: int = 4) -> List[tuple]:
    n = rng.randint(1, max_stmts)
    stmts: List[tuple] = []
    for _ in range(n):
        st = _rand_stmt(rng, vars_, depth)
        stmts.append(st)
        if st[0] == "return":
            break
    return stmts


def make_random_program(seed: int = 0, depth: int = 4) -> AstProgram:
    rng = random.Random(seed)
    vars_ = ["x", "y", "z"]
    return build_program(_rand_block(rng, vars_, depth, max_stmts=6))
