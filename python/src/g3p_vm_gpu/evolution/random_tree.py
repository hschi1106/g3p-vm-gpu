from __future__ import annotations

import random
from enum import Enum
from typing import List

from .genome import Limits


class RType(str, Enum):
    NUM = "NUM"
    BOOL = "BOOL"
    NONE = "NONE"
    CONTAINER = "CONTAINER"


_NUM_BUILTINS = ("abs", "min", "max", "clip", "len")
_BIN_NUM = ("add", "sub", "mul", "div", "mod")
_BIN_CMP = ("lt", "le", "gt", "ge", "eq", "ne")
_BIN_BOOL = ("and", "or")


def rand_const(rng: random.Random, value_type: RType):
    if value_type == RType.NUM:
        if rng.random() < 0.5:
            return ("const", rng.randint(-8, 8))
        return ("const", round(rng.uniform(-8.0, 8.0), 3))
    if value_type == RType.BOOL:
        return ("const", rng.choice([True, False]))
    if value_type == RType.CONTAINER:
        if rng.random() < 0.5:
            return ("const", "".join(rng.choice("abcxyz") for _ in range(rng.randint(0, 6))))
        return ("const", [rng.randint(-3, 3) for _ in range(rng.randint(0, 6))])
    return ("const", None)


def rand_expr(rng: random.Random, depth: int, value_type: RType) -> tuple:
    if depth <= 1:
        return rand_const(rng, value_type)

    if value_type == RType.NUM:
        mode = rng.randint(0, 5)
        if mode == 0:
            return rand_const(rng, RType.NUM)
        if mode == 1:
            return ("neg", rand_expr(rng, depth - 1, RType.NUM))
        if mode == 2:
            op = rng.choice(_BIN_NUM)
            return (op, rand_expr(rng, depth - 1, RType.NUM), rand_expr(rng, depth - 1, RType.NUM))
        if mode == 3:
            builtin = rng.choice(_NUM_BUILTINS)
            if builtin == "abs":
                return ("call", builtin, [rand_expr(rng, depth - 1, RType.NUM)])
            if builtin in ("min", "max"):
                return (
                    "call",
                    builtin,
                    [rand_expr(rng, depth - 1, RType.NUM), rand_expr(rng, depth - 1, RType.NUM)],
                )
            if builtin == "len":
                return ("call", builtin, [rand_expr(rng, depth - 1, RType.CONTAINER)])
            return (
                "call",
                builtin,
                [
                    rand_expr(rng, depth - 1, RType.NUM),
                    rand_expr(rng, depth - 1, RType.NUM),
                    rand_expr(rng, depth - 1, RType.NUM),
                ],
            )
        if mode == 4:
            return (
                "if_expr",
                rand_expr(rng, depth - 1, RType.BOOL),
                rand_expr(rng, depth - 1, RType.NUM),
                rand_expr(rng, depth - 1, RType.NUM),
            )
        return ("call", "index", [rand_expr(rng, depth - 1, RType.CONTAINER), ("const", rng.randint(-6, 6))])

    if value_type == RType.BOOL:
        mode = rng.randint(0, 4)
        if mode == 0:
            return rand_const(rng, RType.BOOL)
        if mode == 1:
            return ("not", rand_expr(rng, depth - 1, RType.BOOL))
        if mode == 2:
            op = rng.choice(_BIN_CMP)
            if op in ("eq", "ne") and rng.random() < 0.35:
                return (op, rand_const(rng, RType.NONE), rand_const(rng, RType.NONE))
            return (op, rand_expr(rng, depth - 1, RType.NUM), rand_expr(rng, depth - 1, RType.NUM))
        if mode == 3:
            op = rng.choice(_BIN_BOOL)
            return (op, rand_expr(rng, depth - 1, RType.BOOL), rand_expr(rng, depth - 1, RType.BOOL))
        return (
            "if_expr",
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_expr(rng, depth - 1, RType.BOOL),
        )

    if value_type == RType.CONTAINER:
        mode = rng.randint(0, 3)
        if mode == 0:
            return rand_const(rng, RType.CONTAINER)
        if mode == 1:
            return ("call", "concat", [rand_expr(rng, depth - 1, RType.CONTAINER), rand_expr(rng, depth - 1, RType.CONTAINER)])
        if mode == 2:
            return (
                "call",
                "slice",
                [rand_expr(rng, depth - 1, RType.CONTAINER), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
            )
        return (
            "if_expr",
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_expr(rng, depth - 1, RType.CONTAINER),
            rand_expr(rng, depth - 1, RType.CONTAINER),
        )

    return rand_const(rng, RType.NONE)


def rand_stmt(rng: random.Random, depth: int, limits: Limits) -> tuple:
    if depth <= 1:
        if rng.random() < 0.25:
            return ("return", rand_expr(rng, 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))
        return (
            "assign",
            rng.choice(["x", "y", "z", "w"]),
            rand_expr(rng, 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])),
        )

    mode = rng.randint(0, 3)
    if mode == 0:
        return (
            "assign",
            rng.choice(["x", "y", "z", "w", "u", "v"]),
            rand_expr(rng, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])),
        )
    if mode == 1:
        return (
            "if",
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_block(rng, depth - 1, limits, force_return=False),
            rand_block(rng, depth - 1, limits, force_return=False),
        )
    if mode == 2:
        return (
            "for",
            rng.choice(["i", "j", "k"]),
            ("const", rng.randint(0, max(0, limits.max_for_k))),
            rand_block(rng, depth - 1, limits, force_return=False),
        )
    return ("return", rand_expr(rng, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))


def rand_block(rng: random.Random, depth: int, limits: Limits, force_return: bool) -> List[tuple]:
    count = rng.randint(1, max(1, limits.max_stmts_per_block))
    out: List[tuple] = []
    for _ in range(count):
        stmt = rand_stmt(rng, depth, limits)
        out.append(stmt)
        if stmt[0] == "return":
            break
    if force_return and not any(stmt[0] == "return" for stmt in out):
        out.append(("return", rand_expr(rng, max(1, depth - 1), RType.NUM)))
    return out[: limits.max_stmts_per_block]
