from __future__ import annotations

import random
from enum import Enum
from typing import List

from ..core.ast import make_num_list, make_string_list
from .genome import Limits


class RType(str, Enum):
    NUM = "NUM"
    BOOL = "BOOL"
    NONE = "NONE"
    STRING = "STRING"
    NUM_LIST = "NUM_LIST"
    STRING_LIST = "STRING_LIST"


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
    if value_type == RType.STRING:
        if rng.random() < 0.5:
            return ("const", "".join(rng.choice("abcxyz") for _ in range(rng.randint(0, 6))))
        return ("const", "")
    if value_type == RType.NUM_LIST:
        return ("const", make_num_list(rng.randint(-3, 3) for _ in range(rng.randint(0, 6))))
    if value_type == RType.STRING_LIST:
        return ("const", make_string_list("".join(rng.choice("abcxyz") for _ in range(rng.randint(0, 3))) for _ in range(rng.randint(0, 6))))
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
                return ("call", builtin, [rand_expr(rng, depth - 1, rng.choice([RType.STRING, RType.NUM_LIST, RType.STRING_LIST]))])
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
        return ("call", "index", [rand_expr(rng, depth - 1, RType.NUM_LIST), ("const", rng.randint(-6, 6))])

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

    if value_type == RType.STRING:
        mode = rng.randint(0, 3)
        if mode == 0:
            return rand_const(rng, RType.STRING)
        if mode == 1:
            return ("call", "concat", [rand_expr(rng, depth - 1, RType.STRING), rand_expr(rng, depth - 1, RType.STRING)])
        if mode == 2:
            return (
                "call",
                "slice",
                [rand_expr(rng, depth - 1, RType.STRING), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
            )
        if mode == 3:
            return ("call", "reverse", [rand_expr(rng, depth - 1, RType.STRING)])
        return (
            "if_expr",
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_expr(rng, depth - 1, RType.STRING),
            rand_expr(rng, depth - 1, RType.STRING),
        )

    if value_type == RType.NUM_LIST:
        mode = rng.randint(0, 4)
        if mode == 0:
            return rand_const(rng, RType.NUM_LIST)
        if mode == 1:
            return ("call", "concat", [rand_expr(rng, depth - 1, RType.NUM_LIST), rand_expr(rng, depth - 1, RType.NUM_LIST)])
        if mode == 2:
            return (
                "call",
                "slice",
                [rand_expr(rng, depth - 1, RType.NUM_LIST), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
            )
        if mode == 3:
            return ("call", "append", [rand_expr(rng, depth - 1, RType.NUM_LIST), rand_expr(rng, depth - 1, RType.NUM)])
        if mode == 4:
            return ("call", "reverse", [rand_expr(rng, depth - 1, RType.NUM_LIST)])
        return (
            "if_expr",
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_expr(rng, depth - 1, RType.NUM_LIST),
            rand_expr(rng, depth - 1, RType.NUM_LIST),
        )

    if value_type == RType.STRING_LIST:
        mode = rng.randint(0, 4)
        if mode == 0:
            return rand_const(rng, RType.STRING_LIST)
        if mode == 1:
            return ("call", "concat", [rand_expr(rng, depth - 1, RType.STRING_LIST), rand_expr(rng, depth - 1, RType.STRING_LIST)])
        if mode == 2:
            return (
                "call",
                "slice",
                [rand_expr(rng, depth - 1, RType.STRING_LIST), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
            )
        if mode == 3:
            return ("call", "append", [rand_expr(rng, depth - 1, RType.STRING_LIST), rand_expr(rng, depth - 1, RType.STRING)])
        if mode == 4:
            return ("call", "reverse", [rand_expr(rng, depth - 1, RType.STRING_LIST)])
        return (
            "if_expr",
            rand_expr(rng, depth - 1, RType.BOOL),
            rand_expr(rng, depth - 1, RType.STRING_LIST),
            rand_expr(rng, depth - 1, RType.STRING_LIST),
        )

    return rand_const(rng, RType.NONE)


def rand_stmt(rng: random.Random, depth: int, limits: Limits) -> tuple:
    if depth <= 1:
        if rng.random() < 0.25:
            return ("return", rand_expr(rng, 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE, RType.STRING, RType.NUM_LIST, RType.STRING_LIST])))
        return (
            "assign",
            rng.choice(["x", "y", "z", "w"]),
            rand_expr(rng, 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE, RType.STRING, RType.NUM_LIST, RType.STRING_LIST])),
        )

    mode = rng.randint(0, 3)
    if mode == 0:
        return (
            "assign",
            rng.choice(["x", "y", "z", "w", "u", "v"]),
            rand_expr(rng, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE, RType.STRING, RType.NUM_LIST, RType.STRING_LIST])),
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
    return ("return", rand_expr(rng, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE, RType.STRING, RType.NUM_LIST, RType.STRING_LIST])))


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
