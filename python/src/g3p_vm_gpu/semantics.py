from __future__ import annotations

from .ast import Val
from .errors import Err, ErrCode


def is_num(v: Val) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def promote_numeric(a: Val, b: Val) -> tuple[int | float, int | float] | Err:
    if not is_num(a) or not is_num(b):
        return Err(ErrCode.TYPE, "numeric operands required")
    if isinstance(a, float) or isinstance(b, float):
        return float(a), float(b)
    return int(a), int(b)


def compare_values(op: str, a: Val, b: Val) -> bool | Err:
    if is_num(a) and is_num(b):
        prom = promote_numeric(a, b)
        if isinstance(prom, Err):
            return prom
        a2, b2 = prom
        if op == "LT":
            return a2 < b2
        if op == "LE":
            return a2 <= b2
        if op == "GT":
            return a2 > b2
        if op == "GE":
            return a2 >= b2
        if op == "EQ":
            return a2 == b2
        if op == "NE":
            return a2 != b2
        return Err(ErrCode.TYPE, f"unknown comparison op: {op}")

    if isinstance(a, bool) and isinstance(b, bool):
        if op == "EQ":
            return a == b
        if op == "NE":
            return a != b
        return Err(ErrCode.TYPE, "ordering comparison on bool not supported")

    if a is None or b is None:
        if op == "EQ":
            return a is b
        if op == "NE":
            return a is not b
        return Err(ErrCode.TYPE, "ordering comparison on None not supported")

    return Err(ErrCode.TYPE, "unsupported comparison operand types")
