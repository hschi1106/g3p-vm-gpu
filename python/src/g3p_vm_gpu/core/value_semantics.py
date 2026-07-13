from __future__ import annotations

from .ast import Char, FloatList, IntList, StringList, Val
from .errors import Err, ErrCode


def is_num(v: Val) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def promote_numeric(a: Val, b: Val) -> tuple[int | float, int | float] | Err:
    if not is_num(a) or not is_num(b):
        return Err(ErrCode.TYPE, "numeric operands required")
    if type(a) is not type(b):
        return Err(ErrCode.TYPE, "numeric operands require exact matching runtime types")
    if isinstance(a, float) or isinstance(b, float):
        return float(a), float(b)
    return int(a), int(b)


def compare_values(op: str, a: Val, b: Val) -> bool | Err:
    if op in {"EQ", "NE"}:
        if type(a) is not type(b):
            return Err(ErrCode.TYPE, "equality requires exact matching runtime types")
        if isinstance(a, (bool, int, float, Char, str, IntList, FloatList, StringList)):
            return (a == b) if op == "EQ" else (a != b)
        return Err(ErrCode.TYPE, "unsupported equality operand types")

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
        return Err(ErrCode.TYPE, f"unknown comparison op: {op}")

    if isinstance(a, bool) and isinstance(b, bool):
        return Err(ErrCode.TYPE, "ordering comparison on bool not supported")

    if isinstance(a, (Char, str, IntList, FloatList, StringList)) and isinstance(b, (Char, str, IntList, FloatList, StringList)):
        return Err(ErrCode.TYPE, "ordering comparison on char/string/list not supported")

    return Err(ErrCode.TYPE, "unsupported comparison operand types")
