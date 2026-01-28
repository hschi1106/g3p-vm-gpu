from __future__ import annotations

from typing import List, Union

from .errors import Err, ErrCode
from .ast import Val


def _is_num(v: Val) -> bool:
    # bool is NOT numeric in this subset
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _promote(a: Union[int, float], b: Union[int, float]) -> tuple[Union[int, float], Union[int, float], type]:
    if isinstance(a, float) or isinstance(b, float):
        return float(a), float(b), float
    return int(a), int(b), int


def builtin_call(name: str, args: List[Val]) -> Val | Err:
    if name == "abs":
        if len(args) != 1:
            return Err(ErrCode.TYPE, "abs expects 1 argument")
        x = args[0]
        if not _is_num(x):
            return Err(ErrCode.TYPE, "abs expects a numeric argument")
        return -x if x < 0 else x

    if name in ("min", "max"):
        if len(args) != 2:
            return Err(ErrCode.TYPE, f"{name} expects 2 arguments")
        a, b = args
        if not _is_num(a) or not _is_num(b):
            return Err(ErrCode.TYPE, f"{name} expects numeric arguments")
        a2, b2, _t = _promote(a, b)
        return a2 if (a2 <= b2 if name == "min" else a2 >= b2) else b2

    if name == "clip":
        if len(args) != 3:
            return Err(ErrCode.TYPE, "clip expects 3 arguments: clip(x, lo, hi)")
        x, lo, hi = args
        if not _is_num(x) or not _is_num(lo) or not _is_num(hi):
            return Err(ErrCode.TYPE, "clip expects numeric arguments")
        # promote all to float if any float appears
        any_float = any(isinstance(v, float) for v in (x, lo, hi))
        if any_float:
            x, lo, hi = float(x), float(lo), float(hi)
        else:
            x, lo, hi = int(x), int(lo), int(hi)
        if lo > hi:
            return Err(ErrCode.VALUE, "clip requires lo <= hi")
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    return Err(ErrCode.NAME, f"unknown builtin: {name}")
