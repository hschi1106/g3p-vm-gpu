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

    if name == "len":
        if len(args) != 1:
            return Err(ErrCode.TYPE, "len expects 1 argument")
        x = args[0]
        if not isinstance(x, (str, list)):
            return Err(ErrCode.TYPE, "len expects string/list argument")
        return len(x)

    if name == "concat":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "concat expects 2 arguments")
        a, b = args
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        return Err(ErrCode.TYPE, "concat expects (string,string) or (list,list)")

    if name == "slice":
        if len(args) != 3:
            return Err(ErrCode.TYPE, "slice expects 3 arguments: slice(x, lo, hi)")
        x, lo, hi = args
        if not isinstance(x, (str, list)):
            return Err(ErrCode.TYPE, "slice expects string/list as first argument")
        if not (isinstance(lo, int) and not isinstance(lo, bool) and isinstance(hi, int) and not isinstance(hi, bool)):
            return Err(ErrCode.TYPE, "slice expects integer lo/hi")
        return x[lo:hi]

    if name == "index":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "index expects 2 arguments: index(x, i)")
        x, i = args
        if not isinstance(x, (str, list)):
            return Err(ErrCode.TYPE, "index expects string/list as first argument")
        if not (isinstance(i, int) and not isinstance(i, bool)):
            return Err(ErrCode.TYPE, "index expects integer index")
        n = len(x)
        j = i + n if i < 0 else i
        if j < 0 or j >= n:
            return Err(ErrCode.VALUE, "index out of range")
        return x[j]

    return Err(ErrCode.NAME, f"unknown builtin: {name}")
