from __future__ import annotations

from typing import List, Union

from ..core.errors import Err, ErrCode
from ..core.ast import NumList, StringList, Val, make_num_list, make_string_list


def _is_num(v: Val) -> bool:
    # bool is NOT numeric in this subset
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _promote(a: Union[int, float], b: Union[int, float]) -> tuple[Union[int, float], Union[int, float], type]:
    if isinstance(a, float) or isinstance(b, float):
        return float(a), float(b), float
    return int(a), int(b), int


def _is_num_list(v: Val) -> bool:
    return isinstance(v, NumList)


def _is_string_list(v: Val) -> bool:
    return isinstance(v, StringList)


def _is_sequence(v: Val) -> bool:
    return isinstance(v, (str, NumList, StringList))


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
        if not _is_sequence(x):
            return Err(ErrCode.TYPE, "len expects string/typed-list argument")
        return len(x)

    if name == "concat":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "concat expects 2 arguments")
        a, b = args
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if _is_num_list(a) and _is_num_list(b):
            return make_num_list(a.items + b.items)
        if _is_string_list(a) and _is_string_list(b):
            return make_string_list(a.items + b.items)
        return Err(ErrCode.TYPE, "concat expects matching string/typed-list arguments")

    if name == "slice":
        if len(args) != 3:
            return Err(ErrCode.TYPE, "slice expects 3 arguments: slice(x, lo, hi)")
        x, lo, hi = args
        if not _is_sequence(x):
            return Err(ErrCode.TYPE, "slice expects string/typed-list as first argument")
        if not (isinstance(lo, int) and not isinstance(lo, bool) and isinstance(hi, int) and not isinstance(hi, bool)):
            return Err(ErrCode.TYPE, "slice expects integer lo/hi")
        if isinstance(x, str):
            return x[lo:hi]
        if _is_num_list(x):
            return make_num_list(x.items[lo:hi])
        return make_string_list(x.items[lo:hi])

    if name == "index":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "index expects 2 arguments: index(x, i)")
        x, i = args
        if not _is_sequence(x):
            return Err(ErrCode.TYPE, "index expects string/typed-list as first argument")
        if not (isinstance(i, int) and not isinstance(i, bool)):
            return Err(ErrCode.TYPE, "index expects integer index")
        n = len(x)
        j = i + n if i < 0 else i
        if j < 0 or j >= n:
            return Err(ErrCode.VALUE, "index out of range")
        if isinstance(x, str):
            return x[j]
        return x.items[j]

    if name == "append":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "append expects 2 arguments")
        xs, value = args
        if _is_num_list(xs) and _is_num(value):
            return make_num_list(xs.items + (value,))
        if _is_string_list(xs) and isinstance(value, str):
            return make_string_list(xs.items + (value,))
        return Err(ErrCode.TYPE, "append expects (num_list,num) or (string_list,string)")

    if name == "reverse":
        if len(args) != 1:
            return Err(ErrCode.TYPE, "reverse expects 1 argument")
        x = args[0]
        if isinstance(x, str):
            return x[::-1]
        if _is_num_list(x):
            return make_num_list(tuple(reversed(x.items)))
        if _is_string_list(x):
            return make_string_list(tuple(reversed(x.items)))
        return Err(ErrCode.TYPE, "reverse expects string/typed-list argument")

    if name == "find":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "find expects 2 arguments")
        haystack, needle = args
        if not isinstance(haystack, str) or not isinstance(needle, str):
            return Err(ErrCode.TYPE, "find expects (string,string)")
        return haystack.find(needle)

    if name == "contains":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "contains expects 2 arguments")
        haystack, needle = args
        if not isinstance(haystack, str) or not isinstance(needle, str):
            return Err(ErrCode.TYPE, "contains expects (string,string)")
        return needle in haystack

    if name == "is_int":
        if len(args) != 1:
            return Err(ErrCode.TYPE, "is_int expects 1 argument")
        x = args[0]
        return isinstance(x, int) and not isinstance(x, bool)

    return Err(ErrCode.NAME, f"unknown builtin: {name}")
