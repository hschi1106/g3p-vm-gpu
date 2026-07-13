from __future__ import annotations

import math
from typing import List, Union

from ..core.errors import Err, ErrCode
from ..core.ast import (
    Char,
    FloatList,
    IntList,
    StringList,
    Val,
    make_char,
    make_float_list,
    make_int_list,
    make_string_list,
)

BUILTIN_ID_BY_NAME = {
    "abs": 0,
    "min": 1,
    "max": 2,
    "clip": 3,
    "len": 4,
    "concat": 5,
    "slice": 6,
    "index": 7,
    "append": 8,
    "reverse": 9,
    "find": 10,
    "contains": 11,
    "is_int": 12,
    "idiv0": 13,
    "imod0": 14,
    "prepend": 15,
    "char_to_string": 16,
    "string_to_char": 17,
    "ord": 18,
    "chr": 19,
    "is_letter": 20,
    "is_digit": 21,
    "is_space": 22,
    "is_vowel": 23,
    "to_lower": 24,
    "to_upper": 25,
    "to_string": 26,
    "singleton": 27,
}

BUILTIN_NAME_BY_ID = {idx: name for name, idx in BUILTIN_ID_BY_NAME.items()}


def _is_num(v: Val) -> bool:
    # bool is NOT numeric in this subset
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _format_float_current(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    out = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if out == "-0" else out


def _promote(a: Union[int, float], b: Union[int, float]) -> tuple[Union[int, float], Union[int, float], type]:
    if type(a) is not type(b):
        raise TypeError("numeric arguments require exact matching runtime types")
    if isinstance(a, float) or isinstance(b, float):
        return float(a), float(b), float
    return int(a), int(b), int


def _is_int_list(v: Val) -> bool:
    return isinstance(v, IntList)


def _is_float_list(v: Val) -> bool:
    return isinstance(v, FloatList)


def _is_string_list(v: Val) -> bool:
    return isinstance(v, StringList)


def _is_sequence(v: Val) -> bool:
    return isinstance(v, (str, IntList, FloatList, StringList))


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
        try:
            a2, b2, _t = _promote(a, b)
        except TypeError:
            return Err(ErrCode.TYPE, f"{name} expects matching numeric argument types")
        return a2 if (a2 <= b2 if name == "min" else a2 >= b2) else b2

    if name == "clip":
        if len(args) != 3:
            return Err(ErrCode.TYPE, "clip expects 3 arguments: clip(x, lo, hi)")
        x, lo, hi = args
        if not _is_num(x) or not _is_num(lo) or not _is_num(hi):
            return Err(ErrCode.TYPE, "clip expects numeric arguments")
        if type(x) is not type(lo) or type(x) is not type(hi):
            return Err(ErrCode.TYPE, "clip expects matching numeric argument types")
        return min(max(x, lo), hi)

    if name == "idiv0":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "idiv0 expects 2 arguments")
        a, b = args
        if not (isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int) and not isinstance(b, bool)):
            return Err(ErrCode.TYPE, "idiv0 expects integer arguments")
        if b == 0:
            return 0
        q = abs(a) // abs(b)
        return -q if (a < 0) != (b < 0) else q

    if name == "imod0":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "imod0 expects 2 arguments")
        a, b = args
        if not (isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int) and not isinstance(b, bool)):
            return Err(ErrCode.TYPE, "imod0 expects integer arguments")
        return 0 if b == 0 else a % b

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
        if _is_int_list(a) and _is_int_list(b):
            return make_int_list(a.items + b.items)
        if _is_float_list(a) and _is_float_list(b):
            return make_float_list(a.items + b.items)
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
        if _is_int_list(x):
            return make_int_list(x.items[lo:hi])
        if _is_float_list(x):
            return make_float_list(x.items[lo:hi])
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
            return make_char(x[j])
        return x.items[j]

    if name == "append":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "append expects 2 arguments")
        xs, value = args
        if _is_int_list(xs) and isinstance(value, int) and not isinstance(value, bool):
            return make_int_list(xs.items + (value,))
        if _is_float_list(xs) and isinstance(value, float):
            return make_float_list(xs.items + (value,))
        if _is_string_list(xs) and isinstance(value, str):
            return make_string_list(xs.items + (value,))
        return Err(ErrCode.TYPE, "append expects (int_list,int), (float_list,float), or (string_list,string)")

    if name == "prepend":
        if len(args) != 2:
            return Err(ErrCode.TYPE, "prepend expects 2 arguments")
        xs, value = args
        if _is_int_list(xs) and isinstance(value, int) and not isinstance(value, bool):
            return make_int_list((value,) + xs.items)
        if _is_float_list(xs) and isinstance(value, float):
            return make_float_list((value,) + xs.items)
        if _is_string_list(xs) and isinstance(value, str):
            return make_string_list((value,) + xs.items)
        return Err(ErrCode.TYPE, "prepend expects (int_list,int), (float_list,float), or (string_list,string)")

    if name == "reverse":
        if len(args) != 1:
            return Err(ErrCode.TYPE, "reverse expects 1 argument")
        x = args[0]
        if isinstance(x, str):
            return x[::-1]
        if _is_int_list(x):
            return make_int_list(tuple(reversed(x.items)))
        if _is_float_list(x):
            return make_float_list(tuple(reversed(x.items)))
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

    if name == "char_to_string":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "char_to_string expects char")
        return args[0].value

    if name == "string_to_char":
        if len(args) != 1 or not isinstance(args[0], str):
            return Err(ErrCode.TYPE, "string_to_char expects string")
        if len(args[0]) != 1:
            return Err(ErrCode.VALUE, "string_to_char expects length-1 string")
        return make_char(args[0])

    if name == "ord":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "ord expects char")
        return ord(args[0].value)

    if name == "chr":
        if len(args) != 1 or not (isinstance(args[0], int) and not isinstance(args[0], bool)):
            return Err(ErrCode.TYPE, "chr expects int")
        try:
            return make_char(chr(args[0]))
        except (ValueError, OverflowError):
            return Err(ErrCode.VALUE, "invalid character code point")

    if name == "is_letter":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "is_letter expects char")
        return args[0].value.isalpha()

    if name == "is_digit":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "is_digit expects char")
        return args[0].value in "0123456789"

    if name == "is_space":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "is_space expects char")
        return args[0].value.isspace()

    if name == "is_vowel":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "is_vowel expects char")
        return args[0].value in "aeiouAEIOU"

    if name == "to_lower":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "to_lower expects char")
        return make_char(args[0].value.lower())

    if name == "to_upper":
        if len(args) != 1 or not isinstance(args[0], Char):
            return Err(ErrCode.TYPE, "to_upper expects char")
        return make_char(args[0].value.upper())

    if name == "to_string":
        if len(args) != 1 or not _is_num(args[0]):
            return Err(ErrCode.TYPE, "to_string expects int or float")
        if isinstance(args[0], float):
            return _format_float_current(args[0])
        return str(args[0])

    if name == "singleton":
        if len(args) != 1:
            return Err(ErrCode.TYPE, "singleton expects 1 argument")
        x = args[0]
        if isinstance(x, Char):
            return x.value
        if isinstance(x, int) and not isinstance(x, bool):
            return make_int_list((x,))
        if isinstance(x, float):
            return make_float_list((x,))
        if isinstance(x, str):
            return make_string_list((x,))
        return Err(ErrCode.TYPE, "singleton expects int, float, string, or char")

    return Err(ErrCode.NAME, f"unknown builtin: {name}")
