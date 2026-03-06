from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrCode(str, Enum):
    NAME = "NameError"
    TYPE = "TypeError"
    ZERODIV = "ZeroDivisionError"
    VALUE = "ValueError"
    TIMEOUT = "Timeout"


@dataclass(frozen=True)
class Err:
    code: ErrCode
    message: str = ""


@dataclass(frozen=True)
class Normal:
    pass


@dataclass(frozen=True)
class Returned:
    value: object


@dataclass(frozen=True)
class Failed:
    err: Err


Out = Normal | Returned | Failed
