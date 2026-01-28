from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union


# ---------- Values ----------
Val = Union[int, float, bool, None]


# ---------- Operators ----------
class UOp(str, Enum):
    NEG = "NEG"
    NOT = "NOT"


class BOp(str, Enum):
    # arithmetic
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    # comparisons
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    EQ = "EQ"
    NE = "NE"
    # boolean
    AND = "AND"
    OR = "OR"


# ---------- Expressions ----------
class Expr:
    pass


@dataclass(frozen=True)
class Const(Expr):
    value: Val


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Unary(Expr):
    op: UOp
    e: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: BOp
    a: Expr
    b: Expr


@dataclass(frozen=True)
class IfExpr(Expr):
    cond: Expr
    then_e: Expr
    else_e: Expr


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: Sequence[Expr]


# ---------- Statements / Blocks ----------
class Stmt:
    pass


@dataclass(frozen=True)
class Assign(Stmt):
    name: str
    e: Expr


@dataclass(frozen=True)
class IfStmt(Stmt):
    cond: Expr
    then_block: "Block"
    else_block: "Block"


@dataclass(frozen=True)
class ForRange(Stmt):
    var: str
    k: int  # must be non-negative int constant
    body: "Block"


@dataclass(frozen=True)
class Return(Stmt):
    e: Expr


@dataclass(frozen=True)
class Block:
    stmts: List[Stmt]
