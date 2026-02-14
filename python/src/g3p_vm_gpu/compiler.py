from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .ast import (
    Assign,
    BOp,
    Binary,
    Block,
    Call,
    Const,
    Expr,
    ForRange,
    IfExpr,
    IfStmt,
    Return,
    Stmt,
    UOp,
    Unary,
    Var,
    Val,
)


@dataclass(frozen=True)
class Instr:
    op: str
    a: int | None = None
    b: int | None = None


@dataclass(frozen=True)
class BytecodeProgram:
    consts: List[Val]
    code: List[Instr]
    n_locals: int
    var2idx: Dict[str, int]


class _Compiler:
    def __init__(self) -> None:
        self.consts: List[Val] = []
        self.code: List[Instr | tuple[str, str, int | None]] = []
        self.labels: Dict[str, int] = {}
        self.var2idx: Dict[str, int] = {}
        self._label_counter = 0
        self._tmp_counter = 0

    def _const(self, v: Val) -> int:
        self.consts.append(v)
        return len(self.consts) - 1

    def _emit(self, op: str, a: int | None = None, b: int | None = None) -> None:
        self.code.append(Instr(op=op, a=a, b=b))

    def _emit_jump(self, op: str, label: str) -> None:
        self.code.append(("JUMP_LABEL", op, label))

    def _label(self, name: str) -> None:
        self.labels[name] = len(self.code)

    def _new_label(self, prefix: str) -> str:
        name = f"{prefix}_{self._label_counter}"
        self._label_counter += 1
        return name

    def _local(self, name: str) -> int:
        idx = self.var2idx.get(name)
        if idx is not None:
            return idx
        idx = len(self.var2idx)
        self.var2idx[name] = idx
        return idx

    def _new_temp(self) -> str:
        name = f"\x00for_i_{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def compile_expr(self, e: Expr) -> None:
        if isinstance(e, Const):
            self._emit("PUSH_CONST", self._const(e.value))
            return

        if isinstance(e, Var):
            self._emit("LOAD", self._local(e.name))
            return

        if isinstance(e, Unary):
            self.compile_expr(e.e)
            if e.op == UOp.NEG:
                self._emit("NEG")
                return
            if e.op == UOp.NOT:
                self._emit("NOT")
                return
            raise ValueError(f"unknown unary op: {e.op}")

        if isinstance(e, Binary):
            if e.op == BOp.AND:
                false_l = self._new_label("and_false")
                end_l = self._new_label("and_end")
                self.compile_expr(e.a)
                self._emit_jump("JMP_IF_FALSE", false_l)
                self.compile_expr(e.b)
                self._emit("NOT")
                self._emit("NOT")
                self._emit_jump("JMP", end_l)
                self._label(false_l)
                self._emit("PUSH_CONST", self._const(False))
                self._label(end_l)
                return

            if e.op == BOp.OR:
                true_l = self._new_label("or_true")
                end_l = self._new_label("or_end")
                self.compile_expr(e.a)
                self._emit_jump("JMP_IF_TRUE", true_l)
                self.compile_expr(e.b)
                self._emit("NOT")
                self._emit("NOT")
                self._emit_jump("JMP", end_l)
                self._label(true_l)
                self._emit("PUSH_CONST", self._const(True))
                self._label(end_l)
                return

            self.compile_expr(e.a)
            self.compile_expr(e.b)
            opmap = {
                BOp.ADD: "ADD",
                BOp.SUB: "SUB",
                BOp.MUL: "MUL",
                BOp.DIV: "DIV",
                BOp.MOD: "MOD",
                BOp.LT: "LT",
                BOp.LE: "LE",
                BOp.GT: "GT",
                BOp.GE: "GE",
                BOp.EQ: "EQ",
                BOp.NE: "NE",
            }
            op = opmap.get(e.op)
            if op is None:
                raise ValueError(f"unsupported binary op in compiler: {e.op}")
            self._emit(op)
            return

        if isinstance(e, IfExpr):
            else_l = self._new_label("ifexpr_else")
            end_l = self._new_label("ifexpr_end")
            self.compile_expr(e.cond)
            self._emit_jump("JMP_IF_FALSE", else_l)
            self.compile_expr(e.then_e)
            self._emit_jump("JMP", end_l)
            self._label(else_l)
            self.compile_expr(e.else_e)
            self._label(end_l)
            return

        if isinstance(e, Call):
            for arg in e.args:
                self.compile_expr(arg)
            bid = {"abs": 0, "min": 1, "max": 2, "clip": 3}.get(e.name, -1)
            self._emit("CALL_BUILTIN", bid, len(e.args))
            return

        raise ValueError(f"unknown Expr node: {type(e).__name__}")

    def compile_stmt(self, s: Stmt) -> None:
        if isinstance(s, Assign):
            self.compile_expr(s.e)
            self._emit("STORE", self._local(s.name))
            return

        if isinstance(s, Return):
            self.compile_expr(s.e)
            self._emit("RETURN")
            return

        if isinstance(s, IfStmt):
            else_l = self._new_label("if_else")
            end_l = self._new_label("if_end")
            self.compile_expr(s.cond)
            self._emit_jump("JMP_IF_FALSE", else_l)
            self.compile_block(s.then_block)
            self._emit_jump("JMP", end_l)
            self._label(else_l)
            self.compile_block(s.else_block)
            self._label(end_l)
            return

        if isinstance(s, ForRange):
            idx_k = self._const(s.k)
            idx_0 = self._const(0)
            idx_1 = self._const(1)
            counter_name = self._new_temp()
            counter_i = self._local(counter_name)
            user_i = self._local(s.var)

            loop_l = self._new_label("for_loop")
            end_l = self._new_label("for_end")

            self._emit("PUSH_CONST", idx_0)
            self._emit("STORE", counter_i)

            self._label(loop_l)
            self._emit("LOAD", counter_i)
            self._emit("PUSH_CONST", idx_k)
            self._emit("LT")
            self._emit_jump("JMP_IF_FALSE", end_l)

            self._emit("LOAD", counter_i)
            self._emit("STORE", user_i)

            self.compile_block(s.body)

            self._emit("LOAD", counter_i)
            self._emit("PUSH_CONST", idx_1)
            self._emit("ADD")
            self._emit("STORE", counter_i)
            self._emit_jump("JMP", loop_l)
            self._label(end_l)
            return

        raise ValueError(f"unknown Stmt node: {type(s).__name__}")

    def compile_block(self, b: Block) -> None:
        for st in b.stmts:
            self.compile_stmt(st)

    def finalize(self) -> BytecodeProgram:
        out_code: List[Instr] = []
        for item in self.code:
            if isinstance(item, Instr):
                out_code.append(item)
                continue
            _tag, op, label = item
            addr = self.labels.get(label)
            if addr is None:
                raise ValueError(f"undefined label: {label}")
            out_code.append(Instr(op=op, a=addr))

        return BytecodeProgram(
            consts=self.consts,
            code=out_code,
            n_locals=len(self.var2idx),
            var2idx=dict(self.var2idx),
        )


def compile_program(p: Block) -> BytecodeProgram:
    c = _Compiler()
    c.compile_block(p)
    return c.finalize()
