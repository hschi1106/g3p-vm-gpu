from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .ast import AstProgram, NodeKind, Val, prefix_subtree_end, validate_prefix_program


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
    def __init__(self, p: AstProgram) -> None:
        self.p = p
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

    def _compile_expr(self, idx: int) -> int:
        n = self.p.nodes[idx]
        k = n.kind

        if k == NodeKind.CONST:
            self._emit("PUSH_CONST", self._const(self.p.consts[n.i0]))
            return idx + 1

        if k == NodeKind.VAR:
            self._emit("LOAD", self._local(self.p.names[n.i0]))
            return idx + 1

        if k == NodeKind.NEG:
            j = self._compile_expr(idx + 1)
            self._emit("NEG")
            return j

        if k == NodeKind.NOT:
            j = self._compile_expr(idx + 1)
            self._emit("NOT")
            return j

        if k == NodeKind.AND:
            false_l = self._new_label("and_false")
            end_l = self._new_label("and_end")
            j = self._compile_expr(idx + 1)
            self._emit_jump("JMP_IF_FALSE", false_l)
            h = self._compile_expr(j)
            self._emit("NOT")
            self._emit("NOT")
            self._emit_jump("JMP", end_l)
            self._label(false_l)
            self._emit("PUSH_CONST", self._const(False))
            self._label(end_l)
            return h

        if k == NodeKind.OR:
            true_l = self._new_label("or_true")
            end_l = self._new_label("or_end")
            j = self._compile_expr(idx + 1)
            self._emit_jump("JMP_IF_TRUE", true_l)
            h = self._compile_expr(j)
            self._emit("NOT")
            self._emit("NOT")
            self._emit_jump("JMP", end_l)
            self._label(true_l)
            self._emit("PUSH_CONST", self._const(True))
            self._label(end_l)
            return h

        if k in {NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV, NodeKind.MOD, NodeKind.LT, NodeKind.LE, NodeKind.GT, NodeKind.GE, NodeKind.EQ, NodeKind.NE}:
            j = self._compile_expr(idx + 1)
            h = self._compile_expr(j)
            op = {
                NodeKind.ADD: "ADD",
                NodeKind.SUB: "SUB",
                NodeKind.MUL: "MUL",
                NodeKind.DIV: "DIV",
                NodeKind.MOD: "MOD",
                NodeKind.LT: "LT",
                NodeKind.LE: "LE",
                NodeKind.GT: "GT",
                NodeKind.GE: "GE",
                NodeKind.EQ: "EQ",
                NodeKind.NE: "NE",
            }[k]
            self._emit(op)
            return h

        if k == NodeKind.IF_EXPR:
            else_l = self._new_label("ifexpr_else")
            end_l = self._new_label("ifexpr_end")
            j = self._compile_expr(idx + 1)
            self._emit_jump("JMP_IF_FALSE", else_l)
            h = self._compile_expr(j)
            self._emit_jump("JMP", end_l)
            self._label(else_l)
            t = self._compile_expr(h)
            self._label(end_l)
            return t

        if k in {NodeKind.CALL_ABS, NodeKind.CALL_MIN, NodeKind.CALL_MAX, NodeKind.CALL_CLIP}:
            argc = {NodeKind.CALL_ABS: 1, NodeKind.CALL_MIN: 2, NodeKind.CALL_MAX: 2, NodeKind.CALL_CLIP: 3}[k]
            cur = idx + 1
            for _ in range(argc):
                cur = self._compile_expr(cur)
            bid = {NodeKind.CALL_ABS: 0, NodeKind.CALL_MIN: 1, NodeKind.CALL_MAX: 2, NodeKind.CALL_CLIP: 3}[k]
            self._emit("CALL_BUILTIN", bid, argc)
            return cur

        raise ValueError(f"expected Expr at index {idx}, got {k}")

    def _compile_stmt(self, idx: int) -> int:
        n = self.p.nodes[idx]
        k = n.kind

        if k == NodeKind.ASSIGN:
            j = self._compile_expr(idx + 1)
            self._emit("STORE", self._local(self.p.names[n.i0]))
            return j

        if k == NodeKind.RETURN:
            j = self._compile_expr(idx + 1)
            self._emit("RETURN")
            return j

        if k == NodeKind.IF_STMT:
            else_l = self._new_label("if_else")
            end_l = self._new_label("if_end")
            j = self._compile_expr(idx + 1)
            self._emit_jump("JMP_IF_FALSE", else_l)
            h = self._compile_block(j)
            self._emit_jump("JMP", end_l)
            self._label(else_l)
            t = self._compile_block(h)
            self._label(end_l)
            return t

        if k == NodeKind.FOR_RANGE:
            kval = n.i1
            if not isinstance(kval, int) or isinstance(kval, bool) or kval < 0:
                self._emit("PUSH_CONST", self._const(True))
                self._emit("NEG")
                return prefix_subtree_end(self.p.nodes, idx)

            idx_k = self._const(kval)
            idx_0 = self._const(0)
            idx_1 = self._const(1)
            counter_i = self._local(self._new_temp())
            user_i = self._local(self.p.names[n.i0])
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
            j = self._compile_block(idx + 1)
            self._emit("LOAD", counter_i)
            self._emit("PUSH_CONST", idx_1)
            self._emit("ADD")
            self._emit("STORE", counter_i)
            self._emit_jump("JMP", loop_l)
            self._label(end_l)
            return j

        raise ValueError(f"expected Stmt at index {idx}, got {k}")

    def _compile_block(self, idx: int) -> int:
        n = self.p.nodes[idx]
        if n.kind == NodeKind.BLOCK_NIL:
            return idx + 1
        if n.kind != NodeKind.BLOCK_CONS:
            raise ValueError(f"expected Block at index {idx}, got {n.kind}")
        j = self._compile_stmt(idx + 1)
        return self._compile_block(j)

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


def compile_program(p: AstProgram) -> BytecodeProgram:
    validate_prefix_program(p)
    c = _Compiler(p)
    end = c._compile_block(1)
    if end != len(p.nodes):
        raise ValueError("invalid trailing tokens")
    return c.finalize()
