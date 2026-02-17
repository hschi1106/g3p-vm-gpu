from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Union


Val = Union[int, float, bool, None]


class UOp(str, Enum):
    NEG = "NEG"
    NOT = "NOT"


class BOp(str, Enum):
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    EQ = "EQ"
    NE = "NE"
    AND = "AND"
    OR = "OR"


class NodeKind(str, Enum):
    PROGRAM = "PROGRAM"
    BLOCK_NIL = "BLOCK_NIL"
    BLOCK_CONS = "BLOCK_CONS"
    ASSIGN = "ASSIGN"
    IF_STMT = "IF_STMT"
    FOR_RANGE = "FOR_RANGE"
    RETURN = "RETURN"
    CONST = "CONST"
    VAR = "VAR"
    NEG = "NEG"
    NOT = "NOT"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    EQ = "EQ"
    NE = "NE"
    AND = "AND"
    OR = "OR"
    IF_EXPR = "IF_EXPR"
    CALL_ABS = "CALL_ABS"
    CALL_MIN = "CALL_MIN"
    CALL_MAX = "CALL_MAX"
    CALL_CLIP = "CALL_CLIP"


NODE_ARITY: Dict[NodeKind, int] = {
    NodeKind.PROGRAM: 1,
    NodeKind.BLOCK_NIL: 0,
    NodeKind.BLOCK_CONS: 2,
    NodeKind.ASSIGN: 1,
    NodeKind.IF_STMT: 3,
    NodeKind.FOR_RANGE: 1,
    NodeKind.RETURN: 1,
    NodeKind.CONST: 0,
    NodeKind.VAR: 0,
    NodeKind.NEG: 1,
    NodeKind.NOT: 1,
    NodeKind.ADD: 2,
    NodeKind.SUB: 2,
    NodeKind.MUL: 2,
    NodeKind.DIV: 2,
    NodeKind.MOD: 2,
    NodeKind.LT: 2,
    NodeKind.LE: 2,
    NodeKind.GT: 2,
    NodeKind.GE: 2,
    NodeKind.EQ: 2,
    NodeKind.NE: 2,
    NodeKind.AND: 2,
    NodeKind.OR: 2,
    NodeKind.IF_EXPR: 3,
    NodeKind.CALL_ABS: 1,
    NodeKind.CALL_MIN: 2,
    NodeKind.CALL_MAX: 2,
    NodeKind.CALL_CLIP: 3,
}


EXPR_KINDS = {
    NodeKind.CONST,
    NodeKind.VAR,
    NodeKind.NEG,
    NodeKind.NOT,
    NodeKind.ADD,
    NodeKind.SUB,
    NodeKind.MUL,
    NodeKind.DIV,
    NodeKind.MOD,
    NodeKind.LT,
    NodeKind.LE,
    NodeKind.GT,
    NodeKind.GE,
    NodeKind.EQ,
    NodeKind.NE,
    NodeKind.AND,
    NodeKind.OR,
    NodeKind.IF_EXPR,
    NodeKind.CALL_ABS,
    NodeKind.CALL_MIN,
    NodeKind.CALL_MAX,
    NodeKind.CALL_CLIP,
}


STMT_KINDS = {
    NodeKind.ASSIGN,
    NodeKind.IF_STMT,
    NodeKind.FOR_RANGE,
    NodeKind.RETURN,
}


@dataclass(frozen=True)
class AstNode:
    kind: NodeKind
    i0: int = 0
    i1: int = 0


@dataclass(frozen=True)
class AstProgram:
    nodes: Sequence[AstNode]
    names: Sequence[str]
    consts: Sequence[Val]
    version: str = "ast-prefix-v1"


def prefix_subtree_end(nodes: Sequence[AstNode], start: int) -> int:
    if start >= len(nodes):
        raise ValueError("invalid prefix AST: out-of-range subtree start")
    arity = NODE_ARITY[nodes[start].kind]
    idx = start + 1
    for _ in range(arity):
        idx = prefix_subtree_end(nodes, idx)
    return idx


def prefix_child_index(nodes: Sequence[AstNode], start: int, child_no: int) -> int:
    arity = NODE_ARITY[nodes[start].kind]
    if child_no < 0 or child_no >= arity:
        raise ValueError("child_no out of range")
    idx = start + 1
    for _ in range(child_no):
        idx = prefix_subtree_end(nodes, idx)
    return idx


def validate_prefix_program(program: AstProgram) -> None:
    if program.version != "ast-prefix-v1":
        raise ValueError(f"unsupported ast version: {program.version}")
    if not program.nodes:
        raise ValueError("invalid prefix AST: empty node list")
    if program.nodes[0].kind != NodeKind.PROGRAM:
        raise ValueError("invalid prefix AST: root must be PROGRAM")

    end = prefix_subtree_end(program.nodes, 0)
    if end != len(program.nodes):
        raise ValueError("invalid prefix AST: trailing tokens")

    for n in program.nodes:
        if n.kind == NodeKind.CONST and not (0 <= n.i0 < len(program.consts)):
            raise ValueError("invalid prefix AST: const index out of range")
        if n.kind in (NodeKind.VAR, NodeKind.ASSIGN, NodeKind.FOR_RANGE) and not (0 <= n.i0 < len(program.names)):
            raise ValueError("invalid prefix AST: name index out of range")


def prefix_repr(program: AstProgram) -> str:
    validate_prefix_program(program)
    return "AstPrefix(" + ",".join(f"{n.kind.value}:{n.i0}:{n.i1}" for n in program.nodes) + ")"


def node_count(program: AstProgram) -> int:
    validate_prefix_program(program)
    return len(program.nodes)


def _expr_depth_from(nodes: Sequence[AstNode], idx: int) -> tuple[int, int]:
    kind = nodes[idx].kind
    if kind not in EXPR_KINDS:
        raise ValueError(f"expected Expr at index {idx}, got {kind}")
    arity = NODE_ARITY[kind]
    if arity == 0:
        return 1, idx + 1
    max_child = 0
    cur = idx + 1
    for _ in range(arity):
        d, cur = _expr_depth_from(nodes, cur)
        if d > max_child:
            max_child = d
    return 1 + max_child, cur


def _stmt_max_expr_depth(nodes: Sequence[AstNode], idx: int) -> tuple[int, int]:
    kind = nodes[idx].kind
    if kind == NodeKind.ASSIGN:
        d, j = _expr_depth_from(nodes, idx + 1)
        return d, j
    if kind == NodeKind.RETURN:
        d, j = _expr_depth_from(nodes, idx + 1)
        return d, j
    if kind == NodeKind.IF_STMT:
        dc, j = _expr_depth_from(nodes, idx + 1)
        dt, k = _block_max_expr_depth(nodes, j)
        de, h = _block_max_expr_depth(nodes, k)
        return max(dc, dt, de), h
    if kind == NodeKind.FOR_RANGE:
        d, j = _block_max_expr_depth(nodes, idx + 1)
        return d, j
    raise ValueError(f"expected Stmt at index {idx}, got {kind}")


def _block_max_expr_depth(nodes: Sequence[AstNode], idx: int) -> tuple[int, int]:
    kind = nodes[idx].kind
    if kind == NodeKind.BLOCK_NIL:
        return 0, idx + 1
    if kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {kind}")
    d0, j = _stmt_max_expr_depth(nodes, idx + 1)
    d1, k = _block_max_expr_depth(nodes, j)
    return max(d0, d1), k


def max_expr_depth(program: AstProgram) -> int:
    validate_prefix_program(program)
    d, end = _block_max_expr_depth(program.nodes, 1)
    if end != len(program.nodes):
        raise ValueError("invalid trailing tokens")
    return d


def _block_contains_return(nodes: Sequence[AstNode], idx: int) -> tuple[bool, int]:
    kind = nodes[idx].kind
    if kind == NodeKind.BLOCK_NIL:
        return False, idx + 1
    if kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {kind}")

    st = nodes[idx + 1].kind
    if st == NodeKind.RETURN:
        j = prefix_subtree_end(nodes, idx + 1)
        _, k = _block_contains_return(nodes, j)
        return True, k

    if st == NodeKind.ASSIGN:
        j = prefix_subtree_end(nodes, idx + 1)
        rtail, k = _block_contains_return(nodes, j)
        return rtail, k

    if st == NodeKind.FOR_RANGE:
        body_idx = idx + 2
        _, body_end = _block_contains_return(nodes, body_idx)
        rtail, k = _block_contains_return(nodes, body_end)
        return rtail, k

    if st == NodeKind.IF_STMT:
        cond_idx = idx + 2
        cond_end = prefix_subtree_end(nodes, cond_idx)
        _, then_end = _block_contains_return(nodes, cond_end)
        _, else_end = _block_contains_return(nodes, then_end)
        rtail, k = _block_contains_return(nodes, else_end)
        return rtail, k

    raise ValueError(f"expected Stmt at index {idx + 1}, got {st}")


def top_level_has_return(program: AstProgram) -> bool:
    validate_prefix_program(program)
    has_ret, _ = _block_contains_return(program.nodes, 1)
    return has_ret


def build_program(stmt_specs: Sequence[tuple]) -> AstProgram:
    names: List[str] = []
    name_to_idx: Dict[str, int] = {}
    consts: List[Val] = []
    const_to_idx: Dict[tuple[str, str], int] = {}
    nodes: List[AstNode] = [AstNode(NodeKind.PROGRAM)]

    def name_id(name: str) -> int:
        idx = name_to_idx.get(name)
        if idx is not None:
            return idx
        idx = len(names)
        names.append(name)
        name_to_idx[name] = idx
        return idx

    def const_id(v: Val) -> int:
        key = (type(v).__name__, repr(v))
        idx = const_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(consts)
        consts.append(v)
        const_to_idx[key] = idx
        return idx

    def emit_expr(spec: tuple) -> None:
        tag = spec[0]
        if tag == "const":
            nodes.append(AstNode(NodeKind.CONST, i0=const_id(spec[1])))
            return
        if tag == "var":
            nodes.append(AstNode(NodeKind.VAR, i0=name_id(spec[1])))
            return
        if tag == "neg":
            nodes.append(AstNode(NodeKind.NEG))
            emit_expr(spec[1])
            return
        if tag == "not":
            nodes.append(AstNode(NodeKind.NOT))
            emit_expr(spec[1])
            return
        if tag in {
            "add", "sub", "mul", "div", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or",
        }:
            nodes.append(AstNode(NodeKind(tag.upper())))
            emit_expr(spec[1])
            emit_expr(spec[2])
            return
        if tag == "if_expr":
            nodes.append(AstNode(NodeKind.IF_EXPR))
            emit_expr(spec[1])
            emit_expr(spec[2])
            emit_expr(spec[3])
            return
        if tag == "call":
            name = spec[1]
            args = list(spec[2])
            nk = {
                "abs": NodeKind.CALL_ABS,
                "min": NodeKind.CALL_MIN,
                "max": NodeKind.CALL_MAX,
                "clip": NodeKind.CALL_CLIP,
            }.get(name)
            if nk is None:
                raise ValueError(f"unknown builtin: {name}")
            if len(args) != NODE_ARITY[nk]:
                raise ValueError(f"bad builtin arity for {name}: got {len(args)}")
            nodes.append(AstNode(nk))
            for a in args:
                emit_expr(a)
            return
        raise ValueError(f"unknown expr tag: {tag}")

    def emit_stmt(spec: tuple) -> None:
        tag = spec[0]
        if tag == "assign":
            nodes.append(AstNode(NodeKind.ASSIGN, i0=name_id(spec[1])))
            emit_expr(spec[2])
            return
        if tag == "return":
            nodes.append(AstNode(NodeKind.RETURN))
            emit_expr(spec[1])
            return
        if tag == "if":
            nodes.append(AstNode(NodeKind.IF_STMT))
            emit_expr(spec[1])
            emit_block(spec[2])
            emit_block(spec[3])
            return
        if tag == "for":
            nodes.append(AstNode(NodeKind.FOR_RANGE, i0=name_id(spec[1]), i1=int(spec[2])))
            emit_block(spec[3])
            return
        raise ValueError(f"unknown stmt tag: {tag}")

    def emit_block(stmts: Sequence[tuple]) -> None:
        if not stmts:
            nodes.append(AstNode(NodeKind.BLOCK_NIL))
            return
        nodes.append(AstNode(NodeKind.BLOCK_CONS))
        emit_stmt(stmts[0])
        emit_block(stmts[1:])

    emit_block(stmt_specs)
    program = AstProgram(nodes=tuple(nodes), names=tuple(names), consts=tuple(consts))
    validate_prefix_program(program)
    return program
