from __future__ import annotations

from typing import List, Sequence

from ..core.ast import AstProgram, NodeKind, build_program
from .genome import ProgramGenome, as_genome


def expr_from_prefix(program: AstProgram, idx: int) -> tuple[tuple, int]:
    node = program.nodes[idx]
    kind = node.kind
    if kind == NodeKind.CONST:
        return ("const", program.consts[node.i0]), idx + 1
    if kind == NodeKind.VAR:
        return ("var", program.names[node.i0]), idx + 1
    if kind == NodeKind.NEG:
        expr, nxt = expr_from_prefix(program, idx + 1)
        return ("neg", expr), nxt
    if kind == NodeKind.NOT:
        expr, nxt = expr_from_prefix(program, idx + 1)
        return ("not", expr), nxt
    if kind in {
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
    }:
        left, mid = expr_from_prefix(program, idx + 1)
        right, nxt = expr_from_prefix(program, mid)
        return (kind.value.lower(), left, right), nxt
    if kind == NodeKind.IF_EXPR:
        cond, mid = expr_from_prefix(program, idx + 1)
        then_expr, tail = expr_from_prefix(program, mid)
        else_expr, nxt = expr_from_prefix(program, tail)
        return ("if_expr", cond, then_expr, else_expr), nxt
    if kind in {
        NodeKind.CALL_ABS,
        NodeKind.CALL_MIN,
        NodeKind.CALL_MAX,
        NodeKind.CALL_CLIP,
        NodeKind.CALL_LEN,
        NodeKind.CALL_CONCAT,
        NodeKind.CALL_SLICE,
        NodeKind.CALL_INDEX,
    }:
        name = {
            NodeKind.CALL_ABS: "abs",
            NodeKind.CALL_MIN: "min",
            NodeKind.CALL_MAX: "max",
            NodeKind.CALL_CLIP: "clip",
            NodeKind.CALL_LEN: "len",
            NodeKind.CALL_CONCAT: "concat",
            NodeKind.CALL_SLICE: "slice",
            NodeKind.CALL_INDEX: "index",
        }[kind]
        argc = {
            NodeKind.CALL_ABS: 1,
            NodeKind.CALL_MIN: 2,
            NodeKind.CALL_MAX: 2,
            NodeKind.CALL_CLIP: 3,
            NodeKind.CALL_LEN: 1,
            NodeKind.CALL_CONCAT: 2,
            NodeKind.CALL_SLICE: 3,
            NodeKind.CALL_INDEX: 2,
        }[kind]
        args: List[tuple] = []
        cur = idx + 1
        for _ in range(argc):
            arg, cur = expr_from_prefix(program, cur)
            args.append(arg)
        return ("call", name, args), cur
    raise ValueError(f"expected Expr at index {idx}, got {kind}")


def stmt_from_prefix(program: AstProgram, idx: int) -> tuple[tuple, int]:
    node = program.nodes[idx]
    if node.kind == NodeKind.ASSIGN:
        expr, nxt = expr_from_prefix(program, idx + 1)
        return ("assign", program.names[node.i0], expr), nxt
    if node.kind == NodeKind.RETURN:
        expr, nxt = expr_from_prefix(program, idx + 1)
        return ("return", expr), nxt
    if node.kind == NodeKind.IF_STMT:
        cond, mid = expr_from_prefix(program, idx + 1)
        then_block, tail = block_from_prefix(program, mid)
        else_block, nxt = block_from_prefix(program, tail)
        return ("if", cond, then_block, else_block), nxt
    if node.kind == NodeKind.FOR_RANGE:
        body, nxt = block_from_prefix(program, idx + 1)
        return ("for", program.names[node.i0], node.i1, body), nxt
    raise ValueError(f"expected Stmt at index {idx}, got {node.kind}")


def block_from_prefix(program: AstProgram, idx: int) -> tuple[List[tuple], int]:
    node = program.nodes[idx]
    if node.kind == NodeKind.BLOCK_NIL:
        return [], idx + 1
    if node.kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {node.kind}")
    stmt, mid = stmt_from_prefix(program, idx + 1)
    tail, nxt = block_from_prefix(program, mid)
    return [stmt] + tail, nxt


def top_level_statements(program: AstProgram) -> List[tuple]:
    block, end = block_from_prefix(program, 1)
    if end != len(program.nodes):
        raise ValueError("invalid trailing tokens")
    return block


def genome_from_statements(statements: Sequence[tuple]) -> ProgramGenome:
    return as_genome(build_program(statements))
