from __future__ import annotations

from typing import List, Sequence

from ..core.ast import AstProgram, BUILTIN_NAME_BY_NODE, NODE_ARITY, NodeKind, build_program, linear_rec_binders_for_node
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
    if kind in BUILTIN_NAME_BY_NODE:
        name = BUILTIN_NAME_BY_NODE[kind]
        argc = NODE_ARITY[kind]
        args: List[tuple] = []
        cur = idx + 1
        for _ in range(argc):
            arg, cur = expr_from_prefix(program, cur)
            args.append(arg)
        return ("call", name, args), cur
    if kind == NodeKind.BOUND_VAR:
        return ("bound", program.names[node.i0]), idx + 1
    if kind == NodeKind.MAP_LIST:
        source, mid = expr_from_prefix(program, idx + 1)
        body, nxt = expr_from_prefix(program, mid)
        out_type = {1: "int", 2: "float", 3: "string"}.get(node.i1)
        if out_type is None:
            raise ValueError(f"unknown MapList output tag: {node.i1}")
        return ("map_list", program.names[node.i0], source, body, out_type), nxt
    if kind == NodeKind.FILTER_LIST:
        source, mid = expr_from_prefix(program, idx + 1)
        pred, nxt = expr_from_prefix(program, mid)
        return ("filter_list", program.names[node.i0], source, pred), nxt
    if kind == NodeKind.LINEAR_REC:
        binders = linear_rec_binders_for_node(program, idx)
        source, mid = expr_from_prefix(program, idx + 1)
        start, tail = expr_from_prefix(program, mid)
        empty_case, step_idx = expr_from_prefix(program, tail)
        step, last_idx = expr_from_prefix(program, step_idx)
        last, nxt = expr_from_prefix(program, last_idx)
        return (
            "linear_rec",
            program.names[binders.elem_name],
            program.names[binders.accum_name],
            program.names[binders.index_name],
            source,
            start,
            empty_case,
            step,
            last,
        ), nxt
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
        bound, mid = expr_from_prefix(program, idx + 1)
        body, nxt = block_from_prefix(program, mid)
        return ("for", program.names[node.i0], bound, body), nxt
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
