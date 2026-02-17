from __future__ import annotations

from typing import Dict, List

from .ast import AstProgram, NodeKind, Val, validate_prefix_program
from .builtins import builtin_call
from .errors import Err, ErrCode, Failed, Normal, Out, Returned
from .semantics import compare_values, is_num, promote_numeric


Env = Dict[str, Val]


def _consume_fuel(fuel: int) -> int | Err:
    if fuel <= 0:
        return Err(ErrCode.TIMEOUT, "fuel exhausted")
    return fuel - 1


def _skip_expr_end(p: AstProgram, idx: int) -> int:
    cur = idx
    stack = 1
    while stack > 0:
        n = p.nodes[cur]
        cur += 1
        arity = {
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
        }.get(n.kind)
        if arity is None:
            raise ValueError(f"expected Expr at index {cur - 1}, got {n.kind}")
        stack += arity - 1
    return cur


def _skip_stmt_end(p: AstProgram, idx: int) -> int:
    n = p.nodes[idx]
    if n.kind in (NodeKind.ASSIGN, NodeKind.RETURN):
        return _skip_expr_end(p, idx + 1)
    if n.kind == NodeKind.IF_STMT:
        j = _skip_expr_end(p, idx + 1)
        j = _skip_block_end(p, j)
        return _skip_block_end(p, j)
    if n.kind == NodeKind.FOR_RANGE:
        return _skip_block_end(p, idx + 1)
    raise ValueError(f"expected Stmt at index {idx}, got {n.kind}")


def _skip_block_end(p: AstProgram, idx: int) -> int:
    n = p.nodes[idx]
    if n.kind == NodeKind.BLOCK_NIL:
        return idx + 1
    if n.kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {n.kind}")
    j = _skip_stmt_end(p, idx + 1)
    return _skip_block_end(p, j)


def _eval_expr(p: AstProgram, idx: int, env: Env, fuel: int) -> tuple[Val | Err, int, int]:
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return fuel2, _skip_expr_end(p, idx), 0
    fuel = fuel2

    n = p.nodes[idx]
    k = n.kind

    if k == NodeKind.CONST:
        return p.consts[n.i0], idx + 1, fuel

    if k == NodeKind.VAR:
        name = p.names[n.i0]
        if name not in env:
            return Err(ErrCode.NAME, f"undefined variable: {name}"), idx + 1, fuel
        return env[name], idx + 1, fuel

    if k == NodeKind.NEG:
        r, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(r, Err):
            return r, j, fuel
        if not is_num(r):
            return Err(ErrCode.TYPE, "unary '-' expects numeric"), j, fuel
        return -r, j, fuel

    if k == NodeKind.NOT:
        r, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(r, Err):
            return r, j, fuel
        if not isinstance(r, bool):
            return Err(ErrCode.TYPE, "'not' expects bool"), j, fuel
        return (not r), j, fuel

    if k == NodeKind.AND:
        ra, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        if not isinstance(ra, bool):
            return Err(ErrCode.TYPE, "'and' expects bool operands"), j, fuel
        if not ra:
            return False, _skip_expr_end(p, j), fuel
        rb, h, fuel = _eval_expr(p, j, env, fuel)
        if isinstance(rb, Err):
            return rb, h, fuel
        if not isinstance(rb, bool):
            return Err(ErrCode.TYPE, "'and' expects bool operands"), h, fuel
        return rb, h, fuel

    if k == NodeKind.OR:
        ra, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        if not isinstance(ra, bool):
            return Err(ErrCode.TYPE, "'or' expects bool operands"), j, fuel
        if ra:
            return True, _skip_expr_end(p, j), fuel
        rb, h, fuel = _eval_expr(p, j, env, fuel)
        if isinstance(rb, Err):
            return rb, h, fuel
        if not isinstance(rb, bool):
            return Err(ErrCode.TYPE, "'or' expects bool operands"), h, fuel
        return rb, h, fuel

    if k in (NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV, NodeKind.MOD):
        ra, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        rb, h, fuel = _eval_expr(p, j, env, fuel)
        if isinstance(rb, Err):
            return rb, h, fuel
        prom = promote_numeric(ra, rb)
        if isinstance(prom, Err):
            return prom, h, fuel
        a2, b2 = prom
        if k == NodeKind.ADD:
            return a2 + b2, h, fuel
        if k == NodeKind.SUB:
            return a2 - b2, h, fuel
        if k == NodeKind.MUL:
            return a2 * b2, h, fuel
        if k == NodeKind.DIV:
            if b2 == 0:
                return Err(ErrCode.ZERODIV, "division by zero"), h, fuel
            return float(a2) / float(b2), h, fuel
        if b2 == 0:
            return Err(ErrCode.ZERODIV, "modulo by zero"), h, fuel
        return a2 % b2, h, fuel

    if k in (NodeKind.LT, NodeKind.LE, NodeKind.GT, NodeKind.GE, NodeKind.EQ, NodeKind.NE):
        ra, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        rb, h, fuel = _eval_expr(p, j, env, fuel)
        if isinstance(rb, Err):
            return rb, h, fuel
        op = {
            NodeKind.LT: "LT",
            NodeKind.LE: "LE",
            NodeKind.GT: "GT",
            NodeKind.GE: "GE",
            NodeKind.EQ: "EQ",
            NodeKind.NE: "NE",
        }[k]
        rcmp = compare_values(op, ra, rb)
        if isinstance(rcmp, Err):
            return rcmp, h, fuel
        return rcmp, h, fuel

    if k == NodeKind.IF_EXPR:
        rc, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(rc, Err):
            return rc, j, fuel
        if not isinstance(rc, bool):
            return Err(ErrCode.TYPE, "ternary condition must be bool"), j, fuel
        if rc:
            rt, h, fuel = _eval_expr(p, j, env, fuel)
            return rt, _skip_expr_end(p, h), fuel
        h = _skip_expr_end(p, j)
        rf, t, fuel = _eval_expr(p, h, env, fuel)
        return rf, t, fuel

    if k in (NodeKind.CALL_ABS, NodeKind.CALL_MIN, NodeKind.CALL_MAX, NodeKind.CALL_CLIP):
        name = {
            NodeKind.CALL_ABS: "abs",
            NodeKind.CALL_MIN: "min",
            NodeKind.CALL_MAX: "max",
            NodeKind.CALL_CLIP: "clip",
        }[k]
        argc = {NodeKind.CALL_ABS: 1, NodeKind.CALL_MIN: 2, NodeKind.CALL_MAX: 2, NodeKind.CALL_CLIP: 3}[k]
        vals: List[Val] = []
        cur = idx + 1
        for _ in range(argc):
            rv, cur, fuel = _eval_expr(p, cur, env, fuel)
            if isinstance(rv, Err):
                return rv, cur, fuel
            vals.append(rv)
        out = builtin_call(name, vals)
        return out, cur, fuel

    return Err(ErrCode.TYPE, f"unknown Expr node: {k}"), _skip_expr_end(p, idx), fuel


def _exec_stmt(p: AstProgram, idx: int, env: Env, fuel: int) -> tuple[Env, Out, int, int]:
    next_idx = _skip_stmt_end(p, idx)
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return env, Failed(fuel2), next_idx, 0
    fuel = fuel2

    n = p.nodes[idx]
    k = n.kind

    if k == NodeKind.ASSIGN:
        r, _j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(r, Err):
            return env, Failed(r), next_idx, fuel
        env2 = dict(env)
        env2[p.names[n.i0]] = r
        return env2, Normal(), next_idx, fuel

    if k == NodeKind.RETURN:
        r, _j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(r, Err):
            return env, Failed(r), next_idx, fuel
        return env, Returned(r), next_idx, fuel

    if k == NodeKind.IF_STMT:
        rc, j, fuel = _eval_expr(p, idx + 1, env, fuel)
        if isinstance(rc, Err):
            return env, Failed(rc), next_idx, fuel
        if not isinstance(rc, bool):
            return env, Failed(Err(ErrCode.TYPE, "if condition must be bool")), next_idx, fuel
        if rc:
            env2, out2, _after_then, fuel = _exec_block(p, j, env, fuel)
            return env2, out2, next_idx, fuel
        else_j = _skip_block_end(p, j)
        env3, out3, _after_else, fuel = _exec_block(p, else_j, env, fuel)
        return env3, out3, next_idx, fuel

    if k == NodeKind.FOR_RANGE:
        kval = n.i1
        body_idx = idx + 1
        if not isinstance(kval, int) or isinstance(kval, bool) or kval < 0:
            return env, Failed(Err(ErrCode.TYPE, "range(K) requires non-negative int constant K")), next_idx, fuel
        cur_env = dict(env)
        name = p.names[n.i0]
        for i in range(kval):
            cur_env[name] = i
            cur_env, out, _after_body, fuel = _exec_block(p, body_idx, cur_env, fuel)
            if not isinstance(out, Normal):
                return cur_env, out, next_idx, fuel
        return cur_env, Normal(), next_idx, fuel

    return env, Failed(Err(ErrCode.TYPE, f"unknown Stmt node: {k}")), next_idx, fuel


def _exec_block(p: AstProgram, idx: int, env: Env, fuel: int) -> tuple[Env, Out, int, int]:
    n = p.nodes[idx]
    if n.kind == NodeKind.BLOCK_NIL:
        return dict(env), Normal(), idx + 1, fuel
    if n.kind != NodeKind.BLOCK_CONS:
        return env, Failed(Err(ErrCode.TYPE, f"expected Block at index {idx}, got {n.kind}")), idx, fuel

    env1, out1, j, fuel = _exec_stmt(p, idx + 1, dict(env), fuel)
    if not isinstance(out1, Normal):
        return env1, out1, _skip_block_end(p, j), fuel
    return _exec_block(p, j, env1, fuel)


def eval_expr(p: AstProgram, inputs: Env | None = None, fuel: int = 10_000) -> tuple[Val | Err, int]:
    validate_prefix_program(p)
    env = dict(inputs) if inputs else {}
    v, end, fuel_left = _eval_expr(p, 1, env, fuel)
    if end != len(p.nodes):
        return Err(ErrCode.TYPE, "eval_expr expects single Expr root"), fuel_left
    return v, fuel_left


def run_program(p: AstProgram, inputs: Env | None = None, fuel: int = 10_000) -> tuple[Env, Out]:
    validate_prefix_program(p)
    env0 = dict(inputs) if inputs else {}
    env1, out, end, _fuel_left = _exec_block(p, 1, env0, fuel)
    if end != len(p.nodes):
        return env1, Failed(Err(ErrCode.TYPE, "invalid trailing tokens"))
    if isinstance(out, Normal):
        return env1, Failed(Err(ErrCode.VALUE, "program finished without return"))
    return env1, out
