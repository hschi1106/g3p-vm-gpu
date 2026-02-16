from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .ast import (
    Val, Expr, Const, Var, Unary, Binary, IfExpr, Call,
    Stmt, Assign, IfStmt, ForRange, Return, Block,
    UOp, BOp,
)
from .builtins import builtin_call
from .errors import Err, ErrCode, Normal, Returned, Failed, Out
from .semantics import compare_values, is_num, promote_numeric


Env = Dict[str, Val]


def _consume_fuel(fuel: int) -> int | Err:
    if fuel <= 0:
        return Err(ErrCode.TIMEOUT, "fuel exhausted")
    return fuel - 1


def eval_expr(e: Expr, env: Env, fuel: int) -> tuple[Val | Err, int]:
    """Big-step expression evaluation with strict typing and fuel.

    Returns: (value_or_error, fuel_left)
    """
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return fuel2, 0
    fuel = fuel2

    if isinstance(e, Const):
        return e.value, fuel

    if isinstance(e, Var):
        if e.name not in env:
            return Err(ErrCode.NAME, f"undefined variable: {e.name}"), fuel
        return env[e.name], fuel

    if isinstance(e, Unary):
        r, fuel = eval_expr(e.e, env, fuel)
        if isinstance(r, Err):
            return r, fuel
        if e.op == UOp.NEG:
            if not is_num(r):
                return Err(ErrCode.TYPE, "unary '-' expects numeric"), fuel
            return (-r), fuel
        if e.op == UOp.NOT:
            if not isinstance(r, bool):
                return Err(ErrCode.TYPE, "'not' expects bool"), fuel
            return (not r), fuel
        return Err(ErrCode.TYPE, f"unknown unary op: {e.op}"), fuel

    if isinstance(e, Binary):
        op = e.op

        # short-circuit boolean ops
        if op == BOp.AND:
            ra, fuel = eval_expr(e.a, env, fuel)
            if isinstance(ra, Err):
                return ra, fuel
            if not isinstance(ra, bool):
                return Err(ErrCode.TYPE, "'and' expects bool operands"), fuel
            if ra is False:
                return False, fuel
            rb, fuel = eval_expr(e.b, env, fuel)
            if isinstance(rb, Err):
                return rb, fuel
            if not isinstance(rb, bool):
                return Err(ErrCode.TYPE, "'and' expects bool operands"), fuel
            return rb, fuel

        if op == BOp.OR:
            ra, fuel = eval_expr(e.a, env, fuel)
            if isinstance(ra, Err):
                return ra, fuel
            if not isinstance(ra, bool):
                return Err(ErrCode.TYPE, "'or' expects bool operands"), fuel
            if ra is True:
                return True, fuel
            rb, fuel = eval_expr(e.b, env, fuel)
            if isinstance(rb, Err):
                return rb, fuel
            if not isinstance(rb, bool):
                return Err(ErrCode.TYPE, "'or' expects bool operands"), fuel
            return rb, fuel

        # non-short-circuit ops evaluate both sides
        ra, fuel = eval_expr(e.a, env, fuel)
        if isinstance(ra, Err):
            return ra, fuel
        rb, fuel = eval_expr(e.b, env, fuel)
        if isinstance(rb, Err):
            return rb, fuel

        # arithmetic
        if op in (BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD):
            prom = promote_numeric(ra, rb)
            if isinstance(prom, Err):
                return prom, fuel
            a2, b2 = prom
            if op == BOp.ADD:
                return a2 + b2, fuel
            if op == BOp.SUB:
                return a2 - b2, fuel
            if op == BOp.MUL:
                return a2 * b2, fuel
            if op == BOp.DIV:
                if b2 == 0:
                    return Err(ErrCode.ZERODIV, "division by zero"), fuel
                # '/' always returns float (matches Python behavior)
                return float(a2) / float(b2), fuel
            if op == BOp.MOD:
                if b2 == 0:
                    return Err(ErrCode.ZERODIV, "modulo by zero"), fuel
                return a2 % b2, fuel

        # comparisons
        if op in (BOp.LT, BOp.LE, BOp.GT, BOp.GE, BOp.EQ, BOp.NE):
            rcmp = compare_values(op.value, ra, rb)
            if isinstance(rcmp, Err):
                return rcmp, fuel
            return rcmp, fuel

        return Err(ErrCode.TYPE, f"unknown binary op: {op}"), fuel

    if isinstance(e, IfExpr):
        rc, fuel = eval_expr(e.cond, env, fuel)
        if isinstance(rc, Err):
            return rc, fuel
        if not isinstance(rc, bool):
            return Err(ErrCode.TYPE, "ternary condition must be bool"), fuel
        branch = e.then_e if rc else e.else_e
        return eval_expr(branch, env, fuel)

    if isinstance(e, Call):
        vals: List[Val] = []
        for arg in e.args:
            r, fuel = eval_expr(arg, env, fuel)
            if isinstance(r, Err):
                return r, fuel
            vals.append(r)
        out = builtin_call(e.name, list(vals))
        return out, fuel  # out may be Err

    return Err(ErrCode.TYPE, f"unknown Expr node: {type(e).__name__}"), fuel


def exec_stmt(s: Stmt, env: Env, fuel: int) -> tuple[Env, Out, int]:
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return env, Failed(fuel2), 0
    fuel = fuel2

    if isinstance(s, Assign):
        r, fuel = eval_expr(s.e, env, fuel)
        if isinstance(r, Err):
            return env, Failed(r), fuel
        env2 = dict(env)
        env2[s.name] = r
        return env2, Normal(), fuel

    if isinstance(s, Return):
        r, fuel = eval_expr(s.e, env, fuel)
        if isinstance(r, Err):
            return env, Failed(r), fuel
        return env, Returned(r), fuel

    if isinstance(s, IfStmt):
        rc, fuel = eval_expr(s.cond, env, fuel)
        if isinstance(rc, Err):
            return env, Failed(rc), fuel
        if not isinstance(rc, bool):
            return env, Failed(Err(ErrCode.TYPE, "if condition must be bool")), fuel
        blk = s.then_block if rc else s.else_block
        return exec_block(blk, env, fuel)

    if isinstance(s, ForRange):
        if not isinstance(s.k, int) or isinstance(s.k, bool) or s.k < 0:
            return env, Failed(Err(ErrCode.TYPE, "range(K) requires non-negative int constant K")), fuel
        cur_env = dict(env)
        for i in range(s.k):
            cur_env[s.var] = i
            cur_env, out, fuel = exec_block(s.body, cur_env, fuel)
            if not isinstance(out, Normal):
                return cur_env, out, fuel
        return cur_env, Normal(), fuel

    return env, Failed(Err(ErrCode.TYPE, f"unknown Stmt node: {type(s).__name__}")), fuel


def exec_block(b: Block, env: Env, fuel: int) -> tuple[Env, Out, int]:
    cur_env = dict(env)
    cur_out: Out = Normal()
    for st in b.stmts:
        cur_env, cur_out, fuel = exec_stmt(st, cur_env, fuel)
        if not isinstance(cur_out, Normal):
            return cur_env, cur_out, fuel
    return cur_env, cur_out, fuel


def run_program(p: Block, inputs: Env | None = None, fuel: int = 10_000) -> tuple[Env, Out]:
    env0 = dict(inputs) if inputs else {}
    env1, out, _fuel_left = exec_block(p, env0, fuel)
    if isinstance(out, Normal):
        return env1, Failed(Err(ErrCode.VALUE, "program finished without return"))
    return env1, out
