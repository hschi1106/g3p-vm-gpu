from __future__ import annotations

from typing import Dict, List

from ..core.ast import (
    BUILTIN_NAME_BY_NODE,
    EXPR_KINDS,
    NODE_ARITY,
    AstProgram,
    FloatList,
    IntList,
    NodeKind,
    StringList,
    Val,
    asgp_dc_binders_for_node,
    asgp_dp1d_spec_for_node,
    asgp_dp2d_spec_for_node,
    empty_list_for_tag,
    linear_rec_binders_for_node,
    list_tag_for_value,
    validate_prefix_program,
)
from .builtins import builtin_call
from ..core.errors import Err, ErrCode, Failed, Normal, Out, Returned
from ..core.value_semantics import compare_values, is_num, promote_numeric


Env = Dict[str, Val]
BinderEnv = Dict[int, List[Val]]
AsgpDcSource = str | IntList | FloatList | StringList


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
        if n.kind not in EXPR_KINDS:
            raise ValueError(f"expected Expr at index {cur - 1}, got {n.kind}")
        arity = NODE_ARITY.get(n.kind)
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
        j = _skip_expr_end(p, idx + 1)
        return _skip_block_end(p, j)
    raise ValueError(f"expected Stmt at index {idx}, got {n.kind}")


def _skip_block_end(p: AstProgram, idx: int) -> int:
    n = p.nodes[idx]
    if n.kind == NodeKind.BLOCK_NIL:
        return idx + 1
    if n.kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {n.kind}")
    j = _skip_stmt_end(p, idx + 1)
    return _skip_block_end(p, j)


def _with_bound(binders: BinderEnv, name_idx: int, value: Val) -> BinderEnv:
    out = {k: list(v) for k, v in binders.items()}
    out.setdefault(name_idx, []).append(value)
    return out


def _with_bound_values(binders: BinderEnv, values: tuple[tuple[int, Val], ...]) -> BinderEnv:
    out = {k: list(v) for k, v in binders.items()}
    for name_idx, value in values:
        out.setdefault(name_idx, []).append(value)
    return out


def _is_asgp_dc_source(value: Val) -> bool:
    return isinstance(value, (str, IntList, FloatList, StringList))


def _asgp_dc_source_len(value: AsgpDcSource) -> int:
    return len(value)


def _asgp_dc_source_slice(value: AsgpDcSource, start: int, end: int) -> AsgpDcSource:
    if isinstance(value, str):
        return value[start:end]
    if isinstance(value, IntList):
        return IntList(tuple(value.items[start:end]))
    if isinstance(value, FloatList):
        return FloatList(tuple(value.items[start:end]))
    return StringList(tuple(value.items[start:end]))


def _subtree_contains_asgp(p: AstProgram, idx: int) -> bool:
    end = _skip_expr_end(p, idx)
    return any(
        node.kind in {NodeKind.ASGP_DC, NodeKind.ASGP_DP1D, NodeKind.ASGP_DP2D}
        for node in p.nodes[idx:end]
    )


def _asgp_dp1d_deps(spec, state: int) -> tuple[int, ...]:
    if spec.dep_kind in {NodeKind.DP1_BACKWARD1, NodeKind.DP1_BACKWARD2, NodeKind.DP1_BACKWARD3}:
        return tuple(state - offset for offset in spec.dep_offsets)
    return tuple(state + offset for offset in spec.dep_offsets)


def _asgp_dp2d_deps(spec, i: int, j: int) -> tuple[tuple[int, int], ...]:
    if spec.dep_kind == NodeKind.DP2_CROSS_BACKWARD:
        return ((i - 1, j), (i, j - 1))
    if spec.dep_kind == NodeKind.DP2_CROSS_FORWARD:
        return ((i + 1, j), (i, j + 1))
    if spec.dep_kind == NodeKind.DP2_DIAGONAL_BACKWARD:
        return ((i - 1, j - 1),)
    if spec.dep_kind == NodeKind.DP2_DIAGONAL_FORWARD:
        return ((i + 1, j + 1),)
    if spec.dep_kind == NodeKind.DP2_NEIGHBORHOOD_BACKWARD3:
        return ((i - 1, j), (i, j - 1), (i - 1, j - 1))
    if spec.dep_kind == NodeKind.DP2_NEIGHBORHOOD_FORWARD3:
        return ((i + 1, j), (i, j + 1), (i + 1, j + 1))
    raise ValueError(f"unknown ASGP-DP2D dependency kind: {spec.dep_kind}")


def _asgp_dc_frame(
    p: AstProgram,
    node_idx: int,
    source: AsgpDcSource,
    lo: int,
    solve_idx: int,
    divide_idx: int,
    combine_idx: int,
    dc_end: int,
    fuel: int,
) -> tuple[Val | Err, int]:
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return fuel2, 0
    fuel = fuel2

    binders = asgp_dc_binders_for_node(p, node_idx)
    n = _asgp_dc_source_len(source)
    if n <= 1:
        solve_binders = {
            binders.solve_xs_name: [source],
            binders.solve_n_name: [n],
            binders.solve_lo_name: [lo],
        }
        out, end, fuel = _eval_expr(p, solve_idx, {}, solve_binders, fuel)
        if isinstance(out, Err):
            return out, fuel
        if end != divide_idx:
            return Err(ErrCode.TYPE, "ASGP-DC solve phase did not consume exactly one expression"), fuel
        return out, fuel

    divide_binders = {binders.divide_n_name: [n]}
    raw_split, end, fuel = _eval_expr(p, divide_idx, {}, divide_binders, fuel)
    if isinstance(raw_split, Err):
        return raw_split, fuel
    if end != combine_idx:
        return Err(ErrCode.TYPE, "ASGP-DC divide phase did not consume exactly one expression"), fuel
    if not isinstance(raw_split, int) or isinstance(raw_split, bool):
        return Err(ErrCode.TYPE, "ASGP-DC divide phase must return int"), fuel

    split = max(1, min(raw_split, n - 1))
    left, fuel = _asgp_dc_frame(
        p,
        node_idx,
        _asgp_dc_source_slice(source, 0, split),
        lo,
        solve_idx,
        divide_idx,
        combine_idx,
        dc_end,
        fuel,
    )
    if isinstance(left, Err):
        return left, fuel
    right, fuel = _asgp_dc_frame(
        p,
        node_idx,
        _asgp_dc_source_slice(source, split, n),
        lo + split,
        solve_idx,
        divide_idx,
        combine_idx,
        dc_end,
        fuel,
    )
    if isinstance(right, Err):
        return right, fuel
    if type(left) is not type(right):
        return Err(ErrCode.TYPE, "ASGP-DC recursive results must have matching types"), fuel

    combine_binders = {
        binders.combine_left_name: [left],
        binders.combine_right_name: [right],
    }
    out, end, fuel = _eval_expr(p, combine_idx, {}, combine_binders, fuel)
    if isinstance(out, Err):
        return out, fuel
    if end != dc_end:
        return Err(ErrCode.TYPE, "ASGP-DC combine phase did not consume exactly one expression"), fuel
    if type(out) is not type(left):
        return Err(ErrCode.TYPE, "ASGP-DC combine result type must match recursive result type"), fuel
    return out, fuel


def _asgp_dp1d_frame(
    p: AstProgram,
    node_idx: int,
    state: int,
    solve_idx: int,
    transition_idx: int,
    dp_end: int,
    memo: Dict[int, Val],
    fuel: int,
) -> tuple[Val | Err, int]:
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return fuel2, 0
    fuel = fuel2

    spec = asgp_dp1d_spec_for_node(p, node_idx)
    if state < spec.lo or state > spec.hi:
        return p.consts[spec.boundary_const], fuel

    if state == spec.base_state:
        solve_binders = {spec.solve_state_name: [state]}
        out, end, fuel = _eval_expr(p, solve_idx, {}, solve_binders, fuel)
        if isinstance(out, Err):
            return out, fuel
        if end != transition_idx:
            return Err(ErrCode.TYPE, "ASGP-DP1D solve phase did not consume exactly one expression"), fuel
        return out, fuel

    cached = memo.get(state)
    if cached is not None:
        return cached, fuel

    dep_values: List[Val] = []
    for dep_state in _asgp_dp1d_deps(spec, state):
        dep_value, fuel = _asgp_dp1d_frame(p, node_idx, dep_state, solve_idx, transition_idx, dp_end, memo, fuel)
        if isinstance(dep_value, Err):
            return dep_value, fuel
        dep_values.append(dep_value)
    if dep_values and any(type(value) is not type(dep_values[0]) for value in dep_values[1:]):
        return Err(ErrCode.TYPE, "ASGP-DP1D dependency result types must match"), fuel

    transition_binders = {spec.transition_state_name: [state]}
    for name_idx, value in zip(spec.transition_dep_names, dep_values):
        transition_binders[name_idx] = [value]
    out, end, fuel = _eval_expr(p, transition_idx, {}, transition_binders, fuel)
    if isinstance(out, Err):
        return out, fuel
    if end != dp_end:
        return Err(ErrCode.TYPE, "ASGP-DP1D transition phase did not consume exactly one expression"), fuel
    if dep_values and type(out) is not type(dep_values[0]):
        return Err(ErrCode.TYPE, "ASGP-DP1D transition result type must match dependency result type"), fuel
    memo[state] = out
    return out, fuel


def _asgp_dp2d_frame(
    p: AstProgram,
    node_idx: int,
    i: int,
    j: int,
    solve_idx: int,
    transition_idx: int,
    dp_end: int,
    memo: Dict[tuple[int, int], Val],
    fuel: int,
) -> tuple[Val | Err, int]:
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return fuel2, 0
    fuel = fuel2

    spec = asgp_dp2d_spec_for_node(p, node_idx)
    if i < spec.i_lo or i > spec.i_hi or j < spec.j_lo or j > spec.j_hi:
        return p.consts[spec.boundary_const], fuel

    if i == spec.base_i and j == spec.base_j:
        solve_binders = {
            spec.solve_i_name: [i],
            spec.solve_j_name: [j],
        }
        out, end, fuel = _eval_expr(p, solve_idx, {}, solve_binders, fuel)
        if isinstance(out, Err):
            return out, fuel
        if end != transition_idx:
            return Err(ErrCode.TYPE, "ASGP-DP2D solve phase did not consume exactly one expression"), fuel
        return out, fuel

    key = (i, j)
    cached = memo.get(key)
    if cached is not None:
        return cached, fuel

    dep_values: List[Val] = []
    for dep_i, dep_j in _asgp_dp2d_deps(spec, i, j):
        dep_value, fuel = _asgp_dp2d_frame(
            p,
            node_idx,
            dep_i,
            dep_j,
            solve_idx,
            transition_idx,
            dp_end,
            memo,
            fuel,
        )
        if isinstance(dep_value, Err):
            return dep_value, fuel
        dep_values.append(dep_value)
    if dep_values and any(type(value) is not type(dep_values[0]) for value in dep_values[1:]):
        return Err(ErrCode.TYPE, "ASGP-DP2D dependency result types must match"), fuel

    transition_binders = {
        spec.transition_i_name: [i],
        spec.transition_j_name: [j],
    }
    for name_idx, value in zip(spec.transition_dep_names, dep_values):
        transition_binders[name_idx] = [value]
    out, end, fuel = _eval_expr(p, transition_idx, {}, transition_binders, fuel)
    if isinstance(out, Err):
        return out, fuel
    if end != dp_end:
        return Err(ErrCode.TYPE, "ASGP-DP2D transition phase did not consume exactly one expression"), fuel
    if dep_values and type(out) is not type(dep_values[0]):
        return Err(ErrCode.TYPE, "ASGP-DP2D transition result type must match dependency result type"), fuel
    memo[key] = out
    return out, fuel


def _eval_expr(p: AstProgram, idx: int, env: Env, binders: BinderEnv, fuel: int) -> tuple[Val | Err, int, int]:
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

    if k == NodeKind.BOUND_VAR:
        stack = binders.get(n.i0)
        if not stack:
            return Err(ErrCode.NAME, f"undefined binder: {p.names[n.i0]}"), idx + 1, fuel
        return stack[-1], idx + 1, fuel

    if k == NodeKind.NEG:
        r, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(r, Err):
            return r, j, fuel
        if not is_num(r):
            return Err(ErrCode.TYPE, "unary '-' expects numeric"), j, fuel
        return -r, j, fuel

    if k == NodeKind.NOT:
        r, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(r, Err):
            return r, j, fuel
        if not isinstance(r, bool):
            return Err(ErrCode.TYPE, "'not' expects bool"), j, fuel
        return (not r), j, fuel

    if k == NodeKind.AND:
        ra, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        if not isinstance(ra, bool):
            return Err(ErrCode.TYPE, "'and' expects bool operands"), j, fuel
        if not ra:
            return False, _skip_expr_end(p, j), fuel
        rb, h, fuel = _eval_expr(p, j, env, binders, fuel)
        if isinstance(rb, Err):
            return rb, h, fuel
        if not isinstance(rb, bool):
            return Err(ErrCode.TYPE, "'and' expects bool operands"), h, fuel
        return rb, h, fuel

    if k == NodeKind.OR:
        ra, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        if not isinstance(ra, bool):
            return Err(ErrCode.TYPE, "'or' expects bool operands"), j, fuel
        if ra:
            return True, _skip_expr_end(p, j), fuel
        rb, h, fuel = _eval_expr(p, j, env, binders, fuel)
        if isinstance(rb, Err):
            return rb, h, fuel
        if not isinstance(rb, bool):
            return Err(ErrCode.TYPE, "'or' expects bool operands"), h, fuel
        return rb, h, fuel

    if k in (NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV, NodeKind.MOD):
        ra, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        rb, h, fuel = _eval_expr(p, j, env, binders, fuel)
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
        ra, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(ra, Err):
            return ra, j, fuel
        rb, h, fuel = _eval_expr(p, j, env, binders, fuel)
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
        rc, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(rc, Err):
            return rc, j, fuel
        if not isinstance(rc, bool):
            return Err(ErrCode.TYPE, "ternary condition must be bool"), j, fuel
        if rc:
            rt, h, fuel = _eval_expr(p, j, env, binders, fuel)
            return rt, _skip_expr_end(p, h), fuel
        h = _skip_expr_end(p, j)
        rf, t, fuel = _eval_expr(p, h, env, binders, fuel)
        return rf, t, fuel

    if k == NodeKind.MAP_LIST:
        xs, body_idx, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        body_end = _skip_expr_end(p, body_idx)
        if isinstance(xs, Err):
            return xs, body_end, fuel
        if list_tag_for_value(xs) is None:
            return Err(ErrCode.TYPE, "MapList source must be a typed list"), body_end, fuel
        try:
            out = empty_list_for_tag(n.i1)
        except ValueError as exc:
            return Err(ErrCode.TYPE, str(exc)), body_end, fuel
        for item in xs.items:
            body_binders = _with_bound(binders, n.i0, item)
            rv, end, fuel = _eval_expr(p, body_idx, env, body_binders, fuel)
            if isinstance(rv, Err):
                return rv, body_end, fuel
            if end != body_end:
                return Err(ErrCode.TYPE, "MapList body did not consume exactly one expression"), body_end, fuel
            appended = builtin_call("append", [out, rv])
            if isinstance(appended, Err):
                return appended, body_end, fuel
            out = appended
        return out, body_end, fuel

    if k == NodeKind.FILTER_LIST:
        xs, pred_idx, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        pred_end = _skip_expr_end(p, pred_idx)
        if isinstance(xs, Err):
            return xs, pred_end, fuel
        if list_tag_for_value(xs) is None:
            return Err(ErrCode.TYPE, "FilterList source must be a typed list"), pred_end, fuel
        out = empty_list_for_tag(int(list_tag_for_value(xs)))
        for item in xs.items:
            pred_binders = _with_bound(binders, n.i0, item)
            keep, end, fuel = _eval_expr(p, pred_idx, env, pred_binders, fuel)
            if isinstance(keep, Err):
                return keep, pred_end, fuel
            if end != pred_end:
                return Err(ErrCode.TYPE, "FilterList predicate did not consume exactly one expression"), pred_end, fuel
            if not isinstance(keep, bool):
                return Err(ErrCode.TYPE, "FilterList predicate must return bool"), pred_end, fuel
            if keep:
                appended = builtin_call("append", [out, item])
                if isinstance(appended, Err):
                    return appended, pred_end, fuel
                out = appended
        return out, pred_end, fuel

    if k == NodeKind.LINEAR_REC:
        linear_binders = linear_rec_binders_for_node(p, idx)
        xs, start_idx_expr, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        start_idx_end = _skip_expr_end(p, start_idx_expr)
        empty_idx = _skip_expr_end(p, start_idx_expr)
        step_idx = _skip_expr_end(p, empty_idx)
        last_idx = _skip_expr_end(p, step_idx)
        linear_end = _skip_expr_end(p, last_idx)
        if isinstance(xs, Err):
            return xs, linear_end, fuel
        if list_tag_for_value(xs) is None:
            return Err(ErrCode.TYPE, "LinearRec source must be a typed list"), linear_end, fuel

        start_idx_value, _start_end, fuel = _eval_expr(p, start_idx_expr, env, binders, fuel)
        if isinstance(start_idx_value, Err):
            return start_idx_value, linear_end, fuel
        if not isinstance(start_idx_value, int) or isinstance(start_idx_value, bool):
            return Err(ErrCode.TYPE, "LinearRec start_idx must be int"), linear_end, fuel

        if len(xs.items) == 0:
            empty_value, empty_end, fuel = _eval_expr(p, empty_idx, env, binders, fuel)
            if isinstance(empty_value, Err):
                return empty_value, linear_end, fuel
            if empty_end != step_idx:
                return Err(ErrCode.TYPE, "LinearRec empty case did not consume exactly one expression"), linear_end, fuel
            return empty_value, linear_end, fuel

        last_item = xs.items[-1]
        last_binders = _with_bound_values(
            binders,
            (
                (linear_binders.elem_name, last_item),
                (linear_binders.index_name, start_idx_value + len(xs.items) - 1),
            ),
        )
        acc, last_end, fuel = _eval_expr(p, last_idx, env, last_binders, fuel)
        if isinstance(acc, Err):
            return acc, linear_end, fuel
        if last_end != linear_end:
            return Err(ErrCode.TYPE, "LinearRec last body did not consume exactly one expression"), linear_end, fuel

        for offset in range(len(xs.items) - 2, -1, -1):
            step_binders = _with_bound_values(
                binders,
                (
                    (linear_binders.elem_name, xs.items[offset]),
                    (linear_binders.accum_name, acc),
                    (linear_binders.index_name, start_idx_value + offset),
                ),
            )
            step_value, step_end, fuel = _eval_expr(p, step_idx, env, step_binders, fuel)
            if isinstance(step_value, Err):
                return step_value, linear_end, fuel
            if step_end != last_idx:
                return Err(ErrCode.TYPE, "LinearRec step body did not consume exactly one expression"), linear_end, fuel
            if type(step_value) is not type(acc):
                return Err(ErrCode.TYPE, "LinearRec step result type must match accumulator type"), linear_end, fuel
            acc = step_value
        return acc, linear_end, fuel

    if k == NodeKind.ASGP_DC:
        source, solve_idx, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        divide_idx = _skip_expr_end(p, solve_idx)
        combine_idx = _skip_expr_end(p, divide_idx)
        dc_end = _skip_expr_end(p, combine_idx)
        if isinstance(source, Err):
            return source, dc_end, fuel
        if not _is_asgp_dc_source(source):
            return Err(ErrCode.TYPE, "ASGP-DC source must be String, IntList, FloatList, or StringList"), dc_end, fuel
        if (
            _subtree_contains_asgp(p, solve_idx)
            or _subtree_contains_asgp(p, divide_idx)
            or _subtree_contains_asgp(p, combine_idx)
        ):
            return Err(ErrCode.TYPE, "ASGP-DC phase bodies must not contain ASGP source forms"), dc_end, fuel
        out, fuel = _asgp_dc_frame(p, idx, source, 0, solve_idx, divide_idx, combine_idx, dc_end, fuel)
        return out, dc_end, fuel

    if k == NodeKind.ASGP_DP1D:
        state, solve_idx, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        transition_idx = _skip_expr_end(p, solve_idx)
        dp_end = _skip_expr_end(p, transition_idx)
        if isinstance(state, Err):
            return state, dp_end, fuel
        if not isinstance(state, int) or isinstance(state, bool):
            return Err(ErrCode.TYPE, "ASGP-DP1D state must be int"), dp_end, fuel
        if _subtree_contains_asgp(p, solve_idx) or _subtree_contains_asgp(p, transition_idx):
            return Err(ErrCode.TYPE, "ASGP-DP1D phase bodies must not contain ASGP source forms"), dp_end, fuel
        out, fuel = _asgp_dp1d_frame(p, idx, state, solve_idx, transition_idx, dp_end, {}, fuel)
        return out, dp_end, fuel

    if k == NodeKind.ASGP_DP2D:
        state_i, state_j_idx, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        state_j, solve_idx, fuel = _eval_expr(p, state_j_idx, env, binders, fuel)
        transition_idx = _skip_expr_end(p, solve_idx)
        dp_end = _skip_expr_end(p, transition_idx)
        if isinstance(state_i, Err):
            return state_i, dp_end, fuel
        if isinstance(state_j, Err):
            return state_j, dp_end, fuel
        if not isinstance(state_i, int) or isinstance(state_i, bool):
            return Err(ErrCode.TYPE, "ASGP-DP2D state_i must be int"), dp_end, fuel
        if not isinstance(state_j, int) or isinstance(state_j, bool):
            return Err(ErrCode.TYPE, "ASGP-DP2D state_j must be int"), dp_end, fuel
        if _subtree_contains_asgp(p, solve_idx) or _subtree_contains_asgp(p, transition_idx):
            return Err(ErrCode.TYPE, "ASGP-DP2D phase bodies must not contain ASGP source forms"), dp_end, fuel
        out, fuel = _asgp_dp2d_frame(p, idx, state_i, state_j, solve_idx, transition_idx, dp_end, {}, fuel)
        return out, dp_end, fuel

    if k in BUILTIN_NAME_BY_NODE:
        name = BUILTIN_NAME_BY_NODE[k]
        argc = NODE_ARITY[k]
        vals: List[Val] = []
        cur = idx + 1
        for _ in range(argc):
            rv, cur, fuel = _eval_expr(p, cur, env, binders, fuel)
            if isinstance(rv, Err):
                return rv, cur, fuel
            vals.append(rv)
        out = builtin_call(name, vals)
        return out, cur, fuel

    return Err(ErrCode.TYPE, f"unknown Expr node: {k}"), _skip_expr_end(p, idx), fuel


def _exec_stmt(p: AstProgram, idx: int, env: Env, binders: BinderEnv, fuel: int) -> tuple[Env, Out, int, int]:
    next_idx = _skip_stmt_end(p, idx)
    fuel2 = _consume_fuel(fuel)
    if isinstance(fuel2, Err):
        return env, Failed(fuel2), next_idx, 0
    fuel = fuel2

    n = p.nodes[idx]
    k = n.kind

    if k == NodeKind.ASSIGN:
        r, _j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(r, Err):
            return env, Failed(r), next_idx, fuel
        env2 = dict(env)
        env2[p.names[n.i0]] = r
        return env2, Normal(), next_idx, fuel

    if k == NodeKind.RETURN:
        r, _j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(r, Err):
            return env, Failed(r), next_idx, fuel
        return env, Returned(r), next_idx, fuel

    if k == NodeKind.IF_STMT:
        rc, j, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(rc, Err):
            return env, Failed(rc), next_idx, fuel
        if not isinstance(rc, bool):
            return env, Failed(Err(ErrCode.TYPE, "if condition must be bool")), next_idx, fuel
        if rc:
            env2, out2, _after_then, fuel = _exec_block(p, j, env, binders, fuel)
            return env2, out2, next_idx, fuel
        else_j = _skip_block_end(p, j)
        encurrent, out3, _after_else, fuel = _exec_block(p, else_j, env, binders, fuel)
        return encurrent, out3, next_idx, fuel

    if k == NodeKind.FOR_RANGE:
        bound, body_idx, fuel = _eval_expr(p, idx + 1, env, binders, fuel)
        if isinstance(bound, Err):
            return env, Failed(bound), next_idx, fuel
        if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
            return env, Failed(Err(ErrCode.TYPE, "range(e) requires non-negative int bound")), next_idx, fuel
        cur_env = dict(env)
        name = p.names[n.i0]
        for i in range(bound):
            cur_env[name] = i
            cur_env, out, _after_body, fuel = _exec_block(p, body_idx, cur_env, binders, fuel)
            if not isinstance(out, Normal):
                return cur_env, out, next_idx, fuel
        return cur_env, Normal(), next_idx, fuel

    return env, Failed(Err(ErrCode.TYPE, f"unknown Stmt node: {k}")), next_idx, fuel


def _exec_block(p: AstProgram, idx: int, env: Env, binders: BinderEnv, fuel: int) -> tuple[Env, Out, int, int]:
    n = p.nodes[idx]
    if n.kind == NodeKind.BLOCK_NIL:
        return dict(env), Normal(), idx + 1, fuel
    if n.kind != NodeKind.BLOCK_CONS:
        return env, Failed(Err(ErrCode.TYPE, f"expected Block at index {idx}, got {n.kind}")), idx, fuel

    env1, out1, j, fuel = _exec_stmt(p, idx + 1, dict(env), binders, fuel)
    if not isinstance(out1, Normal):
        return env1, out1, _skip_block_end(p, j), fuel
    return _exec_block(p, j, env1, binders, fuel)


def eval_expr(p: AstProgram, inputs: Env | None = None, fuel: int = 10_000) -> tuple[Val | Err, int]:
    validate_prefix_program(p)
    env = dict(inputs) if inputs else {}
    v, end, fuel_left = _eval_expr(p, 1, env, {}, fuel)
    if end != len(p.nodes):
        return Err(ErrCode.TYPE, "eval_expr expects single Expr root"), fuel_left
    return v, fuel_left


def run_program(p: AstProgram, inputs: Env | None = None, fuel: int = 10_000) -> tuple[Env, Out]:
    validate_prefix_program(p)
    env0 = dict(inputs) if inputs else {}
    env1, out, end, _fuel_left = _exec_block(p, 1, env0, {}, fuel)
    if end != len(p.nodes):
        return env1, Failed(Err(ErrCode.TYPE, "invalid trailing tokens"))
    if isinstance(out, Normal):
        return env1, Failed(Err(ErrCode.VALUE, "program finished without return"))
    return env1, out
