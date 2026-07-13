from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..core.ast import (
    BUILTIN_NAME_BY_NODE,
    NODE_ARITY,
    AstProgram,
    ListTypeTag,
    NodeKind,
    Val,
    asgp_dc_binders_for_node,
    asgp_dp1d_spec_for_node,
    asgp_dp2d_spec_for_node,
    linear_rec_binders_for_node,
    validate_prefix_program,
)
from .builtins import BUILTIN_ID_BY_NAME


@dataclass(frozen=True)
class Instr:
    op: str
    a: int | None = None
    b: int | None = None


@dataclass(frozen=True)
class PhaseProgram:
    consts: List[Val]
    code: List[Instr]
    n_locals: int
    var2idx: Dict[str, int]
    binder_locals: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AsgpDcSegment:
    solve_xs_name: int
    solve_n_name: int
    solve_lo_name: int
    divide_n_name: int
    combine_left_name: int
    combine_right_name: int
    solve: PhaseProgram
    divide: PhaseProgram
    combine: PhaseProgram


@dataclass(frozen=True)
class AsgpDp1dSegment:
    lo: int
    hi: int
    base_state: int
    boundary_value: Val
    dep_kind: NodeKind
    dep_offsets: tuple[int, ...]
    solve_state_name: int
    transition_state_name: int
    transition_dep_names: tuple[int, ...]
    solve: PhaseProgram
    transition: PhaseProgram


@dataclass(frozen=True)
class AsgpDp2dSegment:
    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    base_i: int
    base_j: int
    boundary_value: Val
    dep_kind: NodeKind
    solve_i_name: int
    solve_j_name: int
    transition_i_name: int
    transition_j_name: int
    transition_dep_names: tuple[int, ...]
    solve: PhaseProgram
    transition: PhaseProgram


@dataclass(frozen=True)
class BytecodeProgram:
    consts: List[Val]
    code: List[Instr]
    n_locals: int
    var2idx: Dict[str, int]
    asgp_dc_segments: List[AsgpDcSegment] = field(default_factory=list)
    asgp_dp1d_segments: List[AsgpDp1dSegment] = field(default_factory=list)
    asgp_dp2d_segments: List[AsgpDp2dSegment] = field(default_factory=list)


class _Compiler:
    def __init__(self, p: AstProgram) -> None:
        self.p = p
        self.consts: List[Val] = []
        self.code: List[Instr | tuple[str, str, int | None]] = []
        self.labels: Dict[str, int] = {}
        self.var2idx: Dict[str, int] = {}
        self._binder_stack: Dict[int, List[int]] = {}
        self.asgp_dc_segments: List[AsgpDcSegment] = []
        self.asgp_dp1d_segments: List[AsgpDp1dSegment] = []
        self.asgp_dp2d_segments: List[AsgpDp2dSegment] = []
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
        name = f"\x00tmp_{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def _push_binder(self, name_idx: int, local_idx: int) -> None:
        self._binder_stack.setdefault(name_idx, []).append(local_idx)

    def _pop_binder(self, name_idx: int) -> None:
        stack = self._binder_stack.get(name_idx)
        if not stack:
            raise ValueError("internal binder stack underflow")
        stack.pop()

    def _bound_local(self, name_idx: int) -> int:
        stack = self._binder_stack.get(name_idx)
        if not stack:
            raise ValueError(f"undefined binder: {self.p.names[name_idx]}")
        return stack[-1]

    def _compile_for_loop_body(self, bound_local: int, user_local: int, body_idx: int) -> int:
        idx_0 = self._const(0)
        idx_1 = self._const(1)
        counter_i = self._local(self._new_temp())
        loop_l = self._new_label("for_loop")
        end_l = self._new_label("for_end")

        self._emit("PUSH_CONST", idx_0)
        self._emit("STORE", counter_i)
        self._label(loop_l)
        self._emit("LOAD", counter_i)
        self._emit("LOAD", bound_local)
        self._emit("LT")
        self._emit_jump("JMP_IF_FALSE", end_l)
        self._emit("LOAD", counter_i)
        self._emit("STORE", user_local)
        j = self._compile_block(body_idx)
        self._emit("LOAD", counter_i)
        self._emit("PUSH_CONST", idx_1)
        self._emit("ADD")
        self._emit("STORE", counter_i)
        self._emit_jump("JMP", loop_l)
        self._label(end_l)
        return j

    def _compile_map_list(self, idx: int) -> int:
        n = self.p.nodes[idx]
        xs_local = self._local(self._new_temp())
        out_local = self._local(self._new_temp())
        val_local = self._local(self._new_temp())
        counter_i = self._local(self._new_temp())
        binder_local = self._local(self._new_temp())
        loop_l = self._new_label("map_loop")
        end_l = self._new_label("map_end")

        body_idx = self._compile_expr(idx + 1)
        body_end = self._expr_end(body_idx)

        self._emit("CHECK_LIST")
        self._emit("STORE", xs_local)
        self._emit("EMPTY_LIST", n.i1)
        self._emit("STORE", out_local)
        self._emit("PUSH_CONST", self._const(0))
        self._emit("STORE", counter_i)

        self._label(loop_l)
        self._emit("LOAD", counter_i)
        self._emit("LOAD", xs_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["len"], 1)
        self._emit("LT")
        self._emit_jump("JMP_IF_FALSE", end_l)
        self._emit("LOAD", xs_local)
        self._emit("LOAD", counter_i)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["index"], 2)
        self._emit("STORE", binder_local)
        self._push_binder(n.i0, binder_local)
        compiled_body_end = self._compile_expr(body_idx)
        self._pop_binder(n.i0)
        if compiled_body_end != body_end:
            raise ValueError("MapList body did not consume exactly one expression")
        self._emit("STORE", val_local)
        self._emit("LOAD", out_local)
        self._emit("LOAD", val_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["append"], 2)
        self._emit("STORE", out_local)
        self._emit("LOAD", counter_i)
        self._emit("PUSH_CONST", self._const(1))
        self._emit("ADD")
        self._emit("STORE", counter_i)
        self._emit_jump("JMP", loop_l)
        self._label(end_l)
        self._emit("LOAD", out_local)
        return body_end

    def _compile_filter_list(self, idx: int) -> int:
        n = self.p.nodes[idx]
        xs_local = self._local(self._new_temp())
        out_local = self._local(self._new_temp())
        counter_i = self._local(self._new_temp())
        binder_local = self._local(self._new_temp())
        loop_l = self._new_label("filter_loop")
        skip_l = self._new_label("filter_skip")
        end_l = self._new_label("filter_end")

        pred_idx = self._compile_expr(idx + 1)
        pred_end = self._expr_end(pred_idx)

        self._emit("CHECK_LIST")
        self._emit("STORE", xs_local)
        self._emit("LOAD", xs_local)
        self._emit("EMPTY_LIST_LIKE")
        self._emit("STORE", out_local)
        self._emit("PUSH_CONST", self._const(0))
        self._emit("STORE", counter_i)

        self._label(loop_l)
        self._emit("LOAD", counter_i)
        self._emit("LOAD", xs_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["len"], 1)
        self._emit("LT")
        self._emit_jump("JMP_IF_FALSE", end_l)
        self._emit("LOAD", xs_local)
        self._emit("LOAD", counter_i)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["index"], 2)
        self._emit("STORE", binder_local)
        self._push_binder(n.i0, binder_local)
        compiled_pred_end = self._compile_expr(pred_idx)
        self._pop_binder(n.i0)
        if compiled_pred_end != pred_end:
            raise ValueError("FilterList predicate did not consume exactly one expression")
        self._emit_jump("JMP_IF_FALSE", skip_l)
        self._emit("LOAD", out_local)
        self._emit("LOAD", binder_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["append"], 2)
        self._emit("STORE", out_local)
        self._label(skip_l)
        self._emit("LOAD", counter_i)
        self._emit("PUSH_CONST", self._const(1))
        self._emit("ADD")
        self._emit("STORE", counter_i)
        self._emit_jump("JMP", loop_l)
        self._label(end_l)
        self._emit("LOAD", out_local)
        return pred_end

    def _compile_linear_rec(self, idx: int) -> int:
        binders = linear_rec_binders_for_node(self.p, idx)
        xs_local = self._local(self._new_temp())
        start_local = self._local(self._new_temp())
        len_local = self._local(self._new_temp())
        counter_local = self._local(self._new_temp())
        elem_local = self._local(self._new_temp())
        accum_local = self._local(self._new_temp())
        index_local = self._local(self._new_temp())
        nonempty_l = self._new_label("linear_nonempty")
        step_check_l = self._new_label("linear_step_check")
        done_l = self._new_label("linear_done")
        end_l = self._new_label("linear_end")

        source_idx = idx + 1
        start_idx = self._compile_expr(source_idx)
        empty_idx = self._compile_expr(start_idx)
        step_idx = self._expr_end(empty_idx)
        last_idx = self._expr_end(step_idx)
        linear_end = self._expr_end(last_idx)

        self._emit("CHECK_INT")
        self._emit("STORE", start_local)
        self._emit("CHECK_LIST")
        self._emit("STORE", xs_local)
        self._emit("LOAD", xs_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["len"], 1)
        self._emit("STORE", len_local)
        self._emit("LOAD", len_local)
        self._emit("PUSH_CONST", self._const(0))
        self._emit("EQ")
        self._emit_jump("JMP_IF_FALSE", nonempty_l)
        compiled_empty_end = self._compile_expr(empty_idx)
        if compiled_empty_end != step_idx:
            raise ValueError("LinearRec empty case did not consume exactly one expression")
        self._emit_jump("JMP", end_l)

        self._label(nonempty_l)
        self._emit("LOAD", len_local)
        self._emit("PUSH_CONST", self._const(1))
        self._emit("SUB")
        self._emit("STORE", counter_local)
        self._emit("LOAD", xs_local)
        self._emit("LOAD", counter_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["index"], 2)
        self._emit("STORE", elem_local)
        self._emit("LOAD", start_local)
        self._emit("LOAD", counter_local)
        self._emit("ADD")
        self._emit("STORE", index_local)
        self._push_binder(binders.elem_name, elem_local)
        self._push_binder(binders.index_name, index_local)
        compiled_last_end = self._compile_expr(last_idx)
        self._pop_binder(binders.index_name)
        self._pop_binder(binders.elem_name)
        if compiled_last_end != linear_end:
            raise ValueError("LinearRec last body did not consume exactly one expression")
        self._emit("STORE", accum_local)

        self._label(step_check_l)
        self._emit("LOAD", counter_local)
        self._emit("PUSH_CONST", self._const(0))
        self._emit("GT")
        self._emit_jump("JMP_IF_FALSE", done_l)
        self._emit("LOAD", counter_local)
        self._emit("PUSH_CONST", self._const(1))
        self._emit("SUB")
        self._emit("STORE", counter_local)
        self._emit("LOAD", xs_local)
        self._emit("LOAD", counter_local)
        self._emit("CALL_BUILTIN", BUILTIN_ID_BY_NAME["index"], 2)
        self._emit("STORE", elem_local)
        self._emit("LOAD", start_local)
        self._emit("LOAD", counter_local)
        self._emit("ADD")
        self._emit("STORE", index_local)
        self._push_binder(binders.elem_name, elem_local)
        self._push_binder(binders.accum_name, accum_local)
        self._push_binder(binders.index_name, index_local)
        compiled_step_end = self._compile_expr(step_idx)
        self._pop_binder(binders.index_name)
        self._pop_binder(binders.accum_name)
        self._pop_binder(binders.elem_name)
        if compiled_step_end != last_idx:
            raise ValueError("LinearRec step body did not consume exactly one expression")
        self._emit("STORE", accum_local)
        self._emit_jump("JMP", step_check_l)

        self._label(done_l)
        self._emit("LOAD", accum_local)
        self._label(end_l)
        return linear_end

    def _subtree_contains_asgp(self, idx: int) -> bool:
        end = self._expr_end(idx)
        return any(
            node.kind in {NodeKind.ASGP_DC, NodeKind.ASGP_DP1D, NodeKind.ASGP_DP2D}
            for node in self.p.nodes[idx:end]
        )

    def _compile_phase_expr(self, expr_idx: int, expected_end: int, binder_names: tuple[int, ...]) -> PhaseProgram:
        phase = _Compiler(self.p)
        binder_locals: Dict[int, int] = {}
        for offset, name_idx in enumerate(binder_names):
            local_idx = phase._local(f"\x00asgp_binder_{offset}_{name_idx}")
            binder_locals[name_idx] = local_idx
            phase._push_binder(name_idx, local_idx)
        end = phase._compile_expr(expr_idx)
        for name_idx in reversed(binder_names):
            phase._pop_binder(name_idx)
        if end != expected_end:
            raise ValueError("ASGP-DC phase did not consume exactly one expression")
        return PhaseProgram(
            consts=phase.consts,
            code=phase._resolved_code(),
            n_locals=len(phase.var2idx),
            var2idx=dict(phase.var2idx),
            binder_locals=binder_locals,
        )

    def _compile_asgp_dc(self, idx: int) -> int:
        binders = asgp_dc_binders_for_node(self.p, idx)
        source_idx = idx + 1
        solve_idx = self._compile_expr(source_idx)
        divide_idx = self._expr_end(solve_idx)
        combine_idx = self._expr_end(divide_idx)
        dc_end = self._expr_end(combine_idx)
        if (
            self._subtree_contains_asgp(solve_idx)
            or self._subtree_contains_asgp(divide_idx)
            or self._subtree_contains_asgp(combine_idx)
        ):
            raise ValueError("ASGP-DC phase bodies must not contain ASGP source forms")
        segment = AsgpDcSegment(
            solve_xs_name=binders.solve_xs_name,
            solve_n_name=binders.solve_n_name,
            solve_lo_name=binders.solve_lo_name,
            divide_n_name=binders.divide_n_name,
            combine_left_name=binders.combine_left_name,
            combine_right_name=binders.combine_right_name,
            solve=self._compile_phase_expr(
                solve_idx,
                divide_idx,
                (binders.solve_xs_name, binders.solve_n_name, binders.solve_lo_name),
            ),
            divide=self._compile_phase_expr(divide_idx, combine_idx, (binders.divide_n_name,)),
            combine=self._compile_phase_expr(
                combine_idx,
                dc_end,
                (binders.combine_left_name, binders.combine_right_name),
            ),
        )
        segment_idx = len(self.asgp_dc_segments)
        self.asgp_dc_segments.append(segment)
        self._emit("ASGP_DC", segment_idx)
        return dc_end

    def _compile_asgp_dp1d(self, idx: int) -> int:
        spec = asgp_dp1d_spec_for_node(self.p, idx)
        state_idx = idx + 1
        solve_idx = self._compile_expr(state_idx)
        transition_idx = self._expr_end(solve_idx)
        dp_end = self._expr_end(transition_idx)
        if self._subtree_contains_asgp(solve_idx) or self._subtree_contains_asgp(transition_idx):
            raise ValueError("ASGP-DP1D phase bodies must not contain ASGP source forms")
        segment = AsgpDp1dSegment(
            lo=spec.lo,
            hi=spec.hi,
            base_state=spec.base_state,
            boundary_value=self.p.consts[spec.boundary_const],
            dep_kind=spec.dep_kind,
            dep_offsets=spec.dep_offsets,
            solve_state_name=spec.solve_state_name,
            transition_state_name=spec.transition_state_name,
            transition_dep_names=spec.transition_dep_names,
            solve=self._compile_phase_expr(solve_idx, transition_idx, (spec.solve_state_name,)),
            transition=self._compile_phase_expr(
                transition_idx,
                dp_end,
                (spec.transition_state_name, *spec.transition_dep_names),
            ),
        )
        segment_idx = len(self.asgp_dp1d_segments)
        self.asgp_dp1d_segments.append(segment)
        self._emit("ASGP_DP1D", segment_idx)
        return dp_end

    def _compile_asgp_dp2d(self, idx: int) -> int:
        spec = asgp_dp2d_spec_for_node(self.p, idx)
        state_i_idx = idx + 1
        state_j_idx = self._compile_expr(state_i_idx)
        solve_idx = self._compile_expr(state_j_idx)
        transition_idx = self._expr_end(solve_idx)
        dp_end = self._expr_end(transition_idx)
        if self._subtree_contains_asgp(solve_idx) or self._subtree_contains_asgp(transition_idx):
            raise ValueError("ASGP-DP2D phase bodies must not contain ASGP source forms")
        segment = AsgpDp2dSegment(
            i_lo=spec.i_lo,
            i_hi=spec.i_hi,
            j_lo=spec.j_lo,
            j_hi=spec.j_hi,
            base_i=spec.base_i,
            base_j=spec.base_j,
            boundary_value=self.p.consts[spec.boundary_const],
            dep_kind=spec.dep_kind,
            solve_i_name=spec.solve_i_name,
            solve_j_name=spec.solve_j_name,
            transition_i_name=spec.transition_i_name,
            transition_j_name=spec.transition_j_name,
            transition_dep_names=spec.transition_dep_names,
            solve=self._compile_phase_expr(solve_idx, transition_idx, (spec.solve_i_name, spec.solve_j_name)),
            transition=self._compile_phase_expr(
                transition_idx,
                dp_end,
                (spec.transition_i_name, spec.transition_j_name, *spec.transition_dep_names),
            ),
        )
        segment_idx = len(self.asgp_dp2d_segments)
        self.asgp_dp2d_segments.append(segment)
        self._emit("ASGP_DP2D", segment_idx)
        return dp_end

    def _expr_end(self, idx: int) -> int:
        arity = NODE_ARITY[self.p.nodes[idx].kind]
        cur = idx + 1
        for _ in range(arity):
            cur = self._expr_end(cur)
        return cur

    def _compile_expr(self, idx: int) -> int:
        n = self.p.nodes[idx]
        k = n.kind

        if k == NodeKind.CONST:
            self._emit("PUSH_CONST", self._const(self.p.consts[n.i0]))
            return idx + 1

        if k == NodeKind.VAR:
            self._emit("LOAD", self._local(self.p.names[n.i0]))
            return idx + 1

        if k == NodeKind.BOUND_VAR:
            self._emit("LOAD", self._bound_local(n.i0))
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

        if k == NodeKind.MAP_LIST:
            if n.i1 not in set(ListTypeTag):
                raise ValueError(f"invalid MapList output tag: {n.i1}")
            return self._compile_map_list(idx)

        if k == NodeKind.FILTER_LIST:
            return self._compile_filter_list(idx)

        if k == NodeKind.LINEAR_REC:
            return self._compile_linear_rec(idx)

        if k == NodeKind.ASGP_DC:
            return self._compile_asgp_dc(idx)

        if k == NodeKind.ASGP_DP1D:
            return self._compile_asgp_dp1d(idx)

        if k == NodeKind.ASGP_DP2D:
            return self._compile_asgp_dp2d(idx)

        if k in BUILTIN_NAME_BY_NODE:
            argc = NODE_ARITY[k]
            cur = idx + 1
            for _ in range(argc):
                cur = self._compile_expr(cur)
            bid = BUILTIN_ID_BY_NAME[BUILTIN_NAME_BY_NODE[k]]
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
            bound_local = self._local(self._new_temp())
            user_i = self._local(self.p.names[n.i0])
            valid_l = self._new_label("for_valid")
            j = self._compile_expr(idx + 1)
            self._emit("STORE", bound_local)
            self._emit("LOAD", bound_local)
            self._emit("CALL_BUILTIN", 12, 1)
            self._emit_jump("JMP_IF_FALSE", valid_l + "_bad")
            self._emit("LOAD", bound_local)
            self._emit("PUSH_CONST", self._const(0))
            self._emit("LT")
            self._emit_jump("JMP_IF_FALSE", valid_l)
            self._label(valid_l + "_bad")
            self._emit("PUSH_CONST", self._const(True))
            self._emit("NEG")
            self._label(valid_l)
            return self._compile_for_loop_body(bound_local, user_i, j)

        raise ValueError(f"expected Stmt at index {idx}, got {k}")

    def _compile_block(self, idx: int) -> int:
        n = self.p.nodes[idx]
        if n.kind == NodeKind.BLOCK_NIL:
            return idx + 1
        if n.kind != NodeKind.BLOCK_CONS:
            raise ValueError(f"expected Block at index {idx}, got {n.kind}")
        j = self._compile_stmt(idx + 1)
        return self._compile_block(j)

    def _resolved_code(self) -> List[Instr]:
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
        return out_code

    def finalize(self) -> BytecodeProgram:
        return BytecodeProgram(
            consts=self.consts,
            code=self._resolved_code(),
            n_locals=len(self.var2idx),
            var2idx=dict(self.var2idx),
            asgp_dc_segments=list(self.asgp_dc_segments),
            asgp_dp1d_segments=list(self.asgp_dp1d_segments),
            asgp_dp2d_segments=list(self.asgp_dp2d_segments),
        )


def compile_program(p: AstProgram) -> BytecodeProgram:
    validate_prefix_program(p)
    c = _Compiler(p)
    end = c._compile_block(1)
    if end != len(p.nodes):
        raise ValueError("invalid trailing tokens")
    return c.finalize()
