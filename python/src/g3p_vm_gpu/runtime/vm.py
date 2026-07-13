from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..core.ast import FloatList, IntList, NodeKind, StringList, Val, empty_list_for_tag, list_tag_for_value
from .builtins import BUILTIN_NAME_BY_ID, builtin_call
from .compiler import AsgpDcSegment, AsgpDp1dSegment, AsgpDp2dSegment, BytecodeProgram, Instr, PhaseProgram
from ..core.errors import Err, ErrCode
from ..core.value_semantics import compare_values, is_num, promote_numeric


@dataclass(frozen=True)
class ExecReturn:
    value: Val


@dataclass(frozen=True)
class ExecError:
    err: Err


ExecResult = ExecReturn | ExecError


AsgpDcSource = str | IntList | FloatList | StringList


def _is_asgp_dc_source(value: Val) -> bool:
    return isinstance(value, (str, IntList, FloatList, StringList))


def _asgp_dc_source_slice(value: AsgpDcSource, start: int, end: int) -> AsgpDcSource:
    if isinstance(value, str):
        return value[start:end]
    if isinstance(value, IntList):
        return IntList(tuple(value.items[start:end]))
    if isinstance(value, FloatList):
        return FloatList(tuple(value.items[start:end]))
    return StringList(tuple(value.items[start:end]))


def exec_bytecode(program: BytecodeProgram, inputs: Dict[str, Val] | None = None, fuel: int = 10_000) -> ExecResult:
    UNSET = object()

    def fail(code_: ErrCode, msg: str) -> ExecError:
        return ExecError(Err(code_, msg))

    def run_phase(phase: PhaseProgram, bindings: Dict[int, Val], fuel_left: int) -> tuple[Val | Err, int]:
        preset_locals: Dict[int, Val] = {}
        for name_idx, value in bindings.items():
            local_idx = phase.binder_locals.get(name_idx)
            if local_idx is not None:
                preset_locals[local_idx] = value
        return run_code(
            consts=phase.consts,
            code=phase.code,
            n_locals=phase.n_locals,
            var2idx=phase.var2idx,
            input_values=None,
            preset_locals=preset_locals,
            fuel_left=fuel_left,
            require_return=False,
        )

    def eval_asgp_dc(segment: AsgpDcSegment, source: Val, lo: int, fuel_left: int) -> tuple[Val | Err, int]:
        if fuel_left <= 0:
            return Err(ErrCode.TIMEOUT, "out of fuel"), 0
        fuel_left -= 1
        if not _is_asgp_dc_source(source):
            return Err(ErrCode.TYPE, "ASGP-DC source must be String, IntList, FloatList, or StringList"), fuel_left
        n = len(source)
        if n <= 1:
            return run_phase(
                segment.solve,
                {
                    segment.solve_xs_name: source,
                    segment.solve_n_name: n,
                    segment.solve_lo_name: lo,
                },
                fuel_left,
            )

        raw_split, fuel_left = run_phase(segment.divide, {segment.divide_n_name: n}, fuel_left)
        if isinstance(raw_split, Err):
            return raw_split, fuel_left
        if not isinstance(raw_split, int) or isinstance(raw_split, bool):
            return Err(ErrCode.TYPE, "ASGP-DC divide phase must return int"), fuel_left

        split = max(1, min(raw_split, n - 1))
        left, fuel_left = eval_asgp_dc(segment, _asgp_dc_source_slice(source, 0, split), lo, fuel_left)
        if isinstance(left, Err):
            return left, fuel_left
        right, fuel_left = eval_asgp_dc(segment, _asgp_dc_source_slice(source, split, n), lo + split, fuel_left)
        if isinstance(right, Err):
            return right, fuel_left
        if type(left) is not type(right):
            return Err(ErrCode.TYPE, "ASGP-DC recursive results must have matching types"), fuel_left
        out, fuel_left = run_phase(
            segment.combine,
            {
                segment.combine_left_name: left,
                segment.combine_right_name: right,
            },
            fuel_left,
        )
        if isinstance(out, Err):
            return out, fuel_left
        if type(out) is not type(left):
            return Err(ErrCode.TYPE, "ASGP-DC combine result type must match recursive result type"), fuel_left
        return out, fuel_left

    def asgp_dp1d_deps(segment: AsgpDp1dSegment, state: int) -> tuple[int, ...]:
        if segment.dep_kind in {NodeKind.DP1_BACKWARD1, NodeKind.DP1_BACKWARD2, NodeKind.DP1_BACKWARD3}:
            return tuple(state - offset for offset in segment.dep_offsets)
        return tuple(state + offset for offset in segment.dep_offsets)

    def asgp_dp2d_deps(segment: AsgpDp2dSegment, i: int, j: int) -> tuple[tuple[int, int], ...]:
        if segment.dep_kind == NodeKind.DP2_CROSS_BACKWARD:
            return ((i - 1, j), (i, j - 1))
        if segment.dep_kind == NodeKind.DP2_CROSS_FORWARD:
            return ((i + 1, j), (i, j + 1))
        if segment.dep_kind == NodeKind.DP2_DIAGONAL_BACKWARD:
            return ((i - 1, j - 1),)
        if segment.dep_kind == NodeKind.DP2_DIAGONAL_FORWARD:
            return ((i + 1, j + 1),)
        if segment.dep_kind == NodeKind.DP2_NEIGHBORHOOD_BACKWARD3:
            return ((i - 1, j), (i, j - 1), (i - 1, j - 1))
        if segment.dep_kind == NodeKind.DP2_NEIGHBORHOOD_FORWARD3:
            return ((i + 1, j), (i, j + 1), (i + 1, j + 1))
        raise ValueError(f"unknown ASGP-DP2D dependency kind: {segment.dep_kind}")

    def eval_asgp_dp1d(
        segment: AsgpDp1dSegment,
        state: int,
        memo: Dict[int, Val],
        fuel_left: int,
    ) -> tuple[Val | Err, int]:
        if fuel_left <= 0:
            return Err(ErrCode.TIMEOUT, "out of fuel"), 0
        fuel_left -= 1

        if state < segment.lo or state > segment.hi:
            return segment.boundary_value, fuel_left

        if state == segment.base_state:
            return run_phase(segment.solve, {segment.solve_state_name: state}, fuel_left)

        cached = memo.get(state)
        if cached is not None:
            return cached, fuel_left

        dep_values: List[Val] = []
        for dep_state in asgp_dp1d_deps(segment, state):
            dep_value, fuel_left = eval_asgp_dp1d(segment, dep_state, memo, fuel_left)
            if isinstance(dep_value, Err):
                return dep_value, fuel_left
            dep_values.append(dep_value)
        if dep_values and any(type(value) is not type(dep_values[0]) for value in dep_values[1:]):
            return Err(ErrCode.TYPE, "ASGP-DP1D dependency result types must match"), fuel_left

        bindings = {segment.transition_state_name: state}
        for name_idx, value in zip(segment.transition_dep_names, dep_values):
            bindings[name_idx] = value
        out, fuel_left = run_phase(segment.transition, bindings, fuel_left)
        if isinstance(out, Err):
            return out, fuel_left
        if dep_values and type(out) is not type(dep_values[0]):
            return Err(ErrCode.TYPE, "ASGP-DP1D transition result type must match dependency result type"), fuel_left
        memo[state] = out
        return out, fuel_left

    def eval_asgp_dp2d(
        segment: AsgpDp2dSegment,
        i: int,
        j: int,
        memo: Dict[tuple[int, int], Val],
        fuel_left: int,
    ) -> tuple[Val | Err, int]:
        if fuel_left <= 0:
            return Err(ErrCode.TIMEOUT, "out of fuel"), 0
        fuel_left -= 1

        if i < segment.i_lo or i > segment.i_hi or j < segment.j_lo or j > segment.j_hi:
            return segment.boundary_value, fuel_left

        if i == segment.base_i and j == segment.base_j:
            return run_phase(
                segment.solve,
                {
                    segment.solve_i_name: i,
                    segment.solve_j_name: j,
                },
                fuel_left,
            )

        key = (i, j)
        cached = memo.get(key)
        if cached is not None:
            return cached, fuel_left

        dep_values: List[Val] = []
        for dep_i, dep_j in asgp_dp2d_deps(segment, i, j):
            dep_value, fuel_left = eval_asgp_dp2d(segment, dep_i, dep_j, memo, fuel_left)
            if isinstance(dep_value, Err):
                return dep_value, fuel_left
            dep_values.append(dep_value)
        if dep_values and any(type(value) is not type(dep_values[0]) for value in dep_values[1:]):
            return Err(ErrCode.TYPE, "ASGP-DP2D dependency result types must match"), fuel_left

        bindings = {
            segment.transition_i_name: i,
            segment.transition_j_name: j,
        }
        for name_idx, value in zip(segment.transition_dep_names, dep_values):
            bindings[name_idx] = value
        out, fuel_left = run_phase(segment.transition, bindings, fuel_left)
        if isinstance(out, Err):
            return out, fuel_left
        if dep_values and type(out) is not type(dep_values[0]):
            return Err(ErrCode.TYPE, "ASGP-DP2D transition result type must match dependency result type"), fuel_left
        memo[key] = out
        return out, fuel_left

    def run_code(
        consts: List[Val],
        code: List[Instr],
        n_locals: int,
        var2idx: Dict[str, int],
        input_values: Dict[str, Val] | None,
        preset_locals: Dict[int, Val],
        fuel_left: int,
        require_return: bool,
    ) -> tuple[Val | Err, int]:
        stack: List[Val] = []
        locals_: List[Val | object] = [UNSET for _ in range(n_locals)]

        if input_values:
            for name, value in input_values.items():
                idx = var2idx.get(name)
                if idx is not None:
                    locals_[idx] = value
        for idx, value in preset_locals.items():
            if idx < 0 or idx >= len(locals_):
                return Err(ErrCode.NAME, "local index out of range"), fuel_left
            locals_[idx] = value

        ip = 0
        while ip < len(code):
            if fuel_left <= 0:
                return Err(ErrCode.TIMEOUT, "out of fuel"), 0
            fuel_left -= 1

            ins = code[ip]
            ip += 1

            if ins.op == "PUSH_CONST":
                if ins.a is None or ins.a < 0 or ins.a >= len(consts):
                    return Err(ErrCode.VALUE, "const index out of range"), fuel_left
                stack.append(consts[ins.a])
                continue

            if ins.op == "LOAD":
                if ins.a is None or ins.a < 0 or ins.a >= len(locals_):
                    return Err(ErrCode.NAME, "local index out of range"), fuel_left
                v = locals_[ins.a]
                if v is UNSET:
                    return Err(ErrCode.NAME, "read of uninitialized local"), fuel_left
                stack.append(v)  # type: ignore[arg-type]
                continue

            if ins.op == "STORE":
                if ins.a is None or ins.a < 0 or ins.a >= len(locals_):
                    return Err(ErrCode.NAME, "local index out of range"), fuel_left
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                locals_[ins.a] = stack.pop()
                continue

            if ins.op == "CHECK_LIST":
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                if list_tag_for_value(stack[-1]) is None:
                    return Err(ErrCode.TYPE, "structured list source must be a typed list"), fuel_left
                continue

            if ins.op == "CHECK_INT":
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                x = stack[-1]
                if not isinstance(x, int) or isinstance(x, bool):
                    return Err(ErrCode.TYPE, "expected int"), fuel_left
                continue

            if ins.op == "EMPTY_LIST":
                if ins.a is None:
                    return Err(ErrCode.TYPE, "EMPTY_LIST requires list type tag"), fuel_left
                try:
                    stack.append(empty_list_for_tag(ins.a))
                except ValueError as exc:
                    return Err(ErrCode.TYPE, str(exc)), fuel_left
                continue

            if ins.op == "EMPTY_LIST_LIKE":
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                source = stack.pop()
                tag = list_tag_for_value(source)
                if tag is None:
                    return Err(ErrCode.TYPE, "EMPTY_LIST_LIKE expects typed list"), fuel_left
                stack.append(empty_list_for_tag(int(tag)))
                continue

            if ins.op in {"NEG", "NOT"}:
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                x = stack.pop()
                if ins.op == "NEG":
                    if not is_num(x):
                        return Err(ErrCode.TYPE, "NEG expects numeric"), fuel_left
                    stack.append(-x)  # type: ignore[operator]
                else:
                    if not isinstance(x, bool):
                        return Err(ErrCode.TYPE, "NOT expects bool"), fuel_left
                    stack.append(not x)
                continue

            if ins.op in {"ADD", "SUB", "MUL", "DIV", "MOD"}:
                if len(stack) < 2:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                b = stack.pop()
                a = stack.pop()
                prom = promote_numeric(a, b)
                if isinstance(prom, Err):
                    return Err(ErrCode.TYPE, f"{ins.op} expects numeric operands"), fuel_left
                a2, b2 = prom
                if ins.op == "ADD":
                    stack.append(a2 + b2)
                elif ins.op == "SUB":
                    stack.append(a2 - b2)
                elif ins.op == "MUL":
                    stack.append(a2 * b2)
                elif ins.op == "DIV":
                    if b2 == 0:
                        return Err(ErrCode.ZERODIV, "division by zero"), fuel_left
                    stack.append(float(a2) / float(b2))
                else:
                    if b2 == 0:
                        return Err(ErrCode.ZERODIV, "modulo by zero"), fuel_left
                    stack.append(a2 % b2)
                continue

            if ins.op in {"LT", "LE", "GT", "GE", "EQ", "NE"}:
                if len(stack) < 2:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                b = stack.pop()
                a = stack.pop()
                r = compare_values(ins.op, a, b)
                if isinstance(r, Err):
                    return r, fuel_left
                stack.append(r)
                continue

            if ins.op == "JMP":
                if ins.a is None or ins.a < 0 or ins.a > len(code):
                    return Err(ErrCode.VALUE, "jump target out of range"), fuel_left
                ip = ins.a
                continue

            if ins.op in {"JMP_IF_FALSE", "JMP_IF_TRUE"}:
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                if ins.a is None or ins.a < 0 or ins.a > len(code):
                    return Err(ErrCode.VALUE, "jump target out of range"), fuel_left
                c = stack.pop()
                if not isinstance(c, bool):
                    return Err(ErrCode.TYPE, "jump condition must be bool"), fuel_left
                if ins.op == "JMP_IF_FALSE" and not c:
                    ip = ins.a
                if ins.op == "JMP_IF_TRUE" and c:
                    ip = ins.a
                continue

            if ins.op == "CALL_BUILTIN":
                bid = ins.a if ins.a is not None else -1
                argc = ins.b if ins.b is not None else -1
                if argc < 0:
                    return Err(ErrCode.TYPE, "invalid builtin argc"), fuel_left
                if len(stack) < argc:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                args = stack[-argc:]
                del stack[-argc:]
                name = BUILTIN_NAME_BY_ID.get(bid)
                if name is None:
                    return Err(ErrCode.NAME, "unknown builtin id"), fuel_left
                out = builtin_call(name, args)
                if isinstance(out, Err):
                    return out, fuel_left
                stack.append(out)
                continue

            if ins.op == "ASGP_DC":
                if ins.a is None or ins.a < 0 or ins.a >= len(program.asgp_dc_segments):
                    return Err(ErrCode.VALUE, "ASGP-DC segment index out of range"), fuel_left
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                out, fuel_left = eval_asgp_dc(program.asgp_dc_segments[ins.a], stack.pop(), 0, fuel_left)
                if isinstance(out, Err):
                    return out, fuel_left
                stack.append(out)
                continue

            if ins.op == "ASGP_DP1D":
                if ins.a is None or ins.a < 0 or ins.a >= len(program.asgp_dp1d_segments):
                    return Err(ErrCode.VALUE, "ASGP-DP1D segment index out of range"), fuel_left
                if not stack:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                state = stack.pop()
                if not isinstance(state, int) or isinstance(state, bool):
                    return Err(ErrCode.TYPE, "ASGP-DP1D state must be int"), fuel_left
                out, fuel_left = eval_asgp_dp1d(program.asgp_dp1d_segments[ins.a], state, {}, fuel_left)
                if isinstance(out, Err):
                    return out, fuel_left
                stack.append(out)
                continue

            if ins.op == "ASGP_DP2D":
                if ins.a is None or ins.a < 0 or ins.a >= len(program.asgp_dp2d_segments):
                    return Err(ErrCode.VALUE, "ASGP-DP2D segment index out of range"), fuel_left
                if len(stack) < 2:
                    return Err(ErrCode.VALUE, "stack underflow"), fuel_left
                state_j = stack.pop()
                state_i = stack.pop()
                if not isinstance(state_i, int) or isinstance(state_i, bool):
                    return Err(ErrCode.TYPE, "ASGP-DP2D state_i must be int"), fuel_left
                if not isinstance(state_j, int) or isinstance(state_j, bool):
                    return Err(ErrCode.TYPE, "ASGP-DP2D state_j must be int"), fuel_left
                out, fuel_left = eval_asgp_dp2d(program.asgp_dp2d_segments[ins.a], state_i, state_j, {}, fuel_left)
                if isinstance(out, Err):
                    return out, fuel_left
                stack.append(out)
                continue

            if ins.op == "RETURN":
                if not stack:
                    return Err(ErrCode.VALUE, "return requires value on stack"), fuel_left
                return stack.pop(), fuel_left

            return Err(ErrCode.TYPE, f"unknown opcode: {ins.op}"), fuel_left

        if require_return:
            return Err(ErrCode.VALUE, "program finished without return"), fuel_left
        if not stack:
            return Err(ErrCode.VALUE, "expression segment produced no value"), fuel_left
        return stack.pop(), fuel_left

    out, _fuel_left = run_code(
        consts=program.consts,
        code=program.code,
        n_locals=program.n_locals,
        var2idx=program.var2idx,
        input_values=inputs,
        preset_locals={},
        fuel_left=fuel,
        require_return=True,
    )
    if isinstance(out, Err):
        return ExecError(out)
    return ExecReturn(out)
