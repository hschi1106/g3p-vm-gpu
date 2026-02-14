from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .ast import Val
from .builtins import builtin_call
from .compiler import BytecodeProgram
from .errors import Err, ErrCode


@dataclass(frozen=True)
class VMReturn:
    value: Val


@dataclass(frozen=True)
class VMError:
    err: Err


VMResult = VMReturn | VMError


def _is_num(v: Val) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _promote(a: int | float, b: int | float) -> tuple[int | float, int | float]:
    if isinstance(a, float) or isinstance(b, float):
        return float(a), float(b)
    return int(a), int(b)


def _cmp(op: str, a: Val, b: Val) -> bool | Err:
    if _is_num(a) and _is_num(b):
        a2, b2 = _promote(a, b)  # type: ignore[arg-type]
        if op == "LT":
            return a2 < b2
        if op == "LE":
            return a2 <= b2
        if op == "GT":
            return a2 > b2
        if op == "GE":
            return a2 >= b2
        if op == "EQ":
            return a2 == b2
        if op == "NE":
            return a2 != b2

    if isinstance(a, bool) and isinstance(b, bool):
        if op == "EQ":
            return a == b
        if op == "NE":
            return a != b
        return Err(ErrCode.TYPE, "ordering comparison on bool not supported")

    if a is None or b is None:
        if op == "EQ":
            return a is b
        if op == "NE":
            return a is not b
        return Err(ErrCode.TYPE, "ordering comparison on None not supported")

    return Err(ErrCode.TYPE, "unsupported comparison operand types")


def run_bytecode(program: BytecodeProgram, inputs: Dict[str, Val] | None = None, fuel: int = 10_000) -> VMResult:
    stack: List[Val] = []
    UNSET = object()
    locals_: List[Val | object] = [UNSET for _ in range(program.n_locals)]

    if inputs:
        for name, value in inputs.items():
            idx = program.var2idx.get(name)
            if idx is not None:
                locals_[idx] = value

    ip = 0
    code = program.code

    def fail(code_: ErrCode, msg: str) -> VMError:
        return VMError(Err(code_, msg))

    while ip < len(code):
        if fuel <= 0:
            return fail(ErrCode.TIMEOUT, "out of fuel")
        fuel -= 1

        ins = code[ip]
        ip += 1

        if ins.op == "PUSH_CONST":
            if ins.a is None or ins.a < 0 or ins.a >= len(program.consts):
                return fail(ErrCode.VALUE, "const index out of range")
            stack.append(program.consts[ins.a])
            continue

        if ins.op == "LOAD":
            if ins.a is None or ins.a < 0 or ins.a >= len(locals_):
                return fail(ErrCode.NAME, "local index out of range")
            v = locals_[ins.a]
            if v is UNSET:
                return fail(ErrCode.NAME, "read of uninitialized local")
            stack.append(v)  # type: ignore[arg-type]
            continue

        if ins.op == "STORE":
            if ins.a is None or ins.a < 0 or ins.a >= len(locals_):
                return fail(ErrCode.NAME, "local index out of range")
            if not stack:
                return fail(ErrCode.VALUE, "stack underflow")
            locals_[ins.a] = stack.pop()
            continue

        if ins.op in {"NEG", "NOT"}:
            if not stack:
                return fail(ErrCode.VALUE, "stack underflow")
            x = stack.pop()
            if ins.op == "NEG":
                if not _is_num(x):
                    return fail(ErrCode.TYPE, "NEG expects numeric")
                stack.append(-x)  # type: ignore[operator]
            else:
                if not isinstance(x, bool):
                    return fail(ErrCode.TYPE, "NOT expects bool")
                stack.append(not x)
            continue

        if ins.op in {"ADD", "SUB", "MUL", "DIV", "MOD"}:
            if len(stack) < 2:
                return fail(ErrCode.VALUE, "stack underflow")
            b = stack.pop()
            a = stack.pop()
            if not _is_num(a) or not _is_num(b):
                return fail(ErrCode.TYPE, f"{ins.op} expects numeric operands")
            a2, b2 = _promote(a, b)  # type: ignore[arg-type]
            if ins.op == "ADD":
                stack.append(a2 + b2)
            elif ins.op == "SUB":
                stack.append(a2 - b2)
            elif ins.op == "MUL":
                stack.append(a2 * b2)
            elif ins.op == "DIV":
                if b2 == 0:
                    return fail(ErrCode.ZERODIV, "division by zero")
                stack.append(float(a2) / float(b2))
            else:
                if b2 == 0:
                    return fail(ErrCode.ZERODIV, "modulo by zero")
                stack.append(a2 % b2)
            continue

        if ins.op in {"LT", "LE", "GT", "GE", "EQ", "NE"}:
            if len(stack) < 2:
                return fail(ErrCode.VALUE, "stack underflow")
            b = stack.pop()
            a = stack.pop()
            r = _cmp(ins.op, a, b)
            if isinstance(r, Err):
                return VMError(r)
            stack.append(r)
            continue

        if ins.op == "JMP":
            if ins.a is None or ins.a < 0 or ins.a > len(code):
                return fail(ErrCode.VALUE, "jump target out of range")
            ip = ins.a
            continue

        if ins.op in {"JMP_IF_FALSE", "JMP_IF_TRUE"}:
            if not stack:
                return fail(ErrCode.VALUE, "stack underflow")
            if ins.a is None or ins.a < 0 or ins.a > len(code):
                return fail(ErrCode.VALUE, "jump target out of range")
            c = stack.pop()
            if not isinstance(c, bool):
                return fail(ErrCode.TYPE, "jump condition must be bool")
            if ins.op == "JMP_IF_FALSE" and not c:
                ip = ins.a
            if ins.op == "JMP_IF_TRUE" and c:
                ip = ins.a
            continue

        if ins.op == "CALL_BUILTIN":
            bid = ins.a if ins.a is not None else -1
            argc = ins.b if ins.b is not None else -1
            if argc < 0:
                return fail(ErrCode.TYPE, "invalid builtin argc")
            if len(stack) < argc:
                return fail(ErrCode.VALUE, "stack underflow")
            args = stack[-argc:]
            del stack[-argc:]
            name = {0: "abs", 1: "min", 2: "max", 3: "clip"}.get(bid)
            if name is None:
                return fail(ErrCode.NAME, "unknown builtin id")
            out = builtin_call(name, args)
            if isinstance(out, Err):
                return VMError(out)
            stack.append(out)
            continue

        if ins.op == "RETURN":
            if not stack:
                return fail(ErrCode.VALUE, "return requires value on stack")
            return VMReturn(stack.pop())

        return fail(ErrCode.TYPE, f"unknown opcode: {ins.op}")

    return fail(ErrCode.VALUE, "program finished without return")
