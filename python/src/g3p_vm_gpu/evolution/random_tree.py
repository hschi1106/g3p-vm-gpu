from __future__ import annotations

import random
from enum import Enum
from typing import List

from ..core.ast import make_num_list, make_string_list
from .genome import Limits
from .grammar_config import DEFAULT_GRAMMAR_CONFIG, GrammarConfig


class RType(str, Enum):
    NUM = "NUM"
    BOOL = "BOOL"
    NONE = "NONE"
    STRING = "STRING"
    NUM_LIST = "NUM_LIST"
    STRING_LIST = "STRING_LIST"


_BIN_NUM = ("add", "sub", "mul", "div", "mod")
_BIN_CMP = ("lt", "le", "gt", "ge", "eq", "ne")
_BIN_BOOL = ("and", "or")
_VALUE_KEY_BY_RTYPE = {
    RType.BOOL: "bool",
    RType.NONE: "none",
    RType.STRING: "string",
    RType.NUM_LIST: "num_list",
    RType.STRING_LIST: "string_list",
}
_ALL_RTYPES = (RType.NUM, RType.BOOL, RType.NONE, RType.STRING, RType.NUM_LIST, RType.STRING_LIST)


def _grammar(grammar_config: GrammarConfig | None) -> GrammarConfig:
    return grammar_config or DEFAULT_GRAMMAR_CONFIG


def _type_enabled(grammar: GrammarConfig, value_type: RType) -> bool:
    if value_type == RType.NUM:
        return grammar.allow_num()
    return grammar.allow_value(_VALUE_KEY_BY_RTYPE[value_type])


def _enabled_types(grammar: GrammarConfig, candidates: tuple[RType, ...] = _ALL_RTYPES) -> list[RType]:
    return [value_type for value_type in candidates if _type_enabled(grammar, value_type)]


def _choose_type(rng: random.Random, grammar: GrammarConfig, candidates: tuple[RType, ...] = _ALL_RTYPES) -> RType:
    enabled = _enabled_types(grammar, candidates)
    if enabled:
        return rng.choice(enabled)
    return RType.NUM


def _coerce_type(rng: random.Random, grammar: GrammarConfig, value_type: RType) -> RType:
    if _type_enabled(grammar, value_type):
        return value_type
    return _choose_type(rng, grammar)


def rand_const(rng: random.Random, value_type: RType, grammar_config: GrammarConfig | None = None):
    grammar = _grammar(grammar_config)
    value_type = _coerce_type(rng, grammar, value_type)
    if value_type == RType.NUM:
        if grammar.allow_value("int") and (not grammar.allow_value("float") or rng.random() < 0.5):
            return ("const", rng.randint(-8, 8))
        return ("const", round(rng.uniform(-8.0, 8.0), 3))
    if value_type == RType.BOOL:
        return ("const", rng.choice([True, False]))
    if value_type == RType.STRING:
        if rng.random() < 0.5:
            return ("const", "".join(rng.choice("abcxyz") for _ in range(rng.randint(0, 6))))
        return ("const", "")
    if value_type == RType.NUM_LIST:
        return ("const", make_num_list(rng.randint(-3, 3) for _ in range(rng.randint(0, 6))))
    if value_type == RType.STRING_LIST:
        return ("const", make_string_list("".join(rng.choice("abcxyz") for _ in range(rng.randint(0, 3))) for _ in range(rng.randint(0, 6))))
    return ("const", None)


def rand_expr(rng: random.Random, depth: int, value_type: RType, grammar_config: GrammarConfig | None = None) -> tuple:
    grammar = _grammar(grammar_config)
    value_type = _coerce_type(rng, grammar, value_type)
    if depth <= 1:
        return rand_const(rng, value_type, grammar)

    if value_type == RType.NUM:
        makers = [lambda: rand_const(rng, RType.NUM, grammar)]
        if grammar.allow_unary("neg"):
            makers.append(lambda: ("neg", rand_expr(rng, depth - 1, RType.NUM, grammar)))
        bin_num = [op for op in _BIN_NUM if grammar.allow_binary(op)]
        if bin_num:
            makers.append(
                lambda: (
                    rng.choice(bin_num),
                    rand_expr(rng, depth - 1, RType.NUM, grammar),
                    rand_expr(rng, depth - 1, RType.NUM, grammar),
                )
            )
        num_builtins = [name for name in ("abs", "min", "max", "clip") if grammar.allow_builtin(name)]
        len_arg_types = _enabled_types(grammar, (RType.STRING, RType.NUM_LIST, RType.STRING_LIST))
        if grammar.allow_builtin("len") and len_arg_types:
            num_builtins.append("len")
        if grammar.allow_builtin("index") and _type_enabled(grammar, RType.NUM_LIST):
            num_builtins.append("index")
        if grammar.allow_builtin("find") and _type_enabled(grammar, RType.STRING):
            num_builtins.append("find")
        if num_builtins:
            def make_builtin():
                builtin = rng.choice(num_builtins)
                if builtin == "abs":
                    return ("call", builtin, [rand_expr(rng, depth - 1, RType.NUM, grammar)])
                if builtin in ("min", "max"):
                    return (
                        "call",
                        builtin,
                        [rand_expr(rng, depth - 1, RType.NUM, grammar), rand_expr(rng, depth - 1, RType.NUM, grammar)],
                    )
                if builtin == "len":
                    return ("call", builtin, [rand_expr(rng, depth - 1, rng.choice(len_arg_types), grammar)])
                if builtin == "index":
                    return (
                        "call",
                        builtin,
                        [rand_expr(rng, depth - 1, RType.NUM_LIST, grammar), ("const", rng.randint(-6, 6))],
                    )
                if builtin == "find":
                    return (
                        "call",
                        builtin,
                        [rand_expr(rng, depth - 1, RType.STRING, grammar), rand_expr(rng, depth - 1, RType.STRING, grammar)],
                    )
                return (
                    "call",
                    builtin,
                    [
                        rand_expr(rng, depth - 1, RType.NUM, grammar),
                        rand_expr(rng, depth - 1, RType.NUM, grammar),
                        rand_expr(rng, depth - 1, RType.NUM, grammar),
                    ],
                )
            makers.append(make_builtin)
        if grammar.allow_expression("if_expr") and _type_enabled(grammar, RType.BOOL):
            makers.append(
                lambda: (
                    "if_expr",
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.NUM, grammar),
                    rand_expr(rng, depth - 1, RType.NUM, grammar),
                )
            )
        return rng.choice(makers)()

    if value_type == RType.BOOL:
        makers = [lambda: rand_const(rng, RType.BOOL, grammar)]
        if grammar.allow_unary("not"):
            makers.append(lambda: ("not", rand_expr(rng, depth - 1, RType.BOOL, grammar)))
        cmp_ops = [op for op in _BIN_CMP if grammar.allow_binary(op)]
        if cmp_ops and _type_enabled(grammar, RType.NUM):
            def make_cmp():
                op = rng.choice(cmp_ops)
                if op in ("eq", "ne") and grammar.allow_value("none") and rng.random() < 0.35:
                    return (op, rand_const(rng, RType.NONE, grammar), rand_const(rng, RType.NONE, grammar))
                return (op, rand_expr(rng, depth - 1, RType.NUM, grammar), rand_expr(rng, depth - 1, RType.NUM, grammar))
            makers.append(make_cmp)
        bin_bool = [op for op in _BIN_BOOL if grammar.allow_binary(op)]
        if bin_bool:
            makers.append(
                lambda: (
                    rng.choice(bin_bool),
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                )
            )
        if grammar.allow_builtin("contains") and _type_enabled(grammar, RType.STRING):
            makers.append(
                lambda: (
                    "call",
                    "contains",
                    [rand_expr(rng, depth - 1, RType.STRING, grammar), rand_expr(rng, depth - 1, RType.STRING, grammar)],
                )
            )
        if grammar.allow_expression("if_expr"):
            makers.append(
                lambda: (
                    "if_expr",
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                )
            )
        return rng.choice(makers)()

    if value_type == RType.STRING:
        makers = [lambda: rand_const(rng, RType.STRING, grammar)]
        if grammar.allow_builtin("concat"):
            makers.append(
                lambda: (
                    "call",
                    "concat",
                    [rand_expr(rng, depth - 1, RType.STRING, grammar), rand_expr(rng, depth - 1, RType.STRING, grammar)],
                )
            )
        if grammar.allow_builtin("slice"):
            makers.append(
                lambda: (
                    "call",
                    "slice",
                    [rand_expr(rng, depth - 1, RType.STRING, grammar), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
                )
            )
        if grammar.allow_builtin("reverse"):
            makers.append(lambda: ("call", "reverse", [rand_expr(rng, depth - 1, RType.STRING, grammar)]))
        if grammar.allow_expression("if_expr") and _type_enabled(grammar, RType.BOOL):
            makers.append(
                lambda: (
                    "if_expr",
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.STRING, grammar),
                    rand_expr(rng, depth - 1, RType.STRING, grammar),
                )
            )
        return rng.choice(makers)()

    if value_type == RType.NUM_LIST:
        makers = [lambda: rand_const(rng, RType.NUM_LIST, grammar)]
        if grammar.allow_builtin("concat"):
            makers.append(
                lambda: (
                    "call",
                    "concat",
                    [rand_expr(rng, depth - 1, RType.NUM_LIST, grammar), rand_expr(rng, depth - 1, RType.NUM_LIST, grammar)],
                )
            )
        if grammar.allow_builtin("slice"):
            makers.append(
                lambda: (
                    "call",
                    "slice",
                    [rand_expr(rng, depth - 1, RType.NUM_LIST, grammar), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
                )
            )
        if grammar.allow_builtin("append") and _type_enabled(grammar, RType.NUM):
            makers.append(
                lambda: (
                    "call",
                    "append",
                    [rand_expr(rng, depth - 1, RType.NUM_LIST, grammar), rand_expr(rng, depth - 1, RType.NUM, grammar)],
                )
            )
        if grammar.allow_builtin("reverse"):
            makers.append(lambda: ("call", "reverse", [rand_expr(rng, depth - 1, RType.NUM_LIST, grammar)]))
        if grammar.allow_expression("if_expr") and _type_enabled(grammar, RType.BOOL):
            makers.append(
                lambda: (
                    "if_expr",
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.NUM_LIST, grammar),
                    rand_expr(rng, depth - 1, RType.NUM_LIST, grammar),
                )
            )
        return rng.choice(makers)()

    if value_type == RType.STRING_LIST:
        makers = [lambda: rand_const(rng, RType.STRING_LIST, grammar)]
        if grammar.allow_builtin("concat"):
            makers.append(
                lambda: (
                    "call",
                    "concat",
                    [rand_expr(rng, depth - 1, RType.STRING_LIST, grammar), rand_expr(rng, depth - 1, RType.STRING_LIST, grammar)],
                )
            )
        if grammar.allow_builtin("slice"):
            makers.append(
                lambda: (
                    "call",
                    "slice",
                    [rand_expr(rng, depth - 1, RType.STRING_LIST, grammar), ("const", rng.randint(-6, 6)), ("const", rng.randint(-6, 6))],
                )
            )
        if grammar.allow_builtin("append") and _type_enabled(grammar, RType.STRING):
            makers.append(
                lambda: (
                    "call",
                    "append",
                    [rand_expr(rng, depth - 1, RType.STRING_LIST, grammar), rand_expr(rng, depth - 1, RType.STRING, grammar)],
                )
            )
        if grammar.allow_builtin("reverse"):
            makers.append(lambda: ("call", "reverse", [rand_expr(rng, depth - 1, RType.STRING_LIST, grammar)]))
        if grammar.allow_expression("if_expr") and _type_enabled(grammar, RType.BOOL):
            makers.append(
                lambda: (
                    "if_expr",
                    rand_expr(rng, depth - 1, RType.BOOL, grammar),
                    rand_expr(rng, depth - 1, RType.STRING_LIST, grammar),
                    rand_expr(rng, depth - 1, RType.STRING_LIST, grammar),
                )
            )
        return rng.choice(makers)()

    return rand_const(rng, RType.NONE, grammar)


def rand_stmt(rng: random.Random, depth: int, limits: Limits, grammar_config: GrammarConfig | None = None) -> tuple:
    grammar = _grammar(grammar_config)
    assign_types = tuple(_enabled_types(grammar))
    if not assign_types:
        assign_types = (RType.NUM,)
    if depth <= 1:
        if grammar.allow_statement("return") and (not grammar.allow_statement("assign") or rng.random() < 0.25):
            return ("return", rand_expr(rng, 1, rng.choice(assign_types), grammar))
        return (
            "assign",
            rng.choice(["x", "y", "z", "w"]),
            rand_expr(rng, 1, rng.choice(assign_types), grammar),
        )

    modes: list[str] = []
    if grammar.allow_statement("assign"):
        modes.append("assign")
    if grammar.allow_statement("if_stmt") and _type_enabled(grammar, RType.BOOL):
        modes.append("if")
    if grammar.allow_statement("for_range") and grammar.allow_value("int"):
        modes.append("for")
    if grammar.allow_statement("return"):
        modes.append("return")
    mode = rng.choice(modes or ["return"])
    if mode == "assign":
        return (
            "assign",
            rng.choice(["x", "y", "z", "w", "u", "v"]),
            rand_expr(rng, depth - 1, rng.choice(assign_types), grammar),
        )
    if mode == "if":
        return (
            "if",
            rand_expr(rng, depth - 1, RType.BOOL, grammar),
            rand_block(rng, depth - 1, limits, force_return=False, grammar_config=grammar),
            rand_block(rng, depth - 1, limits, force_return=False, grammar_config=grammar),
        )
    if mode == "for":
        return (
            "for",
            rng.choice(["i", "j", "k"]),
            ("const", rng.randint(0, max(0, limits.max_for_k))),
            rand_block(rng, depth - 1, limits, force_return=False, grammar_config=grammar),
        )
    return ("return", rand_expr(rng, depth - 1, rng.choice(assign_types), grammar))


def rand_block(
    rng: random.Random,
    depth: int,
    limits: Limits,
    force_return: bool,
    grammar_config: GrammarConfig | None = None,
) -> List[tuple]:
    grammar = _grammar(grammar_config)
    count = rng.randint(1, max(1, limits.max_stmts_per_block))
    out: List[tuple] = []
    for _ in range(count):
        stmt = rand_stmt(rng, depth, limits, grammar)
        out.append(stmt)
        if stmt[0] == "return":
            break
    if force_return and not any(stmt[0] == "return" for stmt in out):
        out.append(("return", rand_expr(rng, max(1, depth - 1), _choose_type(rng, grammar), grammar)))
    return out[: limits.max_stmts_per_block]
