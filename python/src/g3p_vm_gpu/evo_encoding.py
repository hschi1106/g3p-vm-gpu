from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

from .ast import (
    AstProgram,
    NodeKind,
    build_program,
    max_expr_depth,
    node_count,
    prefix_repr,
    top_level_has_return,
    validate_prefix_program,
)
from .compiler import BytecodeProgram, compile_program


class RType(str, Enum):
    NUM = "NUM"
    BOOL = "BOOL"
    NONE = "NONE"


@dataclass(frozen=True)
class Limits:
    max_expr_depth: int = 5
    max_stmts_per_block: int = 6
    max_total_nodes: int = 80
    max_for_k: int = 16
    max_call_args: int = 3


@dataclass(frozen=True)
class GenomeMeta:
    node_count: int
    max_depth: int
    uses_builtins: bool
    hash_key: str


@dataclass(frozen=True)
class ProgramGenome:
    ast: AstProgram
    meta: GenomeMeta


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: Tuple[str, ...]


_BUILTINS = ("abs", "min", "max", "clip")
_BIN_NUM = ("add", "sub", "mul", "div", "mod")
_BIN_CMP = ("lt", "le", "gt", "ge", "eq", "ne")
_BIN_BOOL = ("and", "or")


def _build_meta(ast: AstProgram) -> GenomeMeta:
    digest = hashlib.sha1(prefix_repr(ast).encode("utf-8")).hexdigest()[:16]
    uses_builtins = any(n.kind in {NodeKind.CALL_ABS, NodeKind.CALL_MIN, NodeKind.CALL_MAX, NodeKind.CALL_CLIP} for n in ast.nodes)
    return GenomeMeta(
        node_count=node_count(ast),
        max_depth=max_expr_depth(ast),
        uses_builtins=uses_builtins,
        hash_key=digest,
    )


def _as_genome(stmts: Sequence[tuple]) -> ProgramGenome:
    p = build_program(stmts)
    return ProgramGenome(ast=p, meta=_build_meta(p))


def _rand_const(rng: random.Random, t: RType):
    if t == RType.NUM:
        return ("const", rng.randint(-8, 8) if rng.random() < 0.5 else round(rng.uniform(-8.0, 8.0), 3))
    if t == RType.BOOL:
        return ("const", rng.choice([True, False]))
    return ("const", None)


def _rand_expr(rng: random.Random, depth: int, t: RType) -> tuple:
    if depth <= 1:
        return _rand_const(rng, t)

    if t == RType.NUM:
        mode = rng.randint(0, 5)
        if mode == 0:
            return _rand_const(rng, RType.NUM)
        if mode == 1:
            return ("neg", _rand_expr(rng, depth - 1, RType.NUM))
        if mode == 2:
            op = rng.choice(_BIN_NUM)
            return (op, _rand_expr(rng, depth - 1, RType.NUM), _rand_expr(rng, depth - 1, RType.NUM))
        if mode == 3:
            b = rng.choice(_BUILTINS)
            if b == "abs":
                return ("call", b, [_rand_expr(rng, depth - 1, RType.NUM)])
            if b in ("min", "max"):
                return ("call", b, [_rand_expr(rng, depth - 1, RType.NUM), _rand_expr(rng, depth - 1, RType.NUM)])
            return (
                "call",
                b,
                [_rand_expr(rng, depth - 1, RType.NUM), _rand_expr(rng, depth - 1, RType.NUM), _rand_expr(rng, depth - 1, RType.NUM)],
            )
        if mode == 4:
            return (
                "if_expr",
                _rand_expr(rng, depth - 1, RType.BOOL),
                _rand_expr(rng, depth - 1, RType.NUM),
                _rand_expr(rng, depth - 1, RType.NUM),
            )
        return (rng.choice(_BIN_NUM), _rand_expr(rng, depth - 1, RType.NUM), _rand_expr(rng, depth - 1, RType.NUM))

    if t == RType.BOOL:
        mode = rng.randint(0, 4)
        if mode == 0:
            return _rand_const(rng, RType.BOOL)
        if mode == 1:
            return ("not", _rand_expr(rng, depth - 1, RType.BOOL))
        if mode == 2:
            op = rng.choice(_BIN_CMP)
            if op in ("eq", "ne") and rng.random() < 0.35:
                return (op, _rand_const(rng, RType.NONE), _rand_const(rng, RType.NONE))
            return (op, _rand_expr(rng, depth - 1, RType.NUM), _rand_expr(rng, depth - 1, RType.NUM))
        if mode == 3:
            op = rng.choice(_BIN_BOOL)
            return (op, _rand_expr(rng, depth - 1, RType.BOOL), _rand_expr(rng, depth - 1, RType.BOOL))
        return (
            "if_expr",
            _rand_expr(rng, depth - 1, RType.BOOL),
            _rand_expr(rng, depth - 1, RType.BOOL),
            _rand_expr(rng, depth - 1, RType.BOOL),
        )

    return _rand_const(rng, RType.NONE)


def _rand_stmt(rng: random.Random, depth: int, limits: Limits) -> tuple:
    if depth <= 1:
        if rng.random() < 0.25:
            return ("return", _rand_expr(rng, 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))
        return ("assign", rng.choice(["x", "y", "z", "w"]), _rand_expr(rng, 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))

    mode = rng.randint(0, 3)
    if mode == 0:
        return ("assign", rng.choice(["x", "y", "z", "w", "u", "v"]), _rand_expr(rng, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))
    if mode == 1:
        return (
            "if",
            _rand_expr(rng, depth - 1, RType.BOOL),
            _rand_block(rng, depth - 1, limits, force_return=False),
            _rand_block(rng, depth - 1, limits, force_return=False),
        )
    if mode == 2:
        return (
            "for",
            rng.choice(["i", "j", "k"]),
            rng.randint(0, max(0, limits.max_for_k)),
            _rand_block(rng, depth - 1, limits, force_return=False),
        )
    return ("return", _rand_expr(rng, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))


def _rand_block(rng: random.Random, depth: int, limits: Limits, force_return: bool) -> List[tuple]:
    n = rng.randint(1, max(1, limits.max_stmts_per_block))
    out: List[tuple] = []
    for _ in range(n):
        st = _rand_stmt(rng, depth, limits)
        out.append(st)
        if st[0] == "return":
            break
    if force_return and not any(st[0] == "return" for st in out):
        out.append(("return", _rand_expr(rng, max(1, depth - 1), RType.NUM)))
    return out[: limits.max_stmts_per_block]


def _expr_from_prefix(p: AstProgram, idx: int) -> tuple[tuple, int]:
    n = p.nodes[idx]
    k = n.kind
    if k == NodeKind.CONST:
        return ("const", p.consts[n.i0]), idx + 1
    if k == NodeKind.VAR:
        return ("var", p.names[n.i0]), idx + 1
    if k == NodeKind.NEG:
        e, j = _expr_from_prefix(p, idx + 1)
        return ("neg", e), j
    if k == NodeKind.NOT:
        e, j = _expr_from_prefix(p, idx + 1)
        return ("not", e), j
    if k in {
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
        a, j = _expr_from_prefix(p, idx + 1)
        b, h = _expr_from_prefix(p, j)
        return (k.value.lower(), a, b), h
    if k == NodeKind.IF_EXPR:
        c, j = _expr_from_prefix(p, idx + 1)
        t, h = _expr_from_prefix(p, j)
        f, z = _expr_from_prefix(p, h)
        return ("if_expr", c, t, f), z
    if k in {NodeKind.CALL_ABS, NodeKind.CALL_MIN, NodeKind.CALL_MAX, NodeKind.CALL_CLIP}:
        name = {
            NodeKind.CALL_ABS: "abs",
            NodeKind.CALL_MIN: "min",
            NodeKind.CALL_MAX: "max",
            NodeKind.CALL_CLIP: "clip",
        }[k]
        argc = {NodeKind.CALL_ABS: 1, NodeKind.CALL_MIN: 2, NodeKind.CALL_MAX: 2, NodeKind.CALL_CLIP: 3}[k]
        args: List[tuple] = []
        cur = idx + 1
        for _ in range(argc):
            a, cur = _expr_from_prefix(p, cur)
            args.append(a)
        return ("call", name, args), cur
    raise ValueError(f"expected Expr at index {idx}, got {k}")


def _stmt_from_prefix(p: AstProgram, idx: int) -> tuple[tuple, int]:
    n = p.nodes[idx]
    if n.kind == NodeKind.ASSIGN:
        e, j = _expr_from_prefix(p, idx + 1)
        return ("assign", p.names[n.i0], e), j
    if n.kind == NodeKind.RETURN:
        e, j = _expr_from_prefix(p, idx + 1)
        return ("return", e), j
    if n.kind == NodeKind.IF_STMT:
        c, j = _expr_from_prefix(p, idx + 1)
        tb, h = _block_from_prefix(p, j)
        eb, z = _block_from_prefix(p, h)
        return ("if", c, tb, eb), z
    if n.kind == NodeKind.FOR_RANGE:
        b, j = _block_from_prefix(p, idx + 1)
        return ("for", p.names[n.i0], n.i1, b), j
    raise ValueError(f"expected Stmt at index {idx}, got {n.kind}")


def _block_from_prefix(p: AstProgram, idx: int) -> tuple[List[tuple], int]:
    n = p.nodes[idx]
    if n.kind == NodeKind.BLOCK_NIL:
        return [], idx + 1
    if n.kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {n.kind}")
    st, j = _stmt_from_prefix(p, idx + 1)
    tail, h = _block_from_prefix(p, j)
    return [st] + tail, h


def _top_stmts(p: AstProgram) -> List[tuple]:
    b, end = _block_from_prefix(p, 1)
    if end != len(p.nodes):
        raise ValueError("invalid trailing tokens")
    return b


def _validate_stmt_limits(st: tuple, limits: Limits, errors: List[str], depth: int = 0) -> None:
    tag = st[0]
    if tag == "for":
        k = st[2]
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            errors.append("ForRange.k must be non-negative int and not bool")
        elif k > limits.max_for_k:
            errors.append(f"ForRange.k {k} exceeds max_for_k {limits.max_for_k}")
        _validate_block_limits(st[3], limits, errors, depth + 1)
    elif tag == "if":
        _validate_block_limits(st[2], limits, errors, depth + 1)
        _validate_block_limits(st[3], limits, errors, depth + 1)


def _validate_block_limits(stmts: Sequence[tuple], limits: Limits, errors: List[str], depth: int = 0) -> None:
    if len(stmts) > limits.max_stmts_per_block:
        errors.append(f"block has {len(stmts)} statements (> {limits.max_stmts_per_block})")
    for st in stmts:
        _validate_stmt_limits(st, limits, errors, depth)


def make_random_genome(seed: int = 0, limits: Limits | None = None, type_policy: str = "strong") -> ProgramGenome:
    del type_policy
    limits = limits or Limits()
    rng = random.Random(seed)
    for _ in range(256):
        stmts = _rand_block(rng, limits.max_expr_depth, limits, force_return=True)
        g = _as_genome(stmts)
        if validate_genome(g, limits).is_valid:
            return g
    return _as_genome([("return", ("const", 0))])


def validate_genome(genome: ProgramGenome, limits: Limits | None = None) -> ValidationResult:
    limits = limits or Limits()
    errors: List[str] = []

    try:
        validate_prefix_program(genome.ast)
    except Exception as exc:
        return ValidationResult(False, (str(exc),))

    if node_count(genome.ast) > limits.max_total_nodes:
        errors.append(f"node count exceeds max_total_nodes {limits.max_total_nodes}")
    if max_expr_depth(genome.ast) > limits.max_expr_depth:
        errors.append(f"expression depth exceeds max_expr_depth {limits.max_expr_depth}")
    if not top_level_has_return(genome.ast):
        errors.append("top-level block must contain at least one Return")

    try:
        stmts = _top_stmts(genome.ast)
        _validate_block_limits(stmts, limits, errors)
    except Exception as exc:
        errors.append(f"invalid AST decode: {exc}")

    try:
        compile_program(genome.ast)
    except Exception as exc:
        errors.append(f"compile failed: {exc}")

    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def compile_for_eval(genome: ProgramGenome) -> BytecodeProgram:
    return compile_program(genome.ast)


def mutate(genome: ProgramGenome, seed: int = 0, limits: Limits | None = None) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    try:
        stmts = list(_top_stmts(genome.ast))
    except Exception:
        return make_random_genome(seed=seed, limits=limits)

    if not stmts:
        return make_random_genome(seed=seed, limits=limits)

    op = rng.randint(0, 2)
    if op == 0:
        i = rng.randrange(len(stmts))
        stmts[i] = _rand_stmt(rng, limits.max_expr_depth, limits)
    elif op == 1 and len(stmts) < limits.max_stmts_per_block:
        i = rng.randint(0, len(stmts))
        stmts.insert(i, _rand_stmt(rng, limits.max_expr_depth, limits))
    elif len(stmts) > 1:
        i = rng.randrange(len(stmts))
        del stmts[i]

    if not any(st[0] == "return" for st in stmts):
        stmts.append(("return", _rand_expr(rng, max(1, limits.max_expr_depth - 1), RType.NUM)))

    stmts = stmts[: limits.max_stmts_per_block]
    out = _as_genome(stmts)
    if validate_genome(out, limits).is_valid:
        return out
    return make_random_genome(seed=seed + 1, limits=limits)


def crossover_top_level(parent_a: ProgramGenome, parent_b: ProgramGenome, seed: int = 0, limits: Limits | None = None) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    try:
        a = _top_stmts(parent_a.ast)
        b = _top_stmts(parent_b.ast)
    except Exception:
        return make_random_genome(seed=seed, limits=limits)

    cut_a = rng.randint(0, len(a)) if a else 0
    cut_b = rng.randint(0, len(b)) if b else 0
    child = list(a[:cut_a]) + list(b[cut_b:])
    if not child:
        child = [("return", ("const", 0))]
    child = child[: limits.max_stmts_per_block]
    if not any(st[0] == "return" for st in child):
        child.append(("return", _rand_expr(rng, max(1, limits.max_expr_depth - 1), RType.NUM)))
        child = child[: limits.max_stmts_per_block]

    out = _as_genome(child)
    if validate_genome(out, limits).is_valid:
        return out
    return make_random_genome(seed=seed + 7, limits=limits)


def crossover_typed_subtree(parent_a: ProgramGenome, parent_b: ProgramGenome, seed: int = 0, limits: Limits | None = None) -> ProgramGenome:
    return crossover_top_level(parent_a, parent_b, seed=seed, limits=limits)


def crossover(
    parent_a: ProgramGenome,
    parent_b: ProgramGenome,
    seed: int = 0,
    limits: Limits | None = None,
    method: str = "top_level_splice",
) -> ProgramGenome:
    if method == "top_level_splice":
        return crossover_top_level(parent_a, parent_b, seed=seed, limits=limits)
    if method == "typed_subtree":
        return crossover_typed_subtree(parent_a, parent_b, seed=seed, limits=limits)
    if method == "hybrid":
        rng = random.Random(seed)
        if rng.random() < 0.7:
            return crossover_typed_subtree(parent_a, parent_b, seed=seed, limits=limits)
        return crossover_top_level(parent_a, parent_b, seed=seed, limits=limits)
    raise ValueError(f"unknown crossover method: {method}")
