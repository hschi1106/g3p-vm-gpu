from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple

from .ast import (
    Assign,
    BOp,
    Binary,
    Block,
    Call,
    Const,
    Expr,
    ForRange,
    IfExpr,
    IfStmt,
    Return,
    Stmt,
    UOp,
    Unary,
    Var,
)
from .compiler import BytecodeProgram, compile_program


class RType(str, Enum):
    NUM = "NUM"
    BOOL = "BOOL"
    NONE = "NONE"
    ANY = "ANY"
    INVALID = "INVALID"


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
    ast: Block
    meta: GenomeMeta


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: Tuple[str, ...]


@dataclass
class _GenContext:
    vars_by_type: Dict[RType, Set[str]]
    all_vars: Set[str]
    tmp_idx: int = 0

    @classmethod
    def empty(cls) -> "_GenContext":
        return cls(
            vars_by_type={
                RType.NUM: set(),
                RType.BOOL: set(),
                RType.NONE: set(),
            },
            all_vars=set(),
            tmp_idx=0,
        )

    def clone(self) -> "_GenContext":
        return _GenContext(
            vars_by_type={
                RType.NUM: set(self.vars_by_type[RType.NUM]),
                RType.BOOL: set(self.vars_by_type[RType.BOOL]),
                RType.NONE: set(self.vars_by_type[RType.NONE]),
            },
            all_vars=set(self.all_vars),
            tmp_idx=self.tmp_idx,
        )

    def add_var(self, name: str, typ: RType) -> None:
        self.all_vars.add(name)
        for t in (RType.NUM, RType.BOOL, RType.NONE):
            if name in self.vars_by_type[t]:
                self.vars_by_type[t].remove(name)
        if typ in (RType.NUM, RType.BOOL, RType.NONE):
            self.vars_by_type[typ].add(name)

    def vars_for(self, typ: RType) -> List[str]:
        if typ == RType.ANY:
            return sorted(self.all_vars)
        return sorted(self.vars_by_type.get(typ, set()))

    def next_tmp_name(self) -> str:
        name = f"t{self.tmp_idx}"
        self.tmp_idx += 1
        return name


_BASE_NAMES = ("x", "y", "z", "w", "u", "v")
_BUILTIN_ARITY = {"abs": 1, "min": 2, "max": 2, "clip": 3}


def _build_meta(ast: Block) -> GenomeMeta:
    node_count = _count_block_nodes(ast)
    max_depth = _max_block_expr_depth(ast)
    uses_builtins = _block_uses_builtins(ast)
    digest = hashlib.sha1(repr(ast).encode("utf-8")).hexdigest()[:16]
    return GenomeMeta(
        node_count=node_count,
        max_depth=max_depth,
        uses_builtins=uses_builtins,
        hash_key=digest,
    )


def _as_genome(ast: Block) -> ProgramGenome:
    return ProgramGenome(ast=ast, meta=_build_meta(ast))


def _choose_var_name(rng: random.Random, ctx: _GenContext) -> str:
    if not ctx.all_vars:
        return _new_var_name(ctx)
    if rng.random() < 0.4:
        return _new_var_name(ctx)
    return rng.choice(sorted(ctx.all_vars))


def _new_var_name(ctx: _GenContext) -> str:
    for name in _BASE_NAMES:
        if name not in ctx.all_vars:
            return name
    return ctx.next_tmp_name()


def _rand_num_const(rng: random.Random) -> Const:
    if rng.random() < 0.5:
        return Const(rng.randint(-8, 8))
    return Const(round(rng.uniform(-8.0, 8.0), 3))


def _rand_leaf_expr(rng: random.Random, ctx: _GenContext, target: RType) -> Expr:
    if target == RType.NUM:
        vars_ = ctx.vars_for(RType.NUM)
        if vars_ and rng.random() < 0.45:
            return Var(rng.choice(vars_))
        return _rand_num_const(rng)
    if target == RType.BOOL:
        vars_ = ctx.vars_for(RType.BOOL)
        if vars_ and rng.random() < 0.45:
            return Var(rng.choice(vars_))
        return Const(rng.choice([True, False]))
    if target == RType.NONE:
        vars_ = ctx.vars_for(RType.NONE)
        if vars_ and rng.random() < 0.45:
            return Var(rng.choice(vars_))
        return Const(None)

    # ANY
    choices: List[Expr] = [Const(None), Const(rng.choice([True, False])), _rand_num_const(rng)]
    for typ in (RType.NUM, RType.BOOL, RType.NONE):
        vars_ = ctx.vars_for(typ)
        if vars_:
            choices.append(Var(rng.choice(vars_)))
    return rng.choice(choices)


def _gen_bool_compare(rng: random.Random, ctx: _GenContext, depth: int) -> Expr:
    mode = rng.choice(["num", "bool", "none"])
    if mode == "num":
        op = rng.choice([BOp.LT, BOp.LE, BOp.GT, BOp.GE, BOp.EQ, BOp.NE])
        return Binary(op, _gen_expr(rng, ctx, depth - 1, RType.NUM), _gen_expr(rng, ctx, depth - 1, RType.NUM))
    if mode == "bool":
        op = rng.choice([BOp.EQ, BOp.NE])
        return Binary(op, _gen_expr(rng, ctx, depth - 1, RType.BOOL), _gen_expr(rng, ctx, depth - 1, RType.BOOL))
    op = rng.choice([BOp.EQ, BOp.NE])
    return Binary(op, _gen_expr(rng, ctx, depth - 1, RType.NONE), _gen_expr(rng, ctx, depth - 1, RType.NONE))


def _gen_expr(rng: random.Random, ctx: _GenContext, depth: int, target: RType) -> Expr:
    if target == RType.ANY:
        target = rng.choice([RType.NUM, RType.BOOL, RType.NONE])

    if depth <= 0:
        return _rand_leaf_expr(rng, ctx, target)

    if target == RType.NUM:
        choice = rng.randint(0, 5)
        if choice == 0:
            return _rand_leaf_expr(rng, ctx, RType.NUM)
        if choice == 1:
            return Unary(UOp.NEG, _gen_expr(rng, ctx, depth - 1, RType.NUM))
        if choice == 2:
            return Binary(
                rng.choice([BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD]),
                _gen_expr(rng, ctx, depth - 1, RType.NUM),
                _gen_expr(rng, ctx, depth - 1, RType.NUM),
            )
        if choice == 3:
            f = rng.choice(["abs", "min", "max", "clip"])
            if f == "abs":
                return Call(f, [_gen_expr(rng, ctx, depth - 1, RType.NUM)])
            if f in ("min", "max"):
                return Call(f, [_gen_expr(rng, ctx, depth - 1, RType.NUM), _gen_expr(rng, ctx, depth - 1, RType.NUM)])
            return Call(
                f,
                [
                    _gen_expr(rng, ctx, depth - 1, RType.NUM),
                    _gen_expr(rng, ctx, depth - 1, RType.NUM),
                    _gen_expr(rng, ctx, depth - 1, RType.NUM),
                ],
            )
        return IfExpr(
            _gen_expr(rng, ctx, depth - 1, RType.BOOL),
            _gen_expr(rng, ctx, depth - 1, RType.NUM),
            _gen_expr(rng, ctx, depth - 1, RType.NUM),
        )

    if target == RType.BOOL:
        choice = rng.randint(0, 4)
        if choice == 0:
            return _rand_leaf_expr(rng, ctx, RType.BOOL)
        if choice == 1:
            return Unary(UOp.NOT, _gen_expr(rng, ctx, depth - 1, RType.BOOL))
        if choice == 2:
            return _gen_bool_compare(rng, ctx, depth)
        if choice == 3:
            return Binary(
                rng.choice([BOp.AND, BOp.OR]),
                _gen_expr(rng, ctx, depth - 1, RType.BOOL),
                _gen_expr(rng, ctx, depth - 1, RType.BOOL),
            )
        return IfExpr(
            _gen_expr(rng, ctx, depth - 1, RType.BOOL),
            _gen_expr(rng, ctx, depth - 1, RType.BOOL),
            _gen_expr(rng, ctx, depth - 1, RType.BOOL),
        )

    return _rand_leaf_expr(rng, ctx, RType.NONE)


def _gen_stmt(rng: random.Random, ctx: _GenContext, depth: int, limits: Limits) -> Stmt:
    if depth <= 0:
        if rng.random() < 0.75:
            return _gen_assign_stmt(rng, ctx, depth)
        return Return(_gen_expr(rng, ctx, 0, RType.ANY))

    choice = rng.randint(0, 3)
    if choice == 0:
        return _gen_assign_stmt(rng, ctx, depth - 1)
    if choice == 1:
        then_ctx = ctx.clone()
        else_ctx = ctx.clone()
        return IfStmt(
            cond=_gen_expr(rng, ctx, depth - 1, RType.BOOL),
            then_block=_gen_block(
                rng,
                then_ctx,
                depth - 1,
                limits,
                force_return=False,
                max_stmts=max(1, min(2, limits.max_stmts_per_block)),
            ),
            else_block=_gen_block(
                rng,
                else_ctx,
                depth - 1,
                limits,
                force_return=False,
                max_stmts=max(1, min(2, limits.max_stmts_per_block)),
            ),
        )
    if choice == 2:
        loop_var = rng.choice(["i", "j", "k"])
        if loop_var in ctx.all_vars:
            loop_var = f"{loop_var}{rng.randint(0, 9)}"
        body_ctx = ctx.clone()
        body_ctx.add_var(loop_var, RType.NUM)
        return ForRange(
            var=loop_var,
            k=rng.randint(0, max(0, limits.max_for_k)),
            body=_gen_block(
                rng,
                body_ctx,
                depth - 1,
                limits,
                force_return=False,
                max_stmts=max(1, min(2, limits.max_stmts_per_block)),
            ),
        )
    return Return(_gen_expr(rng, ctx, depth - 1, rng.choice([RType.NUM, RType.BOOL, RType.NONE])))


def _gen_assign_stmt(rng: random.Random, ctx: _GenContext, depth: int) -> Assign:
    name = _choose_var_name(rng, ctx)
    typ = rng.choice([RType.NUM, RType.BOOL, RType.NONE])
    expr = _gen_expr(rng, ctx, max(0, depth), typ)
    ctx.add_var(name, typ)
    return Assign(name, expr)


def _gen_block(
    rng: random.Random,
    ctx: _GenContext,
    depth: int,
    limits: Limits,
    force_return: bool,
    max_stmts: int | None = None,
) -> Block:
    max_n = limits.max_stmts_per_block if max_stmts is None else max_stmts
    n = rng.randint(1, max(1, max_n))
    stmts: List[Stmt] = []
    for _ in range(n):
        st = _gen_stmt(rng, ctx, depth, limits)
        stmts.append(st)
        if isinstance(st, Return):
            break
    if force_return and not any(isinstance(st, Return) for st in stmts):
        stmts.append(Return(_gen_expr(rng, ctx, max(0, depth - 1), rng.choice([RType.NUM, RType.BOOL]))))
    return Block(stmts[:max_n])


def make_random_genome(seed: int = 0, limits: Limits | None = None, type_policy: str = "strong") -> ProgramGenome:
    del type_policy  # v0.1 implementation always uses strong typing.
    limits = limits or Limits()
    rng = random.Random(seed)
    for _ in range(128):
        ctx = _GenContext.empty()
        ast = _gen_block(rng, ctx, limits.max_expr_depth, limits, force_return=True)
        ast = _repair_limits(ast, rng, limits, ctx)
        genome = _as_genome(ast)
        valid = validate_genome(genome, limits)
        if valid.is_valid:
            return genome
    return _as_genome(Block([Return(Const(0))]))


def _infer_expr_type(e: Expr, env: Dict[str, RType]) -> RType:
    if isinstance(e, Const):
        if e.value is None:
            return RType.NONE
        if isinstance(e.value, bool):
            return RType.BOOL
        if isinstance(e.value, (int, float)):
            return RType.NUM
        return RType.INVALID
    if isinstance(e, Var):
        return env.get(e.name, RType.ANY)
    if isinstance(e, Unary):
        t = _infer_expr_type(e.e, env)
        if e.op == UOp.NEG:
            return RType.NUM if t == RType.NUM else RType.INVALID
        if e.op == UOp.NOT:
            return RType.BOOL if t == RType.BOOL else RType.INVALID
        return RType.INVALID
    if isinstance(e, Binary):
        ta = _infer_expr_type(e.a, env)
        tb = _infer_expr_type(e.b, env)
        if e.op in (BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD):
            return RType.NUM if ta == RType.NUM and tb == RType.NUM else RType.INVALID
        if e.op in (BOp.AND, BOp.OR):
            return RType.BOOL if ta == RType.BOOL and tb == RType.BOOL else RType.INVALID
        if e.op in (BOp.LT, BOp.LE, BOp.GT, BOp.GE):
            return RType.BOOL if ta == RType.NUM and tb == RType.NUM else RType.INVALID
        if e.op in (BOp.EQ, BOp.NE):
            if ta == tb and ta in (RType.NUM, RType.BOOL, RType.NONE):
                return RType.BOOL
            return RType.INVALID
        return RType.INVALID
    if isinstance(e, IfExpr):
        tc = _infer_expr_type(e.cond, env)
        tt = _infer_expr_type(e.then_e, env)
        tf = _infer_expr_type(e.else_e, env)
        if tc != RType.BOOL:
            return RType.INVALID
        if tt == tf and tt in (RType.NUM, RType.BOOL, RType.NONE):
            return tt
        return RType.INVALID
    if isinstance(e, Call):
        if e.name not in _BUILTIN_ARITY:
            return RType.INVALID
        if len(e.args) != _BUILTIN_ARITY[e.name]:
            return RType.INVALID
        for arg in e.args:
            if _infer_expr_type(arg, env) != RType.NUM:
                return RType.INVALID
        return RType.NUM
    return RType.INVALID


def _validate_expr(e: Expr, env: Dict[str, RType], errors: List[str]) -> RType:
    t = _infer_expr_type(e, env)
    if t == RType.INVALID:
        errors.append(f"invalid expression typing: {type(e).__name__}")
    return t


def _validate_block(
    b: Block,
    limits: Limits,
    env: Dict[str, RType],
    errors: List[str],
    *,
    top_level: bool = False,
) -> None:
    if len(b.stmts) > limits.max_stmts_per_block:
        errors.append(f"block has {len(b.stmts)} statements (> {limits.max_stmts_per_block})")

    if top_level and not any(isinstance(st, Return) for st in b.stmts):
        errors.append("top-level block must contain at least one Return")

    local_env = dict(env)
    for st in b.stmts:
        if isinstance(st, Assign):
            t = _validate_expr(st.e, local_env, errors)
            if t in (RType.NUM, RType.BOOL, RType.NONE):
                local_env[st.name] = t
        elif isinstance(st, Return):
            _validate_expr(st.e, local_env, errors)
        elif isinstance(st, IfStmt):
            tc = _validate_expr(st.cond, local_env, errors)
            if tc != RType.BOOL:
                errors.append("IfStmt condition must be Bool")
            _validate_block(st.then_block, limits, dict(local_env), errors)
            _validate_block(st.else_block, limits, dict(local_env), errors)
        elif isinstance(st, ForRange):
            if not isinstance(st.k, int) or isinstance(st.k, bool) or st.k < 0:
                errors.append("ForRange.k must be non-negative int and not bool")
            if st.k > limits.max_for_k:
                errors.append(f"ForRange.k {st.k} exceeds max_for_k {limits.max_for_k}")
            body_env = dict(local_env)
            body_env[st.var] = RType.NUM
            _validate_block(st.body, limits, body_env, errors)
        else:
            errors.append(f"unknown statement type: {type(st).__name__}")


def validate_genome(genome: ProgramGenome, limits: Limits | None = None) -> ValidationResult:
    limits = limits or Limits()
    errors: List[str] = []
    if _count_block_nodes(genome.ast) > limits.max_total_nodes:
        errors.append(f"node count exceeds max_total_nodes {limits.max_total_nodes}")
    if _max_block_expr_depth(genome.ast) > limits.max_expr_depth:
        errors.append(f"expression depth exceeds max_expr_depth {limits.max_expr_depth}")
    _validate_block(genome.ast, limits, {}, errors, top_level=True)
    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def compile_for_eval(genome: ProgramGenome) -> BytecodeProgram:
    return compile_program(genome.ast)


def _mutate_expr(
    e: Expr,
    rng: random.Random,
    ctx: _GenContext,
    depth: int,
    target: RType,
) -> Expr:
    if depth <= 0 or rng.random() < 0.3:
        return _gen_expr(rng, ctx, max(0, depth), target)

    if isinstance(e, Const):
        return _gen_expr(rng, ctx, max(0, depth - 1), target)
    if isinstance(e, Var):
        vars_ = ctx.vars_for(target)
        if vars_ and rng.random() < 0.6:
            return Var(rng.choice(vars_))
        return _gen_expr(rng, ctx, max(0, depth - 1), target)
    if isinstance(e, Unary):
        return Unary(e.op, _mutate_expr(e.e, rng, ctx, depth - 1, target if e.op == UOp.NEG else RType.BOOL))
    if isinstance(e, Binary):
        if e.op in (BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD):
            op = rng.choice([BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD]) if rng.random() < 0.3 else e.op
            if rng.random() < 0.5:
                return Binary(op, _mutate_expr(e.a, rng, ctx, depth - 1, RType.NUM), e.b)
            return Binary(op, e.a, _mutate_expr(e.b, rng, ctx, depth - 1, RType.NUM))
        if e.op in (BOp.AND, BOp.OR):
            op = rng.choice([BOp.AND, BOp.OR]) if rng.random() < 0.3 else e.op
            if rng.random() < 0.5:
                return Binary(op, _mutate_expr(e.a, rng, ctx, depth - 1, RType.BOOL), e.b)
            return Binary(op, e.a, _mutate_expr(e.b, rng, ctx, depth - 1, RType.BOOL))
        if e.op in (BOp.LT, BOp.LE, BOp.GT, BOp.GE, BOp.EQ, BOp.NE):
            return _gen_expr(rng, ctx, depth - 1, RType.BOOL)
    if isinstance(e, IfExpr):
        t = _infer_expr_type(e.then_e, _infer_env_from_program(Block([Return(e.then_e)])))
        t = target if t == RType.INVALID else t
        mode = rng.randint(0, 2)
        if mode == 0:
            return IfExpr(_mutate_expr(e.cond, rng, ctx, depth - 1, RType.BOOL), e.then_e, e.else_e)
        if mode == 1:
            return IfExpr(e.cond, _mutate_expr(e.then_e, rng, ctx, depth - 1, t), e.else_e)
        return IfExpr(e.cond, e.then_e, _mutate_expr(e.else_e, rng, ctx, depth - 1, t))
    if isinstance(e, Call):
        if not e.args:
            return _gen_expr(rng, ctx, depth - 1, RType.NUM)
        idx = rng.randrange(len(e.args))
        args = list(e.args)
        args[idx] = _mutate_expr(args[idx], rng, ctx, depth - 1, RType.NUM)
        return Call(e.name, args)
    return _gen_expr(rng, ctx, depth - 1, target)


def _mutate_stmt(st: Stmt, rng: random.Random, ctx: _GenContext, limits: Limits) -> Stmt:
    if isinstance(st, Assign):
        if rng.random() < 0.3:
            name = _choose_var_name(rng, ctx)
            t = _infer_expr_type(st.e, _infer_env_from_program(Block([st])))
            if t not in (RType.NUM, RType.BOOL, RType.NONE):
                t = rng.choice([RType.NUM, RType.BOOL, RType.NONE])
            e = _gen_expr(rng, ctx, limits.max_expr_depth - 1, t)
            ctx.add_var(name, t)
            return Assign(name, e)
        env = _infer_env_from_program(Block([st]))
        t = _infer_expr_type(st.e, env)
        if t == RType.INVALID:
            t = rng.choice([RType.NUM, RType.BOOL, RType.NONE])
        new_e = _mutate_expr(st.e, rng, ctx, limits.max_expr_depth - 1, t)
        ctx.add_var(st.name, t)
        return Assign(st.name, new_e)

    if isinstance(st, Return):
        t = _infer_expr_type(st.e, _infer_env_from_program(Block([st])))
        if t == RType.INVALID:
            t = rng.choice([RType.NUM, RType.BOOL, RType.NONE])
        return Return(_mutate_expr(st.e, rng, ctx, limits.max_expr_depth - 1, t))

    if isinstance(st, IfStmt):
        mode = rng.randint(0, 3)
        if mode == 0:
            return IfStmt(_mutate_expr(st.cond, rng, ctx, limits.max_expr_depth - 1, RType.BOOL), st.then_block, st.else_block)
        if mode == 1:
            return IfStmt(st.cond, st.else_block, st.then_block)
        if mode == 2:
            then_ctx = ctx.clone()
            return IfStmt(
                st.cond,
                _gen_block(
                    rng,
                    then_ctx,
                    max(1, limits.max_expr_depth - 2),
                    limits,
                    force_return=False,
                    max_stmts=max(1, min(2, limits.max_stmts_per_block)),
                ),
                st.else_block,
            )
        else_ctx = ctx.clone()
        return IfStmt(
            st.cond,
            st.then_block,
            _gen_block(
                rng,
                else_ctx,
                max(1, limits.max_expr_depth - 2),
                limits,
                force_return=False,
                max_stmts=max(1, min(2, limits.max_stmts_per_block)),
            ),
        )

    if isinstance(st, ForRange):
        mode = rng.randint(0, 2)
        if mode == 0:
            delta = rng.choice([-2, -1, 1, 2])
            new_k = max(0, min(limits.max_for_k, st.k + delta))
            return ForRange(st.var, new_k, st.body)
        if mode == 1:
            body_ctx = ctx.clone()
            body_ctx.add_var(st.var, RType.NUM)
            return ForRange(
                st.var,
                st.k,
                _gen_block(
                    rng,
                    body_ctx,
                    max(1, limits.max_expr_depth - 2),
                    limits,
                    force_return=False,
                    max_stmts=max(1, min(2, limits.max_stmts_per_block)),
                ),
            )
        body_ctx = ctx.clone()
        body_ctx.add_var(st.var, RType.NUM)
        return ForRange(st.var, rng.randint(0, limits.max_for_k), st.body)

    return st


def _stmt_tree_size(st: Stmt) -> int:
    if isinstance(st, IfStmt):
        return 1 + _block_stmt_tree_size(st.then_block) + _block_stmt_tree_size(st.else_block)
    if isinstance(st, ForRange):
        return 1 + _block_stmt_tree_size(st.body)
    return 1


def _block_stmt_tree_size(b: Block) -> int:
    return sum(_stmt_tree_size(st) for st in b.stmts)


def _mutate_stmt_in_stmt(
    st: Stmt,
    target: int,
    rng: random.Random,
    ctx: _GenContext,
    limits: Limits,
) -> Tuple[Stmt, int]:
    if target == 0:
        return _mutate_stmt(st, rng, ctx, limits), -1

    idx = target - 1
    if isinstance(st, IfStmt):
        then_size = _block_stmt_tree_size(st.then_block)
        if idx < then_size:
            new_then, rem = _mutate_block_at_index(st.then_block, idx, rng, ctx.clone(), limits)
            return IfStmt(st.cond, new_then, st.else_block), rem
        idx -= then_size
        else_size = _block_stmt_tree_size(st.else_block)
        if idx < else_size:
            new_else, rem = _mutate_block_at_index(st.else_block, idx, rng, ctx.clone(), limits)
            return IfStmt(st.cond, st.then_block, new_else), rem
    elif isinstance(st, ForRange):
        body_size = _block_stmt_tree_size(st.body)
        if idx < body_size:
            body_ctx = ctx.clone()
            body_ctx.add_var(st.var, RType.NUM)
            new_body, rem = _mutate_block_at_index(st.body, idx, rng, body_ctx, limits)
            return ForRange(st.var, st.k, new_body), rem

    return st, target


def _mutate_block_at_index(
    b: Block,
    target: int,
    rng: random.Random,
    ctx: _GenContext,
    limits: Limits,
) -> Tuple[Block, int]:
    out: List[Stmt] = []
    cur = target
    for st in b.stmts:
        if cur < 0:
            out.append(st)
            continue
        size = _stmt_tree_size(st)
        if cur >= size:
            out.append(st)
            cur -= size
            continue
        new_st, rem = _mutate_stmt_in_stmt(st, cur, rng, ctx, limits)
        out.append(new_st)
        cur = rem
    return Block(out), cur


def _repair_limits(ast: Block, rng: random.Random, limits: Limits, ctx: _GenContext | None = None) -> Block:
    ctx = ctx or _GenContext.empty()
    fixed = _shrink_block_expr_depth(ast, limits.max_expr_depth)
    fixed = _clamp_block_stmt_count(fixed, limits.max_stmts_per_block)
    fixed = _clamp_for_k(fixed, limits.max_for_k)
    if not any(isinstance(st, Return) for st in fixed.stmts):
        fixed = Block(list(fixed.stmts) + [Return(_gen_expr(rng, ctx, max(0, limits.max_expr_depth - 1), RType.NUM))])
    if _count_block_nodes(fixed) > limits.max_total_nodes:
        return Block([Return(Const(0))])
    return fixed


def mutate(genome: ProgramGenome, seed: int = 0, limits: Limits | None = None) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    ast = genome.ast
    total = _block_stmt_tree_size(ast)
    if total <= 0:
        return make_random_genome(seed=seed, limits=limits)
    ctx = _infer_context_from_program(ast)
    target = rng.randrange(total)
    mutated, _ = _mutate_block_at_index(ast, target, rng, ctx, limits)
    repaired = _repair_limits(mutated, rng, limits, _infer_context_from_program(mutated))
    out = _as_genome(repaired)
    if validate_genome(out, limits).is_valid:
        return out
    return make_random_genome(seed=seed + 1, limits=limits)


def _crossover_top_level_splice_block(a: Block, b: Block, rng: random.Random, limits: Limits) -> Block:
    stmts_a = a.stmts
    stmts_b = b.stmts
    if not stmts_a and not stmts_b:
        return Block([Return(Const(0))])
    cut_a = rng.randint(0, len(stmts_a)) if stmts_a else 0
    cut_b = rng.randint(0, len(stmts_b)) if stmts_b else 0
    child = list(stmts_a[:cut_a]) + list(stmts_b[cut_b:])
    if not child:
        child = [Return(Const(0))]
    return Block(child[: limits.max_stmts_per_block])


def _collect_typed_expr_nodes(b: Block, env: Dict[str, RType] | None = None) -> List[Tuple[Expr, RType]]:
    out: List[Tuple[Expr, RType]] = []
    env = dict(env or {})

    def walk_expr(e: Expr, local_env: Dict[str, RType]) -> None:
        t = _infer_expr_type(e, local_env)
        if t in (RType.NUM, RType.BOOL, RType.NONE):
            out.append((e, t))
        if isinstance(e, Unary):
            walk_expr(e.e, local_env)
            return
        if isinstance(e, Binary):
            walk_expr(e.a, local_env)
            walk_expr(e.b, local_env)
            return
        if isinstance(e, IfExpr):
            walk_expr(e.cond, local_env)
            walk_expr(e.then_e, local_env)
            walk_expr(e.else_e, local_env)
            return
        if isinstance(e, Call):
            for arg in e.args:
                walk_expr(arg, local_env)

    def walk_block(block: Block, local_env: Dict[str, RType]) -> Dict[str, RType]:
        cur = dict(local_env)
        for st in block.stmts:
            if isinstance(st, Assign):
                walk_expr(st.e, cur)
                t = _infer_expr_type(st.e, cur)
                if t in (RType.NUM, RType.BOOL, RType.NONE):
                    cur[st.name] = t
            elif isinstance(st, Return):
                walk_expr(st.e, cur)
            elif isinstance(st, IfStmt):
                walk_expr(st.cond, cur)
                walk_block(st.then_block, dict(cur))
                walk_block(st.else_block, dict(cur))
            elif isinstance(st, ForRange):
                body_env = dict(cur)
                body_env[st.var] = RType.NUM
                walk_block(st.body, body_env)
        return cur

    walk_block(b, env)
    return out


def _collect_stmt_nodes(b: Block) -> List[Stmt]:
    out: List[Stmt] = []

    def walk_block(block: Block) -> None:
        for st in block.stmts:
            out.append(st)
            if isinstance(st, IfStmt):
                walk_block(st.then_block)
                walk_block(st.else_block)
            elif isinstance(st, ForRange):
                walk_block(st.body)

    walk_block(b)
    return out


def _replace_expr_in_expr(e: Expr, target: Expr, repl: Expr) -> Expr:
    if e is target:
        return repl
    if isinstance(e, Unary):
        return Unary(e.op, _replace_expr_in_expr(e.e, target, repl))
    if isinstance(e, Binary):
        return Binary(
            e.op,
            _replace_expr_in_expr(e.a, target, repl),
            _replace_expr_in_expr(e.b, target, repl),
        )
    if isinstance(e, IfExpr):
        return IfExpr(
            _replace_expr_in_expr(e.cond, target, repl),
            _replace_expr_in_expr(e.then_e, target, repl),
            _replace_expr_in_expr(e.else_e, target, repl),
        )
    if isinstance(e, Call):
        return Call(e.name, [_replace_expr_in_expr(arg, target, repl) for arg in e.args])
    return e


def _replace_expr_in_stmt(st: Stmt, target: Expr, repl: Expr) -> Stmt:
    if isinstance(st, Assign):
        return Assign(st.name, _replace_expr_in_expr(st.e, target, repl))
    if isinstance(st, Return):
        return Return(_replace_expr_in_expr(st.e, target, repl))
    if isinstance(st, IfStmt):
        return IfStmt(
            _replace_expr_in_expr(st.cond, target, repl),
            _replace_expr_in_block(st.then_block, target, repl),
            _replace_expr_in_block(st.else_block, target, repl),
        )
    if isinstance(st, ForRange):
        return ForRange(st.var, st.k, _replace_expr_in_block(st.body, target, repl))
    return st


def _replace_expr_in_block(b: Block, target: Expr, repl: Expr) -> Block:
    return Block([_replace_expr_in_stmt(st, target, repl) for st in b.stmts])


def _replace_stmt_in_stmt(st: Stmt, target: Stmt, repl: Stmt) -> Stmt:
    if st is target:
        return repl
    if isinstance(st, IfStmt):
        return IfStmt(
            st.cond,
            _replace_stmt_in_block(st.then_block, target, repl),
            _replace_stmt_in_block(st.else_block, target, repl),
        )
    if isinstance(st, ForRange):
        return ForRange(st.var, st.k, _replace_stmt_in_block(st.body, target, repl))
    return st


def _replace_stmt_in_block(b: Block, target: Stmt, repl: Stmt) -> Block:
    return Block([_replace_stmt_in_stmt(st, target, repl) for st in b.stmts])


def crossover_top_level(parent_a: ProgramGenome, parent_b: ProgramGenome, seed: int = 0, limits: Limits | None = None) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    child_block = _crossover_top_level_splice_block(parent_a.ast, parent_b.ast, rng, limits)
    repaired = _repair_limits(child_block, rng, limits, _infer_context_from_program(child_block))
    out = _as_genome(repaired)
    if validate_genome(out, limits).is_valid:
        return out
    return make_random_genome(seed=seed + 7, limits=limits)


def crossover_typed_subtree(
    parent_a: ProgramGenome, parent_b: ProgramGenome, seed: int = 0, limits: Limits | None = None
) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)

    exprs_a = _collect_typed_expr_nodes(parent_a.ast)
    exprs_b = _collect_typed_expr_nodes(parent_b.ast)

    child_block: Block | None = None
    common_types = sorted({t for _, t in exprs_a} & {t for _, t in exprs_b}, key=lambda x: x.value)
    if common_types:
        chosen_type = rng.choice(common_types)
        expr_pool_a = [e for e, t in exprs_a if t == chosen_type]
        expr_pool_b = [e for e, t in exprs_b if t == chosen_type]
        if expr_pool_a and expr_pool_b:
            target = rng.choice(expr_pool_a)
            donor = rng.choice(expr_pool_b)
            child_block = _replace_expr_in_block(parent_a.ast, target, donor)

    if child_block is None:
        stmts_a = _collect_stmt_nodes(parent_a.ast)
        stmts_b = _collect_stmt_nodes(parent_b.ast)
        common_stmt_types = sorted({type(s) for s in stmts_a} & {type(s) for s in stmts_b}, key=lambda c: c.__name__)
        if common_stmt_types:
            chosen_cls = rng.choice(common_stmt_types)
            stmt_pool_a = [s for s in stmts_a if isinstance(s, chosen_cls)]
            stmt_pool_b = [s for s in stmts_b if isinstance(s, chosen_cls)]
            if stmt_pool_a and stmt_pool_b:
                target_s = rng.choice(stmt_pool_a)
                donor_s = rng.choice(stmt_pool_b)
                child_block = _replace_stmt_in_block(parent_a.ast, target_s, donor_s)

    if child_block is None:
        child_block = _crossover_top_level_splice_block(parent_a.ast, parent_b.ast, rng, limits)

    repaired = _repair_limits(child_block, rng, limits, _infer_context_from_program(child_block))
    out = _as_genome(repaired)
    if validate_genome(out, limits).is_valid:
        return out
    return make_random_genome(seed=seed + 17, limits=limits)


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


def _count_expr_nodes(e: Expr) -> int:
    if isinstance(e, (Const, Var)):
        return 1
    if isinstance(e, Unary):
        return 1 + _count_expr_nodes(e.e)
    if isinstance(e, Binary):
        return 1 + _count_expr_nodes(e.a) + _count_expr_nodes(e.b)
    if isinstance(e, IfExpr):
        return 1 + _count_expr_nodes(e.cond) + _count_expr_nodes(e.then_e) + _count_expr_nodes(e.else_e)
    if isinstance(e, Call):
        return 1 + sum(_count_expr_nodes(arg) for arg in e.args)
    return 1


def _count_stmt_nodes(st: Stmt) -> int:
    if isinstance(st, Assign):
        return 1 + _count_expr_nodes(st.e)
    if isinstance(st, Return):
        return 1 + _count_expr_nodes(st.e)
    if isinstance(st, IfStmt):
        return 1 + _count_expr_nodes(st.cond) + _count_block_nodes(st.then_block) + _count_block_nodes(st.else_block)
    if isinstance(st, ForRange):
        return 1 + _count_block_nodes(st.body)
    return 1


def _count_block_nodes(b: Block) -> int:
    return 1 + sum(_count_stmt_nodes(st) for st in b.stmts)


def _expr_depth(e: Expr) -> int:
    if isinstance(e, (Const, Var)):
        return 1
    if isinstance(e, Unary):
        return 1 + _expr_depth(e.e)
    if isinstance(e, Binary):
        return 1 + max(_expr_depth(e.a), _expr_depth(e.b))
    if isinstance(e, IfExpr):
        return 1 + max(_expr_depth(e.cond), _expr_depth(e.then_e), _expr_depth(e.else_e))
    if isinstance(e, Call):
        if not e.args:
            return 1
        return 1 + max(_expr_depth(arg) for arg in e.args)
    return 1


def _stmt_expr_depth(st: Stmt) -> int:
    if isinstance(st, Assign):
        return _expr_depth(st.e)
    if isinstance(st, Return):
        return _expr_depth(st.e)
    if isinstance(st, IfStmt):
        return max(_expr_depth(st.cond), _max_block_expr_depth(st.then_block), _max_block_expr_depth(st.else_block))
    if isinstance(st, ForRange):
        return _max_block_expr_depth(st.body)
    return 0


def _max_block_expr_depth(b: Block) -> int:
    if not b.stmts:
        return 0
    return max(_stmt_expr_depth(st) for st in b.stmts)


def _block_uses_builtins(b: Block) -> bool:
    def has_builtin_expr(e: Expr) -> bool:
        if isinstance(e, Call):
            return True
        if isinstance(e, Unary):
            return has_builtin_expr(e.e)
        if isinstance(e, Binary):
            return has_builtin_expr(e.a) or has_builtin_expr(e.b)
        if isinstance(e, IfExpr):
            return has_builtin_expr(e.cond) or has_builtin_expr(e.then_e) or has_builtin_expr(e.else_e)
        return False

    for st in b.stmts:
        if isinstance(st, (Assign, Return)) and has_builtin_expr(st.e):
            return True
        if isinstance(st, IfStmt):
            if has_builtin_expr(st.cond) or _block_uses_builtins(st.then_block) or _block_uses_builtins(st.else_block):
                return True
        if isinstance(st, ForRange) and _block_uses_builtins(st.body):
            return True
    return False


def _fallback_const_for_type(t: RType) -> Const:
    if t == RType.BOOL:
        return Const(False)
    if t == RType.NONE:
        return Const(None)
    return Const(0)


def _shrink_expr_depth(e: Expr, depth: int, expected: RType = RType.ANY) -> Expr:
    if depth <= 1:
        return _fallback_const_for_type(expected if expected != RType.ANY else RType.NUM)
    if isinstance(e, (Const, Var)):
        return e
    if isinstance(e, Unary):
        target = RType.NUM if e.op == UOp.NEG else RType.BOOL
        return Unary(e.op, _shrink_expr_depth(e.e, depth - 1, target))
    if isinstance(e, Binary):
        if e.op in (BOp.ADD, BOp.SUB, BOp.MUL, BOp.DIV, BOp.MOD):
            return Binary(e.op, _shrink_expr_depth(e.a, depth - 1, RType.NUM), _shrink_expr_depth(e.b, depth - 1, RType.NUM))
        if e.op in (BOp.AND, BOp.OR):
            return Binary(e.op, _shrink_expr_depth(e.a, depth - 1, RType.BOOL), _shrink_expr_depth(e.b, depth - 1, RType.BOOL))
        return Binary(e.op, _shrink_expr_depth(e.a, depth - 1, RType.NUM), _shrink_expr_depth(e.b, depth - 1, RType.NUM))
    if isinstance(e, IfExpr):
        t = RType.NUM if expected == RType.ANY else expected
        return IfExpr(
            _shrink_expr_depth(e.cond, depth - 1, RType.BOOL),
            _shrink_expr_depth(e.then_e, depth - 1, t),
            _shrink_expr_depth(e.else_e, depth - 1, t),
        )
    if isinstance(e, Call):
        return Call(e.name, [_shrink_expr_depth(arg, depth - 1, RType.NUM) for arg in e.args])
    return _fallback_const_for_type(expected)


def _shrink_stmt_expr_depth(st: Stmt, depth: int) -> Stmt:
    if isinstance(st, Assign):
        return Assign(st.name, _shrink_expr_depth(st.e, depth))
    if isinstance(st, Return):
        return Return(_shrink_expr_depth(st.e, depth))
    if isinstance(st, IfStmt):
        return IfStmt(
            _shrink_expr_depth(st.cond, depth, RType.BOOL),
            _shrink_block_expr_depth(st.then_block, depth),
            _shrink_block_expr_depth(st.else_block, depth),
        )
    if isinstance(st, ForRange):
        return ForRange(st.var, st.k, _shrink_block_expr_depth(st.body, depth))
    return st


def _shrink_block_expr_depth(b: Block, depth: int) -> Block:
    return Block([_shrink_stmt_expr_depth(st, depth) for st in b.stmts])


def _clamp_block_stmt_count(b: Block, max_stmts: int) -> Block:
    if len(b.stmts) <= max_stmts:
        out: List[Stmt] = []
        for st in b.stmts:
            if isinstance(st, IfStmt):
                out.append(IfStmt(st.cond, _clamp_block_stmt_count(st.then_block, max_stmts), _clamp_block_stmt_count(st.else_block, max_stmts)))
            elif isinstance(st, ForRange):
                out.append(ForRange(st.var, st.k, _clamp_block_stmt_count(st.body, max_stmts)))
            else:
                out.append(st)
        return Block(out)
    trimmed = list(b.stmts[:max_stmts])
    return Block(trimmed)


def _clamp_for_k(b: Block, max_for_k: int) -> Block:
    out: List[Stmt] = []
    for st in b.stmts:
        if isinstance(st, IfStmt):
            out.append(IfStmt(st.cond, _clamp_for_k(st.then_block, max_for_k), _clamp_for_k(st.else_block, max_for_k)))
        elif isinstance(st, ForRange):
            k = st.k
            if not isinstance(k, int) or isinstance(k, bool):
                k = 0
            k = max(0, min(max_for_k, k))
            out.append(ForRange(st.var, k, _clamp_for_k(st.body, max_for_k)))
        else:
            out.append(st)
    return Block(out)


def _infer_env_from_program(b: Block) -> Dict[str, RType]:
    env: Dict[str, RType] = {}
    for st in b.stmts:
        if isinstance(st, Assign):
            t = _infer_expr_type(st.e, env)
            if t in (RType.NUM, RType.BOOL, RType.NONE):
                env[st.name] = t
        elif isinstance(st, ForRange):
            env[st.var] = RType.NUM
    return env


def _infer_context_from_program(b: Block) -> _GenContext:
    ctx = _GenContext.empty()

    def walk_block(block: Block, env: Dict[str, RType]) -> Dict[str, RType]:
        local = dict(env)
        for st in block.stmts:
            if isinstance(st, Assign):
                t = _infer_expr_type(st.e, local)
                if t in (RType.NUM, RType.BOOL, RType.NONE):
                    local[st.name] = t
                    ctx.add_var(st.name, t)
            elif isinstance(st, IfStmt):
                walk_block(st.then_block, dict(local))
                walk_block(st.else_block, dict(local))
            elif isinstance(st, ForRange):
                local[st.var] = RType.NUM
                ctx.add_var(st.var, RType.NUM)
                walk_block(st.body, dict(local))
        return local

    walk_block(b, {})
    return ctx
