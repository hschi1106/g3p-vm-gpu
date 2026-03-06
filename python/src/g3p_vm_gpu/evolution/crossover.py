from __future__ import annotations

import random

from .genome import Limits, ProgramGenome
from .random_genome import make_random_genome
from .random_tree import RType, rand_expr
from .stmt_codec import genome_from_statements, top_level_statements


def _ensure_terminal_return(statements: list[tuple], rng: random.Random, limits: Limits) -> list[tuple]:
    out = list(statements[: limits.max_stmts_per_block])
    if any(stmt[0] == "return" for stmt in out):
        return out
    ret = ("return", rand_expr(rng, max(1, limits.max_expr_depth - 1), RType.NUM))
    if len(out) < limits.max_stmts_per_block:
        out.append(ret)
    elif out:
        out[-1] = ret
    else:
        out = [ret]
    return out


def crossover(
    parent_a: ProgramGenome,
    parent_b: ProgramGenome,
    seed: int = 0,
    limits: Limits | None = None,
) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    try:
        a_stmts = top_level_statements(parent_a.ast)
        b_stmts = top_level_statements(parent_b.ast)
    except Exception:
        return make_random_genome(seed=seed, limits=limits)

    cut_a = rng.randint(0, len(a_stmts)) if a_stmts else 0
    cut_b = rng.randint(0, len(b_stmts)) if b_stmts else 0
    child = list(a_stmts[:cut_a]) + list(b_stmts[cut_b:])
    if not child:
        child = [("return", ("const", 0))]
    child = _ensure_terminal_return(child, rng, limits)

    out = genome_from_statements(child)
    if out.meta.node_count > limits.max_total_nodes:
        return parent_a
    return out
