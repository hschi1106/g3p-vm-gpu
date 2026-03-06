from __future__ import annotations

import random

from .genome import Limits, ProgramGenome
from .random_genome import make_random_genome
from .random_tree import RType, rand_expr, rand_stmt
from .stmt_codec import genome_from_statements, top_level_statements


def _replace_statement_mutation(statements: list[tuple], rng: random.Random, limits: Limits) -> list[tuple]:
    mutated = list(statements)
    if not mutated:
        return mutated
    mutated[rng.randrange(len(mutated))] = rand_stmt(rng, limits.max_expr_depth, limits)
    return mutated


def _structural_mutation(statements: list[tuple], rng: random.Random, limits: Limits) -> list[tuple]:
    mutated = list(statements)
    if len(mutated) < limits.max_stmts_per_block and rng.random() < 0.5:
        mutated.insert(rng.randint(0, len(mutated)), rand_stmt(rng, limits.max_expr_depth, limits))
    elif len(mutated) > 1:
        del mutated[rng.randrange(len(mutated))]
    return mutated


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


def mutate(
    genome: ProgramGenome,
    seed: int = 0,
    limits: Limits | None = None,
    mutation_subtree_prob: float = 0.8,
) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    try:
        statements = top_level_statements(genome.ast)
    except Exception:
        return make_random_genome(seed=seed, limits=limits)
    if not statements:
        return make_random_genome(seed=seed, limits=limits)

    subtree_prob = min(max(mutation_subtree_prob, 0.0), 1.0)
    if rng.random() < subtree_prob:
        mutated = _replace_statement_mutation(statements, rng, limits)
    else:
        mutated = _structural_mutation(statements, rng, limits)

    mutated = _ensure_terminal_return(mutated, rng, limits)

    out = genome_from_statements(mutated)
    if out.meta.node_count > limits.max_total_nodes:
        return genome
    return out
