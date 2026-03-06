from __future__ import annotations

import random

from .genome import Limits, ProgramGenome
from .random_tree import rand_block
from .stmt_codec import genome_from_statements


def make_random_genome(seed: int = 0, limits: Limits | None = None) -> ProgramGenome:
    limits = limits or Limits()
    rng = random.Random(seed)
    for _ in range(256):
        statements = rand_block(rng, limits.max_expr_depth, limits, force_return=True)
        genome = genome_from_statements(statements)
        if genome.meta.node_count <= limits.max_total_nodes:
            return genome
    return genome_from_statements([("return", ("const", 0))])
