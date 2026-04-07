from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..core.ast import AstProgram, NodeKind, max_expr_depth, node_count, prefix_repr
from ..runtime.compiler import BytecodeProgram, compile_program


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


def build_genome_meta(ast: AstProgram) -> GenomeMeta:
    digest = hashlib.sha1(prefix_repr(ast).encode("utf-8")).hexdigest()[:16]
    uses_builtins = any(
        node.kind
        in {
            NodeKind.CALL_ABS,
            NodeKind.CALL_MIN,
            NodeKind.CALL_MAX,
            NodeKind.CALL_CLIP,
            NodeKind.CALL_LEN,
            NodeKind.CALL_CONCAT,
            NodeKind.CALL_SLICE,
            NodeKind.CALL_INDEX,
            NodeKind.CALL_APPEND,
            NodeKind.CALL_REVERSE,
            NodeKind.CALL_FIND,
            NodeKind.CALL_CONTAINS,
        }
        for node in ast.nodes
    )
    return GenomeMeta(
        node_count=node_count(ast),
        max_depth=max_expr_depth(ast),
        uses_builtins=uses_builtins,
        hash_key=digest,
    )


def as_genome(ast: AstProgram) -> ProgramGenome:
    return ProgramGenome(ast=ast, meta=build_genome_meta(ast))


def compile_for_eval(genome: ProgramGenome) -> BytecodeProgram:
    return compile_program(genome.ast)
