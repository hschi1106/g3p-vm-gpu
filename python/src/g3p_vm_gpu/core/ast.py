from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List as TList, Sequence, Union


@dataclass(frozen=True)
class Char:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("Char value must be a string")
        if len(self.value) != 1:
            raise ValueError("Char value must contain exactly one character")


@dataclass(frozen=True)
class IntList:
    items: tuple[int, ...]

    def __post_init__(self) -> None:
        for item in self.items:
            if not isinstance(item, int) or isinstance(item, bool):
                raise TypeError("IntList elements must be int")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int | slice) -> int | tuple[int, ...]:
        return self.items[idx]


@dataclass(frozen=True)
class FloatList:
    items: tuple[float, ...]

    def __post_init__(self) -> None:
        for item in self.items:
            if not isinstance(item, float):
                raise TypeError("FloatList elements must be float")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int | slice) -> float | tuple[float, ...]:
        return self.items[idx]


@dataclass(frozen=True)
class StringList:
    items: tuple[str, ...]

    def __post_init__(self) -> None:
        for item in self.items:
            if not isinstance(item, str):
                raise TypeError("StringList elements must be strings")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int | slice) -> str | tuple[str, ...]:
        return self.items[idx]


Val = Union[int, float, bool, Char, str, IntList, FloatList, StringList]


class ListTypeTag(IntEnum):
    INT = 1
    FLOAT = 2
    STRING = 3


LIST_TYPE_TAG_BY_NAME: Dict[str, ListTypeTag] = {
    "int": ListTypeTag.INT,
    "float": ListTypeTag.FLOAT,
    "string": ListTypeTag.STRING,
}


def make_char(value: str) -> Char:
    return Char(value)


def make_int_list(values: Sequence[int]) -> IntList:
    return IntList(tuple(values))


def make_float_list(values: Sequence[float]) -> FloatList:
    return FloatList(tuple(values))


def make_string_list(values: Sequence[str]) -> StringList:
    return StringList(tuple(values))


def empty_list_for_tag(tag: int) -> IntList | FloatList | StringList:
    if tag == ListTypeTag.INT:
        return make_int_list(())
    if tag == ListTypeTag.FLOAT:
        return make_float_list(())
    if tag == ListTypeTag.STRING:
        return make_string_list(())
    raise ValueError(f"unknown list type tag: {tag}")


def list_tag_for_value(v: Val) -> ListTypeTag | None:
    if isinstance(v, IntList):
        return ListTypeTag.INT
    if isinstance(v, FloatList):
        return ListTypeTag.FLOAT
    if isinstance(v, StringList):
        return ListTypeTag.STRING
    return None


def normalize_value(v: object) -> Val:
    if isinstance(v, Char):
        return make_char(v.value)
    if isinstance(v, IntList):
        return make_int_list(v.items)
    if isinstance(v, FloatList):
        return make_float_list(v.items)
    if isinstance(v, StringList):
        return make_string_list(v.items)
    if isinstance(v, list):
        if not v:
            raise ValueError("ambiguous empty list constant; use IntList(()), FloatList(()), or StringList(())")
        if all(isinstance(item, int) and not isinstance(item, bool) for item in v):
            return make_int_list(v)
        if all(isinstance(item, float) for item in v):
            return make_float_list(v)
        if all(isinstance(item, str) for item in v):
            return make_string_list(v)
        raise ValueError("list constants must be homogeneous int, float, or string lists")
    if v is None:
        raise ValueError("None is not a public current value")
    if isinstance(v, (int, float, bool, str)):
        return v
    raise ValueError(f"unsupported constant value: {v!r}")


class UOp(str, Enum):
    NEG = "NEG"
    NOT = "NOT"


class BOp(str, Enum):
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    EQ = "EQ"
    NE = "NE"
    AND = "AND"
    OR = "OR"


class NodeKind(str, Enum):
    PROGRAM = "PROGRAM"
    BLOCK_NIL = "BLOCK_NIL"
    BLOCK_CONS = "BLOCK_CONS"
    ASSIGN = "ASSIGN"
    IF_STMT = "IF_STMT"
    FOR_RANGE = "FOR_RANGE"
    RETURN = "RETURN"
    CONST = "CONST"
    VAR = "VAR"
    NEG = "NEG"
    NOT = "NOT"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    EQ = "EQ"
    NE = "NE"
    AND = "AND"
    OR = "OR"
    IF_EXPR = "IF_EXPR"
    CALL_ABS = "CALL_ABS"
    CALL_MIN = "CALL_MIN"
    CALL_MAX = "CALL_MAX"
    CALL_CLIP = "CALL_CLIP"
    CALL_IDIV0 = "CALL_IDIV0"
    CALL_IMOD0 = "CALL_IMOD0"
    CALL_LEN = "CALL_LEN"
    CALL_CONCAT = "CALL_CONCAT"
    CALL_SLICE = "CALL_SLICE"
    CALL_INDEX = "CALL_INDEX"
    CALL_APPEND = "CALL_APPEND"
    CALL_REVERSE = "CALL_REVERSE"
    CALL_FIND = "CALL_FIND"
    CALL_CONTAINS = "CALL_CONTAINS"
    CALL_PREPEND = "CALL_PREPEND"
    CALL_CHAR_TO_STRING = "CALL_CHAR_TO_STRING"
    CALL_STRING_TO_CHAR = "CALL_STRING_TO_CHAR"
    CALL_ORD = "CALL_ORD"
    CALL_CHR = "CALL_CHR"
    CALL_IS_LETTER = "CALL_IS_LETTER"
    CALL_IS_DIGIT = "CALL_IS_DIGIT"
    CALL_IS_SPACE = "CALL_IS_SPACE"
    CALL_IS_VOWEL = "CALL_IS_VOWEL"
    CALL_TO_LOWER = "CALL_TO_LOWER"
    CALL_TO_UPPER = "CALL_TO_UPPER"
    CALL_TO_STRING = "CALL_TO_STRING"
    CALL_SINGLETON = "CALL_SINGLETON"
    BOUND_VAR = "BOUND_VAR"
    MAP_LIST = "MAP_LIST"
    FILTER_LIST = "FILTER_LIST"
    LINEAR_REC = "LINEAR_REC"
    ASGP_DC = "ASGP_DC"
    ASGP_DP1D = "ASGP_DP1D"
    ASGP_DP2D = "ASGP_DP2D"
    DP1_BACKWARD1 = "DP1_BACKWARD1"
    DP1_BACKWARD2 = "DP1_BACKWARD2"
    DP1_BACKWARD3 = "DP1_BACKWARD3"
    DP1_FORWARD1 = "DP1_FORWARD1"
    DP1_FORWARD2 = "DP1_FORWARD2"
    DP1_FORWARD3 = "DP1_FORWARD3"
    DP2_CROSS_BACKWARD = "DP2_CROSS_BACKWARD"
    DP2_CROSS_FORWARD = "DP2_CROSS_FORWARD"
    DP2_DIAGONAL_BACKWARD = "DP2_DIAGONAL_BACKWARD"
    DP2_DIAGONAL_FORWARD = "DP2_DIAGONAL_FORWARD"
    DP2_NEIGHBORHOOD_BACKWARD3 = "DP2_NEIGHBORHOOD_BACKWARD3"
    DP2_NEIGHBORHOOD_FORWARD3 = "DP2_NEIGHBORHOOD_FORWARD3"


NODE_ARITY: Dict[NodeKind, int] = {
    NodeKind.PROGRAM: 1,
    NodeKind.BLOCK_NIL: 0,
    NodeKind.BLOCK_CONS: 2,
    NodeKind.ASSIGN: 1,
    NodeKind.IF_STMT: 3,
    NodeKind.FOR_RANGE: 2,
    NodeKind.RETURN: 1,
    NodeKind.CONST: 0,
    NodeKind.VAR: 0,
    NodeKind.NEG: 1,
    NodeKind.NOT: 1,
    NodeKind.ADD: 2,
    NodeKind.SUB: 2,
    NodeKind.MUL: 2,
    NodeKind.DIV: 2,
    NodeKind.MOD: 2,
    NodeKind.LT: 2,
    NodeKind.LE: 2,
    NodeKind.GT: 2,
    NodeKind.GE: 2,
    NodeKind.EQ: 2,
    NodeKind.NE: 2,
    NodeKind.AND: 2,
    NodeKind.OR: 2,
    NodeKind.IF_EXPR: 3,
    NodeKind.CALL_ABS: 1,
    NodeKind.CALL_MIN: 2,
    NodeKind.CALL_MAX: 2,
    NodeKind.CALL_CLIP: 3,
    NodeKind.CALL_IDIV0: 2,
    NodeKind.CALL_IMOD0: 2,
    NodeKind.CALL_LEN: 1,
    NodeKind.CALL_CONCAT: 2,
    NodeKind.CALL_SLICE: 3,
    NodeKind.CALL_INDEX: 2,
    NodeKind.CALL_APPEND: 2,
    NodeKind.CALL_REVERSE: 1,
    NodeKind.CALL_FIND: 2,
    NodeKind.CALL_CONTAINS: 2,
    NodeKind.CALL_PREPEND: 2,
    NodeKind.CALL_CHAR_TO_STRING: 1,
    NodeKind.CALL_STRING_TO_CHAR: 1,
    NodeKind.CALL_ORD: 1,
    NodeKind.CALL_CHR: 1,
    NodeKind.CALL_IS_LETTER: 1,
    NodeKind.CALL_IS_DIGIT: 1,
    NodeKind.CALL_IS_SPACE: 1,
    NodeKind.CALL_IS_VOWEL: 1,
    NodeKind.CALL_TO_LOWER: 1,
    NodeKind.CALL_TO_UPPER: 1,
    NodeKind.CALL_TO_STRING: 1,
    NodeKind.CALL_SINGLETON: 1,
    NodeKind.BOUND_VAR: 0,
    NodeKind.MAP_LIST: 2,
    NodeKind.FILTER_LIST: 2,
    NodeKind.LINEAR_REC: 5,
    NodeKind.ASGP_DC: 4,
    NodeKind.ASGP_DP1D: 3,
    NodeKind.ASGP_DP2D: 4,
    NodeKind.DP1_BACKWARD1: 0,
    NodeKind.DP1_BACKWARD2: 0,
    NodeKind.DP1_BACKWARD3: 0,
    NodeKind.DP1_FORWARD1: 0,
    NodeKind.DP1_FORWARD2: 0,
    NodeKind.DP1_FORWARD3: 0,
    NodeKind.DP2_CROSS_BACKWARD: 0,
    NodeKind.DP2_CROSS_FORWARD: 0,
    NodeKind.DP2_DIAGONAL_BACKWARD: 0,
    NodeKind.DP2_DIAGONAL_FORWARD: 0,
    NodeKind.DP2_NEIGHBORHOOD_BACKWARD3: 0,
    NodeKind.DP2_NEIGHBORHOOD_FORWARD3: 0,
}


EXPR_KINDS = {
    NodeKind.CONST,
    NodeKind.VAR,
    NodeKind.NEG,
    NodeKind.NOT,
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
    NodeKind.IF_EXPR,
    NodeKind.CALL_ABS,
    NodeKind.CALL_MIN,
    NodeKind.CALL_MAX,
    NodeKind.CALL_CLIP,
    NodeKind.CALL_IDIV0,
    NodeKind.CALL_IMOD0,
    NodeKind.CALL_LEN,
    NodeKind.CALL_CONCAT,
    NodeKind.CALL_SLICE,
    NodeKind.CALL_INDEX,
    NodeKind.CALL_APPEND,
    NodeKind.CALL_REVERSE,
    NodeKind.CALL_FIND,
    NodeKind.CALL_CONTAINS,
    NodeKind.CALL_PREPEND,
    NodeKind.CALL_CHAR_TO_STRING,
    NodeKind.CALL_STRING_TO_CHAR,
    NodeKind.CALL_ORD,
    NodeKind.CALL_CHR,
    NodeKind.CALL_IS_LETTER,
    NodeKind.CALL_IS_DIGIT,
    NodeKind.CALL_IS_SPACE,
    NodeKind.CALL_IS_VOWEL,
    NodeKind.CALL_TO_LOWER,
    NodeKind.CALL_TO_UPPER,
    NodeKind.CALL_TO_STRING,
    NodeKind.CALL_SINGLETON,
    NodeKind.BOUND_VAR,
    NodeKind.MAP_LIST,
    NodeKind.FILTER_LIST,
    NodeKind.LINEAR_REC,
    NodeKind.ASGP_DC,
    NodeKind.ASGP_DP1D,
    NodeKind.ASGP_DP2D,
}

BUILTIN_NODE_BY_NAME: Dict[str, NodeKind] = {
    "abs": NodeKind.CALL_ABS,
    "min": NodeKind.CALL_MIN,
    "max": NodeKind.CALL_MAX,
    "clip": NodeKind.CALL_CLIP,
    "idiv0": NodeKind.CALL_IDIV0,
    "imod0": NodeKind.CALL_IMOD0,
    "len": NodeKind.CALL_LEN,
    "concat": NodeKind.CALL_CONCAT,
    "slice": NodeKind.CALL_SLICE,
    "index": NodeKind.CALL_INDEX,
    "append": NodeKind.CALL_APPEND,
    "prepend": NodeKind.CALL_PREPEND,
    "reverse": NodeKind.CALL_REVERSE,
    "find": NodeKind.CALL_FIND,
    "contains": NodeKind.CALL_CONTAINS,
    "char_to_string": NodeKind.CALL_CHAR_TO_STRING,
    "string_to_char": NodeKind.CALL_STRING_TO_CHAR,
    "ord": NodeKind.CALL_ORD,
    "chr": NodeKind.CALL_CHR,
    "is_letter": NodeKind.CALL_IS_LETTER,
    "is_digit": NodeKind.CALL_IS_DIGIT,
    "is_space": NodeKind.CALL_IS_SPACE,
    "is_vowel": NodeKind.CALL_IS_VOWEL,
    "to_lower": NodeKind.CALL_TO_LOWER,
    "to_upper": NodeKind.CALL_TO_UPPER,
    "to_string": NodeKind.CALL_TO_STRING,
    "singleton": NodeKind.CALL_SINGLETON,
}

BUILTIN_NAME_BY_NODE: Dict[NodeKind, str] = {kind: name for name, kind in BUILTIN_NODE_BY_NAME.items()}


STMT_KINDS = {
    NodeKind.ASSIGN,
    NodeKind.IF_STMT,
    NodeKind.FOR_RANGE,
    NodeKind.RETURN,
}


@dataclass(frozen=True)
class AstNode:
    kind: NodeKind
    i0: int = 0
    i1: int = 0


@dataclass(frozen=True)
class LinearRecBinders:
    node_index: int
    elem_name: int
    accum_name: int
    index_name: int


@dataclass(frozen=True)
class AsgpDcBinders:
    node_index: int
    solve_xs_name: int
    solve_n_name: int
    solve_lo_name: int
    divide_n_name: int
    combine_left_name: int
    combine_right_name: int


@dataclass(frozen=True)
class AsgpDp1dSpec:
    node_index: int
    lo: int
    hi: int
    base_state: int
    boundary_const: int
    dep_kind: NodeKind
    dep_offsets: tuple[int, ...]
    solve_state_name: int
    transition_state_name: int
    transition_dep_names: tuple[int, ...]


@dataclass(frozen=True)
class AsgpDp2dSpec:
    node_index: int
    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    base_i: int
    base_j: int
    boundary_const: int
    dep_kind: NodeKind
    solve_i_name: int
    solve_j_name: int
    transition_i_name: int
    transition_j_name: int
    transition_dep_names: tuple[int, ...]


@dataclass(frozen=True)
class AstProgram:
    nodes: Sequence[AstNode]
    names: Sequence[str]
    consts: Sequence[Val]
    linear_rec_binders: Sequence[LinearRecBinders] = ()
    asgp_dc_binders: Sequence[AsgpDcBinders] = ()
    asgp_dp1d_specs: Sequence[AsgpDp1dSpec] = ()
    asgp_dp2d_specs: Sequence[AsgpDp2dSpec] = ()
    version: str = "ast-prefix"


def prefix_subtree_end(nodes: Sequence[AstNode], start: int) -> int:
    if start >= len(nodes):
        raise ValueError("invalid prefix AST: out-of-range subtree start")
    arity = NODE_ARITY[nodes[start].kind]
    idx = start + 1
    for _ in range(arity):
        idx = prefix_subtree_end(nodes, idx)
    return idx


def prefix_child_index(nodes: Sequence[AstNode], start: int, child_no: int) -> int:
    arity = NODE_ARITY[nodes[start].kind]
    if child_no < 0 or child_no >= arity:
        raise ValueError("child_no out of range")
    idx = start + 1
    for _ in range(child_no):
        idx = prefix_subtree_end(nodes, idx)
    return idx


def validate_prefix_program(program: AstProgram) -> None:
    if program.version != "ast-prefix":
        raise ValueError(f"unsupported ast version: {program.version}")
    if not program.nodes:
        raise ValueError("invalid prefix AST: empty node list")
    if program.nodes[0].kind != NodeKind.PROGRAM:
        raise ValueError("invalid prefix AST: root must be PROGRAM")

    end = prefix_subtree_end(program.nodes, 0)
    if end != len(program.nodes):
        raise ValueError("invalid prefix AST: trailing tokens")

    linear_rec_nodes = set()
    asgp_dc_nodes = set()
    asgp_dp1d_nodes = set()
    asgp_dp2d_nodes = set()
    for idx, n in enumerate(program.nodes):
        if n.kind == NodeKind.CONST and not (0 <= n.i0 < len(program.consts)):
            raise ValueError("invalid prefix AST: const index out of range")
        if n.kind in (NodeKind.VAR, NodeKind.BOUND_VAR, NodeKind.ASSIGN, NodeKind.FOR_RANGE) and not (0 <= n.i0 < len(program.names)):
            raise ValueError("invalid prefix AST: name index out of range")
        if n.kind in (NodeKind.MAP_LIST, NodeKind.FILTER_LIST) and not (0 <= n.i0 < len(program.names)):
            raise ValueError("invalid prefix AST: binder name index out of range")
        if n.kind == NodeKind.MAP_LIST and n.i1 not in set(ListTypeTag):
            raise ValueError("invalid prefix AST: map output list type tag out of range")
        if n.kind == NodeKind.LINEAR_REC:
            linear_rec_nodes.add(idx)
        if n.kind == NodeKind.ASGP_DC:
            asgp_dc_nodes.add(idx)
        if n.kind == NodeKind.ASGP_DP1D:
            asgp_dp1d_nodes.add(idx)
        if n.kind == NodeKind.ASGP_DP2D:
            asgp_dp2d_nodes.add(idx)
    seen_linear_rec_metadata = set()
    for binders in program.linear_rec_binders:
        if binders.node_index in seen_linear_rec_metadata:
            raise ValueError("invalid prefix AST: duplicate LinearRec binder metadata")
        seen_linear_rec_metadata.add(binders.node_index)
        if binders.node_index not in linear_rec_nodes:
            raise ValueError("invalid prefix AST: LinearRec binder metadata points at non-LinearRec node")
        for name_idx in (binders.elem_name, binders.accum_name, binders.index_name):
            if not (0 <= name_idx < len(program.names)):
                raise ValueError("invalid prefix AST: LinearRec binder name index out of range")
    if seen_linear_rec_metadata != linear_rec_nodes:
        raise ValueError("invalid prefix AST: LinearRec binder metadata mismatch")
    seen_asgp_dc_metadata = set()
    for binders in program.asgp_dc_binders:
        if binders.node_index in seen_asgp_dc_metadata:
            raise ValueError("invalid prefix AST: duplicate ASGP-DC binder metadata")
        seen_asgp_dc_metadata.add(binders.node_index)
        if binders.node_index not in asgp_dc_nodes:
            raise ValueError("invalid prefix AST: ASGP-DC binder metadata points at non-ASGP-DC node")
        for name_idx in (
            binders.solve_xs_name,
            binders.solve_n_name,
            binders.solve_lo_name,
            binders.divide_n_name,
            binders.combine_left_name,
            binders.combine_right_name,
        ):
            if not (0 <= name_idx < len(program.names)):
                raise ValueError("invalid prefix AST: ASGP-DC binder name index out of range")
    if seen_asgp_dc_metadata != asgp_dc_nodes:
        raise ValueError("invalid prefix AST: ASGP-DC binder metadata mismatch")
    seen_asgp_dp1d_metadata = set()
    dp1_kinds = {
        NodeKind.DP1_BACKWARD1,
        NodeKind.DP1_BACKWARD2,
        NodeKind.DP1_BACKWARD3,
        NodeKind.DP1_FORWARD1,
        NodeKind.DP1_FORWARD2,
        NodeKind.DP1_FORWARD3,
    }
    for spec in program.asgp_dp1d_specs:
        if spec.node_index in seen_asgp_dp1d_metadata:
            raise ValueError("invalid prefix AST: duplicate ASGP-DP1D metadata")
        seen_asgp_dp1d_metadata.add(spec.node_index)
        if spec.node_index not in asgp_dp1d_nodes:
            raise ValueError("invalid prefix AST: ASGP-DP1D metadata points at non-ASGP-DP1D node")
        if spec.lo > spec.hi:
            raise ValueError("invalid prefix AST: ASGP-DP1D bounds must satisfy lo <= hi")
        if not (0 <= spec.boundary_const < len(program.consts)):
            raise ValueError("invalid prefix AST: ASGP-DP1D boundary const index out of range")
        if spec.dep_kind not in dp1_kinds:
            raise ValueError("invalid prefix AST: ASGP-DP1D dependency kind is not DP1")
        expected_k = {
            NodeKind.DP1_BACKWARD1: 1,
            NodeKind.DP1_FORWARD1: 1,
            NodeKind.DP1_BACKWARD2: 2,
            NodeKind.DP1_FORWARD2: 2,
            NodeKind.DP1_BACKWARD3: 3,
            NodeKind.DP1_FORWARD3: 3,
        }[spec.dep_kind]
        if len(spec.dep_offsets) != expected_k or len(spec.transition_dep_names) != expected_k:
            raise ValueError("invalid prefix AST: ASGP-DP1D dependency arity mismatch")
        if any(offset <= 0 for offset in spec.dep_offsets):
            raise ValueError("invalid prefix AST: ASGP-DP1D dependency offsets must be positive")
        for name_idx in (spec.solve_state_name, spec.transition_state_name, *spec.transition_dep_names):
            if not (0 <= name_idx < len(program.names)):
                raise ValueError("invalid prefix AST: ASGP-DP1D binder name index out of range")
    if seen_asgp_dp1d_metadata != asgp_dp1d_nodes:
        raise ValueError("invalid prefix AST: ASGP-DP1D metadata mismatch")
    seen_asgp_dp2d_metadata = set()
    dp2_expected_k = {
        NodeKind.DP2_CROSS_BACKWARD: 2,
        NodeKind.DP2_CROSS_FORWARD: 2,
        NodeKind.DP2_DIAGONAL_BACKWARD: 1,
        NodeKind.DP2_DIAGONAL_FORWARD: 1,
        NodeKind.DP2_NEIGHBORHOOD_BACKWARD3: 3,
        NodeKind.DP2_NEIGHBORHOOD_FORWARD3: 3,
    }
    for spec in program.asgp_dp2d_specs:
        if spec.node_index in seen_asgp_dp2d_metadata:
            raise ValueError("invalid prefix AST: duplicate ASGP-DP2D metadata")
        seen_asgp_dp2d_metadata.add(spec.node_index)
        if spec.node_index not in asgp_dp2d_nodes:
            raise ValueError("invalid prefix AST: ASGP-DP2D metadata points at non-ASGP-DP2D node")
        if spec.i_lo > spec.i_hi or spec.j_lo > spec.j_hi:
            raise ValueError("invalid prefix AST: ASGP-DP2D bounds must satisfy lo <= hi")
        if not (0 <= spec.boundary_const < len(program.consts)):
            raise ValueError("invalid prefix AST: ASGP-DP2D boundary const index out of range")
        expected_k = dp2_expected_k.get(spec.dep_kind)
        if expected_k is None:
            raise ValueError("invalid prefix AST: ASGP-DP2D dependency kind is not DP2")
        if len(spec.transition_dep_names) != expected_k:
            raise ValueError("invalid prefix AST: ASGP-DP2D dependency arity mismatch")
        for name_idx in (
            spec.solve_i_name,
            spec.solve_j_name,
            spec.transition_i_name,
            spec.transition_j_name,
            *spec.transition_dep_names,
        ):
            if not (0 <= name_idx < len(program.names)):
                raise ValueError("invalid prefix AST: ASGP-DP2D binder name index out of range")
    if seen_asgp_dp2d_metadata != asgp_dp2d_nodes:
        raise ValueError("invalid prefix AST: ASGP-DP2D metadata mismatch")
    for c in program.consts:
        normalize_value(c)


def linear_rec_binders_for_node(program: AstProgram, node_index: int) -> LinearRecBinders:
    for binders in program.linear_rec_binders:
        if binders.node_index == node_index:
            return binders
    raise ValueError(f"missing LinearRec binder metadata for node {node_index}")


def asgp_dc_binders_for_node(program: AstProgram, node_index: int) -> AsgpDcBinders:
    for binders in program.asgp_dc_binders:
        if binders.node_index == node_index:
            return binders
    raise ValueError(f"missing ASGP-DC binder metadata for node {node_index}")


def asgp_dp1d_spec_for_node(program: AstProgram, node_index: int) -> AsgpDp1dSpec:
    for spec in program.asgp_dp1d_specs:
        if spec.node_index == node_index:
            return spec
    raise ValueError(f"missing ASGP-DP1D metadata for node {node_index}")


def asgp_dp2d_spec_for_node(program: AstProgram, node_index: int) -> AsgpDp2dSpec:
    for spec in program.asgp_dp2d_specs:
        if spec.node_index == node_index:
            return spec
    raise ValueError(f"missing ASGP-DP2D metadata for node {node_index}")


def prefix_repr(program: AstProgram) -> str:
    validate_prefix_program(program)
    node_repr = ",".join(f"{n.kind.value}:{n.i0}:{n.i1}" for n in program.nodes)
    binder_repr = ",".join(
        f"{b.node_index}:{b.elem_name}:{b.accum_name}:{b.index_name}" for b in program.linear_rec_binders
    )
    asgp_dc_repr = ",".join(
        f"{b.node_index}:{b.solve_xs_name}:{b.solve_n_name}:{b.solve_lo_name}:"
        f"{b.divide_n_name}:{b.combine_left_name}:{b.combine_right_name}"
        for b in program.asgp_dc_binders
    )
    asgp_dp1d_repr = ",".join(
        f"{s.node_index}:{s.lo}:{s.hi}:{s.base_state}:{s.boundary_const}:{s.dep_kind.value}:"
        f"{'/'.join(str(x) for x in s.dep_offsets)}:{s.solve_state_name}:{s.transition_state_name}:"
        f"{'/'.join(str(x) for x in s.transition_dep_names)}"
        for s in program.asgp_dp1d_specs
    )
    asgp_dp2d_repr = ",".join(
        f"{s.node_index}:{s.i_lo}:{s.i_hi}:{s.j_lo}:{s.j_hi}:{s.base_i}:{s.base_j}:"
        f"{s.boundary_const}:{s.dep_kind.value}:{s.solve_i_name}:{s.solve_j_name}:"
        f"{s.transition_i_name}:{s.transition_j_name}:"
        f"{'/'.join(str(x) for x in s.transition_dep_names)}"
        for s in program.asgp_dp2d_specs
    )
    return (
        f"AstPrefix({node_repr};LinearRec={binder_repr};AsgpDC={asgp_dc_repr};"
        f"AsgpDP1D={asgp_dp1d_repr};AsgpDP2D={asgp_dp2d_repr})"
    )


def node_count(program: AstProgram) -> int:
    validate_prefix_program(program)
    return len(program.nodes)


def _expr_depth_from(nodes: Sequence[AstNode], idx: int) -> tuple[int, int]:
    kind = nodes[idx].kind
    if kind not in EXPR_KINDS:
        raise ValueError(f"expected Expr at index {idx}, got {kind}")
    arity = NODE_ARITY[kind]
    if arity == 0:
        return 1, idx + 1
    max_child = 0
    cur = idx + 1
    for _ in range(arity):
        d, cur = _expr_depth_from(nodes, cur)
        if d > max_child:
            max_child = d
    return 1 + max_child, cur


def _stmt_max_expr_depth(nodes: Sequence[AstNode], idx: int) -> tuple[int, int]:
    kind = nodes[idx].kind
    if kind == NodeKind.ASSIGN:
        d, j = _expr_depth_from(nodes, idx + 1)
        return d, j
    if kind == NodeKind.RETURN:
        d, j = _expr_depth_from(nodes, idx + 1)
        return d, j
    if kind == NodeKind.IF_STMT:
        dc, j = _expr_depth_from(nodes, idx + 1)
        dt, k = _block_max_expr_depth(nodes, j)
        de, h = _block_max_expr_depth(nodes, k)
        return max(dc, dt, de), h
    if kind == NodeKind.FOR_RANGE:
        db, j = _expr_depth_from(nodes, idx + 1)
        dbody, h = _block_max_expr_depth(nodes, j)
        return max(db, dbody), h
    raise ValueError(f"expected Stmt at index {idx}, got {kind}")


def _block_max_expr_depth(nodes: Sequence[AstNode], idx: int) -> tuple[int, int]:
    kind = nodes[idx].kind
    if kind == NodeKind.BLOCK_NIL:
        return 0, idx + 1
    if kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {kind}")
    d0, j = _stmt_max_expr_depth(nodes, idx + 1)
    d1, k = _block_max_expr_depth(nodes, j)
    return max(d0, d1), k


def max_expr_depth(program: AstProgram) -> int:
    validate_prefix_program(program)
    d, end = _block_max_expr_depth(program.nodes, 1)
    if end != len(program.nodes):
        raise ValueError("invalid trailing tokens")
    return d


def _block_contains_return(nodes: Sequence[AstNode], idx: int) -> tuple[bool, int]:
    kind = nodes[idx].kind
    if kind == NodeKind.BLOCK_NIL:
        return False, idx + 1
    if kind != NodeKind.BLOCK_CONS:
        raise ValueError(f"expected Block at index {idx}, got {kind}")

    st = nodes[idx + 1].kind
    if st == NodeKind.RETURN:
        j = prefix_subtree_end(nodes, idx + 1)
        _, k = _block_contains_return(nodes, j)
        return True, k

    if st == NodeKind.ASSIGN:
        j = prefix_subtree_end(nodes, idx + 1)
        rtail, k = _block_contains_return(nodes, j)
        return rtail, k

    if st == NodeKind.FOR_RANGE:
        bound_end = prefix_subtree_end(nodes, idx + 2)
        _, body_end = _block_contains_return(nodes, bound_end)
        rtail, k = _block_contains_return(nodes, body_end)
        return rtail, k

    if st == NodeKind.IF_STMT:
        cond_idx = idx + 2
        cond_end = prefix_subtree_end(nodes, cond_idx)
        _, then_end = _block_contains_return(nodes, cond_end)
        _, else_end = _block_contains_return(nodes, then_end)
        rtail, k = _block_contains_return(nodes, else_end)
        return rtail, k

    raise ValueError(f"expected Stmt at index {idx + 1}, got {st}")


def top_level_has_return(program: AstProgram) -> bool:
    validate_prefix_program(program)
    has_ret, _ = _block_contains_return(program.nodes, 1)
    return has_ret


def build_program(stmt_specs: Sequence[tuple]) -> AstProgram:
    names: TList[str] = []
    name_to_idx: Dict[str, int] = {}
    consts: TList[Val] = []
    const_to_idx: Dict[tuple[str, str], int] = {}
    nodes: TList[AstNode] = [AstNode(NodeKind.PROGRAM)]
    linear_rec_binders: TList[LinearRecBinders] = []
    asgp_dc_binders: TList[AsgpDcBinders] = []
    asgp_dp1d_specs: TList[AsgpDp1dSpec] = []
    asgp_dp2d_specs: TList[AsgpDp2dSpec] = []

    dp1_dep_kind_by_name = {
        "backward1": NodeKind.DP1_BACKWARD1,
        "backward2": NodeKind.DP1_BACKWARD2,
        "backward3": NodeKind.DP1_BACKWARD3,
        "forward1": NodeKind.DP1_FORWARD1,
        "forward2": NodeKind.DP1_FORWARD2,
        "forward3": NodeKind.DP1_FORWARD3,
    }
    dp2_dep_kind_by_name = {
        "cross_backward": NodeKind.DP2_CROSS_BACKWARD,
        "cross_forward": NodeKind.DP2_CROSS_FORWARD,
        "diagonal_backward": NodeKind.DP2_DIAGONAL_BACKWARD,
        "diagonal_forward": NodeKind.DP2_DIAGONAL_FORWARD,
        "neighborhood_backward3": NodeKind.DP2_NEIGHBORHOOD_BACKWARD3,
        "neighborhood_forward3": NodeKind.DP2_NEIGHBORHOOD_FORWARD3,
    }
    dp2_dep_arity_by_name = {
        "cross_backward": 2,
        "cross_forward": 2,
        "diagonal_backward": 1,
        "diagonal_forward": 1,
        "neighborhood_backward3": 3,
        "neighborhood_forward3": 3,
    }

    def name_id(name: str) -> int:
        idx = name_to_idx.get(name)
        if idx is not None:
            return idx
        idx = len(names)
        names.append(name)
        name_to_idx[name] = idx
        return idx

    def const_id(v: Val) -> int:
        v = normalize_value(v)
        key = (type(v).__name__, repr(v))
        idx = const_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(consts)
        consts.append(v)
        const_to_idx[key] = idx
        return idx

    def emit_expr(spec: tuple) -> None:
        tag = spec[0]
        if tag == "const":
            nodes.append(AstNode(NodeKind.CONST, i0=const_id(spec[1])))
            return
        if tag == "var":
            nodes.append(AstNode(NodeKind.VAR, i0=name_id(spec[1])))
            return
        if tag == "bound":
            nodes.append(AstNode(NodeKind.BOUND_VAR, i0=name_id(spec[1])))
            return
        if tag == "map_list":
            binder = spec[1]
            source = spec[2]
            body = spec[3]
            out_type = spec[4]
            out_tag = LIST_TYPE_TAG_BY_NAME.get(out_type)
            if out_tag is None:
                raise ValueError(f"unknown map_list output type: {out_type}")
            nodes.append(AstNode(NodeKind.MAP_LIST, i0=name_id(binder), i1=int(out_tag)))
            emit_expr(source)
            emit_expr(body)
            return
        if tag == "filter_list":
            binder = spec[1]
            source = spec[2]
            pred = spec[3]
            nodes.append(AstNode(NodeKind.FILTER_LIST, i0=name_id(binder)))
            emit_expr(source)
            emit_expr(pred)
            return
        if tag == "linear_rec":
            elem_binder = spec[1]
            accum_binder = spec[2]
            index_binder = spec[3]
            source = spec[4]
            start_idx = spec[5]
            empty_case = spec[6]
            step = spec[7]
            last = spec[8]
            node_idx = len(nodes)
            nodes.append(AstNode(NodeKind.LINEAR_REC))
            linear_rec_binders.append(
                LinearRecBinders(
                    node_index=node_idx,
                    elem_name=name_id(elem_binder),
                    accum_name=name_id(accum_binder),
                    index_name=name_id(index_binder),
                )
            )
            emit_expr(source)
            emit_expr(start_idx)
            emit_expr(empty_case)
            emit_expr(step)
            emit_expr(last)
            return
        if tag == "asgp_dc":
            source = spec[1]
            solve_binders = spec[2]
            solve = spec[3]
            divide_binder = spec[4]
            divide = spec[5]
            combine_binders = spec[6]
            combine = spec[7]
            if not (isinstance(solve_binders, tuple) and len(solve_binders) == 3):
                raise ValueError("asgp_dc solve binders must be (xs, n, lo)")
            if not isinstance(divide_binder, str):
                raise ValueError("asgp_dc divide binder must be a name")
            if not (isinstance(combine_binders, tuple) and len(combine_binders) == 2):
                raise ValueError("asgp_dc combine binders must be (r1, r2)")
            node_idx = len(nodes)
            nodes.append(AstNode(NodeKind.ASGP_DC))
            asgp_dc_binders.append(
                AsgpDcBinders(
                    node_index=node_idx,
                    solve_xs_name=name_id(solve_binders[0]),
                    solve_n_name=name_id(solve_binders[1]),
                    solve_lo_name=name_id(solve_binders[2]),
                    divide_n_name=name_id(divide_binder),
                    combine_left_name=name_id(combine_binders[0]),
                    combine_right_name=name_id(combine_binders[1]),
                )
            )
            emit_expr(source)
            emit_expr(solve)
            emit_expr(divide)
            emit_expr(combine)
            return
        if tag == "asgp_dp1d":
            state = spec[1]
            bounds = spec[2]
            solve_binder = spec[3]
            solve = spec[4]
            depselect = spec[5]
            transition_binders = spec[6]
            transition = spec[7]
            if not (isinstance(bounds, tuple) and len(bounds) == 4):
                raise ValueError("asgp_dp1d bounds must be (lo, hi, base_state, boundary_value)")
            lo, hi, base_state, boundary_value = bounds
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, hi, base_state)):
                raise ValueError("asgp_dp1d lo/hi/base_state must be int")
            if not isinstance(solve_binder, str):
                raise ValueError("asgp_dp1d solve binder must be a name")
            if not (isinstance(depselect, tuple) and len(depselect) == 2):
                raise ValueError("asgp_dp1d depselect must be (kind, offsets)")
            dep_name, dep_offsets_raw = depselect
            if dep_name not in dp1_dep_kind_by_name:
                raise ValueError(f"unknown asgp_dp1d dependency kind: {dep_name}")
            if not isinstance(dep_offsets_raw, tuple):
                raise ValueError("asgp_dp1d dependency offsets must be a tuple")
            dep_offsets = tuple(dep_offsets_raw)
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in dep_offsets):
                raise ValueError("asgp_dp1d dependency offsets must be int")
            if not isinstance(transition_binders, tuple) or len(transition_binders) != 1 + len(dep_offsets):
                raise ValueError("asgp_dp1d transition binders must be (s, d1, ..., dK)")
            if not all(isinstance(x, str) for x in transition_binders):
                raise ValueError("asgp_dp1d transition binders must be names")
            node_idx = len(nodes)
            nodes.append(AstNode(NodeKind.ASGP_DP1D))
            asgp_dp1d_specs.append(
                AsgpDp1dSpec(
                    node_index=node_idx,
                    lo=lo,
                    hi=hi,
                    base_state=base_state,
                    boundary_const=const_id(boundary_value),
                    dep_kind=dp1_dep_kind_by_name[dep_name],
                    dep_offsets=dep_offsets,
                    solve_state_name=name_id(solve_binder),
                    transition_state_name=name_id(transition_binders[0]),
                    transition_dep_names=tuple(name_id(x) for x in transition_binders[1:]),
                )
            )
            emit_expr(state)
            emit_expr(solve)
            emit_expr(transition)
            return
        if tag == "asgp_dp2d":
            state_i = spec[1]
            state_j = spec[2]
            bounds = spec[3]
            solve_binders = spec[4]
            solve = spec[5]
            dep_name = spec[6]
            transition_binders = spec[7]
            transition = spec[8]
            if not (isinstance(bounds, tuple) and len(bounds) == 7):
                raise ValueError(
                    "asgp_dp2d bounds must be (i_lo, i_hi, j_lo, j_hi, base_i, base_j, boundary_value)"
                )
            i_lo, i_hi, j_lo, j_hi, base_i, base_j, boundary_value = bounds
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (i_lo, i_hi, j_lo, j_hi, base_i, base_j)):
                raise ValueError("asgp_dp2d bounds/base coordinates must be int")
            if not (isinstance(solve_binders, tuple) and len(solve_binders) == 2):
                raise ValueError("asgp_dp2d solve binders must be (i, j)")
            if not all(isinstance(x, str) for x in solve_binders):
                raise ValueError("asgp_dp2d solve binders must be names")
            if dep_name not in dp2_dep_kind_by_name:
                raise ValueError(f"unknown asgp_dp2d dependency kind: {dep_name}")
            expected_k = dp2_dep_arity_by_name[dep_name]
            if not isinstance(transition_binders, tuple) or len(transition_binders) != 2 + expected_k:
                raise ValueError("asgp_dp2d transition binders must be (i, j, d1, ..., dK)")
            if not all(isinstance(x, str) for x in transition_binders):
                raise ValueError("asgp_dp2d transition binders must be names")
            node_idx = len(nodes)
            nodes.append(AstNode(NodeKind.ASGP_DP2D))
            asgp_dp2d_specs.append(
                AsgpDp2dSpec(
                    node_index=node_idx,
                    i_lo=i_lo,
                    i_hi=i_hi,
                    j_lo=j_lo,
                    j_hi=j_hi,
                    base_i=base_i,
                    base_j=base_j,
                    boundary_const=const_id(boundary_value),
                    dep_kind=dp2_dep_kind_by_name[dep_name],
                    solve_i_name=name_id(solve_binders[0]),
                    solve_j_name=name_id(solve_binders[1]),
                    transition_i_name=name_id(transition_binders[0]),
                    transition_j_name=name_id(transition_binders[1]),
                    transition_dep_names=tuple(name_id(x) for x in transition_binders[2:]),
                )
            )
            emit_expr(state_i)
            emit_expr(state_j)
            emit_expr(solve)
            emit_expr(transition)
            return
        if tag == "neg":
            nodes.append(AstNode(NodeKind.NEG))
            emit_expr(spec[1])
            return
        if tag == "not":
            nodes.append(AstNode(NodeKind.NOT))
            emit_expr(spec[1])
            return
        if tag in {
            "add", "sub", "mul", "div", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or",
        }:
            nodes.append(AstNode(NodeKind(tag.upper())))
            emit_expr(spec[1])
            emit_expr(spec[2])
            return
        if tag == "if_expr":
            nodes.append(AstNode(NodeKind.IF_EXPR))
            emit_expr(spec[1])
            emit_expr(spec[2])
            emit_expr(spec[3])
            return
        if tag == "call":
            name = spec[1]
            args = list(spec[2])
            nk = BUILTIN_NODE_BY_NAME.get(name)
            if nk is None:
                raise ValueError(f"unknown builtin: {name}")
            if len(args) != NODE_ARITY[nk]:
                raise ValueError(f"bad builtin arity for {name}: got {len(args)}")
            nodes.append(AstNode(nk))
            for a in args:
                emit_expr(a)
            return
        raise ValueError(f"unknown expr tag: {tag}")

    def emit_stmt(spec: tuple) -> None:
        tag = spec[0]
        if tag == "assign":
            nodes.append(AstNode(NodeKind.ASSIGN, i0=name_id(spec[1])))
            emit_expr(spec[2])
            return
        if tag == "return":
            nodes.append(AstNode(NodeKind.RETURN))
            emit_expr(spec[1])
            return
        if tag == "if":
            nodes.append(AstNode(NodeKind.IF_STMT))
            emit_expr(spec[1])
            emit_block(spec[2])
            emit_block(spec[3])
            return
        if tag == "for":
            if not isinstance(spec[2], tuple):
                raise ValueError("for expects expression bound, for example ('const', 5)")
            nodes.append(AstNode(NodeKind.FOR_RANGE, i0=name_id(spec[1])))
            emit_expr(spec[2])
            emit_block(spec[3])
            return
        raise ValueError(f"unknown stmt tag: {tag}")

    def emit_block(stmts: Sequence[tuple]) -> None:
        if not stmts:
            nodes.append(AstNode(NodeKind.BLOCK_NIL))
            return
        nodes.append(AstNode(NodeKind.BLOCK_CONS))
        emit_stmt(stmts[0])
        emit_block(stmts[1:])

    emit_block(stmt_specs)
    program = AstProgram(
        nodes=tuple(nodes),
        names=tuple(names),
        consts=tuple(consts),
        linear_rec_binders=tuple(linear_rec_binders),
        asgp_dc_binders=tuple(asgp_dc_binders),
        asgp_dp1d_specs=tuple(asgp_dp1d_specs),
        asgp_dp2d_specs=tuple(asgp_dp2d_specs),
    )
    validate_prefix_program(program)
    return program
