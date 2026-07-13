from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


STATEMENT_KEYS = ("assign", "if_stmt", "for_range", "return")
EXPRESSION_KEYS = (
    "const",
    "var",
    "bound_var",
    "unary",
    "binary",
    "if_expr",
    "call",
    "map_list",
    "filter_list",
    "linear_rec",
    "asgp_dc",
    "asgp_dp1d",
    "asgp_dp2d",
)
UNARY_KEYS = ("neg", "not")
BINARY_KEYS = ("add", "sub", "mul", "div", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or")
BUILTIN_KEYS = (
    "abs",
    "min",
    "max",
    "clip",
    "idiv0",
    "imod0",
    "len",
    "concat",
    "slice",
    "index",
    "append",
    "prepend",
    "reverse",
    "find",
    "contains",
    "singleton",
    "char_to_string",
    "string_to_char",
    "ord",
    "chr",
    "is_letter",
    "is_digit",
    "is_space",
    "is_vowel",
    "to_lower",
    "to_upper",
    "to_string",
)
VALUE_KEYS = ("int", "float", "bool", "char", "string", "int_list", "float_list", "string_list")
ROOT_KEYS = ("format_version", "profile", "values", "statements", "expressions", "builtins", "structured", "asgp", "limits", "compat")


def _all_enabled(keys: tuple[str, ...]) -> dict[str, bool]:
    return {key: True for key in keys}


def _read_bool_map(raw: object, keys: tuple[str, ...], section: str, extra_keys: tuple[str, ...] = ()) -> dict[str, bool]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{section} must be an object")
    allowed = set(keys) | set(extra_keys)
    unknown = sorted(set(raw.keys()) - allowed)
    if unknown:
        raise ValueError(f"{section} has unknown keys: {', '.join(unknown)}")
    out: dict[str, bool] = {}
    for key in keys:
        value = raw.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"{section}.{key} must be a boolean")
        out[key] = value
    return out


def _reject_unknown_map(raw: Mapping[str, object], keys: tuple[str, ...], section: str) -> None:
    unknown = sorted(set(raw.keys()) - set(keys))
    if unknown:
        raise ValueError(f"{section} has unknown keys: {', '.join(unknown)}")


def _optional_object(raw: Mapping[str, object], key: str, section: str) -> Mapping[str, object] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{section}.{key} must be an object")
    return value


def _validate_optional_metadata(raw: Mapping[str, object]) -> None:
    structured = _optional_object(raw, "structured", "root")
    if structured is not None:
        _reject_unknown_map(
            structured,
            ("max_nested_binders", "max_map_body_depth", "max_filter_pred_depth", "max_linear_rec_body_depth"),
            "structured",
        )
    limits = _optional_object(raw, "limits", "root")
    if limits is not None:
        _reject_unknown_map(
            limits,
            ("max_expr_depth", "max_stmts_per_block", "max_total_nodes", "max_for_k", "max_call_args"),
            "limits",
        )
    asgp = _optional_object(raw, "asgp", "root")
    if asgp is not None:
        _reject_unknown_map(asgp, ("max_scheme_nesting", "dc", "dp1d", "dp2d"), "asgp")
        dc = _optional_object(asgp, "dc", "asgp")
        if dc is not None:
            _reject_unknown_map(dc, ("enabled_source_elems", "max_depth"), "asgp.dc")
        dp1d = _optional_object(asgp, "dp1d", "asgp")
        if dp1d is not None:
            _reject_unknown_map(dp1d, ("max_states", "max_step", "dependency_patterns"), "asgp.dp1d")
        dp2d = _optional_object(asgp, "dp2d", "asgp")
        if dp2d is not None:
            _reject_unknown_map(dp2d, ("max_cells", "dependency_patterns"), "asgp.dp2d")
    compat = raw.get("compat")
    if compat is not None and not isinstance(compat, Mapping):
        raise ValueError("root.compat must be null or an object")


@dataclass(frozen=True)
class GrammarConfig:
    statements: Mapping[str, bool]
    expressions: Mapping[str, bool]
    unary: Mapping[str, bool]
    binary: Mapping[str, bool]
    builtins: Mapping[str, bool]
    values: Mapping[str, bool]

    @staticmethod
    def all_enabled() -> "GrammarConfig":
        return GrammarConfig(
            statements=_all_enabled(STATEMENT_KEYS),
            expressions=_all_enabled(EXPRESSION_KEYS),
            unary=_all_enabled(UNARY_KEYS),
            binary=_all_enabled(BINARY_KEYS),
            builtins=_all_enabled(BUILTIN_KEYS),
            values=_all_enabled(VALUE_KEYS),
        )

    @staticmethod
    def from_mapping(raw: Mapping[str, object]) -> "GrammarConfig":
        format_version = raw.get("format_version")
        if format_version != "grammar-config":
            raise ValueError("grammar config must include format_version=grammar-config")
        _reject_unknown_map(raw, ROOT_KEYS, "root")
        _validate_optional_metadata(raw)

        expressions_raw = raw.get("expressions")
        if not isinstance(expressions_raw, Mapping):
            raise ValueError("expressions must be an object")
        expressions = _read_bool_map(expressions_raw, EXPRESSION_KEYS, "expressions")
        builtins = _read_bool_map(raw.get("builtins"), BUILTIN_KEYS, "builtins")
        if not expressions["call"]:
            builtins = {key: False for key in BUILTIN_KEYS}

        cfg = GrammarConfig(
            statements=_read_bool_map(raw.get("statements"), STATEMENT_KEYS, "statements"),
            expressions=expressions,
            unary={key: expressions["unary"] for key in UNARY_KEYS},
            binary={key: expressions["binary"] for key in BINARY_KEYS},
            builtins=builtins,
            values=_read_bool_map(raw.get("values"), VALUE_KEYS, "values"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.allow_statement("return"):
            raise ValueError("grammar config must enable statements.return")
        if not self.allow_expression("const"):
            raise ValueError("grammar config must enable expressions.const")
        if not self.allow_num():
            raise ValueError("grammar config must enable values.int or values.float")
        if self.allow_statement("for_range") and not self.values.get("int", False):
            raise ValueError("statements.for_range requires values.int")

    def allow_statement(self, key: str) -> bool:
        return bool(self.statements.get(key, False))

    def allow_expression(self, key: str) -> bool:
        return bool(self.expressions.get(key, False))

    def allow_unary(self, key: str) -> bool:
        return bool(self.unary.get(key, False))

    def allow_binary(self, key: str) -> bool:
        return bool(self.binary.get(key, False))

    def allow_builtin(self, key: str) -> bool:
        return bool(self.builtins.get(key, False))

    def allow_value(self, key: str) -> bool:
        return bool(self.values.get(key, False))

    def allow_num(self) -> bool:
        return self.allow_value("int") or self.allow_value("float")

    def is_all_enabled(self) -> bool:
        return self == DEFAULT_GRAMMAR_CONFIG


DEFAULT_GRAMMAR_CONFIG = GrammarConfig.all_enabled()


def load_grammar_config(path: str | Path) -> GrammarConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("grammar config must be a JSON object")
    return GrammarConfig.from_mapping(raw)
