from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


STATEMENT_KEYS = ("assign", "if_stmt", "for_range", "return")
EXPRESSION_KEYS = ("const", "var", "if_expr")
UNARY_KEYS = ("neg", "not")
BINARY_KEYS = ("add", "sub", "mul", "div", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or")
BUILTIN_KEYS = (
    "abs",
    "min",
    "max",
    "clip",
    "len",
    "concat",
    "slice",
    "index",
    "append",
    "reverse",
    "find",
    "contains",
)
VALUE_KEYS = ("int", "float", "bool", "none", "string", "num_list", "string_list")


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
        if format_version != "grammar-config-v1":
            raise ValueError("grammar config must include format_version=grammar-config-v1")

        expressions_raw = raw.get("expressions")
        if not isinstance(expressions_raw, Mapping):
            raise ValueError("expressions must be an object")

        cfg = GrammarConfig(
            statements=_read_bool_map(raw.get("statements"), STATEMENT_KEYS, "statements"),
            expressions=_read_bool_map(expressions_raw, EXPRESSION_KEYS, "expressions", ("unary", "binary", "builtins")),
            unary=_read_bool_map(expressions_raw.get("unary"), UNARY_KEYS, "expressions.unary"),
            binary=_read_bool_map(expressions_raw.get("binary"), BINARY_KEYS, "expressions.binary"),
            builtins=_read_bool_map(expressions_raw.get("builtins"), BUILTIN_KEYS, "expressions.builtins"),
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
