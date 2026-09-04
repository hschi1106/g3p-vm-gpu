from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from ..shared.hashing import sha256_file, sha256_text
from ..shared.json_io import load_json_object, stable_json


STATEMENT_KEYS = ("assign", "if_stmt", "for_range", "return")
UNARY_KEYS = ("neg", "not")
BINARY_KEYS = ("add", "sub", "mul", "div", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or")
CURRENT_BUILTIN_KEYS = (
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
CURRENT_VALUE_KEYS = ("int", "float", "bool", "char", "string", "int_list", "float_list", "string_list")
CURRENT_LIST_SCHEMAS = {"int_list", "float_list", "string_list"}
CURRENT_EXPRESSION_KEYS = (
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


def schema_hash(field_schemas: Mapping[str, str]) -> str:
    encoded = json.dumps(field_schemas, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)

def _read_bool_map(raw: Any, keys: tuple[str, ...], section: str, extra_keys: tuple[str, ...] = ()) -> Dict[str, bool]:
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be an object")
    allowed = set(keys) | set(extra_keys)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{section} has unknown keys: {', '.join(unknown)}")
    out: Dict[str, bool] = {}
    for key in keys:
        value = raw.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"{section}.{key} must be a boolean")
        out[key] = value
    return out


def load_fixture_schema(path: Path) -> tuple[Dict[str, str], str | None]:
    payload = load_json_object(path)
    schema_hash_value = None
    meta = payload.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("schema_hash"), str):
        schema_hash_value = meta["schema_hash"]
    source = payload.get("source")
    if schema_hash_value is None and isinstance(source, dict) and isinstance(source.get("schema_hash"), str):
        schema_hash_value = source["schema_hash"]

    schema = payload.get("schema")
    if isinstance(schema, dict):
        inputs = schema.get("inputs")
        expected = schema.get("expected")
        if isinstance(inputs, dict) and isinstance(expected, str):
            field_schemas = {str(k): str(v) for k, v in inputs.items()}
            field_schemas["expected"] = expected
            return field_schemas, schema_hash_value

    if isinstance(source, dict) and isinstance(source.get("field_schemas"), dict):
        return {str(k): str(v) for k, v in source["field_schemas"].items()}, schema_hash_value

    raise ValueError(f"fixture does not include current schema metadata: {path}")


def load_schema_json(path: Path) -> tuple[Dict[str, str], str]:
    payload = load_json_object(path)
    raw = payload.get("field_schemas", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"schema JSON field_schemas must be an object: {path}")
    field_schemas = {str(k): str(v) for k, v in raw.items()}
    return field_schemas, schema_hash(field_schemas)


def _validate_base_config(raw: Mapping[str, Any]) -> None:
    if raw.get("format_version") != "grammar-config":
        raise ValueError("base grammar config must include format_version=grammar-config")
    expressions = raw.get("expressions")
    values = raw.get("values")
    builtins = raw.get("builtins")
    if not isinstance(expressions, dict) or not isinstance(values, dict) or not isinstance(builtins, dict):
        raise ValueError("grammar-config payload is missing expressions, builtins, or values")
    _read_bool_map(raw.get("statements"), STATEMENT_KEYS, "statements")
    _read_bool_map(expressions, CURRENT_EXPRESSION_KEYS, "expressions")
    _read_bool_map(builtins, CURRENT_BUILTIN_KEYS, "builtins")
    _read_bool_map(values, CURRENT_VALUE_KEYS, "values")


def _base_current_profile(profile: str) -> Dict[str, Any]:
    return {
        "format_version": "grammar-config",
        "profile": profile,
        "values": {key: True for key in CURRENT_VALUE_KEYS},
        "statements": {key: True for key in STATEMENT_KEYS},
        "expressions": {
            "const": True,
            "var": True,
            "bound_var": profile == "full",
            "unary": True,
            "binary": True,
            "if_expr": True,
            "call": True,
            "map_list": profile == "full",
            "filter_list": profile == "full",
            "linear_rec": profile == "full",
            "asgp_dc": profile == "full",
            "asgp_dp1d": profile == "full",
            "asgp_dp2d": profile == "full",
        },
        "builtins": {key: True for key in CURRENT_BUILTIN_KEYS},
        "structured": {
            "max_nested_binders": 3 if profile == "full" else 0,
            "max_map_body_depth": 5 if profile == "full" else 0,
            "max_filter_pred_depth": 5 if profile == "full" else 0,
            "max_linear_rec_body_depth": 5 if profile == "full" else 0,
        },
        "asgp": {
            "max_scheme_nesting": 1 if profile == "full" else 0,
            "dc": {"enabled_source_elems": ["int", "float", "string", "char"] if profile == "full" else [], "max_depth": 64 if profile == "full" else 0},
            "dp1d": {
                "max_states": 512 if profile == "full" else 0,
                "max_step": 3 if profile == "full" else 0,
                "dependency_patterns": ["backward1", "backward2", "backward3", "forward1", "forward2", "forward3"] if profile == "full" else [],
            },
            "dp2d": {
                "max_cells": 4096 if profile == "full" else 0,
                "dependency_patterns": [
                    "cross_backward",
                    "cross_forward",
                    "diagonal_backward",
                    "diagonal_forward",
                    "neighborhood_backward3",
                    "neighborhood_forward3",
                ]
                if profile == "full"
                else [],
            },
        },
        "limits": {
            "max_expr_depth": 7,
            "max_stmts_per_block": 6,
            "max_total_nodes": 80,
            "max_for_k": 16,
            "max_call_args": 3,
        },
        "compat": None,
    }


def build_full_config() -> Dict[str, Any]:
    return _base_current_profile("full")


def build_compat_config(
    *,
    base_config_path: Path,
    field_schemas: Mapping[str, str],
    fixture_schema_hash: str,
    profile: str = "compat",
    num_list_mode: str = "schema",
) -> Dict[str, Any]:
    if profile not in {"compat", "compact"}:
        raise ValueError("profile must be compat or compact")
    if num_list_mode not in {"schema", "both"}:
        raise ValueError("num_list_mode must be schema or both")
    base_raw = load_json_object(base_config_path)
    _validate_base_config(base_raw)

    base_expressions = base_raw["expressions"]
    base_values = base_raw["values"]
    base_builtins = base_raw["builtins"]

    schema_values = set(field_schemas.values())
    unknown_list_schemas = sorted(schema_values & {"num_list", "list"})
    if unknown_list_schemas:
        raise ValueError("compatibility requires direct-list fixture schemas, not " + ", ".join(unknown_list_schemas))

    base_num_list = bool(base_values["int_list"]) or bool(base_values["float_list"])
    cfg = _base_current_profile(profile)
    cfg["statements"] = {key: bool(base_raw["statements"][key]) for key in STATEMENT_KEYS}
    cfg["expressions"] = {
        "const": bool(base_expressions["const"]),
        "var": bool(base_expressions["var"]),
        "bound_var": False,
        "unary": bool(base_expressions["unary"]),
        "binary": bool(base_expressions["binary"]),
        "if_expr": bool(base_expressions["if_expr"]),
        "call": bool(base_expressions["call"]),
        "map_list": False,
        "filter_list": False,
        "linear_rec": False,
        "asgp_dc": False,
        "asgp_dp1d": False,
        "asgp_dp2d": False,
    }
    cfg["builtins"] = {key: False for key in CURRENT_BUILTIN_KEYS}
    for key in CURRENT_BUILTIN_KEYS:
        cfg["builtins"][key] = bool(base_builtins[key])
    if num_list_mode == "schema":
        int_list_enabled = base_num_list and "int_list" in schema_values
        float_list_enabled = base_num_list and "float_list" in schema_values
    else:
        int_list_enabled = base_num_list
        float_list_enabled = base_num_list

    cfg["values"] = {
        "int": bool(base_values["int"]),
        "float": bool(base_values["float"]),
        "bool": bool(base_values["bool"]),
        "char": False,
        "string": bool(base_values["string"]),
        "int_list": int_list_enabled,
        "float_list": float_list_enabled,
        "string_list": bool(base_values["string_list"]),
    }
    cfg["compat"] = {
        "mode": profile,
        "source_format_version": "grammar-config",
        "source_path": str(base_config_path),
        "source_hash": sha256_file(base_config_path),
        "fixture_schema_hash": fixture_schema_hash,
        "num_list_mode": num_list_mode,
    }
    return cfg


def write_profile(path: Path, payload: Mapping[str, Any]) -> str:
    text = stable_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate grammar-config profile JSON.")
    parser.add_argument("--profile", required=True, choices=["compat", "compact", "full"])
    parser.add_argument("--base-grammar-config", type=Path)
    parser.add_argument("--fixture-cases", type=Path)
    parser.add_argument("--schema-json", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.profile == "full":
        payload = build_full_config()
    else:
        if args.base_grammar_config is None:
            raise SystemExit("--base-grammar-config is required for compat")
        if args.fixture_cases is not None and args.schema_json is not None:
            raise SystemExit("--fixture-cases and --schema-json are mutually exclusive")
        if args.fixture_cases is not None:
            field_schemas, fixture_schema_hash = load_fixture_schema(args.fixture_cases)
            fixture_schema_hash = fixture_schema_hash or schema_hash(field_schemas)
        elif args.schema_json is not None:
            field_schemas, fixture_schema_hash = load_schema_json(args.schema_json)
        else:
            raise SystemExit("--fixture-cases or --schema-json is required for compat")
        profile = "compact" if args.profile == "compact" else "compat"
        payload = build_compat_config(
            base_config_path=args.base_grammar_config,
            field_schemas=field_schemas,
            fixture_schema_hash=fixture_schema_hash,
            profile=profile,
            num_list_mode="both" if profile == "compact" else "schema",
        )

    generated_hash = write_profile(args.out, payload)
    print(f"GRAMMAR_CONFIG_PROFILE {args.profile}")
    print(f"GRAMMAR_CONFIG_OUT {args.out}")
    print(f"GRAMMAR_CONFIG_HASH {generated_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
