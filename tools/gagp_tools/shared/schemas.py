from __future__ import annotations

from typing import Any, Mapping


FITNESS_CASES = "fitness-cases"
POPULATION_SEEDS = "population-seeds"
PSB_REGRESSION_SUMMARY = "psb-regression-summary"


def require_format(payload: Mapping[str, Any], expected: str, source: object) -> None:
    actual = payload.get("format_version")
    if actual != expected:
        raise ValueError(f"unsupported format in {source}: expected {expected}, got {actual}")
