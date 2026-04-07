#!/usr/bin/env python3

import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "data" / "fixtures"
CASE_COUNT = 1024
SEED = 20260402


def encode_value(value):
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            return {"type": "num_list", "value": value}
        if all(isinstance(x, str) for x in value):
            return {"type": "string_list", "value": value}
        raise TypeError("fixtures only support numeric or string typed lists")
    raise TypeError(f"unsupported value type: {type(value)}")


def make_arrays():
    rng = random.Random(SEED)
    arrays = []
    for _ in range(CASE_COUNT):
        length = rng.choice([3, 4, 5, 6, 7, 8])
        xs = []
        for _ in range(length):
            numer = rng.randint(-96, 96)
            xs.append(numer / 8.0)
        arrays.append(xs)
    return arrays


def make_pairs():
    rng = random.Random(SEED + 1)
    pairs = []
    for _ in range(CASE_COUNT):
        a = rng.randint(-96, 96) / 8.0
        b = rng.randint(-96, 96) / 8.0
        pairs.append([a, b])
    return pairs


def make_payload(arrays, op_name, fn):
    payload = {
        "format_version": "fitness-cases-v1",
        "cases": [],
        "meta": {
            "task": op_name,
            "input_name": "xs",
            "case_count": CASE_COUNT,
            "value_domain": "float_list",
            "seed": SEED,
        },
    }
    for xs in arrays:
        payload["cases"].append(
            {
                "inputs": {"xs": encode_value(xs)},
                "expected": encode_value(float(fn(xs))),
            }
        )
    return payload


def write_fixture(name, payload):
    path = FIXTURES_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main():
    arrays = make_arrays()
    pairs = make_pairs()
    outputs = [
        write_fixture("array_min_1024.json", make_payload(arrays, "array_min", min)),
        write_fixture("array_max_1024.json", make_payload(arrays, "array_max", max)),
        write_fixture("array_avg_1024.json", make_payload(arrays, "array_avg", lambda xs: sum(xs) / len(xs))),
        write_fixture("array_median_1024.json", make_payload(arrays, "array_median", statistics.median)),
        write_fixture("array_head_1024.json", make_payload(arrays, "array_head", lambda xs: xs[0])),
        write_fixture("array_len_1024.json", make_payload(arrays, "array_len", len)),
        write_fixture("array_max2_1024.json", make_payload(pairs, "array_max2", max)),
    ]
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
