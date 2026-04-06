import json
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def decode_value(node):
    t = node["type"]
    if t == "none":
        return None
    if t in {"bool", "int", "float", "string"}:
        return node["value"]
    if t == "list":
        return [decode_value(x) for x in node["value"]]
    raise AssertionError(f"unsupported typed value: {t}")


class TestArrayFixtures(unittest.TestCase):
    def _load_cases(self, rel_path: str):
        path = ROOT / rel_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("format_version"), "fitness-cases-v1")
        self.assertEqual(len(payload["cases"]), 1024)
        return payload["cases"]

    def test_array_min_fixture(self):
        for row in self._load_cases("data/fixtures/array_min_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], min(xs))

    def test_array_max_fixture(self):
        for row in self._load_cases("data/fixtures/array_max_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], max(xs))

    def test_array_avg_fixture(self):
        for row in self._load_cases("data/fixtures/array_avg_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], sum(xs) / len(xs))

    def test_array_median_fixture(self):
        for row in self._load_cases("data/fixtures/array_median_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], statistics.median(xs))

    def test_array_head_fixture(self):
        for row in self._load_cases("data/fixtures/array_head_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], xs[0])

    def test_array_len_fixture(self):
        for row in self._load_cases("data/fixtures/array_len_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], float(len(xs)))

    def test_array_max2_fixture(self):
        for row in self._load_cases("data/fixtures/array_max2_1024.json"):
            xs = decode_value(row["inputs"]["xs"])
            self.assertEqual(len(xs), 2)
            self.assertEqual(row["expected"]["type"], "float")
            self.assertEqual(row["expected"]["value"], max(xs))


if __name__ == "__main__":
    unittest.main()
