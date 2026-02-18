import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestSimpleEvoFixtures(unittest.TestCase):
    def _load_cases(self, rel_path: str):
        path = ROOT / rel_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("cases", payload)
        cases = payload["cases"]
        self.assertEqual(len(cases), 1024)
        xs = [int(row["inputs"]["x"]) for row in cases]
        self.assertEqual(xs[0], -511)
        self.assertEqual(xs[-1], 512)
        self.assertEqual(xs, list(range(-511, 513)))
        return cases

    def test_x_plus_1_fixture(self):
        cases = self._load_cases("data/fixtures/simple_evo_x_plus_1_1024.json")
        for row in cases:
            x = int(row["inputs"]["x"])
            self.assertEqual(int(row["expected"]), x + 1)

    def test_affine_fixture(self):
        cases = self._load_cases("data/fixtures/simple_evo_affine_2x_plus_3_1024.json")
        for row in cases:
            x = int(row["inputs"]["x"])
            self.assertEqual(int(row["expected"]), 2 * x + 3)

    def test_square_fixture(self):
        cases = self._load_cases("data/fixtures/simple_evo_square_x2_1024.json")
        for row in cases:
            x = int(row["inputs"]["x"])
            self.assertEqual(int(row["expected"]), x * x)


if __name__ == "__main__":
    unittest.main()
