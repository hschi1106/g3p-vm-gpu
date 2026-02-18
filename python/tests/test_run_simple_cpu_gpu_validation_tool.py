import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "run_simple_cpu_gpu_validation.py"


spec = importlib.util.spec_from_file_location("run_simple_cpu_gpu_validation", TOOL_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TestRunSimpleCpuGpuValidationTool(unittest.TestCase):
    def test_first_hit_generation_found(self):
        history = [
            {"generation": 0, "best": 1000.0},
            {"generation": 1, "best": 1023.0},
            {"generation": 2, "best": 1024.0},
            {"generation": 3, "best": 1024.0},
        ]
        self.assertEqual(mod._first_hit_generation(history, 1024), 2)

    def test_first_hit_generation_not_found(self):
        history = [
            {"generation": 0, "best": 1000.0},
            {"generation": 1, "best": 1023.0},
            {"generation": 2, "best": 1001.0},
        ]
        self.assertIsNone(mod._first_hit_generation(history, 1024))


if __name__ == "__main__":
    unittest.main()
