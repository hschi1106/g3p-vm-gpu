import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPsb2Tools(unittest.TestCase):
    def test_convert_psb2_to_fitness_cases_basic(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_conv_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            out_test = td_path / "test.json"
            summary = td_path / "summary.json"

            edge.write_text(
                '{"input1":1,"input2":2.5,"output1":3}\n'
                '{"input1":2,"input2":3.0,"output1":5}\n',
                encoding="utf-8",
            )
            rnd.write_text(
                '{"input1":4,"input2":1.0,"output1":5}\n'
                '{"input1":5,"input2":2.0,"output1":7}\n',
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "tools/convert_psb2_to_fitness_cases.py",
                "--edge-file",
                str(edge),
                "--random-file",
                str(rnd),
                "--n-train",
                "3",
                "--n-test",
                "2",
                "--seed",
                "7",
                "--out",
                str(out),
                "--out-test",
                str(out_test),
                "--summary-json",
                str(summary),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("CONVERT_RUNTIME_COMPATIBLE 1", proc.stdout)
            train = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(train["format_version"], "fitness-cases-v1")
            self.assertEqual(len(train["cases"]), 3)
            one = train["cases"][0]
            self.assertIn("inputs", one)
            self.assertEqual(sorted(one["inputs"].keys()), ["input1", "input2"])
            self.assertIn("expected", one)
            self.assertIn("type", one["expected"])
            s = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(s["runtime_compatible"])

    def test_convert_multi_output_to_list_expected(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_multiout_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            edge.write_text('{"input1":1,"output1":2,"output2":3}\n', encoding="utf-8")
            rnd.write_text('{"input1":1,"output1":2}\n', encoding="utf-8")

            cmd = [
                "python3",
                "tools/convert_psb2_to_fitness_cases.py",
                "--edge-file",
                str(edge),
                "--random-file",
                str(rnd),
                "--n-train",
                "1",
                "--n-test",
                "1",
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("CONVERT_RUNTIME_COMPATIBLE 1", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            case0 = payload["cases"][0]
            self.assertEqual(case0["expected"]["type"], "list")
            self.assertEqual(len(case0["expected"]["value"]), 2)

if __name__ == "__main__":
    unittest.main()
