import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPsb1Tools(unittest.TestCase):
    def test_convert_psb1_problem_to_fitness_cases(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb1_conv_") as td:
            td_path = Path(td)
            problem_dir = td_path / "count-odds"
            problem_dir.mkdir(parents=True)
            edge = problem_dir / "count-odds-edge.json"
            rnd = problem_dir / "count-odds-random.json"
            out = td_path / "train.json"
            out_test = td_path / "test.json"
            summary = td_path / "summary.json"

            edge.write_text(
                '{"input1":[],"output1":0}\n'
                '{"input1":[1,2,3],"output1":2}\n',
                encoding="utf-8",
            )
            rnd.write_text(
                '{"input1":[4,5],"output1":1}\n'
                '{"input1":[],"output1":0}\n',
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "tools/convert_psb1_to_fitness_cases.py",
                "--problem",
                "count-odds",
                "--datasets-root",
                str(td_path),
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
            self.assertIn("CONVERT_PSB1_PROBLEM count-odds", proc.stdout)
            self.assertIn("CONVERT_RUNTIME_COMPATIBLE 1", proc.stdout)

            train = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(train["format_version"], "fitness-cases-v1")
            self.assertEqual(train["source"]["suite"], "psb1")
            self.assertEqual(train["source"]["problem"], "count-odds")
            self.assertEqual(train["source"]["field_schemas"]["input1"], "num_list")
            self.assertEqual(len(train["cases"]), 3)
            self.assertTrue(all(row["inputs"]["input1"]["type"] == "num_list" for row in train["cases"]))

            s = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(s["problem"], "count-odds")
            self.assertTrue(s["runtime_compatible"])

    def test_convert_psb1_multi_output_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb1_multiout_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            edge.write_text('{"input1":"a","output1":"a","output2":1}\n', encoding="utf-8")
            rnd.write_text('{"input1":"b","output1":"b","output2":0}\n', encoding="utf-8")

            cmd = [
                "python3",
                "tools/convert_psb1_to_fitness_cases.py",
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
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("multi-output", proc.stderr)


if __name__ == "__main__":
    unittest.main()
