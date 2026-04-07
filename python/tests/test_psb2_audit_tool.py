import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPsb2AuditTool(unittest.TestCase):
    def test_audit_reports_runtime_compatibility_and_multi_output(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_audit_") as td:
            td_path = Path(td)
            ds = td_path / "datasets"
            t_num = ds / "num-task"
            t_mo = ds / "multi-out"
            t_str = ds / "string-task"
            for p in (t_num, t_mo, t_str):
                p.mkdir(parents=True)

            (t_num / "num-task-edge.json").write_text('{"input1":1,"output1":2}\n', encoding="utf-8")
            (t_num / "num-task-random.json").write_text('{"input1":2,"output1":3}\n', encoding="utf-8")

            (t_mo / "multi-out-edge.json").write_text('{"input1":1,"output1":2,"output2":3}\n', encoding="utf-8")
            (t_mo / "multi-out-random.json").write_text('{"input1":2,"output1":3,"output2":4}\n', encoding="utf-8")

            (t_str / "string-task-edge.json").write_text('{"input1":"a","output1":"b"}\n', encoding="utf-8")
            (t_str / "string-task-random.json").write_text('{"input1":"x","output1":"y"}\n', encoding="utf-8")

            out = td_path / "audit.json"
            cmd = [
                "python3",
                "tools/audit_psb2_tasks.py",
                "--datasets-root",
                str(ds),
                "--out-json",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("PSB2_AUDIT_JSON", proc.stdout)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["tasks_total"], 3)
            self.assertEqual(report["runtime_compatible_current"], 2)
            self.assertEqual(report["multi_output_tasks"], 1)

            by_task = {r["task"]: r for r in report["results"]}
            self.assertTrue(by_task["num-task"]["runtime_compatible_current"])
            self.assertTrue(by_task["multi-out"]["has_multi_output"])
            self.assertIn("string", by_task["string-task"]["value_types"])
            self.assertTrue(by_task["string-task"]["runtime_compatible_current"])


if __name__ == "__main__":
    unittest.main()
