import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPsbFetchTools(unittest.TestCase):
    def test_unified_fetch_with_psb1_dry_run_subset(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb1_fetch_") as td:
            out_dir = Path(td) / "psb1"
            cmd = [
                "python3",
                "tools/fetch_psb_datasets.py",
                "--suite",
                "psb1",
                "--out-dir",
                str(out_dir),
                "--problems",
                "checksum,number-io",
                "--dry-run",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("[psb1] fetch checksum", proc.stdout)
            self.assertIn("[psb1] fetch number-io", proc.stdout)
            self.assertIn("total=4", proc.stdout)
            self.assertFalse(out_dir.exists())

    def test_unified_fetch_with_psb1_unknown_problem(self):
        cmd = [
            "python3",
            "tools/fetch_psb_datasets.py",
            "--suite",
            "psb1",
            "--problems",
            "not-a-real-problem",
            "--dry-run",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown PSB1 problem", proc.stderr)

    def test_unified_fetch_with_psb2_dry_run_subset(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_fetch_") as td:
            out_dir = Path(td) / "psb2"
            cmd = [
                "python3",
                "tools/fetch_psb_datasets.py",
                "--suite",
                "psb2",
                "--out-dir",
                str(out_dir),
                "--problems",
                "basement,dice-game",
                "--dry-run",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("[psb2] fetch basement", proc.stdout)
            self.assertIn("[psb2] fetch dice-game", proc.stdout)
            self.assertIn("total=4", proc.stdout)
            self.assertFalse(out_dir.exists())

    def test_unified_fetch_with_psb2_unknown_problem(self):
        cmd = [
            "python3",
            "tools/fetch_psb_datasets.py",
            "--suite",
            "psb2",
            "--problems",
            "not-a-real-problem",
            "--dry-run",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown PSB2 problem", proc.stderr)


if __name__ == "__main__":
    unittest.main()
