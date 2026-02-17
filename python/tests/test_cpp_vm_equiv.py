import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.g3p_vm_gpu.ast import build_program
from src.g3p_vm_gpu.compiler import BytecodeProgram, compile_program
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


ROOT = Path(__file__).resolve().parents[2]


def _encode_value(v):
    if v is None:
        return {"type": "none"}
    if isinstance(v, bool):
        return {"type": "bool", "value": v}
    if isinstance(v, int):
        return {"type": "int", "value": v}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    raise TypeError(f"unsupported value type: {type(v)}")


def _to_json_request(program: BytecodeProgram, fuel: int = 20000):
    one_program = {
        "n_locals": program.n_locals,
        "consts": [_encode_value(c) for c in program.consts],
        "code": [{"op": ins.op, "a": ins.a, "b": ins.b} for ins in program.code],
    }
    return {
        "bytecode_program_inputs": {
            "format_version": "bytecode-json-v0.1",
            "fuel": fuel,
            "programs": [one_program],
            "shared_cases": [[]],
        }
    }


def _program_to_cli_input(program: BytecodeProgram, fuel: int = 20000) -> str:
    return json.dumps(_to_json_request(program, fuel=fuel), ensure_ascii=True)


def _parse_cli_output(text: str):
    first = text.strip().splitlines()[0].strip()
    if first.startswith("ERR "):
        return ("ERR", first.split(" ", 1)[1], None)
    if not first.startswith("OK "):
        raise AssertionError(f"unexpected cpp vm output: {text!r}")

    payload = first[3:]
    if payload == "none":
        return ("OK", None, None)
    t, raw = payload.split(" ", 1)
    if t == "int":
        return ("OK", int(raw), None)
    if t == "float":
        return ("OK", float(raw), None)
    if t == "bool":
        return ("OK", raw == "1", None)
    raise AssertionError(f"unknown cpp value tag: {t}")


class TestCppVMEquiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gxx = shutil.which("g++")
        if gxx is None:
            raise unittest.SkipTest("g++ not found")

        cls._tmpdir = tempfile.TemporaryDirectory(prefix="g3p_cpp_vm_")
        cls._bin = Path(cls._tmpdir.name) / "g3p_vm_cpu_cli"
        cmd = [
            gxx,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "builtins.cpp"),
            str(ROOT / "cpp" / "src" / "vm_cpu.cpp"),
            str(ROOT / "cpp" / "src" / "vm_cpu_cli.cpp"),
            "-o",
            str(cls._bin),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmpdir"):
            cls._tmpdir.cleanup()

    def _run_cpp_vm(self, prog):
        bc = compile_program(prog)
        inp = _program_to_cli_input(bc, fuel=20000)
        proc = subprocess.run(
            [str(self._bin)],
            input=inp,
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        return _parse_cli_output(proc.stdout)

    def _assert_equiv(self, prog):
        bc = compile_program(prog)
        py_out = run_bytecode(bc, {}, fuel=20000)
        cpp_status, cpp_value, _ = self._run_cpp_vm(prog)

        if isinstance(py_out, VMReturn):
            self.assertEqual(cpp_status, "OK")
            if isinstance(py_out.value, float):
                self.assertTrue(math.isclose(py_out.value, cpp_value, rel_tol=1e-12, abs_tol=1e-12))
            else:
                self.assertEqual(py_out.value, cpp_value)
        else:
            self.assertIsInstance(py_out, VMError)
            self.assertEqual(cpp_status, "ERR")
            self.assertEqual(py_out.err.code.value, cpp_value)

    def test_manual_program(self):
        prog = build_program(
            [
                ("assign", "x", ("const", 0)),
                ("for", "i", 5, [("assign", "x", ("add", ("var", "x"), ("const", 2)))]),
                ("return", ("var", "x")),
            ]
        )
        self._assert_equiv(prog)

    def test_none_compare(self):
        self._assert_equiv(build_program([("return", ("eq", ("const", None), ("const", 1)))]))
        self._assert_equiv(build_program([("return", ("ne", ("const", None), ("const", 1)))]))

    def test_no_return(self):
        self._assert_equiv(build_program([("assign", "x", ("const", 1))]))

    def test_fuzz_equivalence(self):
        for i in range(120):
            prog = make_random_program(seed=i, depth=3)
            self._assert_equiv(prog)


if __name__ == "__main__":
    unittest.main()
