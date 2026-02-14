import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.g3p_vm_gpu.ast import Assign, BOp, Binary, Block, Const, ForRange, Return, Var
from src.g3p_vm_gpu.compiler import BytecodeProgram, compile_program
from src.g3p_vm_gpu.fuzz import make_random_program
from src.g3p_vm_gpu.vm import VMError, VMReturn, run_bytecode


ROOT = Path(__file__).resolve().parents[2]


def _encode_value(v):
    if v is None:
        return "none"
    if isinstance(v, bool):
        return f"bool {1 if v else 0}"
    if isinstance(v, int):
        return f"int {v}"
    if isinstance(v, float):
        return f"float {repr(v)}"
    raise TypeError(f"unsupported value type: {type(v)}")


def _program_to_cli_input(program: BytecodeProgram, fuel: int = 20000) -> str:
    lines = []
    lines.append(f"FUEL {fuel}")
    lines.append(f"N_LOCALS {program.n_locals}")
    lines.append(f"N_CONSTS {len(program.consts)}")
    for c in program.consts:
        lines.append(f"CONST {_encode_value(c)}")
    lines.append(f"N_CODE {len(program.code)}")
    for ins in program.code:
        a = "x" if ins.a is None else str(ins.a)
        b = "x" if ins.b is None else str(ins.b)
        lines.append(f"INS {ins.op} {a} {b}")
    lines.append("N_INPUTS 0")
    return "\n".join(lines) + "\n"


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

    def _run_cpp_vm(self, prog: Block):
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

    def _assert_equiv(self, prog: Block):
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
        prog = Block(
            [
                Assign("x", Const(0)),
                ForRange("i", 5, Block([Assign("x", Binary(BOp.ADD, Var("x"), Const(2)))])),
                Return(Var("x")),
            ]
        )
        self._assert_equiv(prog)

    def test_none_compare(self):
        self._assert_equiv(Block([Return(Binary(BOp.EQ, Const(None), Const(1)))]))
        self._assert_equiv(Block([Return(Binary(BOp.NE, Const(None), Const(1)))]))

    def test_no_return(self):
        self._assert_equiv(Block([Assign("x", Const(1))]))

    def test_fuzz_equivalence(self):
        for i in range(120):
            prog = make_random_program(seed=i, depth=3)
            self._assert_equiv(prog)


if __name__ == "__main__":
    unittest.main()
