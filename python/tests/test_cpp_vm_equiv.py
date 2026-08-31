import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.g3p_vm_gpu.core.ast import Char, FloatList, IntList, NodeKind, StringList, build_program, make_char, make_int_list
from src.g3p_vm_gpu.runtime.compiler import BytecodeProgram, compile_program
from src.g3p_vm_gpu.runtime.vm import ExecError, ExecReturn, exec_bytecode


ROOT = Path(__file__).resolve().parents[2]


def _encode_value(v):
    if isinstance(v, bool):
        return {"type": "bool", "value": v}
    if isinstance(v, int):
        return {"type": "int", "value": v}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, Char):
        return {"type": "char", "value": v.value}
    if isinstance(v, str):
        return {"type": "string", "value": v}
    if isinstance(v, IntList):
        return {"type": "int_list", "value": list(v.items)}
    if isinstance(v, FloatList):
        return {"type": "float_list", "value": list(v.items)}
    if isinstance(v, StringList):
        return {"type": "string_list", "value": list(v.items)}
    raise TypeError(f"unsupported current value type: {type(v)}")


def _encode_instr(ins):
    out = {"op": ins.op}
    if ins.a is not None:
        out["a"] = ins.a
    if ins.b is not None:
        out["b"] = ins.b
    return out


def _encode_phase(phase):
    out = {
        "n_locals": phase.n_locals,
        "consts": [_encode_value(c) for c in phase.consts],
        "code": [_encode_instr(ins) for ins in phase.code],
    }
    if phase.binder_locals:
        out["binder_locals"] = [
            {"name": name, "local": local}
            for name, local in sorted(phase.binder_locals.items())
        ]
    return out


def _dp1_dep_kind_code(kind: NodeKind) -> int:
    if kind in {NodeKind.DP1_BACKWARD1, NodeKind.DP1_BACKWARD2, NodeKind.DP1_BACKWARD3}:
        return -1
    return 1


def _dp2_dep_kind_code(kind: NodeKind) -> int:
    return {
        NodeKind.DP2_CROSS_BACKWARD: 0,
        NodeKind.DP2_CROSS_FORWARD: 1,
        NodeKind.DP2_DIAGONAL_BACKWARD: 2,
        NodeKind.DP2_DIAGONAL_FORWARD: 3,
        NodeKind.DP2_NEIGHBORHOOD_BACKWARD3: 4,
        NodeKind.DP2_NEIGHBORHOOD_FORWARD3: 5,
    }[kind]


def _encode_segments(program: BytecodeProgram):
    segments = {}
    if program.asgp_dc_segments:
        segments["asgp_dc"] = [
            {
                "solve_xs_name": segment.solve_xs_name,
                "solve_n_name": segment.solve_n_name,
                "solve_lo_name": segment.solve_lo_name,
                "divide_n_name": segment.divide_n_name,
                "combine_left_name": segment.combine_left_name,
                "combine_right_name": segment.combine_right_name,
                "solve": _encode_phase(segment.solve),
                "divide": _encode_phase(segment.divide),
                "combine": _encode_phase(segment.combine),
            }
            for segment in program.asgp_dc_segments
        ]
    if program.asgp_dp1d_segments:
        segments["asgp_dp1d"] = [
            {
                "lo": segment.lo,
                "hi": segment.hi,
                "base_state": segment.base_state,
                "boundary_value": _encode_value(segment.boundary_value),
                "dep_kind": _dp1_dep_kind_code(segment.dep_kind),
                "dep_offsets": list(segment.dep_offsets),
                "solve_state_name": segment.solve_state_name,
                "transition_state_name": segment.transition_state_name,
                "transition_dep_names": list(segment.transition_dep_names),
                "solve": _encode_phase(segment.solve),
                "transition": _encode_phase(segment.transition),
            }
            for segment in program.asgp_dp1d_segments
        ]
    if program.asgp_dp2d_segments:
        segments["asgp_dp2d"] = [
            {
                "i_lo": segment.i_lo,
                "i_hi": segment.i_hi,
                "j_lo": segment.j_lo,
                "j_hi": segment.j_hi,
                "base_i": segment.base_i,
                "base_j": segment.base_j,
                "boundary_value": _encode_value(segment.boundary_value),
                "dep_kind": _dp2_dep_kind_code(segment.dep_kind),
                "solve_i_name": segment.solve_i_name,
                "solve_j_name": segment.solve_j_name,
                "transition_i_name": segment.transition_i_name,
                "transition_j_name": segment.transition_j_name,
                "transition_dep_names": list(segment.transition_dep_names),
                "solve": _encode_phase(segment.solve),
                "transition": _encode_phase(segment.transition),
            }
            for segment in program.asgp_dp2d_segments
        ]
    return segments


def _to_json_request(program: BytecodeProgram, fuel: int = 20000):
    one_program = {
        "n_locals": program.n_locals,
        "consts": [_encode_value(c) for c in program.consts],
        "code": [_encode_instr(ins) for ins in program.code],
    }
    segments = _encode_segments(program)
    if segments:
        one_program["segments"] = segments
    return {
        "format_version": "bytecode-json",
        "fuel": fuel,
        "programs": [one_program],
        "shared_cases": [[]],
    }


def _program_to_cli_input(program: BytecodeProgram, fuel: int = 20000) -> str:
    return json.dumps(_to_json_request(program, fuel=fuel), ensure_ascii=True)


def _parse_cli_output(text: str):
    first = text.strip().splitlines()[0].strip()
    if first.startswith("ERR "):
        return ("ERR", first.split(" ", 1)[1])
    if not first.startswith("OK "):
        raise AssertionError(f"unexpected cpp vm output: {text!r}")

    payload = first[3:]
    t, raw = payload.split(" ", 1)
    if t == "int":
        return ("OK", int(raw))
    if t == "float":
        return ("OK", float(raw))
    if t == "bool":
        return ("OK", raw == "1")
    if t == "char":
        return ("OK", make_char(chr(int(raw))))
    if t in {"string_hash48", "int_list_hash48", "float_list_hash48", "string_list_hash48"}:
        toks = raw.split()
        return ("OK", (t, int(toks[0]), int(toks[2])))
    if t == "fallback_token":
        return ("OK", ("fallback_token", int(raw)))
    raise AssertionError(f"unknown cpp value tag: {t}")


class TestCppVMEquiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gxx = shutil.which("g++")
        if gxx is None:
            raise unittest.SkipTest("g++ not found")

        cls._tmpdir = tempfile.TemporaryDirectory(prefix="g3p_cpp_vm_")
        cls._bin = Path(cls._tmpdir.name) / "g3p_vm_cli_harness"
        cmd = [
            gxx,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "runtime" / "payload" / "payload.cpp"),
            str(ROOT / "cpp" / "src" / "runtime" / "bytecode_verify.cpp"),
            str(ROOT / "cpp" / "src" / "runtime" / "cpu" / "builtins_cpu.cpp"),
            str(ROOT / "cpp" / "src" / "runtime" / "cpu" / "execute_bytecode_cpu.cpp"),
            str(ROOT / "cpp" / "tests" / "runtime" / "test_vm_cli_harness.cpp"),
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

    def test_bytecode_harness_rejects_old_format_version(self):
        bc = compile_program(build_program([("return", ("const", 1))]))
        request = _to_json_request(bc)
        request["format_version"] = "bytecode-json-old"
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 2)

    def test_bytecode_harness_rejects_old_fixture_format_version(self):
        request = {
            "format_version": "bytecode-fixture-old",
            "fuel": 20000,
            "program": {"n_locals": 0, "consts": [{"type": "int", "value": 1}], "code": [{"op": "PUSH_CONST", "a": 0}, {"op": "RETURN"}]},
            "cases": [{"inputs": [], "expected": {"type": "int", "value": 1}}],
        }
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 2)

    def test_bytecode_harness_executes_current_fixture_cases(self):
        request = {
            "format_version": "bytecode-fixture",
            "fuel": 20000,
            "program": {
                "n_locals": 1,
                "consts": [{"type": "int", "value": 1}],
                "code": [
                    {"op": "LOAD", "a": 0},
                    {"op": "PUSH_CONST", "a": 0},
                    {"op": "ADD"},
                    {"op": "RETURN"},
                ],
            },
            "cases": [
                {"inputs": [{"idx": 0, "value": {"type": "int", "value": 2}}], "expected": {"type": "int", "value": 3}},
                {"inputs": [{"idx": 0, "value": {"type": "int", "value": -1}}], "expected": {"type": "int", "value": 0}},
            ],
        }
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        self.assertEqual(proc.stdout, "OK fixture cases 2 passed 2 failed 0 error 0\n")

    def test_bytecode_harness_counts_current_fixture_mismatches(self):
        request = {
            "format_version": "bytecode-fixture",
            "fuel": 20000,
            "program": {
                "n_locals": 0,
                "consts": [{"type": "int", "value": 1}],
                "code": [{"op": "PUSH_CONST", "a": 0}, {"op": "RETURN"}],
            },
            "cases": [{"inputs": [], "expected": {"type": "int", "value": 2}}],
        }
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        self.assertEqual(proc.stdout, "OK fixture cases 1 passed 0 failed 1 error 0\n")

    def test_bytecode_harness_decodes_asgp_dc_segments(self):
        request = {
            "format_version": "bytecode-json",
            "fuel": 20000,
            "programs": [
                {
                    "n_locals": 0,
                    "consts": [{"type": "int_list", "value": [1, 2]}],
                    "code": [
                        {"op": "PUSH_CONST", "a": 0},
                        {"op": "ASGP_DC", "a": 0},
                        {"op": "RETURN"},
                    ],
                    "segments": {
                        "asgp_dc": [
                            {
                                "solve_xs_name": 0,
                                "solve_n_name": 1,
                                "solve_lo_name": 2,
                                "divide_n_name": 3,
                                "combine_left_name": 4,
                                "combine_right_name": 5,
                                "solve": {
                                    "n_locals": 3,
                                    "consts": [{"type": "int", "value": 0}],
                                    "code": [
                                        {"op": "LOAD", "a": 0},
                                        {"op": "PUSH_CONST", "a": 0},
                                        {"op": "CALL_BUILTIN", "a": 7, "b": 2},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [
                                        {"name": 0, "local": 0},
                                        {"name": 1, "local": 1},
                                        {"name": 2, "local": 2},
                                    ],
                                },
                                "divide": {
                                    "n_locals": 1,
                                    "consts": [{"type": "int", "value": 1}],
                                    "code": [
                                        {"op": "PUSH_CONST", "a": 0},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [{"name": 3, "local": 0}],
                                },
                                "combine": {
                                    "n_locals": 2,
                                    "consts": [],
                                    "code": [
                                        {"op": "LOAD", "a": 0},
                                        {"op": "LOAD", "a": 1},
                                        {"op": "ADD"},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [
                                        {"name": 4, "local": 0},
                                        {"name": 5, "local": 1},
                                    ],
                                },
                            }
                        ]
                    },
                }
            ],
            "shared_cases": [[]],
        }
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        self.assertEqual(_parse_cli_output(proc.stdout), ("OK", 3))

    def test_bytecode_harness_decodes_asgp_dp1d_segments(self):
        request = {
            "format_version": "bytecode-json",
            "fuel": 20000,
            "programs": [
                {
                    "n_locals": 0,
                    "consts": [{"type": "int", "value": 3}],
                    "code": [
                        {"op": "PUSH_CONST", "a": 0},
                        {"op": "ASGP_DP1D", "a": 0},
                        {"op": "RETURN"},
                    ],
                    "segments": {
                        "asgp_dp1d": [
                            {
                                "lo": 0,
                                "hi": 3,
                                "base_state": 0,
                                "boundary_value": {"type": "int", "value": -99},
                                "dep_kind": -1,
                                "dep_offsets": [1],
                                "solve_state_name": 0,
                                "transition_state_name": 1,
                                "transition_dep_names": [2],
                                "solve": {
                                    "n_locals": 1,
                                    "consts": [],
                                    "code": [
                                        {"op": "LOAD", "a": 0},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [{"name": 0, "local": 0}],
                                },
                                "transition": {
                                    "n_locals": 2,
                                    "consts": [{"type": "int", "value": 1}],
                                    "code": [
                                        {"op": "LOAD", "a": 1},
                                        {"op": "PUSH_CONST", "a": 0},
                                        {"op": "ADD"},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [
                                        {"name": 1, "local": 0},
                                        {"name": 2, "local": 1},
                                    ],
                                },
                            }
                        ]
                    },
                }
            ],
            "shared_cases": [[]],
        }
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        self.assertEqual(_parse_cli_output(proc.stdout), ("OK", 3))

    def test_bytecode_harness_decodes_asgp_dp2d_segments(self):
        request = {
            "format_version": "bytecode-json",
            "fuel": 20000,
            "programs": [
                {
                    "n_locals": 0,
                    "consts": [
                        {"type": "int", "value": 1},
                        {"type": "int", "value": 1},
                    ],
                    "code": [
                        {"op": "PUSH_CONST", "a": 0},
                        {"op": "PUSH_CONST", "a": 1},
                        {"op": "ASGP_DP2D", "a": 0},
                        {"op": "RETURN"},
                    ],
                    "segments": {
                        "asgp_dp2d": [
                            {
                                "i_lo": 0,
                                "i_hi": 1,
                                "j_lo": 0,
                                "j_hi": 1,
                                "base_i": 0,
                                "base_j": 0,
                                "boundary_value": {"type": "int", "value": -99},
                                "dep_kind": 2,
                                "solve_i_name": 0,
                                "solve_j_name": 1,
                                "transition_i_name": 2,
                                "transition_j_name": 3,
                                "transition_dep_names": [4],
                                "solve": {
                                    "n_locals": 2,
                                    "consts": [],
                                    "code": [
                                        {"op": "LOAD", "a": 0},
                                        {"op": "LOAD", "a": 1},
                                        {"op": "ADD"},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [
                                        {"name": 0, "local": 0},
                                        {"name": 1, "local": 1},
                                    ],
                                },
                                "transition": {
                                    "n_locals": 3,
                                    "consts": [],
                                    "code": [
                                        {"op": "LOAD", "a": 2},
                                        {"op": "LOAD", "a": 0},
                                        {"op": "ADD"},
                                        {"op": "LOAD", "a": 1},
                                        {"op": "ADD"},
                                        {"op": "RETURN"},
                                    ],
                                    "binder_locals": [
                                        {"name": 2, "local": 0},
                                        {"name": 3, "local": 1},
                                        {"name": 4, "local": 2},
                                    ],
                                },
                            }
                        ]
                    },
                }
            ],
            "shared_cases": [[]],
        }
        proc = subprocess.run(
            [str(self._bin)],
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
        self.assertEqual(_parse_cli_output(proc.stdout), ("OK", 2))

    def test_compiler_generated_asgp_segments_round_trip_through_current_json(self):
        cases = [
            (
                build_program(
                    [
                        (
                            "return",
                            (
                                "asgp_dc",
                                ("const", make_int_list([1, 2])),
                                ("xs", "n", "lo"),
                                ("call", "index", [("bound", "xs"), ("const", 0)]),
                                "n",
                                ("const", 1),
                                ("r1", "r2"),
                                ("add", ("bound", "r1"), ("bound", "r2")),
                            ),
                        )
                    ]
                ),
                3,
            ),
            (
                build_program(
                    [
                        (
                            "return",
                            (
                                "asgp_dp1d",
                                ("const", 1),
                                (0, 3, 0, 0),
                                "s",
                                ("const", 1),
                                ("backward1", (1,)),
                                ("s", "d1"),
                                ("bound", "d1"),
                            ),
                        )
                    ]
                ),
                1,
            ),
            (
                build_program(
                    [
                        (
                            "return",
                            (
                                "asgp_dp2d",
                                ("const", 1),
                                ("const", 1),
                                (0, 3, 0, 3, 0, 0, 0),
                                ("i", "j"),
                                ("const", 1),
                                "diagonal_backward",
                                ("i", "j", "d1"),
                                ("bound", "d1"),
                            ),
                        )
                    ]
                ),
                1,
            ),
        ]

        for program, expected in cases:
            bytecode = compile_program(program)
            proc = subprocess.run(
                [str(self._bin)],
                input=_program_to_cli_input(bytecode),
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
            )
            self.assertEqual(_parse_cli_output(proc.stdout), ("OK", expected))

    def _assert_equiv(self, prog):
        bc = compile_program(prog)
        py_out = exec_bytecode(bc, {}, fuel=20000)
        cpp_status, cpp_value = self._run_cpp_vm(prog)

        if isinstance(py_out, ExecReturn):
            self.assertEqual(cpp_status, "OK")
            if isinstance(py_out.value, float):
                self.assertTrue(math.isclose(py_out.value, cpp_value, rel_tol=1e-12, abs_tol=1e-12))
            elif isinstance(py_out.value, (str, IntList, FloatList, StringList)):
                self.assertIsInstance(cpp_value, tuple)
            else:
                self.assertEqual(py_out.value, cpp_value)
        else:
            self.assertIsInstance(py_out, ExecError)
            self.assertEqual(cpp_status, "ERR")
            self.assertEqual(py_out.err.code.value, cpp_value)

    def test_manual_program(self):
        prog = build_program(
            [
                ("assign", "x", ("const", 0)),
                ("for", "i", ("const", 5), [("assign", "x", ("add", ("var", "x"), ("const", 2)))]),
                ("return", ("var", "x")),
            ]
        )
        self._assert_equiv(prog)

    def test_protected_integer_builtins(self):
        self._assert_equiv(build_program([("return", ("call", "idiv0", [("const", 7), ("const", 0)]))]))
        self._assert_equiv(build_program([("return", ("call", "imod0", [("const", 7), ("const", 0)]))]))
        self._assert_equiv(build_program([("return", ("call", "idiv0", [("const", 7), ("const", 2)]))]))

    def test_char_and_exact_equality(self):
        self._assert_equiv(build_program([("return", ("eq", ("const", make_char("a")), ("const", make_char("a"))))]))
        self._assert_equiv(build_program([("return", ("ne", ("const", make_char("a")), ("const", "a")))]))
        self._assert_equiv(build_program([("return", ("call", "ord", [("const", make_char("A"))]))]))
        self._assert_equiv(build_program([("return", ("call", "to_lower", [("const", make_char("A"))]))]))

    def test_string_index_returns_char(self):
        self._assert_equiv(build_program([("return", ("call", "index", [("const", "abcdef"), ("const", 2)]))]))

    def test_direct_list_builtins_return_scalar(self):
        self._assert_equiv(build_program([("return", ("call", "len", [("const", IntList((1, 2, 3)))]))]))
        self._assert_equiv(
            build_program(
                [
                    (
                        "return",
                        (
                            "call",
                            "len",
                            [("call", "append", [("const", FloatList((1.0, 2.0))), ("const", 3.0)])],
                        ),
                    )
                ]
            )
        )
        self._assert_equiv(
            build_program([("return", ("call", "index", [("const", FloatList((1.0, 2.5))), ("const", -1)]))])
        )


if __name__ == "__main__":
    unittest.main()
