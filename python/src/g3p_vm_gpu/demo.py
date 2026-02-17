from __future__ import annotations

from g3p_vm_gpu.ast import build_program
from g3p_vm_gpu.errors import Failed, Returned
from g3p_vm_gpu.interp import run_program


def main():
    prog = build_program(
        [
            ("assign", "x", ("const", 0)),
            ("for", "i", 3, [("assign", "x", ("add", ("var", "x"), ("const", 1)))]),
            (
                "if",
                ("and", ("eq", ("var", "x"), ("const", 3)), ("const", True)),
                [("return", ("call", "clip", [("var", "x"), ("const", 0), ("const", 2)]))],
                [("return", ("const", 999))],
            ),
        ]
    )

    env, out = run_program(prog, inputs={}, fuel=10_000)
    if isinstance(out, Returned):
        print("Returned:", out.value)
    elif isinstance(out, Failed):
        print("Error:", out.err.code.value, out.err.message)
    else:
        print("Finished without return. Env:", env)


if __name__ == "__main__":
    main()
