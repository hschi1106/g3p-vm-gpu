from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


Command = Callable[[], int]


def _commands() -> dict[tuple[str, str], tuple[str, Command]]:
    from .datasets.convert_psb import main as convert
    from .datasets.fetch_psb import main as fetch
    from .datasets.materialize_psb import main as materialize
    from .experiments.grammar_profiles import main as grammar_profile
    from .experiments.population_seeds import main as population_seeds
    from .experiments.run_psb import main as run_psb
    from .reports.compare_psb import main as compare
    from .reports.psb_manifest import main as psb_manifest
    from .reports.simple_manifest import main as simple_manifest

    return {
        ("psb", "fetch"): ("fetch PSB datasets", fetch),
        ("psb", "convert"): ("convert PSB JSONL to fitness cases", convert),
        ("psb", "materialize"): ("materialize supported PSB fixtures", materialize),
        ("psb", "run"): ("run the PSB regression matrix", run_psb),
        ("psb", "compare"): ("compare compatible PSB summaries", compare),
        ("benchmark", "population-seeds"): ("write deterministic population seeds", population_seeds),
        ("report", "psb-manifest"): ("write compact PSB evidence", psb_manifest),
        ("report", "simple-manifest"): ("write compact simple-expression evidence", simple_manifest),
        ("grammar", "profile"): ("materialize a grammar profile", grammar_profile),
    }


def help_text() -> str:
    lines = ["usage: g3pvm-tools <group> <command> [arguments]", "", "commands:"]
    for (group, command), (description, _) in _commands().items():
        lines.append(f"  {group} {command:<20} {description}")
    lines.append("")
    lines.append("Run 'g3pvm-tools <group> <command> --help' for command arguments.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["-h"], ["--help"]):
        print(help_text(), end="")
        return 0
    if len(arguments) < 2:
        print("g3pvm-tools: expected <group> <command>", file=sys.stderr)
        return 2
    key = (arguments[0], arguments[1])
    command = _commands().get(key)
    if command is None:
        print(f"g3pvm-tools: unknown command: {' '.join(key)}", file=sys.stderr)
        return 2

    previous = sys.argv
    sys.argv = [f"g3pvm-tools {' '.join(key)}", *arguments[2:]]
    try:
        return command[1]()
    finally:
        sys.argv = previous
