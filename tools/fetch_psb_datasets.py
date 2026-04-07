#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PSB1_PROBLEMS: tuple[str, ...] = (
    "checksum",
    "collatz-numbers",
    "compare-string-lengths",
    "count-odds",
    "digits",
    "double-letters",
    "even-squares",
    "for-loop-index",
    "grade",
    "last-index-of-zero",
    "median",
    "mirror-image",
    "negative-to-zero",
    "number-io",
    "pig-latin",
    "replace-space-with-newline",
    "scrabble-score",
    "small-or-large",
    "smallest",
    "string-differences",
    "string-lengths-backwards",
    "sum-of-squares",
    "super-anagrams",
    "syllables",
    "vector-average",
    "vectors-summed",
    "wallis-pi",
    "word-stats",
    "x-word-lines",
)

PSB2_PROBLEMS: tuple[str, ...] = (
    "basement",
    "bouncing-balls",
    "bowling",
    "camel-case",
    "coin-sums",
    "cut-vector",
    "dice-game",
    "find-pair",
    "fizz-buzz",
    "fuel-cost",
    "gcd",
    "indices-of-substring",
    "leaders",
    "luhn",
    "mastermind",
    "middle-character",
    "paired-digits",
    "shopping-list",
    "snow-day",
    "solve-boolean",
    "spin-words",
    "square-digits",
    "substitution-cipher",
    "twitter",
    "vector-distance",
)


@dataclass(frozen=True)
class PsbSuiteConfig:
    label: str
    problems: tuple[str, ...]
    base_url: str
    default_datasets_root: str
    known_empty_targets: frozenset[tuple[str, str]] = frozenset()
    problem_width: int = 20


PSB1_CONFIG = PsbSuiteConfig(
    label="psb1",
    problems=PSB1_PROBLEMS,
    base_url="https://psb2-datasets.s3.amazonaws.com/PSB1/datasets",
    default_datasets_root="data/psb1_datasets",
    known_empty_targets=frozenset(
        {
            ("median", "edge"),
            ("number-io", "edge"),
        }
    ),
    problem_width=24,
)

PSB2_CONFIG = PsbSuiteConfig(
    label="psb2",
    problems=PSB2_PROBLEMS,
    base_url="https://psb2-datasets.s3.amazonaws.com/PSB2/datasets",
    default_datasets_root="data/psb2_datasets",
    problem_width=20,
)


def suite_config(suite: str) -> PsbSuiteConfig:
    normalized = suite.strip().lower()
    if normalized == "psb1":
        return PSB1_CONFIG
    if normalized == "psb2":
        return PSB2_CONFIG
    raise ValueError(f"unsupported PSB suite: {suite}")


def split_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def resolve_problems(raw: str, *, suite_label: str, allowed_problems: tuple[str, ...]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(allowed_problems)
    picked = split_csv(raw)
    unknown = [p for p in picked if p not in allowed_problems]
    if unknown:
        raise ValueError(
            f"unknown {suite_label.upper()} problem(s): "
            + ", ".join(unknown)
            + ". Use --problems all or one of: "
            + ", ".join(allowed_problems)
        )
    return picked


def resolve_splits(raw: str) -> List[str]:
    picked = split_csv(raw)
    allowed = {"edge", "random"}
    unknown = [s for s in picked if s not in allowed]
    if unknown:
        raise ValueError("unsupported split(s): " + ", ".join(unknown) + " (allowed: edge,random)")
    if not picked:
        raise ValueError("no split selected")
    return picked


def iter_targets(
    *,
    base_url: str,
    out_dir: Path,
    problems: Iterable[str],
    splits: Iterable[str],
) -> Iterable[tuple[str, str, Path, str]]:
    for problem in problems:
        for split in splits:
            filename = f"{problem}-{split}.json"
            rel = f"{problem}/{filename}"
            url = f"{base_url}/{rel}"
            dest = out_dir / problem / filename
            yield problem, split, dest, url


def download_with_retries(url: str, dest: Path, timeout_sec: int, retries: int, allow_empty: bool) -> None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_sec) as response:
                data = response.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            if not data and not allow_empty:
                raise RuntimeError("empty response body")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_err = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    assert last_err is not None
    raise RuntimeError(f"download failed after {retries} attempts: {url} ({last_err})")


def run_fetch(
    *,
    suite_label: str,
    base_url: str,
    out_dir: Path,
    problems: Iterable[str],
    splits: Iterable[str],
    timeout_sec: int,
    retries: int,
    force: bool,
    dry_run: bool,
    known_empty_targets: frozenset[tuple[str, str]],
    problem_width: int,
) -> int:
    total = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for problem, split, dest, url in iter_targets(base_url=base_url, out_dir=out_dir, problems=problems, splits=splits):
        total += 1
        if dest.exists() and dest.stat().st_size > 0 and not force:
            skipped += 1
            print(f"[{suite_label}] skip  {problem:{problem_width}s} {split:6s} {dest}")
            continue

        print(f"[{suite_label}] fetch {problem:{problem_width}s} {split:6s} {url}")
        if dry_run:
            continue
        try:
            download_with_retries(
                url=url,
                dest=dest,
                timeout_sec=timeout_sec,
                retries=retries,
                allow_empty=(problem, split) in known_empty_targets,
            )
            downloaded += 1
        except Exception as exc:  # keep going to report all failures
            failed += 1
            print(f"[{suite_label}] fail  {problem:{problem_width}s} {split:6s} {exc}", file=sys.stderr)

    print(
        f"[{suite_label}] done total={total} downloaded={downloaded} skipped={skipped} failed={failed} "
        f"out_dir={out_dir}"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PSB1 or PSB2 datasets into the repository data layout. "
            "By default, downloads all problems for the selected suite."
        )
    )
    parser.add_argument("--suite", required=True, choices=("psb1", "psb2"), help="Dataset suite to fetch.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Target datasets directory; defaults to data/psb1_datasets or data/psb2_datasets.",
    )
    parser.add_argument(
        "--problems",
        default="all",
        help="Comma-separated problem names, or 'all' (default: all).",
    )
    parser.add_argument(
        "--splits",
        default="edge,random",
        help="Comma-separated split names to fetch (default: edge,random).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per file on transient errors (default: 3).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="HTTP timeout per request in seconds (default: 60).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if target file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = suite_config(args.suite)
    try:
        problems = resolve_problems(args.problems, suite_label=config.label, allowed_problems=config.problems)
        splits = resolve_splits(args.splits)
    except ValueError as exc:
        print(f"[{config.label}] argument error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir or config.default_datasets_root)
    return run_fetch(
        suite_label=config.label,
        base_url=config.base_url,
        out_dir=out_dir,
        problems=problems,
        splits=splits,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        force=args.force,
        dry_run=args.dry_run,
        known_empty_targets=config.known_empty_targets,
        problem_width=config.problem_width,
    )


if __name__ == "__main__":
    raise SystemExit(main())
