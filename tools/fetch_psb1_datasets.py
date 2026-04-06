#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
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

BASE_URL = "https://psb2-datasets.s3.amazonaws.com/PSB1/datasets"
KNOWN_EMPTY_TARGETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("median", "edge"),
        ("number-io", "edge"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PSB1 datasets into the repository data layout. "
            "By default, downloads all PSB1 problems."
        )
    )
    parser.add_argument(
        "--out-dir",
        default="data/psb1_datasets",
        help="Target datasets directory (default: data/psb1_datasets).",
    )
    parser.add_argument(
        "--problems",
        default="all",
        help="Comma-separated PSB1 problem names, or 'all' (default: all).",
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


def _split_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolve_problems(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(PSB1_PROBLEMS)
    picked = _split_csv(raw)
    unknown = [p for p in picked if p not in PSB1_PROBLEMS]
    if unknown:
        raise ValueError(
            "unknown PSB1 problem(s): "
            + ", ".join(unknown)
            + ". Use --problems all or one of: "
            + ", ".join(PSB1_PROBLEMS)
        )
    return picked


def _resolve_splits(raw: str) -> List[str]:
    picked = _split_csv(raw)
    allowed = {"edge", "random"}
    unknown = [s for s in picked if s not in allowed]
    if unknown:
        raise ValueError("unsupported split(s): " + ", ".join(unknown) + " (allowed: edge,random)")
    if not picked:
        raise ValueError("no split selected")
    return picked


def _iter_targets(out_dir: Path, problems: Iterable[str], splits: Iterable[str]) -> Iterable[tuple[str, str, Path, str]]:
    for problem in problems:
        for split in splits:
            filename = f"{problem}-{split}.json"
            rel = f"{problem}/{filename}"
            url = f"{BASE_URL}/{rel}"
            dest = out_dir / problem / filename
            yield problem, split, dest, url


def _download_with_retries(url: str, dest: Path, timeout_sec: int, retries: int, allow_empty: bool) -> None:
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


def main() -> int:
    args = parse_args()
    try:
        problems = _resolve_problems(args.problems)
        splits = _resolve_splits(args.splits)
    except ValueError as exc:
        print(f"[psb1] argument error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    total = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for problem, split, dest, url in _iter_targets(out_dir, problems, splits):
        total += 1
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            skipped += 1
            print(f"[psb1] skip  {problem:24s} {split:6s} {dest}")
            continue

        print(f"[psb1] fetch {problem:24s} {split:6s} {url}")
        if args.dry_run:
            continue
        try:
            _download_with_retries(
                url=url,
                dest=dest,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                allow_empty=(problem, split) in KNOWN_EMPTY_TARGETS,
            )
            downloaded += 1
        except Exception as exc:
            failed += 1
            print(f"[psb1] fail  {problem:24s} {split:6s} {exc}", file=sys.stderr)

    print(
        f"[psb1] done total={total} downloaded={downloaded} skipped={skipped} failed={failed} "
        f"out_dir={out_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
