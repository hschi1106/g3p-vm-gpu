#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CPP_CLI="cpp/build/g3pvm_evolve_cli"
FIXTURE_GLOB="data/fixtures/*_1024.json"
POPSIZES=(1024 4096)
GENERATIONS=40
SEED=0
BLOCKSIZE=256
SELECTION_PRESSURE=3
CPP_TIMING="all"
OUTDIR=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_all_fixtures.sh [options]

Run CPU and GPU evolution for every matching fixture.

Options:
  --cpp-cli PATH             C++ evolve CLI path (default: cpp/build/g3pvm_evolve_cli)
  --fixture-glob GLOB        Fixture glob (default: data/fixtures/*_1024.json)
  --popsizes "A B ..."       Space-separated population sizes (default: "1024 4096")
  --generations N            Generations per run (default: 40)
  --seed N                   RNG seed (default: 0)
  --blocksize N              GPU blocksize (default: 256)
  --selection-pressure N     Tournament selection pressure (default: 3)
  --cpp-timing MODE          C++ timing mode (default: all)
  --outdir PATH              Output log directory (default: logs/all_fixtures_<timestamp>)
  --help                     Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpp-cli)
      CPP_CLI="$2"; shift 2 ;;
    --fixture-glob)
      FIXTURE_GLOB="$2"; shift 2 ;;
    --popsizes)
      read -r -a POPSIZES <<<"$2"; shift 2 ;;
    --generations)
      GENERATIONS="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --blocksize)
      BLOCKSIZE="$2"; shift 2 ;;
    --selection-pressure)
      SELECTION_PRESSURE="$2"; shift 2 ;;
    --cpp-timing)
      CPP_TIMING="$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    --help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ! -x "$CPP_CLI" ]]; then
  echo "Missing or non-executable cpp cli: $CPP_CLI" >&2
  exit 2
fi

shopt -s nullglob
FIXTURES=($FIXTURE_GLOB)
shopt -u nullglob
if [[ ${#FIXTURES[@]} -eq 0 ]]; then
  echo "No fixtures matched: $FIXTURE_GLOB" >&2
  exit 2
fi

if [[ -z "$OUTDIR" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="logs/all_fixtures_${TS}"
fi
mkdir -p "$OUTDIR"

echo "[all-fixtures] outdir: $OUTDIR"
echo "[all-fixtures] cpp_cli: $CPP_CLI"
echo "[all-fixtures] fixtures: ${#FIXTURES[@]}"
echo "[all-fixtures] popsizes: ${POPSIZES[*]}"
echo "[all-fixtures] generations=$GENERATIONS seed=$SEED blocksize=$BLOCKSIZE"

for pop in "${POPSIZES[@]}"; do
  for fixture in "${FIXTURES[@]}"; do
    tag="$(basename "$fixture" .json)"

    echo "[all-fixtures] cpu fixture=$tag pop=$pop"
    python3 tools/run_cpp_evolution.py \
      --cases "$fixture" \
      --cpp-cli "$CPP_CLI" \
      --engine cpu \
      --selection-pressure "$SELECTION_PRESSURE" \
      --population-size "$pop" \
      --generations "$GENERATIONS" \
      --seed "$SEED" \
      --cpp-timing "$CPP_TIMING" \
      --log-dir "$OUTDIR" \
      --run-tag "${tag}_pop${pop}_cpu"

    echo "[all-fixtures] gpu fixture=$tag pop=$pop"
    scripts/run_gpu_command.sh -- python3 tools/run_cpp_evolution.py \
      --cases "$fixture" \
      --cpp-cli "$CPP_CLI" \
      --engine gpu \
      --blocksize "$BLOCKSIZE" \
      --selection-pressure "$SELECTION_PRESSURE" \
      --population-size "$pop" \
      --generations "$GENERATIONS" \
      --seed "$SEED" \
      --cpp-timing "$CPP_TIMING" \
      --log-dir "$OUTDIR" \
      --run-tag "${tag}_pop${pop}_gpu"
  done
done

echo "[all-fixtures] done"
