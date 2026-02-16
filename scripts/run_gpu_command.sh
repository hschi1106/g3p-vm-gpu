#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || "$1" != "--" ]]; then
  echo "usage: $0 -- <command> [args...]" >&2
  exit 2
fi
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICES=(0 1)
last_status=1
last_output=""

run_once() {
  local dev="$1"
  shift
  local out
  local rc
  out="$(CUDA_VISIBLE_DEVICES="$dev" "$@" 2>&1)" || rc=$?
  rc=${rc:-0}
  echo "$out"

  if [[ "$out" == *"cuda device unavailable"* || "$out" == *"no CUDA-capable device is detected"* ]]; then
    return 101
  fi
  return "$rc"
}

for dev in "${DEVICES[@]}"; do
  echo "[gpu-wrapper] trying CUDA_VISIBLE_DEVICES=$dev" >&2
  if output="$(run_once "$dev" "$@")"; then
    echo "$output"
    exit 0
  else
    status=$?
    echo "$output"
    if [[ $status -eq 101 ]]; then
      last_status=$status
      last_output="$output"
      continue
    fi
    exit "$status"
  fi

done

echo "[gpu-wrapper] all candidate devices unavailable (tried: ${DEVICES[*]})" >&2
if [[ -n "$last_output" ]]; then
  echo "$last_output"
fi
exit 1
