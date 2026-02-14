#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Python unit tests"
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v

echo "[2/4] Build and run native C++ VM tests"
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure

echo "[3/4] vm_py == vm_cpp on JSON fixtures"
PYTHONPATH=python python3 tools/compare_vm_py_cpp_fixtures.py --fixture data/fixtures/bytecode_cases.json

echo "[4/4] interp_py == vm_py == vm_cpp (non-timeout buckets)"
PYTHONPATH=python python3 tools/compare_interp_vm_py_cpp.py --fixture data/fixtures/bytecode_cases.json

echo "triplet checks: OK"
